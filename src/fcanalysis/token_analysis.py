"""Bounded streaming execution of full-conversation token accounting.

Only messages and tools enter the renderer. Source metadata remains useful for
aggregate attribution, but raw source objects and annotations are not copied
to tokenizer workers. A worker loads one tokenizer, renders bounded batches,
and returns aggregate counts rather than token arrays. Input order and integer
totals are independent of worker count. The largest single conversation must
still fit in one worker; conversations are never truncated to fit a batch.
"""

from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
import multiprocessing
import os
from typing import Any

from .format import ConversationSample
from .tokenization import Qwen35Counter, TokenCounts


@dataclass(frozen=True, slots=True)
class TokenAnalysisConfig:
    """Execution settings; template semantics belong to Qwen35Counter."""

    tokenizer_path: str
    workers: int = 1
    batch_characters: int = 1_000_000
    batch_samples: int = 128
    pending_batches: int | None = None

    def __post_init__(self) -> None:
        for name in ("workers", "batch_characters", "batch_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.pending_batches is not None and (
            type(self.pending_batches) is not int or self.pending_batches < 1
        ):
            raise ValueError("pending_batches must be a positive integer")
        if not self.tokenizer_path:
            raise ValueError("tokenizer_path must name prepared tokenizer assets")


@dataclass(slots=True)
class TokenAnalysisReport:
    samples: int = 0
    tokens: TokenCounts = field(default_factory=TokenCounts)
    by_dataset: dict[str, TokenCounts] = field(default_factory=dict)
    samples_by_dataset: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tokens"] = _counts_dict(self.tokens)
        result["by_dataset"] = {
            dataset: _counts_dict(counts) for dataset, counts in self.by_dataset.items()
        }
        return result


def _counts_dict(counts: TokenCounts) -> dict[str, int]:
    return {
        **asdict(counts),
        "total_tokens": counts.total_tokens,
        "trainable_tokens": counts.trainable_tokens,
    }


def _add(target: TokenCounts, other: TokenCounts) -> None:
    for item in fields(TokenCounts):
        setattr(
            target, item.name, getattr(target, item.name) + getattr(other, item.name)
        )


def _characters(value: Any) -> int:
    """Cheap bounded-batch sizing without repeatedly serializing a sample."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return 2 + sum(len(key) + 4 + _characters(item) for key, item in value.items())
    if isinstance(value, list):
        return 2 + sum(1 + _characters(item) for item in value)
    return 16


def _batches(
    samples: Iterator[ConversationSample], config: TokenAnalysisConfig
) -> Generator[list[ConversationSample], None, None]:
    batch: list[ConversationSample] = []
    characters = 0
    for sample in samples:
        size = _characters(sample.messages) + _characters(sample.tools)
        if batch and (
            len(batch) >= config.batch_samples
            or characters + size > config.batch_characters
        ):
            yield batch
            batch = []
            characters = 0
        batch.append(replace(sample, annotations={}, raw={}))
        characters += size
    if batch:
        yield batch


_worker_counter: Qwen35Counter | None = None


def _load_counter(path: str) -> Qwen35Counter:
    # Outer processes already parallelize rendering and tokenization. Prevent
    # each Rust tokenizer from spawning another full-machine thread pool.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return Qwen35Counter(tokenizer)


def _initialize_worker(path: str) -> None:
    global _worker_counter
    _worker_counter = _load_counter(path)


def _count_batch(
    samples: list[ConversationSample], counter: Qwen35Counter
) -> TokenAnalysisReport:
    report = TokenAnalysisReport()
    for sample, counts in zip(samples, counter.count_batch(samples), strict=True):
        report.samples += 1
        _add(report.tokens, counts)
        _add(report.by_dataset.setdefault(sample.dataset, TokenCounts()), counts)
        report.samples_by_dataset[sample.dataset] = (
            report.samples_by_dataset.get(sample.dataset, 0) + 1
        )
    return report


def _worker_batch(samples: list[ConversationSample]) -> TokenAnalysisReport:
    if _worker_counter is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return _count_batch(samples, _worker_counter)


def _merge(target: TokenAnalysisReport, other: TokenAnalysisReport) -> None:
    target.samples += other.samples
    _add(target.tokens, other.tokens)
    for dataset, counts in other.by_dataset.items():
        _add(target.by_dataset.setdefault(dataset, TokenCounts()), counts)
        target.samples_by_dataset[dataset] = (
            target.samples_by_dataset.get(dataset, 0)
            + other.samples_by_dataset[dataset]
        )


def analyze_tokens(
    samples: Iterable[ConversationSample],
    config: TokenAnalysisConfig,
    *,
    progress: Callable[[TokenAnalysisReport], None] | None = None,
) -> TokenAnalysisReport:
    """Count every supplied conversation once, without padding or truncation.

    ``progress`` receives cumulative completed counts in the caller process.
    At most ``pending_batches`` batches are queued (default twice the worker
    count); each is bounded by both input text size and conversation count.
    Failures propagate and close the input, rather than returning partial totals
    as a complete result. No remote tokenizer downloads occur in workers.
    """
    report = TokenAnalysisReport()
    iterator = iter(samples)
    batches = _batches(iterator, config)
    try:
        if config.workers == 1:
            counter = _load_counter(config.tokenizer_path)
            for batch in batches:
                _merge(report, _count_batch(batch, counter))
                if progress is not None:
                    progress(report)
            return report

        pending: deque[Future[TokenAnalysisReport]] = deque()
        limit = config.pending_batches or config.workers * 2
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(config.tokenizer_path,),
        ) as executor:
            exhausted = False
            try:
                while pending or not exhausted:
                    while not exhausted and len(pending) < limit:
                        batch = next(batches, None)
                        if batch is None:
                            exhausted = True
                        else:
                            pending.append(executor.submit(_worker_batch, batch))
                    if pending:
                        _merge(report, pending.popleft().result())
                        if progress is not None:
                            progress(report)
            finally:
                for future in pending:
                    future.cancel()
        return report
    finally:
        batches.close()
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
