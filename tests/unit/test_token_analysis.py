"""Bounded execution, exact aggregation and cleanup across tokenizer workers."""

from concurrent.futures import Future
from copy import deepcopy
from dataclasses import asdict, replace
import os
from typing import Any

import pytest

from fcanalysis.format import ConversationSample
from fcanalysis import token_analysis as analysis
from fcanalysis.tokenization import TokenCounts


def _sample(number=1, dataset="a", content="answer"):
    return ConversationSample(
        messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": content},
        ],
        tools=[],
        dataset=dataset,
        sample_id=number,
        annotations={"must_not_reach_counter": True},
        raw={"must_not_reach_counter": ["large source"]},
    )


class _Source:
    def __init__(self, samples, fail_at=None):
        self.samples = samples
        self.yielded = 0
        self.closed = 0
        self.fail_at = fail_at

    def __iter__(self):
        return self

    def __next__(self):
        if self.yielded == self.fail_at:
            raise RuntimeError("source failed")
        if self.yielded == len(self.samples):
            raise StopIteration
        sample = self.samples[self.yielded]
        self.yielded += 1
        return sample

    def close(self):
        self.closed += 1


class _Counter:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def count_batch(self, samples):
        self.calls.append([sample.sample_id for sample in samples])
        result = []
        for sample in samples:
            assert sample.raw == {}
            assert sample.annotations == {}
            if sample.sample_id == self.fail_on:
                raise RuntimeError("counter failed")
            value = sample.sample_id
            result.append(
                TokenCounts(
                    reasoning_tokens=value,
                    tool_call_tokens=2 * value,
                    assistant_prose_tokens=3 * value,
                    assistant_formatting_tokens=4 * value,
                    context_tokens=5 * value,
                    samples=1,
                )
            )
        return result


class _DeferredFuture(Future):
    def __init__(self, function, arguments, before_result):
        super().__init__()
        self.function = function
        self.arguments = arguments
        self.before_result = before_result
        self.cancel_called = False

    def result(self, timeout=None):
        self.before_result()
        if not self.done() and self.set_running_or_notify_cancel():
            try:
                self.set_result(self.function(*self.arguments))
            except BaseException as exc:
                self.set_exception(exc)
        return super().result(timeout)

    def cancel(self):
        self.cancel_called = True
        return super().cancel()


@pytest.fixture
def fake_executor(monkeypatch):
    instances = []

    class Executor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.futures = []
            self.closed = False
            self.before_result = lambda: None
            self.maximum_pending = 0
            instances.append(self)

        def __enter__(self):
            self.kwargs["initializer"](*self.kwargs["initargs"])
            return self

        def __exit__(self, *_):
            self.closed = True

        def submit(self, function, *arguments):
            future = _DeferredFuture(function, arguments, lambda: self.before_result())
            self.futures.append(future)
            self.maximum_pending = max(
                self.maximum_pending,
                sum(not item.done() for item in self.futures),
            )
            return future

    monkeypatch.setattr(analysis, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(analysis, "_worker_counter", None)
    return instances


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workers", 0),
        ("workers", True),
        ("workers", 1.5),
        ("batch_characters", -1),
        ("batch_characters", False),
        ("batch_samples", 0),
        ("batch_samples", 2.0),
        ("pending_batches", 0),
        ("pending_batches", False),
        ("tokenizer_path", ""),
    ],
)
def test_invalid_execution_limits_fail_before_input(field, value):
    options: dict[str, Any] = {"tokenizer_path": "local", field: value}
    with pytest.raises(ValueError):
        analysis.TokenAnalysisConfig(**options)


def test_batches_enforce_sample_limit_and_character_boundary():
    sample = _sample()
    size = analysis._characters(sample.messages) + analysis._characters(sample.tools)
    rows = [_sample(number) for number in range(1, 6)]
    config = analysis.TokenAnalysisConfig(
        "local", batch_samples=2, batch_characters=size * 10
    )
    assert [len(batch) for batch in analysis._batches(iter(rows), config)] == [2, 2, 1]
    assert [
        len(batch)
        for batch in analysis._batches(
            iter(rows), replace(config, batch_samples=10, batch_characters=size * 2)
        )
    ] == [2, 2, 1]
    assert [
        len(batch)
        for batch in analysis._batches(
            iter(rows), replace(config, batch_samples=10, batch_characters=size * 2 - 1)
        )
    ] == [1, 1, 1, 1, 1]


def test_oversized_conversation_is_preserved_as_one_batch():
    giant = _sample(2, content="巨大" * 1000)
    rows = [_sample(1), giant, _sample(3)]
    config = analysis.TokenAnalysisConfig("local", batch_characters=100)
    batches = list(analysis._batches(iter(rows), config))
    assert [[sample.sample_id for sample in batch] for batch in batches] == [
        [1],
        [2],
        [3],
    ]
    assert batches[1][0].messages == giant.messages


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_aggregation_preserves_order_metadata_and_all_rows(
    monkeypatch, fake_executor, workers
):
    rows = [_sample(1), _sample(2, "b"), _sample(3), _sample(4, "b")]
    originals = deepcopy(rows)
    source = _Source(rows)
    counter = _Counter()
    loads = []

    def load(path):
        loads.append(path)
        return counter

    monkeypatch.setattr(analysis, "_load_counter", load)
    progress = []
    report = analysis.analyze_tokens(
        source,
        analysis.TokenAnalysisConfig("local", workers=workers, batch_samples=1),
        progress=lambda value: progress.append(value.as_dict()),
    )
    assert source.yielded == 4
    assert source.closed == 1
    assert counter.calls == [[1], [2], [3], [4]]
    assert [item["samples"] for item in progress] == [1, 2, 3, 4]
    assert [item["tokens"]["reasoning_tokens"] for item in progress] == [1, 3, 6, 10]
    assert report.samples == report.tokens.samples == 4
    assert report.samples_by_dataset == {"a": 2, "b": 2}
    assert list(report.by_dataset) == ["a", "b"]
    assert report.by_dataset["a"].reasoning_tokens == 4
    assert report.by_dataset["b"].reasoning_tokens == 6
    assert report.tokens.reasoning_tokens == 10
    assert report.tokens.tool_call_tokens == 20
    assert report.tokens.assistant_prose_tokens == 30
    assert report.tokens.assistant_formatting_tokens == 40
    assert report.tokens.context_tokens == 50
    assert report.as_dict()["tokens"]["total_tokens"] == 150
    assert report.as_dict()["tokens"]["trainable_tokens"] == 100
    assert [asdict(row) for row in rows] == [asdict(row) for row in originals]
    assert loads == ["local"]
    if workers > 1:
        assert fake_executor[0].closed
        assert fake_executor[0].kwargs["max_workers"] == workers
        assert fake_executor[0].kwargs["mp_context"].get_start_method() == "spawn"


@pytest.mark.parametrize("pending", [1, 3, None])
def test_executor_bounds_inflight_batches_and_source_lookahead(
    monkeypatch, fake_executor, pending
):
    source = _Source([_sample(n) for n in range(1, 21)])
    counter = _Counter()
    monkeypatch.setattr(analysis, "_load_counter", lambda _: counter)
    first_result_reads = []
    original = analysis._worker_batch

    def worker(batch):
        if not first_result_reads:
            first_result_reads.append(source.yielded)
        return original(batch)

    monkeypatch.setattr(analysis, "_worker_batch", worker)
    config = analysis.TokenAnalysisConfig(
        "local", workers=2, batch_samples=2, pending_batches=pending
    )
    report = analysis.analyze_tokens(source, config)
    limit = pending or 4
    assert fake_executor[0].maximum_pending <= limit
    assert first_result_reads == [limit * 2 + 1]
    assert report.samples == 20
    assert source.closed == 1


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("failure", ["counter", "progress", "source"])
def test_failures_close_input_and_cancel_unfinished_work(
    monkeypatch, fake_executor, workers, failure
):
    source = _Source(
        [_sample(n) for n in range(1, 11)], fail_at=2 if failure == "source" else None
    )
    counter = _Counter(fail_on=1 if failure == "counter" else None)
    monkeypatch.setattr(analysis, "_load_counter", lambda _: counter)

    def progress(_):
        if failure == "progress":
            raise RuntimeError("progress failed")

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        analysis.analyze_tokens(
            source,
            analysis.TokenAnalysisConfig(
                "local", workers=workers, batch_samples=1, pending_batches=3
            ),
            progress=progress,
        )
    assert source.closed == 1
    if workers > 1:
        executor = fake_executor[0]
        assert executor.closed
        assert all(future.done() for future in executor.futures)
        assert any(future.cancel_called for future in executor.futures)


@pytest.mark.parametrize("failure", ["counter_load", "executor_start"])
def test_initialization_failure_closes_input_before_first_read(monkeypatch, failure):
    source = _Source([_sample()])

    def fail(*_args, **_kwargs):
        raise RuntimeError("initialization failed")

    if failure == "counter_load":
        monkeypatch.setattr(analysis, "_load_counter", fail)
        workers = 1
    else:
        monkeypatch.setattr(analysis, "ProcessPoolExecutor", fail)
        workers = 2
    with pytest.raises(RuntimeError, match="initialization failed"):
        analysis.analyze_tokens(source, analysis.TokenAnalysisConfig("local", workers))
    assert source.yielded == 0
    assert source.closed == 1


@pytest.mark.parametrize("extra", [False, True])
def test_counter_cardinality_mismatch_is_not_a_success(monkeypatch, extra):
    class BadCounter:
        def count_batch(self, _samples):
            return [TokenCounts(), TokenCounts()] if extra else []

    monkeypatch.setattr(analysis, "_load_counter", lambda _: BadCounter())
    source = _Source([_sample()])
    with pytest.raises(ValueError, match="zip"):
        analysis.analyze_tokens(source, analysis.TokenAnalysisConfig("local"))
    assert source.closed == 1


def test_empty_input_returns_zero_report_and_closes(monkeypatch):
    counter = _Counter()
    monkeypatch.setattr(analysis, "_load_counter", lambda _: counter)
    source = _Source([])
    report = analysis.analyze_tokens(source, analysis.TokenAnalysisConfig("local"))
    assert report.samples == report.tokens.total_tokens == 0
    assert report.by_dataset == report.samples_by_dataset == {}
    assert counter.calls == []
    assert source.closed == 1


@pytest.mark.skipif(
    not os.environ.get("FCANALYSIS_TEST_TOKENIZER_PATH"),
    reason="set FCANALYSIS_TEST_TOKENIZER_PATH for the bounded cached-tokenizer check",
)
def test_cached_tokenizer_counts_are_identical_with_one_and_two_workers():
    path = os.environ["FCANALYSIS_TEST_TOKENIZER_PATH"]
    rows = [_sample(1), _sample(2, "b", "A longer answer."), _sample(3)]
    rows[0].messages[-1]["reasoning_content"] = "A short exact reasoning trace."
    config = analysis.TokenAnalysisConfig(path, batch_samples=1, pending_batches=2)
    one = analysis.analyze_tokens(iter(rows), config)
    two = analysis.analyze_tokens(iter(rows), replace(config, workers=2))
    assert one.as_dict() == two.as_dict()
    assert one.samples == 3
    assert one.tokens.reasoning_tokens > 0
