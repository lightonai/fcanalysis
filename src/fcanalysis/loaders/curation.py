"""Scoped, disk-backed curation of validated canonical trajectories.

Adapters supply final-view samples and independently established call/result
bindings. Comparisons never infer parallel pairing from cardinality alone.
Levels 1, 1.5, and 2 run cumulatively; Levels 3--5 describe the final population
in aggregate and never select or annotate a row. Complete comparison bytes
exist only in a temporary SQLite database, without hashes or persisted keys.

Ingestion accepts a one-pass iterable and retains no collection of sample
objects. SQLite groups exact comparison values on disk, while its page cache
is bounded. Winner payloads are separate from scope-indexed audit values, so
aggregate queries do not repeatedly traverse large raw payloads. A selected
sample is spooled losslessly and returned with all its
original values, including ``raw``, source order, and metadata. Object identity
does not survive disk spooling. All temporary resources are removed on normal
completion, exceptions, or explicit/context-manager close. Callers that stop
iteration early must close the result or use its context manager.

Independent dataset/subset/split scopes may be interleaved or processed in
separate workers. Arbitrary fragments of one scope cannot be independently
curated and combined as if the original candidates had been curated once:
cumulative Level 1/1.5 selection must precede Level 2. Complete exact tool
environments are a safe partition because all deletion keys include them;
their Level-4 audit still requires a global count across retained partitions.
A MixtureScope explicitly authorizes comparison of a supplied retained union
across datasets, without changing its original sample dataset fields.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
import pickle
import re
import sqlite3
from tempfile import TemporaryDirectory
from typing import Any, Literal, Protocol
import zlib

import orjson

from fcanalysis.deduplication import EquivalenceAuditCounts
from fcanalysis.format import ConversationSample

from .normalization import json_bytes, parse_json


type CurationLevel = Literal["level_1", "level_1_5", "level_2"]
_LEVELS: tuple[CurationLevel, ...] = ("level_1", "level_1_5", "level_2")
_STRUCTURED_MARKERS = (
    "`",
    "~",
    "{",
    "}",
    "[",
    "]",
    "<",
    ">",
    "|",
    "\\",
    "'",
    '"',
    "(",
    ")",
)
_LITERAL_WORDS = frozenset(
    {"true", "false", "null", "True", "False", "None", "Ellipsis", "..."}
)
_STRUCTURED_LINE = re.compile(r"(?:#{1,6}|>|[-+*]|\d+[.)])\s")
_CODE_LIKE = re.compile(
    r"(?:"
    r"(?:^|\s)(?:class|const|def|enum|export|from|function|import|interface|let|"
    r"package|return|select|struct|type|using|var)\s"
    r"|[A-Za-z_]\w*\s*\(|=>|==|!=|:=|::|;|="
    r")",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class CurationScope:
    """Explicit deletion boundary; no field may be inferred or left unnamed."""

    dataset: str
    subset: str
    split: str

    def __post_init__(self) -> None:
        for name in ("dataset", "subset", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"curation scope {name} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class MixtureScope:
    """Explicit permission to compare datasets inside one named final union.

    This scope applies to the supplied retained population, not the original
    candidates that its source loaders may already have removed.
    """

    name: str
    split: str

    def __post_init__(self) -> None:
        for name in ("name", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"mixture scope {name} must be a nonempty string")


type ComparisonScope = CurationScope | MixtureScope


@dataclass(frozen=True, slots=True)
class BoundCallBatch:
    """Source-proven bindings, expressed in final-view message coordinates.

    ``result_indices[i]`` binds call ``i`` in ``assistant_index`` to exactly
    one result. ``parallel`` explicitly attests that the source protocol makes
    this batch unordered. Only an attested parallel batch may be permuted for
    comparison. Results must occupy the contiguous block after the assistant;
    a protocol with other event structure needs a separately audited adapter.
    """

    assistant_index: int
    result_indices: tuple[int, ...]
    parallel: bool = False


@dataclass(slots=True)
class CurationInput:
    sample: ConversationSample
    batches: tuple[BoundCallBatch, ...] = ()
    # A pipeline may reuse already parsed arguments. This cache is ephemeral
    # and is never included in the spooled sample or a comparison value.
    parsed_arguments: Mapping[tuple[int, int], dict[str, Any]] | None = None


class ComparisonView(Protocol):
    """Ephemeral source-aware comparison, separate from the returned sample.

    A textual action protocol must retain its actions and observations when
    omitting assistant prose or user requests. The ordinary native-call view
    cannot establish those semantics. Adapters can supply this interface to
    the same disk-backed selector; no comparison values enter model input,
    sample annotations or persistent identity metadata.
    """

    def key(self, level: str) -> bytes: ...

    def has_calls(self) -> bool: ...

    def assistant_characters(self) -> int: ...


@dataclass(frozen=True, slots=True)
class CurationConfig:
    max_level: CurationLevel | None = "level_2"
    audit: bool = True
    temporary_directory: str | Path | None = None

    def __post_init__(self) -> None:
        if self.max_level is not None and self.max_level not in _LEVELS:
            raise ValueError(f"unsupported curation level: {self.max_level!r}")


@dataclass(slots=True)
class CurationStageCounts:
    input_samples: int = 0
    output_samples: int = 0
    removed_samples: int = 0
    duplicate_groups: int = 0


@dataclass(slots=True)
class ScopedCurationReport:
    scope: ComparisonScope
    input_samples: int = 0
    output_samples: int = 0
    removed_samples: int = 0
    stages: dict[str, CurationStageCounts] = field(default_factory=dict)
    audit: dict[str, EquivalenceAuditCounts] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json(value: Any) -> bytes:
    try:
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
    except TypeError, ValueError:
        return json_bytes(value)


def _frame(parts: Iterable[bytes]) -> bytes:
    """Unambiguous framing of exact components; this is not a digest."""
    return b"".join(len(part).to_bytes(8, "big") + part for part in parts)


def tool_environment_bytes(tools: Iterable[dict[str, Any]]) -> bytes:
    """Exact ephemeral complete-definition comparison, without parsing calls."""
    return _frame(sorted(_json(tool) for tool in tools))


def tool_name_trace_bytes(
    sample: ConversationSample, batches: Iterable[BoundCallBatch] = ()
) -> bytes:
    """Level-4 comparison using already established optional batch contracts.

    This projection does not establish or validate source linkage. An empty
    binding iterable preserves every batch's call order.
    """
    parallel = {batch.assistant_index for batch in batches if batch.parallel}
    trace = []
    for index, message in enumerate(sample.messages):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        names = [call["function"]["name"] for call in message["tool_calls"]]
        if index in parallel:
            names.sort()
        trace.append(names)
    return _json(trace)


def _plain_prose(text: str) -> str:
    """Normalize line endings and blank outer lines only in safe plain prose."""
    candidate = text.replace("\r\n", "\n").replace("\r", "\n")
    stripped = candidate.strip()
    # Cheap lexical abstention covers both JSON and Python literal forms;
    # parsing arbitrary model prose as Python would add cost and uncertainty.
    # Quotes/containers are handled below. Numeric-leading prose also abstains
    # deliberately: false negatives retain extra rows without deleting code.
    if len(stripped) <= 8 and stripped in _LITERAL_WORDS:
        return text
    start = 1 if stripped[:1] in ("+", "-") else 0
    numeric_start = stripped[start : start + 2]
    if numeric_start[:1].isdigit() or (
        numeric_start.startswith(".") and numeric_start[1:].isdigit()
    ):
        return text
    if any(marker in candidate for marker in _STRUCTURED_MARKERS):
        return text
    if _CODE_LIKE.search(candidate):
        return text
    lines = candidate.split("\n")
    for line in lines:
        if line.strip() and (line[0].isspace() or _STRUCTURED_LINE.match(line)):
            return text
    start, stop = 0, len(lines)
    while start < stop and not lines[start].strip():
        start += 1
    while stop > start and not lines[stop - 1].strip():
        stop -= 1
    return "\n".join(lines[start:stop])


class _ComparisonView:
    """One-row lazy comparison cache, discarded immediately after ingestion."""

    def __init__(self, record: CurationInput) -> None:
        self.sample = record.sample
        self.messages = record.sample.messages
        self.bindings: dict[int, BoundCallBatch] = {}
        occupied: set[int] = set()
        for batch in record.batches:
            index = batch.assistant_index
            if index in self.bindings or not 0 <= index < len(self.messages):
                raise ValueError("duplicate or out-of-range call batch binding")
            message = self.messages[index]
            calls = message.get("tool_calls")
            if message.get("role") != "assistant" or not calls:
                raise ValueError("call batch binding does not point to assistant calls")
            expected = set(range(index + 1, index + len(calls) + 1))
            if (
                len(batch.result_indices) != len(calls)
                or set(batch.result_indices) != expected
                or occupied.intersection(batch.result_indices)
                or any(
                    result_index >= len(self.messages)
                    or self.messages[result_index].get("role") != "tool"
                    for result_index in batch.result_indices
                )
            ):
                raise ValueError(
                    "call batch binding must cover exactly its result block"
                )
            occupied.update(batch.result_indices)
            self.bindings[index] = batch

        # Parse each argument at most once here, or reuse the validation stage's
        # cache. Full call bodies and every other visible field stay exact.
        for index, message in enumerate(self.messages):
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            calls = []
            for call_index, call in enumerate(message["tool_calls"]):
                function = call["function"]
                if (
                    record.parsed_arguments is not None
                    and (index, call_index) in record.parsed_arguments
                ):
                    arguments = record.parsed_arguments[index, call_index]
                else:
                    arguments = function["arguments"]
                    if isinstance(arguments, str):
                        arguments = parse_json(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("curation requires validated object arguments")
                calls.append(
                    {
                        **call,
                        "function": {
                            **function,
                            "arguments": _json(arguments).decode(),
                        },
                    }
                )
            if self.messages is record.sample.messages:
                self.messages = list(self.messages)
            self.messages[index] = {**message, "tool_calls": calls}

        self._tools: bytes | None = None
        self._ordered_messages: list[dict[str, Any]] | None = None
        self._keys: dict[str, bytes] = {}
        self._message_keys: dict[tuple[int, str], bytes] = {}

    def tools(self) -> bytes:
        # Definitions stay exact even at Level 1.5. A schema can contain
        # structured literals under default/enum/const or arbitrary property
        # names, so recursive text-field matching is not safe prose evidence.
        if self._tools is None:
            self._tools = tool_environment_bytes(self.sample.tools)
        return self._tools

    def ordered_messages(self) -> list[dict[str, Any]]:
        if self._ordered_messages is not None:
            return self._ordered_messages
        if not any(batch.parallel for batch in self.bindings.values()):
            self._ordered_messages = self.messages
            return self._ordered_messages
        messages = list(self.messages)
        for index, batch in self.bindings.items():
            if not batch.parallel:
                continue
            assistant = self.messages[index]
            units = [
                (call, self.messages[result_index])
                for call, result_index in zip(
                    assistant["tool_calls"], batch.result_indices, strict=True
                )
            ]
            # Sorting the entire pair preserves same-name calls, multiplicity,
            # argument arrays, and the result associated with each call.
            units.sort(key=lambda unit: _frame((_json(unit[0]), _json(unit[1]))))
            messages[index] = {**assistant, "tool_calls": [call for call, _ in units]}
            for result_index, (_, result) in enumerate(units, start=index + 1):
                messages[result_index] = result
        self._ordered_messages = messages
        return messages

    def key(self, level: str) -> bytes:
        if level in self._keys:
            return self._keys[level]
        if level == "level_5":
            result = self.tools()
        elif level == "level_4":
            result = tool_name_trace_bytes(self.sample, self.bindings.values())
        else:
            messages = []
            for index, message in enumerate(self.ordered_messages()):
                role = message.get("role")
                projected = message
                variant = "exact"
                if level == "level_1_5" and role != "tool":
                    content = message.get("content")
                    if isinstance(content, str):
                        normalized = _plain_prose(content)
                        if normalized != content:
                            projected = {**message, "content": normalized}
                            variant = "prose"
                elif level in {"level_2", "level_3"} and role == "assistant":
                    if level == "level_3" and not message.get("tool_calls"):
                        continue
                    projected = {
                        key: value
                        for key, value in message.items()
                        if key not in {"content", "reasoning_content"}
                    }
                    variant = "omit_assistant_prose"
                elif level == "level_3" and role == "user":
                    projected = {
                        key: value for key, value in message.items() if key != "content"
                    }
                    variant = "omit_user_prose"
                cache_key = (index, variant)
                if cache_key not in self._message_keys:
                    self._message_keys[cache_key] = _json(projected)
                messages.append(self._message_keys[cache_key])
            result = _frame((self.tools(), _frame(messages)))
        self._keys[level] = result
        return result

    def has_calls(self) -> bool:
        return any(
            message.get("role") == "assistant" and message.get("tool_calls")
            for message in self.messages
        )

    def assistant_characters(self) -> int:
        total = 0
        for message in self.messages:
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise TypeError(
                    "curation requires assistant content to be string or null"
                )
            total += len(content or "")
        return total


def canonical_comparison(record: CurationInput) -> ComparisonView:
    """Build the standard cached view for reuse by an audited protocol adapter.

    In particular, an adapter can reuse exact Level 1 and supply its own
    action-aware higher levels without changing either returned samples or
    the shared selector. Do not apply prose-omitting levels to textual actions.
    """
    return _ComparisonView(record)


@dataclass(slots=True)
class CurationResult:
    """One-shot iterator with final per-scope counts and explicit cleanup."""

    reports: dict[ComparisonScope, ScopedCurationReport]
    samples: Iterator[ConversationSample]

    def __iter__(self) -> Iterator[ConversationSample]:
        return self.samples

    def close(self) -> None:
        close = getattr(self.samples, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> CurationResult:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def curate(
    inputs: Iterable[CurationInput],
    *,
    scope: ComparisonScope | Callable[[CurationInput], ComparisonScope],
    config: CurationConfig | None = None,
    on_selected: Callable[[int], None] | None = None,
    compress_comparisons: bool = False,
    compress_payloads: bool = False,
    comparison_factory: Callable[[CurationInput], ComparisonView] | None = None,
) -> CurationResult:
    """Curate final-view rows within explicit dataset/subset/split boundaries.

    The iterable is consumed once. Final selection and aggregate counts are
    complete before the first sample is yielded, so shorter later Level-2
    winners retain their correct source position. ``max_level=None`` disables
    all deletion and permits an audit-only pass. Audit data is attached only
    to ``result.reports``. No sample annotations, IDs, or raw fields are read
    for equality or changed by selection.

    A ``MixtureScope`` explicitly compares the supplied final union across
    datasets; ordinary ``CurationScope`` still requires matching datasets.
    ``on_selected`` receives each winning zero-based input position in source
    order, after selection and before any payload is yielded. Positions are
    temporary storage coordinates and are never added to a sample.

    ``compress_comparisons`` losslessly compresses temporary equality and
    audit BLOBs with deterministic zlib encoding. Compression is injective,
    not a digest; equality and group counts are unchanged. Comparison bytes
    never determine winner ordering. This storage option does not alter
    payloads, reports, or the configured curation levels.

    ``compress_payloads`` also losslessly compresses this process's own
    temporary pickle payloads; selected objects are decompressed before the
    same pickle round trip. Neither option changes returned sample values.

    ``comparison_factory`` selects an audited protocol's ephemeral comparison
    semantics. It never substitutes the spooled/returned sample. Omission keeps
    the existing native-call semantics unchanged.
    """
    selected_config = config or CurationConfig()
    reports: dict[ComparisonScope, ScopedCurationReport] = {}
    return CurationResult(
        reports,
        _run(
            inputs,
            scope,
            selected_config,
            reports,
            on_selected,
            compress_comparisons,
            compress_payloads,
            comparison_factory,
        ),
    )


def _run(
    inputs: Iterable[CurationInput],
    scope_for: ComparisonScope | Callable[[CurationInput], ComparisonScope],
    config: CurationConfig,
    reports: dict[ComparisonScope, ScopedCurationReport],
    on_selected: Callable[[int], None] | None,
    compress_comparisons: bool,
    compress_payloads: bool,
    comparison_factory: Callable[[CurationInput], ComparisonView] | None,
) -> Iterator[ConversationSample]:
    def encoded(value: bytes) -> bytes:
        return zlib.compress(value, level=1) if compress_comparisons else value

    levels = (
        _LEVELS[: _LEVELS.index(config.max_level) + 1]
        if config.max_level is not None
        else ()
    )
    view_factory = _ComparisonView if comparison_factory is None else comparison_factory
    with TemporaryDirectory(
        prefix="fcanalysis-curation-", dir=config.temporary_directory
    ) as temporary:
        connection = sqlite3.connect(Path(temporary) / "comparison.sqlite3")
        try:
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute(
                "CREATE TABLE retained (position INTEGER PRIMARY KEY, scope BLOB, "
                "level_3 BLOB, level_4 BLOB, level_5 BLOB)"
            )
            connection.execute(
                "CREATE TABLE payloads (position INTEGER PRIMARY KEY, sample BLOB)"
            )
            for level in levels:
                connection.execute(
                    f"CREATE TABLE {level} (scope BLOB, comparison BLOB, "
                    "position INTEGER, score INTEGER, count INTEGER, "
                    "PRIMARY KEY (scope, comparison)) WITHOUT ROWID"
                )
            scope_bytes: dict[ComparisonScope, bytes] = {}
            for position, record in enumerate(inputs):
                scope = (
                    scope_for
                    if isinstance(scope_for, (CurationScope, MixtureScope))
                    else scope_for(record)
                )
                if not isinstance(scope, (CurationScope, MixtureScope)):
                    raise TypeError(
                        "curation scope callback must return CurationScope or MixtureScope"
                    )
                if (
                    isinstance(scope, CurationScope)
                    and scope.dataset != record.sample.dataset
                ):
                    raise ValueError(
                        "curation scope dataset must match the sample dataset"
                    )
                if scope not in reports:
                    reports[scope] = ScopedCurationReport(
                        scope, stages={level: CurationStageCounts() for level in levels}
                    )
                    scope_bytes[scope] = _json(asdict(scope))
                report = reports[scope]
                report.input_samples += 1
                encoded_scope = scope_bytes[scope]
                view = view_factory(record)
                score = view.assistant_characters() if "level_2" in levels else 0
                survives = True
                for level in levels:
                    counts = report.stages[level]
                    counts.input_samples += 1
                    comparison = encoded(view.key(level))
                    previous = connection.execute(
                        f"SELECT position, score, count FROM {level} "
                        "WHERE scope=? AND comparison=?",
                        (encoded_scope, comparison),
                    ).fetchone()
                    if previous is None:
                        connection.execute(
                            f"INSERT INTO {level} VALUES (?, ?, ?, ?, 1)",
                            (encoded_scope, comparison, position, score),
                        )
                        counts.output_samples += 1
                        continue
                    previous_position, previous_score, previous_count = previous
                    if previous_count == 1:
                        counts.duplicate_groups += 1
                    counts.removed_samples += 1
                    replace = level == "level_2" and score < previous_score
                    connection.execute(
                        f"UPDATE {level} SET position=?, score=?, count=count+1 "
                        "WHERE scope=? AND comparison=?",
                        (
                            position if replace else previous_position,
                            score if replace else previous_score,
                            encoded_scope,
                            comparison,
                        ),
                    )
                    if replace:
                        connection.execute(
                            "DELETE FROM retained WHERE position=?",
                            (previous_position,),
                        )
                        connection.execute(
                            "DELETE FROM payloads WHERE position=?",
                            (previous_position,),
                        )
                    else:
                        survives = False
                        break
                if not survives:
                    continue
                audit_keys = (
                    (
                        encoded(view.key("level_3")) if view.has_calls() else None,
                        encoded(view.key("level_4")),
                        encoded(view.key("level_5")),
                    )
                    if config.audit
                    else (None, None, None)
                )
                # Pickle is used solely for this process's own normalized
                # objects in a private temporary directory. No external pickle
                # is loaded. It preserves raw Python source-value types too.
                connection.execute(
                    "INSERT INTO retained VALUES (?, ?, ?, ?, ?)",
                    (position, encoded_scope, *audit_keys),
                )
                payload = pickle.dumps(record.sample, protocol=pickle.HIGHEST_PROTOCOL)
                if compress_payloads:
                    payload = zlib.compress(payload, level=1)
                connection.execute(
                    "INSERT INTO payloads VALUES (?, ?)",
                    (position, payload),
                )
            # Build once after ingestion. Every audit visits only its scope's
            # comparison rows, independently of the lossless raw-data spool.
            connection.execute("CREATE INDEX retained_scope ON retained(scope)")
            connection.commit()
            for scope, report in reports.items():
                encoded_scope = scope_bytes[scope]
                report.output_samples = connection.execute(
                    "SELECT COUNT(*) FROM retained WHERE scope=?", (encoded_scope,)
                ).fetchone()[0]
                report.removed_samples = report.input_samples - report.output_samples
                if config.audit:
                    for level in ("level_3", "level_4", "level_5"):
                        counts = connection.execute(
                            "SELECT COALESCE(SUM(n), 0), COUNT(*), "
                            "COALESCE(SUM(n > 1), 0), "
                            "COALESCE(SUM(CASE WHEN n > 1 THEN n ELSE 0 END), 0) "
                            f"FROM (SELECT COUNT(*) AS n FROM retained WHERE scope=? "
                            f"AND {level} IS NOT NULL GROUP BY {level})",
                            (encoded_scope,),
                        ).fetchone()
                        eligible, groups, repeated, repeated_samples = counts
                        report.audit[level] = EquivalenceAuditCounts(
                            eligible_samples=eligible,
                            unique_groups=groups,
                            repeated_groups=repeated,
                            samples_in_repeated_groups=repeated_samples,
                            extra_samples=eligible - groups,
                        )
            if on_selected is not None:
                for (position,) in connection.execute(
                    "SELECT position FROM retained ORDER BY position"
                ):
                    on_selected(position)
            for (payload,) in connection.execute(
                "SELECT sample FROM payloads ORDER BY position"
            ):
                yield pickle.loads(
                    zlib.decompress(payload) if compress_payloads else payload
                )
        finally:
            connection.close()
