"""Deduplicate an explicitly ordered union of retained conversations on disk.

The input is the final retained population supplied by the caller. This pass
does not reconstruct candidates removed by earlier per-source curation and
does not claim equivalence to curating all original source candidates once.
Each named stream is consumed once, in the caller's order. Levels 1 and 1.5
keep the first member; Level 2 keeps the original member with the shortest
assistant content, using that same order to break ties. Selected rows remain
in their original global input order, with reasoning and every other field
unchanged. Source names describe report counters only; neither names, dataset
labels, IDs, annotations nor raw values determine comparison equality.

This utility compares canonical call/result batches in their existing order.
Final loader outputs have already established and aligned their bindings,
but do not carry the source's permission to permute whole call/result pairs.
Adjacency here recovers canonical coordinates, never original-source proof.
No new permutation permission is inferred, including for multi-call batches.
Explicit CurationInput records with parallel bindings are rejected: silently
mixing their unordered comparisons with ordered records could delete a row
whose source did not authorize that permutation. Arguments are read afresh
from the final sample rather than trusting an earlier parsed-argument cache.

Inputs must be qualified for these native-call or genuine text-only comparisons.
Textual action protocols, including Nemotron-Terminal, require their own audited
prose/action separation and are not supported by this utility. An empty tools
list does not establish text-only supervision. See the source contract in
``fcanalysis.loaders.nemotron_terminal``.

The shared curation engine performs cumulative Levels 1/1.5/2 and final
aggregate Level 3/4/5 audits in one explicit MixtureScope. It consumes all
inputs before emitting winners. Payloads and complete comparison values live
in private temporary SQLite databases; RAM does not grow with corpus rows.
Source attribution uses only temporary input-position intervals, requiring
memory proportional to the number of named streams. Disk requirements still
depend on the compressed input plus the largest complete tool environment.
All deletion keys contain exact complete tool definitions. The input is
spooled once with lossless compression, then complete environments are grouped
into size-targeted partitions for shared curation. An environment is never
split, including when it exceeds the target. Level 3 and 5 counts are additive
across those partitions; Level 4 is counted globally over all winners.

Use the result as a context manager, or close it when abandoning iteration.
Opened input iterators and temporary files are closed on completion, errors,
and explicit close. Unopened source iterables are left untouched. Disk
spooling preserves raw values but does not preserve Python object identity.
No hashes, IDs, annotations or provenance fields are created.
"""

from bisect import bisect_right
from collections.abc import Callable, Generator, Iterable, Iterator
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
import pickle
import sqlite3
from tempfile import TemporaryDirectory
import zlib
from typing import Literal

from fcanalysis.deduplication import EquivalenceAuditCounts
from fcanalysis.format import ConversationSample
from fcanalysis.loaders.curation import (
    BoundCallBatch,
    CurationConfig,
    CurationInput,
    CurationStageCounts,
    MixtureScope,
    ScopedCurationReport,
    curate,
    tool_environment_bytes,
    tool_name_trace_bytes,
)


type MixtureProgress = Callable[
    [Literal["spooling", "curating", "yielding"], int], None
]
_PARTITION_ROWS = (
    "SELECT samples.position FROM work_environments "
    "CROSS JOIN samples INDEXED BY samples_environment "
    "ON samples.environment=work_environments.environment "
    "ORDER BY samples.position"
)


@dataclass(frozen=True, slots=True)
class MixtureSource:
    """One named stream; caller order supplies deterministic tie priority."""

    name: str
    samples: Iterable[ConversationSample | CurationInput]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("mixture source name must be a nonempty string")


@dataclass(slots=True)
class MixtureSourceCounts:
    input_samples: int = 0
    output_samples: int = 0
    removed_samples: int = 0


@dataclass(slots=True)
class MixtureReport:
    """Final aggregate curation and attribution to the supplied named streams.

    Counts become final before the first winner is yielded, or when an empty
    result is exhausted. On an ingestion error, ``complete`` stays false.
    Per-source removals include losses within that stream and across streams;
    they are not a separate source-pair overlap audit.
    """

    scope: MixtureScope
    sources: dict[str, MixtureSourceCounts] = field(default_factory=dict)
    curation: ScopedCurationReport | None = None
    complete: bool = False


@dataclass(slots=True)
class MixtureResult:
    """One-shot result with explicit temporary-resource cleanup."""

    report: MixtureReport
    samples: Iterator[ConversationSample]

    def __iter__(self) -> Iterator[ConversationSample]:
        return self.samples

    def close(self) -> None:
        close = getattr(self.samples, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> MixtureResult:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _ordered_record(value: ConversationSample | CurationInput) -> CurationInput:
    record = value if isinstance(value, CurationInput) else CurationInput(value)
    sample = record.sample
    if not isinstance(sample, ConversationSample):
        raise TypeError("mixture inputs must be ConversationSample or CurationInput")
    if any(batch.parallel for batch in record.batches):
        raise ValueError("mixture comparison requires ordered, non-parallel bindings")
    batches: list[BoundCallBatch] = []
    messages = sample.messages
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "tool":
            raise ValueError("canonical mixture input has an orphan tool result")
        calls = message.get("tool_calls")
        if not calls:
            index += 1
            continue
        if message.get("role") != "assistant" or not isinstance(calls, list):
            raise ValueError("canonical mixture calls must be an assistant list")
        end = index + 1
        while end < len(messages) and messages[end].get("role") == "tool":
            end += 1
        if end - index - 1 != len(calls):
            raise ValueError("canonical mixture call/result cardinality differs")
        batches.append(BoundCallBatch(index, tuple(range(index + 1, end))))
        index = end
    ordered = tuple(batches)
    if record.batches and record.batches != ordered:
        raise ValueError("supplied mixture bindings differ from canonical order")
    return CurationInput(sample, ordered)


def _empty_report(scope: MixtureScope, config: CurationConfig) -> ScopedCurationReport:
    levels = ("level_1", "level_1_5", "level_2")
    count = 0 if config.max_level is None else levels.index(config.max_level) + 1
    return ScopedCurationReport(
        scope,
        stages={level: CurationStageCounts() for level in levels[:count]},
        audit=(
            {
                level: EquivalenceAuditCounts(0, 0, 0, 0, 0)
                for level in ("level_3", "level_4", "level_5")
            }
            if config.audit
            else {}
        ),
    )


def deduplicate_mixture(
    sources: Iterable[MixtureSource],
    *,
    scope: MixtureScope,
    config: CurationConfig | None = None,
    partition_size_bytes: int = 64 * 1024 * 1024,
    progress: MixtureProgress | None = None,
) -> MixtureResult:
    """Curate retained inputs across datasets without altering selected rows.

    Source names must be unique, and their iterable order is the explicit
    priority order. An empty source is included in the report. The default
    runs all three deletion levels plus the three aggregate audits; pass an
    explicit CurationConfig to restrict levels, audit, or temporary location.
    No reasoning-removal transform or training policy is applied.

    ``partition_size_bytes`` targets uncompressed spooled payload bytes per
    curation partition. Complete environments may exceed it; it is not a hard
    disk limit. The entire compressed input is spooled before selection.
    Optional ``progress(phase, count)`` reports cumulative records within each
    phase; callbacks may throttle display and their exceptions close the run.
    """
    if not isinstance(scope, MixtureScope):
        raise TypeError("deduplicate_mixture requires an explicit MixtureScope")
    if type(partition_size_bytes) is not int or partition_size_bytes <= 0:
        raise ValueError("partition_size_bytes must be a positive integer")
    ordered = tuple(sources)
    if any(not isinstance(source, MixtureSource) for source in ordered):
        raise TypeError("sources must contain MixtureSource records")
    names = [source.name for source in ordered]
    if len(set(names)) != len(names):
        raise ValueError("mixture source names must be unique")
    report = MixtureReport(
        scope, sources={name: MixtureSourceCounts() for name in names}
    )
    selected_config = config or CurationConfig()
    return MixtureResult(
        report, _run(ordered, selected_config, report, partition_size_bytes, progress)
    )


def _run(
    sources: tuple[MixtureSource, ...],
    config: CurationConfig,
    report: MixtureReport,
    partition_size_bytes: int,
    progress: MixtureProgress | None,
) -> Iterator[ConversationSample]:
    ends: list[int] = []
    aggregate = _empty_report(report.scope, config)
    with TemporaryDirectory(
        prefix="fcanalysis-mixture-", dir=config.temporary_directory
    ) as temporary:
        connection = sqlite3.connect(Path(temporary) / "mixture.sqlite3")
        try:
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("PRAGMA mmap_size=0")
            connection.executescript(
                "CREATE TABLE environments (comparison BLOB UNIQUE, size INTEGER);"
                "CREATE TABLE samples (position INTEGER PRIMARY KEY, environment INTEGER, "
                "payload BLOB);"
                "CREATE TABLE selected_positions (position INTEGER PRIMARY KEY);"
                "CREATE TABLE work_environments (environment INTEGER PRIMARY KEY);"
                "CREATE TABLE work_positions (local INTEGER PRIMARY KEY, position INTEGER);"
                "CREATE TABLE level4 (comparison BLOB PRIMARY KEY, n INTEGER) WITHOUT ROWID;"
            )
            position = 0
            for source in sources:
                counts = report.sources[source.name]
                iterator = iter(source.samples)
                try:
                    for value in iterator:
                        record = _ordered_record(value)
                        comparison = tool_environment_bytes(record.sample.tools)
                        payload = pickle.dumps(
                            record.sample, protocol=pickle.HIGHEST_PROTOCOL
                        )
                        connection.execute(
                            "INSERT INTO environments VALUES (?, ?) "
                            "ON CONFLICT(comparison) DO UPDATE SET size=size+excluded.size",
                            (comparison, len(payload)),
                        )
                        environment = connection.execute(
                            "SELECT rowid FROM environments WHERE comparison=?",
                            (comparison,),
                        ).fetchone()[0]
                        connection.execute(
                            "INSERT INTO samples (position, environment, payload) "
                            "VALUES (?, ?, ?)",
                            (position, environment, zlib.compress(payload, level=1)),
                        )
                        counts.input_samples += 1
                        position += 1
                        if progress is not None:
                            progress("spooling", position)
                finally:
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()
                ends.append(position)
            connection.execute(
                "CREATE INDEX samples_environment ON samples(environment)"
            )
            connection.commit()
            partition_bytes = 0
            for environment, size in connection.execute(
                "SELECT rowid, size FROM environments ORDER BY rowid"
            ):
                if partition_bytes and partition_bytes + size > partition_size_bytes:
                    _curate_partition(
                        connection, temporary, config, aggregate, progress
                    )
                    partition_bytes = 0
                connection.execute(
                    "INSERT INTO work_environments VALUES (?)", (environment,)
                )
                partition_bytes += size
            if partition_bytes:
                _curate_partition(connection, temporary, config, aggregate, progress)

            if config.audit:
                eligible, groups, repeated, repeated_samples = connection.execute(
                    "SELECT COALESCE(SUM(n),0), COUNT(*), COALESCE(SUM(n>1),0), "
                    "COALESCE(SUM(CASE WHEN n>1 THEN n ELSE 0 END),0) FROM level4"
                ).fetchone()
                aggregate.audit["level_4"] = EquivalenceAuditCounts(
                    eligible_samples=eligible,
                    unique_groups=groups,
                    repeated_groups=repeated,
                    samples_in_repeated_groups=repeated_samples,
                    extra_samples=eligible - groups,
                )
            for (position,) in connection.execute(
                "SELECT position FROM selected_positions ORDER BY position"
            ):
                source = sources[bisect_right(ends, position)]
                report.sources[source.name].output_samples += 1
            for counts in report.sources.values():
                counts.removed_samples = counts.input_samples - counts.output_samples
            report.curation = aggregate
            report.complete = True
            for count, (position,) in enumerate(
                connection.execute(
                    "SELECT position FROM selected_positions ORDER BY position"
                ),
                1,
            ):
                # Only this process's own private spool is unpickled.
                if progress is not None:
                    progress("yielding", count)
                payload = connection.execute(
                    "SELECT payload FROM samples WHERE position=?", (position,)
                ).fetchone()[0]
                yield pickle.loads(zlib.decompress(payload))
        finally:
            connection.close()


def _curate_partition(
    connection: sqlite3.Connection,
    temporary: str,
    config: CurationConfig,
    aggregate: ScopedCurationReport,
    progress: MixtureProgress | None,
) -> None:
    def records() -> Generator[CurationInput, None, None]:
        for local, (position,) in enumerate(connection.execute(_PARTITION_ROWS)):
            # Sort only positions. Projecting the payload into the ordered
            # join makes SQLite duplicate every BLOB in its temporary sort.
            payload = connection.execute(
                "SELECT payload FROM samples WHERE position=?", (position,)
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO work_positions VALUES (?, ?)", (local, position)
            )
            if progress is not None:
                progress("curating", aggregate.input_samples + local + 1)
            yield _ordered_record(pickle.loads(zlib.decompress(payload)))

    def selected(local: int) -> None:
        connection.execute(
            "INSERT INTO selected_positions SELECT position "
            "FROM work_positions WHERE local=?",
            (local,),
        )

    inputs = records()
    try:
        with curate(
            inputs,
            scope=aggregate.scope,
            config=replace(config, temporary_directory=temporary),
            on_selected=selected,
            compress_comparisons=True,
            compress_payloads=True,
        ) as result:
            for sample in result:
                if config.audit:
                    key = tool_name_trace_bytes(sample)
                    connection.execute(
                        "INSERT INTO level4 VALUES (?,1) "
                        "ON CONFLICT(comparison) DO UPDATE SET n=n+1",
                        (key,),
                    )
            part = result.reports[aggregate.scope]
    finally:
        inputs.close()
    for name in ("input_samples", "output_samples", "removed_samples"):
        setattr(aggregate, name, getattr(aggregate, name) + getattr(part, name))
    for level, counts in part.stages.items():
        for item in fields(counts):
            current = aggregate.stages[level]
            setattr(
                current,
                item.name,
                getattr(current, item.name) + getattr(counts, item.name),
            )
    for level in ("level_3", "level_5"):
        if config.audit:
            current = aggregate.audit[level]
            aggregate.audit[level] = replace(
                current,
                **{
                    item.name: getattr(current, item.name)
                    + getattr(part.audit[level], item.name)
                    for item in fields(current)
                },
            )
    connection.execute(
        "DELETE FROM samples WHERE environment IN "
        "(SELECT environment FROM work_environments) "
        "AND position NOT IN (SELECT position FROM selected_positions)"
    )
    connection.execute("DELETE FROM work_environments")
    connection.execute("DELETE FROM work_positions")
    connection.commit()
