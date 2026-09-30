"""Exact reconstruction of source-defined assistant-target prefix releases.

Adapters establish the splitting grammar, project only demonstrably omitted
fields, and identify complete anchor messages and required historical donors.
This module never recognizes reasoning syntax or infers families from questions,
row adjacency, tool names, IDs, or future continuations. A donor must end at the
same exact projected message prefix in the same source scope and context.
Different complete donor messages at that boundary make restoration ambiguous.

After verbatim restoration, the adapter validates each complete candidate. Only
then are exact restored prefixes with valid longer continuations removed.
Alternative continuations and reasoning variants remain distinct. Training loss
masks, assistant-target splitting, and history reasoning policy belong to the
trainer, not this stage.

Both prefix comparisons are disk-backed tries of complete message values, not
hashes. One row and its message path are held in memory; no source family is
materialized as a group. Private temporary SQLite state is removed on completion,
exceptions, and explicit/context-manager close. Arbitrary shards of one source
scope require a joint reconstruction pass; independently reconstructed shard
outputs cannot replace the original candidates and donors.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
import sqlite3
from tempfile import TemporaryDirectory
from typing import Any

from .curation import CurationScope
from .normalization import json_bytes


@dataclass(slots=True)
class PrefixRecord[T]:
    """One source prefix and an adapter-certified comparison projection.

    ``value`` is opaque adapter payload, usually the untouched original row and
    its existing source position. ``context`` includes the full external tool
    environment and any source metadata needed to distinguish trajectories;
    systems or tool epochs within messages also remain in the projection.

    ``projected_messages`` has one value per original message. It excludes ONLY
    fields the verified source splitter can omit, preserving all other bytes
    and chronology. ``anchor`` names the final, complete source message, or is
    None for a record that supplies no donor. An explicitly empty/absent native
    field in a complete anchor is evidence; an absent field in stripped history
    is not. ``required`` names messages that must be restored from a unique
    complete donor. The adapter must reject unsupported source shapes before
    constructing this record.
    """

    value: T
    scope: CurationScope
    context: Any
    messages: list[dict[str, Any]]
    projected_messages: Sequence[Any]
    required: tuple[int, ...] = ()
    anchor: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, CurationScope):
            raise TypeError("prefix scope must be CurationScope")
        count = len(self.messages)
        if not count or len(self.projected_messages) != count:
            raise ValueError("prefix projection must cover every nonempty message")
        if self.anchor is not None and self.anchor != count - 1:
            raise ValueError("a complete prefix anchor must be its final message")
        if (
            tuple(sorted(set(self.required))) != self.required
            or any(index < 0 or index >= count for index in self.required)
            or self.anchor in self.required
        ):
            raise ValueError("required donor indices must be unique ordered history")


@dataclass(frozen=True, slots=True)
class ReconstructionConfig:
    temporary_directory: str | Path | None = None


@dataclass(slots=True)
class ReconstructionReport:
    input_rows: int = 0
    missing_donor_rows: int = 0
    ambiguous_donor_rows: int = 0
    restored_rows: int = 0
    restored_messages: int = 0
    validation_dropped_rows: int = 0
    validated_rows: int = 0
    removed_prefix_rows: int = 0
    output_rows: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class ReconstructionResult[T]:
    report: ReconstructionReport
    rows: Iterator[T]

    def __iter__(self) -> Iterator[T]:
        return self.rows

    def close(self) -> None:
        close = getattr(self.rows, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> ReconstructionResult[T]:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def reconstruct_prefixes[T, R](
    records: Iterable[PrefixRecord[T]],
    validate: Callable[[T, list[dict[str, Any]]], R | None],
    *,
    config: ReconstructionConfig | None = None,
) -> ReconstructionResult[R]:
    """Restore source-proven fields, validate, then retain maximal exact prefixes.

    ``validate`` receives the original payload and independently owned restored
    messages. It converts and validates the complete trajectory, including any
    requested final-view transforms; None quarantines a candidate. Programming
    exceptions propagate. Prefix selection compares restored source messages
    before the callback can transform them, never a reasoning-stripped view.
    The callback must not mutate the original payload. Selected callback values
    are returned in original input order; exact same-length duplicates remain
    for the separate curation stage. ``raw`` and sample metadata are never read
    or changed by this module. Final counts precede the first yielded value.
    """
    report = ReconstructionReport()
    return ReconstructionResult(
        report, _run(records, validate, config or ReconstructionConfig(), report)
    )


def _dump(value: Any) -> bytes:
    # Only our own in-process objects enter this private temporary spool. Never
    # read externally supplied pickle data.
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _node(connection: sqlite3.Connection, parent: int, body: bytes) -> int:
    existing = connection.execute(
        "SELECT id FROM nodes WHERE parent=? AND body=?", (parent, body)
    ).fetchone()
    if existing is not None:
        return existing[0]
    inserted = connection.execute(
        "INSERT INTO nodes (parent, body) VALUES (?, ?)", (parent, body)
    )
    assert inserted.lastrowid is not None
    return inserted.lastrowid


def _path(
    connection: sqlite3.Connection, root: int, bodies: Iterable[bytes]
) -> list[int]:
    result = []
    parent = root
    for body in bodies:
        parent = _node(connection, parent, body)
        result.append(parent)
    return result


def _run[T, R](
    records: Iterable[PrefixRecord[T]],
    validate: Callable[[T, list[dict[str, Any]]], R | None],
    config: ReconstructionConfig,
    report: ReconstructionReport,
) -> Iterator[R]:
    with TemporaryDirectory(
        prefix="fcanalysis-reconstruction-", dir=config.temporary_directory
    ) as temporary:
        connection = sqlite3.connect(Path(temporary) / "prefixes.sqlite3")
        try:
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute(
                "CREATE TABLE nodes (id INTEGER PRIMARY KEY, parent INTEGER, "
                "body BLOB, extended INTEGER NOT NULL DEFAULT 0, "
                "UNIQUE(parent, body))"
            )
            connection.execute(
                "CREATE TABLE donors (node INTEGER, body BLOB, payload BLOB, "
                "PRIMARY KEY(node, body)) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE TABLE candidates (position INTEGER PRIMARY KEY, "
                "root INTEGER, path BLOB, payload BLOB)"
            )
            connection.execute(
                "CREATE TABLE valid (position INTEGER PRIMARY KEY, "
                "node INTEGER, payload BLOB)"
            )
            for position, record in enumerate(records):
                report.input_rows += 1
                # Two disjoint roots separate omission-aware donor lookup from
                # complete reconstructed-content prefix selection.
                context = json_bytes([asdict(record.scope), record.context])
                candidate_root = _node(connection, 0, context)
                complete_root = _node(connection, -1, context)
                projected = [
                    json_bytes(message) for message in record.projected_messages
                ]
                required = set(record.required)
                for index, message in enumerate(record.messages):
                    if index != record.anchor and index not in required:
                        if json_bytes(message) != projected[index]:
                            raise ValueError(
                                "projection cannot omit known non-anchor history"
                            )
                path = _path(connection, candidate_root, projected)
                if record.anchor is not None:
                    anchor = record.messages[record.anchor]
                    connection.execute(
                        "INSERT OR IGNORE INTO donors VALUES (?, ?, ?)",
                        (path[record.anchor], json_bytes(anchor), _dump(anchor)),
                    )
                connection.execute(
                    "INSERT INTO candidates VALUES (?, ?, ?, ?)",
                    (position, complete_root, _dump(path), _dump(record)),
                )
            connection.commit()
            for position, root, path_blob, payload in connection.execute(
                "SELECT position, root, path, payload FROM candidates ORDER BY position"
            ):
                record = pickle.loads(payload)
                path = pickle.loads(path_blob)
                messages = deepcopy(record.messages)
                replacements = []
                for index in record.required:
                    donors = connection.execute(
                        "SELECT payload FROM donors WHERE node=? LIMIT 2",
                        (path[index],),
                    ).fetchall()
                    if not donors:
                        report.missing_donor_rows += 1
                        break
                    if len(donors) > 1:
                        report.ambiguous_donor_rows += 1
                        break
                    replacements.append((index, pickle.loads(donors[0][0])))
                else:
                    for index, message in replacements:
                        messages[index] = message
                    # Capture proof values before a callback mutates its owned
                    # messages, for example by removing reasoning or source IDs.
                    proof = [json_bytes(message) for message in messages]
                    result = validate(record.value, messages)
                    if result is None:
                        report.validation_dropped_rows += 1
                        continue
                    report.validated_rows += 1
                    if replacements:
                        report.restored_rows += 1
                        report.restored_messages += len(replacements)
                    complete_path = _path(connection, root, proof)
                    connection.executemany(
                        "UPDATE nodes SET extended=1 WHERE id=? AND extended=0",
                        ((node,) for node in complete_path[:-1]),
                    )
                    connection.execute(
                        "INSERT INTO valid VALUES (?, ?, ?)",
                        (position, complete_path[-1], _dump(result)),
                    )
            connection.commit()
            report.output_rows = connection.execute(
                "SELECT COUNT(*) FROM valid JOIN nodes ON valid.node=nodes.id "
                "WHERE nodes.extended=0"
            ).fetchone()[0]
            report.removed_prefix_rows = report.validated_rows - report.output_rows
            for (payload,) in connection.execute(
                "SELECT valid.payload FROM valid JOIN nodes ON valid.node=nodes.id "
                "WHERE nodes.extended=0 ORDER BY valid.position"
            ):
                yield pickle.loads(payload)
        finally:
            connection.close()
