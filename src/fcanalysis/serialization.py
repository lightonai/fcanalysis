"""Read serialized canonical conversations without rerunning source loaders.

JSONL artifacts contain normalized samples, not necessarily their original
source rows. In particular, loader fixtures omit ``raw``; reading them leaves
that field empty and does not recover or certify historical runtime objects.
This module checks the serialized container shape, not loader validity.
"""

from collections.abc import Generator
import gzip
from pathlib import Path
from typing import Any

from .format import ConversationSample
from .loaders.normalization import parse_json


_REQUIRED_FIELDS = frozenset({"messages", "tools", "dataset", "sample_id"})
_OPTIONAL_FIELDS = frozenset({"annotations", "raw"})


def _sample_from_payload(payload: Any) -> ConversationSample:
    if not isinstance(payload, dict):
        raise ValueError("sample must be a JSON object")
    missing = _REQUIRED_FIELDS - payload.keys()
    if missing:
        raise ValueError(f"missing sample fields: {', '.join(sorted(missing))}")
    unknown = payload.keys() - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
    if unknown:
        raise ValueError(f"unknown sample fields: {', '.join(sorted(unknown))}")
    for name in ("messages", "tools"):
        value = payload[name]
        if not isinstance(value, list) or any(
            not isinstance(item, dict) for item in value
        ):
            raise ValueError(f"{name} must be an array of objects")
    if not isinstance(payload["dataset"], str):
        raise ValueError("dataset must be a string")
    if type(payload["sample_id"]) not in (str, int):
        raise ValueError("sample_id must be a string or integer")
    for name in ("annotations", "raw"):
        if name in payload and not isinstance(payload[name], dict):
            raise ValueError(f"{name} must be an object")
    return ConversationSample(**payload)


def iter_samples_jsonl(path: str | Path) -> Generator[ConversationSample, None, None]:
    """Stream UTF-8 canonical JSONL, decompressing paths ending in ``.gz``.

    Preserve row order and every nested JSON value, including arbitrary-size
    integers, argument strings, annotations, and explicitly serialized raw
    fields. Missing annotations/raw receive separate empty dictionaries.
    Required fields and shallow types are checked; unknown root fields,
    malformed JSON, duplicate keys, nonfinite numbers, and blank rows fail
    with path/line context. Empty files and an unterminated final line are
    supported. No messages, tools, or reasoning are normalized or validated.

    Memory is bounded by one decoded row, whose size is not capped. The
    iterator owns its file; callers stopping early must close the generator.
    """
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rb") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                payload = parse_json(line.decode("utf-8"))
                sample = _sample_from_payload(payload)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
            yield sample
            del sample, payload
