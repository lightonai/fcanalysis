"""Audited ToolMind result bundles, kept separate from ordinary JSON outputs.

At Nanbeige/ToolMind@8020ed1c03c367e4eb720ac3828ab4b0b95d8baf,
all 32,983 BUTTONInstruct tool messages are nonempty lists of exact
{name, arguments, results} entries. 32,963 echo the preceding calls in source
order; 20 repeat identical calls and cannot establish unique pairing.
All 1,543 ToolACE tool messages contain exact {name, results} entries;
1,526 match unique call names in source order, 14 repeat names, and three
contradict the calls. These two released subsets have explicit result envelopes.
Other subsets' JSON lists are ordinary payloads and must not be split.

Canonical expansion preserves every complete entry and its order. Only results
supply external-state evidence; echoed names/arguments are linkage information,
not new facts. No identifiers, ordering rule or result value is invented.
"""

from collections import Counter
from collections.abc import Iterator
from typing import Any

from .context import json_scalar_values
from .normalization import Reject, normalize_arguments, parse_json, serialize_result

BUTTON = "open_datasets/BUTTONInstruct-query.jsonl"
TOOLACE = "open_datasets/ToolACE-query.jsonl"
_KEYS = {
    BUTTON: frozenset({"name", "arguments", "results"}),
    TOOLACE: frozenset({"name", "results"}),
}


def expand_result(
    value: Any, *, source_file: str | None, transforms: Counter[str] | None = None
) -> list[dict[str, Any]]:
    """Split only the two audited bundle protocols, preserving full entries."""
    if source_file not in _KEYS:
        return [{"role": "tool", "content": serialize_result(value)}]
    entries = parse_json(value) if isinstance(value, str) else value
    if not isinstance(entries, list) or not entries:
        raise Reject("invalid_source_result_bundle")
    keys = _KEYS[source_file]
    if any(
        not isinstance(entry, dict)
        or set(entry) != keys
        or not isinstance(entry["name"], str)
        or not entry["name"]
        for entry in entries
    ):
        raise Reject("invalid_source_result_bundle")
    if transforms is not None:
        transforms["source_result_bundles_expanded"] += 1
        transforms["source_result_entries_emitted"] += len(entries)
    return [{"role": "tool", "content": serialize_result(entry)} for entry in entries]


def result_echo(
    message: dict[str, Any], *, source_file: str | None
) -> tuple[str, dict[str, Any] | None] | None:
    """Read exact source linkage from a retained complete entry."""
    if source_file not in _KEYS:
        return None
    value = parse_json(message["content"])
    if not isinstance(value, dict) or set(value) != _KEYS[source_file]:
        raise Reject("invalid_source_result_bundle")
    name = value["name"]
    if not isinstance(name, str) or not name:
        raise Reject("invalid_source_result_bundle")
    arguments = (
        normalize_arguments(value["arguments"])[1] if source_file == BUTTON else None
    )
    return name, arguments


def result_values(value: Any, *, source_file: str | None) -> Iterator[str]:
    """The audited entry's results, never its echoed arguments, are evidence."""
    if source_file in _KEYS:
        if not isinstance(value, dict) or set(value) != _KEYS[source_file]:
            raise Reject("invalid_source_result_bundle")
        value = value["results"]
    yield from json_scalar_values(value)
