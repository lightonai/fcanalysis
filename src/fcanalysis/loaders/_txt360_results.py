"""TxT360 low's bounded inherited name/result array interpretation.

The pinned agent low release has 2,964 tool-role arrays of {name, results}
entries: 2,894 ordered unique-name matches, 29 repeated-name cases and 41
mismatches in the full-corpus census. High/medium have no such arrays. Runtime
labels are absent, so shape alone cannot declare an ordinary payload a protocol.
Only a complete contiguous call/result batch with matching cardinality, unique
names and matching released order proves this encoding. Other payloads retain
their original interpretation; ambiguous parallel linkage still quarantines.

Proved bundles split into singleton arrays, keeping full entries and order.
The same per-batch proof is rebuilt on each final linkage/context check using
transient message references. No sample metadata, IDs or state fields are added.
Only a proved entry's results supply external state; ordinary payloads keep
ordinary result-value semantics.
"""

from collections import Counter
from collections.abc import Iterator
from typing import Any

from .context import json_scalar_values
from .normalization import Reject, parse_json, serialize_result


def _entries(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, str):
        try:
            value = parse_json(value)
        except Reject:
            return None
    if (
        isinstance(value, list)
        and value
        and all(
            isinstance(entry, dict)
            and set(entry) == {"name", "results"}
            and isinstance(entry["name"], str)
            and entry["name"]
            for entry in value
        )
    ):
        return value
    return None


def _batches(
    messages: list[dict[str, Any]], *, canonical: bool
) -> Iterator[list[tuple[int, list[dict[str, Any]]]]]:
    for index, message in enumerate(messages):
        calls = message.get("tool_calls", []) if message["role"] == "assistant" else []
        if not calls or any(not isinstance(call, dict) for call in calls):
            continue
        names = [
            call.get("function", {}).get("name") if canonical else call.get("name")
            for call in calls
        ]
        if not all(isinstance(name, str) and name for name in names) or len(
            set(names)
        ) != len(names):
            continue
        results = []
        position = index + 1
        while position < len(messages) and messages[position]["role"] == "tool":
            entries = _entries(messages[position].get("content"))
            if entries is None or (canonical and len(entries) != 1):
                results = []
                break
            results.append((position, entries))
            position += 1
        if (
            results
            and [entry["name"] for _, entries in results for entry in entries] == names
        ):
            yield results


_WORDLE = "Get Today's Word"
_CHAMPION = "Get League of Legends Champion Meta Data"


def _reject_known_call_corruption(messages: list[dict[str, Any]]) -> None:
    """Four pinned low rows embed a second call inside Wordle's difficulty.

    low-00006-of-00007.parquet row 26948 is one example: the losslessly
    parsed string includes Champion Meta Data(rankname=...), while the result
    contains two named operation entries for the one recorded call. This is a
    specific released parser defect, not a general test for code-looking prose.
    """
    for index, message in enumerate(messages[:-1]):
        calls = message.get("tool_calls", []) if message["role"] == "assistant" else []
        if (
            len(calls) != 1
            or not isinstance(calls[0], dict)
            or calls[0].get("name") != _WORDLE
        ):
            continue
        result = messages[index + 1]
        if result["role"] != "tool":
            continue
        entries = _entries(result.get("content"))
        if entries is None or [entry["name"] for entry in entries] != [
            _WORDLE,
            _CHAMPION,
        ]:
            continue
        arguments = calls[0].get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = parse_json(arguments)
            except Reject:
                continue
        if (
            isinstance(arguments, dict)
            and set(arguments) == {"difficulty", "name"}
            and isinstance(arguments.get("difficulty"), str)
            and _CHAMPION + "(rankname=" in arguments["difficulty"]
        ):
            raise Reject("corrupted_source_call_serialization")


def expanded_results(
    messages: list[dict[str, Any]],
    *,
    split: str,
    transforms: Counter[str] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Recognize complete source batches before changing any result boundary."""
    expanded = {}
    if split == "low":
        _reject_known_call_corruption(messages)
        for results in _batches(messages, canonical=False):
            for index, entries in results:
                expanded[index] = [
                    {"role": "tool", "content": serialize_result([entry])}
                    for entry in entries
                ]
                if transforms is not None:
                    transforms["source_result_bundles_expanded"] += 1
                    transforms["source_result_entries_emitted"] += len(entries)
    return expanded


def proved_results(messages: list[dict[str, Any]], *, split: str | None) -> set[int]:
    """Rebuild contextual proof from the exact current canonical message view."""
    return {
        id(messages[index])
        for results in (_batches(messages, canonical=True) if split == "low" else ())
        for index, _ in results
    }


def result_echo(
    message: dict[str, Any], *, proved: set[int]
) -> tuple[str, dict[str, Any] | None] | None:
    if id(message) not in proved:
        return None
    entries = _entries(message["content"])
    assert entries is not None and len(entries) == 1
    return entries[0]["name"], None


def result_values(
    value: Any, message: dict[str, Any], *, proved: set[int]
) -> Iterator[str]:
    if id(message) in proved:
        entries = _entries(value)
        assert entries is not None and len(entries) == 1
        value = entries[0]["results"]
    yield from json_scalar_values(value)
