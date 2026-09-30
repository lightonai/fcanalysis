"""Lossless JSON and function-definition primitives for source adapters.

Only JSON object-key order is ignored for equality. Source envelopes, embedded
system templates, compatibility projections and legacy repairs are adapter
decisions; none are guessed here. Mutable model-visible values are copied once
at conversion, never by every subsequent stage.
"""

import json
import math
from copy import deepcopy
from typing import Any

import orjson


class Reject(ValueError):
    """A deterministic quarantine reason, not an unexpected programming error."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise Reject("duplicate_json_key")
        result[key] = value
    return result


def _nonfinite(value: str) -> Any:
    raise Reject("nonfinite_json_number")


def parse_json(value: str) -> Any:
    """Reject ambiguous duplicate keys and non-JSON numerical values."""
    try:
        result = json.loads(value, object_pairs_hook=_object, parse_constant=_nonfinite)
        _check_json(result)
        return result
    except (ValueError, TypeError, RecursionError) as exc:
        if isinstance(exc, Reject):
            raise
        raise Reject("malformed_json") from exc


def _check_json(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise Reject("non_json_value")
            _check_json(item)
    elif isinstance(value, list):
        for item in value:
            _check_json(item)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise Reject("nonfinite_json_number")
    elif value is not None and type(value) not in (str, int, bool):
        raise Reject("non_json_value")


def json_bytes(value: Any) -> bytes:
    """Ephemeral exact comparison, preserving scalar types and array order."""
    _check_json(value)
    try:
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
    except TypeError, ValueError:
        # JSON integers have no 64-bit limit. Keep their exact decimal value;
        # converting them to floats would corrupt arguments and comparisons.
        # The checked stdlib fallback also rejects invalid Unicode on encoding.
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise Reject("non_json_value") from exc


def normalize_arguments(value: Any) -> tuple[str, dict[str, Any]]:
    parsed = parse_json(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise Reject("non_object_arguments")
    # Source JSON strings already are the model-visible payload. Parse once for
    # validation/comparison without rewriting their whitespace or key order.
    return value if isinstance(value, str) else json_bytes(parsed).decode(), parsed


def serialize_result(value: Any) -> str:
    return value if isinstance(value, str) else json_bytes(value).decode()


def normalize_tools(value: Any, *, bare: bool = False) -> list[dict[str, Any]]:
    """Normalize a known bare/enveloped list, retaining every source field."""
    parsed = parse_json(value) if isinstance(value, str) else deepcopy(value)
    if not isinstance(parsed, list):
        raise Reject("malformed_tool_definitions")
    result = []
    for item in parsed:
        if not isinstance(item, dict):
            raise Reject("malformed_tool_definitions")
        tool = {"type": "function", "function": item} if bare else item
        function = tool.get("function")
        if (
            tool.get("type") != "function"
            or not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"]
        ):
            raise Reject("malformed_tool_definitions")
        # Unknown model-visible fields are retained; adapters classify their
        # source schema explicitly. Schema support is checked before calls.
        json_bytes(tool)
        result.append(tool)
    return result


def reconcile_tools(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Stable first-seen union; incompatible exact names are quarantined."""
    if not isinstance(tools, list):
        raise Reject("malformed_tool_definitions")
    seen: dict[str, bytes] = {}
    result = []
    repeats = 0
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise Reject("malformed_tool_definitions")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise Reject("malformed_tool_definitions")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise Reject("malformed_tool_definitions")
        if "description" in function and not isinstance(function["description"], str):
            raise Reject("invalid_tool_description")
        if "strict" in function and not isinstance(
            function["strict"], (bool, type(None))
        ):
            raise Reject("invalid_tool_strict")
        body = json_bytes(tool)
        if name in seen:
            if seen[name] != body:
                raise Reject("conflicting_duplicate_tool_names")
            repeats += 1
        else:
            seen[name] = body
            result.append(tool)
    return result, repeats
