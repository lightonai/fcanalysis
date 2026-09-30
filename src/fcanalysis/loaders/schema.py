"""Explicit JSON Schema support, separate from argument validity.

Use the declared supported draft, otherwise Draft 2020-12. Unknown keywords,
unknown formats, remote references and unsupported vocabularies quarantine the
row rather than silently relaxing its definition. The supported format set is
exactly ``date``, ``ipv4``, ``ipv6``, and ``uuid``: the upstream checkers for
these use only Python's standard library. All other formats are unsupported,
including email, uri, and date-time. Ambient optional dependencies or later
global checker registrations cannot expand this contract.

Compiled validators use a bounded 8,192-entry LRU shared across rows. A pinned
20,000-row Dolci comparison reduced runtime from 27.62s to 14.54s versus 1,024
entries, with unchanged acceptance and about 42MB additional peak memory.
No source content is mutated or defaulted.
"""

from functools import lru_cache
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from typing import Any, cast

from jsonschema import Draft202012Validator, FormatChecker, validators
from jsonschema.exceptions import SchemaError

from .normalization import Reject, json_bytes, parse_json

_SUPPORTED_FORMATS = ("date", "ipv4", "ipv6", "uuid")
# Capture only the fixed supported checkers once; FormatChecker's default
# global set varies with optional imports and must never define our policy.
_FORMAT_CHECKER = FormatChecker(formats=_SUPPORTED_FORMATS)

_ANNOTATIONS = {
    "$schema",
    "$id",
    "$anchor",
    "$comment",
    "$defs",
    "definitions",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
}
_MAPS = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
_ARRAYS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_SINGLE = {
    "items",
    "additionalItems",
    "additionalProperties",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "propertyNames",
    "unevaluatedItems",
    "unevaluatedProperties",
}


def normalize_schema_types(
    schema: Any,
    aliases: Mapping[str, str],
    *,
    transforms: MutableMapping[str, int] | None = None,
) -> Any:
    """Map adapter-certified primitive aliases only in schema type positions.

    The adapter establishes the source encoding and supplies its exact map.
    Unknown types and union entries remain unchanged for support validation;
    defaults, enums, constants and other instance data are never traversed as
    schemas. Make one owned copy and preserve every other source field/value.
    """
    result = deepcopy(schema)

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        kind = node.get("type")
        if isinstance(kind, str) and kind in aliases:
            node["type"] = aliases[kind]
            if transforms is not None:
                key = "source_schema_type_aliases"
                transforms[key] = transforms.get(key, 0) + 1
        for key, value in node.items():
            if key in _MAPS and isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif key in _ARRAYS and isinstance(value, list):
                for child in value:
                    visit(child)
            elif key in _SINGLE:
                for child in value if isinstance(value, list) else [value]:
                    visit(child)
            elif key == "dependencies" and isinstance(value, dict):
                for child in value.values():
                    visit(child)

    visit(result)
    return result


def _support(schema: Any, cls: Any) -> None:
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise Reject("invalid_tool_schema")
    for key, value in schema.items():
        if key not in cls.VALIDATORS and key not in _ANNOTATIONS:
            raise Reject("unsupported_schema_keyword:" + key)
        if key in ("$ref", "$dynamicRef") and (
            not isinstance(value, str) or not value.startswith("#")
        ):
            raise Reject("unsupported_external_schema_reference")
        if key == "format" and value not in _SUPPORTED_FORMATS:
            raise Reject("unsupported_schema_format:" + str(value))
        if (
            key == "$schema"
            and validators.validator_for(schema, default=cast(Any, None)) is None
        ):
            raise Reject("unsupported_schema_draft")
        if key in _MAPS and isinstance(value, dict):
            for child in value.values():
                _support(child, cls)
        elif key in _ARRAYS and isinstance(value, list):
            for child in value:
                _support(child, cls)
        elif key in _SINGLE:
            if isinstance(value, list):
                for child in value:
                    _support(child, cls)
            else:
                _support(value, cls)
        elif key == "dependencies" and isinstance(value, dict):
            for child in value.values():
                if isinstance(child, (dict, bool)):
                    _support(child, cls)


@lru_cache(maxsize=8192)
def _compile(body: bytes) -> Any:
    schema = parse_json(body.decode())
    if not isinstance(schema, (dict, bool)):
        raise Reject("invalid_tool_schema")
    if (
        isinstance(schema, dict)
        and "$schema" in schema
        and not isinstance(schema["$schema"], str)
    ):
        raise Reject("invalid_tool_schema")
    cls = validators.validator_for(schema, default=Draft202012Validator)
    try:
        cls.check_schema(schema)
    except SchemaError as exc:
        raise Reject("invalid_tool_schema") from exc
    _support(schema, cls)
    return cls(schema, format_checker=_FORMAT_CHECKER)


def compile_schema(schema: Any) -> Any:
    return _compile(json_bytes(schema))


def check_arguments(arguments: dict[str, Any], validator: Any) -> None:
    try:
        error = next(validator.iter_errors(arguments), None)
    except Exception as exc:
        # Broken local references, invalid regexes and recursion are schema
        # uncertainty, never evidence that the argument is valid.
        raise Reject("unresolved_tool_schema") from exc
    if error is not None:
        raise Reject("invalid_arguments")
