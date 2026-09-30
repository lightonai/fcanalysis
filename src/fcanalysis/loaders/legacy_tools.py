"""Opt-in conversion of the audited ToolMind/UltraData legacy tool grammar.

This stage preserves all definition annotations and instance values. It unwraps
one redundant function envelope, converts explicit argument/property maps,
lifts nonconflicting function-level required, and maps exact Python primitive
aliases in schema type positions. It never infers required fields, defaults,
nullability or unknown type encodings. Adapters must certify this grammar before
selecting it; ordinary canonical tools do not require these conversions.

Extracted unchanged from the ToolMind adapter after the same encodings were
observed in pinned UltraData GraphSyn, memory and search definitions.
"""

from collections import Counter
from copy import deepcopy
from typing import Any

from .normalization import Reject, json_bytes, normalize_tools
from .schema import normalize_schema_types

_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
}


def _is_parameter_map(parameters: dict[str, Any], required: Any = None) -> bool:
    # The released legacy encoding is a map of named parameter descriptors.
    # Abstain when the same object can be a valid JSON Schema keyword object.
    schema_keywords = {
        "not",
        "if",
        "then",
        "else",
        "items",
        "additionalItems",
        "additionalProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
        "contains",
        "propertyNames",
        "contentSchema",
        "default",
        "enum",
        "const",
    }
    if not parameters:
        return False
    if parameters.keys() & schema_keywords and not (
        isinstance(required, list)
        and required
        and all(isinstance(name, str) and name in parameters for name in required)
    ):
        return False
    return all(
        isinstance(value, dict)
        and (
            isinstance(value.get("type"), (str, list))
            or isinstance(value.get("description"), str)
        )
        for value in parameters.values()
    )


def _normalize_schema(schema: Any, counts: Counter[str]) -> Any:
    return normalize_schema_types(schema, _TYPE_MAP, transforms=counts)


def normalize_legacy_tool(
    tool: dict[str, Any], counts: Counter[str] | None = None
) -> dict[str, Any]:
    transforms = counts if counts is not None else Counter()
    if not isinstance(tool, dict) or tool.keys() - {
        "type",
        "function",
        "input_description",
        "output_description",
        "functionality",
        "output_structure",
    }:
        raise Reject("unresolved_source_definition_envelope")
    normalized = deepcopy(tool)
    if normalized.get("type") != "function" or not isinstance(
        normalized.get("function"), dict
    ):
        raise Reject("malformed_tool_definitions")
    function = normalized["function"]
    if set(function) == {"type", "function"}:
        if function["type"] != "function" or not isinstance(function["function"], dict):
            raise Reject("malformed_tool_definitions")
        function = function["function"]
        normalized["function"] = function
        transforms["double_function_envelopes_unwrapped"] += 1
    if function.keys() - {
        "name",
        "description",
        "parameters",
        "arguments",
        "required",
        "strict",
        "response",
    }:
        raise Reject("unknown_source_definition_fields")
    if "arguments" in function:
        if "parameters" in function or not isinstance(function["arguments"], dict):
            raise Reject("ambiguous_source_parameter_encoding")
        props = function.pop("arguments")
        if any(not isinstance(value, dict) for value in props.values()):
            raise Reject("invalid_source_property_map")
        function["parameters"] = {"type": "object", "properties": props}
        transforms["source_argument_maps_wrapped"] += 1
    if isinstance(function.get("parameters"), dict) and _is_parameter_map(
        function["parameters"], function.get("required")
    ):
        function["parameters"] = {
            "type": "object",
            "properties": function["parameters"],
        }
        transforms["source_parameter_maps_wrapped"] += 1
    if "parameters" in function:
        function["parameters"] = _normalize_schema(function["parameters"], transforms)
    if "required" in function:
        required = function.pop("required")
        if required is None:
            transforms["null_function_required_placeholders_removed"] += 1
        else:
            parameters = function.get("parameters")
            if not isinstance(required, list) or not isinstance(parameters, dict):
                raise Reject("invalid_legacy_required")
            if "required" in parameters and json_bytes(
                parameters["required"]
            ) != json_bytes(required):
                raise Reject("conflicting_legacy_required")
            parameters["required"] = required
            transforms["legacy_function_required_lifted"] += 1
    return normalize_tools([normalized])[0]
