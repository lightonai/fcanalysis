import json
from typing import Any

from .format import ConversationSample


def _matches_single_type(value: Any, type_name: str) -> bool:
    # bool is a subclass of int in Python; JSON Schema "integer" and "number"
    # must not match True/False, hence the explicit exclusion.
    match type_name:
        case "string":
            return isinstance(value, str)
        case "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        case "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        case "boolean":
            return isinstance(value, bool)
        case "array":
            return isinstance(value, list)
        case "object":
            return isinstance(value, dict)
        case "null":
            return value is None
        case _:
            return True


def _value_matches_schema_type(value: Any, schema: dict[str, Any]) -> bool:
    match schema:
        case {"enum": enum}:
            return value in enum
        case {"const": const}:
            return value == const
        case {"anyOf": any_of}:
            return any(_value_matches_schema_type(value, sub) for sub in any_of)

    schema_type = schema.get("type")
    if schema_type is None:
        return True
    if isinstance(schema_type, list):
        return any(_matches_single_type(value, t) for t in schema_type)
    return _matches_single_type(value, schema_type)


def validate_arguments(
    arguments: dict[str, Any],
    parameters: dict[str, Any],
) -> list[str]:
    violations: list[str] = []

    for name in parameters.get("required", []):
        if name not in arguments:
            violations.append("missing_required")

    properties = parameters.get("properties")
    if isinstance(properties, dict):
        for key in arguments:
            if key not in properties:
                violations.append("extra_param")

        for key, value in arguments.items():
            schema = properties.get(key)
            if isinstance(schema, dict) and not _value_matches_schema_type(
                value, schema
            ):
                violations.append("type_mismatch")

    return violations


def _function_definitions(
    sample: ConversationSample,
) -> dict[str, dict[str, Any] | None]:
    result: dict[str, dict[str, Any] | None] = {}
    for tool in sample.tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        func = tool.get("function")
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not isinstance(name, str):
            continue
        params = func.get("parameters")
        result[name] = params if isinstance(params, dict) else None
    return result


def has_undefined_function_calls(sample: ConversationSample) -> bool:
    defined = _function_definitions(sample)
    for msg in sample.messages:
        if msg["role"] != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            if tc["function"]["name"] not in defined:
                return True
    return False


def has_invalid_arguments(sample: ConversationSample) -> bool:
    schema_map = {
        name: params
        for name, params in _function_definitions(sample).items()
        if params is not None
    }
    for msg in sample.messages:
        if msg["role"] != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            fn_name = tc["function"]["name"]
            if fn_name not in schema_map:
                continue

            args_raw = tc["function"].get("arguments", "")
            if isinstance(args_raw, str):
                try:
                    parsed = json.loads(args_raw)
                except json.JSONDecodeError, ValueError:
                    continue
            else:
                parsed = args_raw

            if not isinstance(parsed, dict):
                return True

            if validate_arguments(parsed, schema_map[fn_name]):
                return True

    return False
