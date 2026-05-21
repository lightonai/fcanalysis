"""Unit tests for fcanalysis.validation: tool call schema checks.

Covers the public predicates (has_undefined_function_calls,
has_invalid_arguments, validate_arguments) and the type-matching
helpers they compose. Behavior is the contract used by the universal
filters require_defined_functions and require_valid_arguments.
"""

from fcanalysis.validation import (
    _matches_single_type,
    _value_matches_schema_type,
    has_invalid_arguments,
    has_undefined_function_calls,
    validate_arguments,
)

from tests.helpers import assistant, call, func, sample


class TestMatchesSingleType:
    def test_string(self) -> None:
        assert _matches_single_type("x", "string") is True
        assert _matches_single_type(1, "string") is False

    def test_integer_rejects_bool(self) -> None:
        # bool is a subclass of int in Python; JSON Schema integer must reject it
        assert _matches_single_type(1, "integer") is True
        assert _matches_single_type(True, "integer") is False

    def test_number_rejects_bool(self) -> None:
        assert _matches_single_type(1.5, "number") is True
        assert _matches_single_type(1, "number") is True
        assert _matches_single_type(False, "number") is False

    def test_boolean(self) -> None:
        assert _matches_single_type(True, "boolean") is True
        assert _matches_single_type(1, "boolean") is False

    def test_array(self) -> None:
        assert _matches_single_type([1, 2], "array") is True
        assert _matches_single_type((1, 2), "array") is False
        assert _matches_single_type("x", "array") is False

    def test_object(self) -> None:
        assert _matches_single_type({"a": 1}, "object") is True
        assert _matches_single_type([], "object") is False

    def test_null(self) -> None:
        assert _matches_single_type(None, "null") is True
        assert _matches_single_type(0, "null") is False

    def test_unrecognized_type_is_permissive(self) -> None:
        assert _matches_single_type("anything", "weirdtype") is True


class TestValueMatchesSchemaType:
    def test_no_type_passes(self) -> None:
        assert _value_matches_schema_type(123, {}) is True

    def test_simple_type(self) -> None:
        assert _value_matches_schema_type("x", {"type": "string"}) is True
        assert _value_matches_schema_type(1, {"type": "string"}) is False

    def test_type_as_list(self) -> None:
        schema = {"type": ["string", "null"]}
        assert _value_matches_schema_type("x", schema) is True
        assert _value_matches_schema_type(None, schema) is True
        assert _value_matches_schema_type(1, schema) is False

    def test_enum(self) -> None:
        schema = {"enum": ["a", "b"]}
        assert _value_matches_schema_type("a", schema) is True
        assert _value_matches_schema_type("c", schema) is False

    def test_const(self) -> None:
        schema = {"const": 42}
        assert _value_matches_schema_type(42, schema) is True
        assert _value_matches_schema_type(43, schema) is False

    def test_any_of(self) -> None:
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        assert _value_matches_schema_type("x", schema) is True
        assert _value_matches_schema_type(1, schema) is True
        assert _value_matches_schema_type(1.5, schema) is False

    def test_enum_takes_priority_over_type(self) -> None:
        schema = {"type": "integer", "enum": ["only-this"]}
        assert _value_matches_schema_type("only-this", schema) is True
        assert _value_matches_schema_type(1, schema) is False


class TestValidateArguments:
    def test_all_required_present(self) -> None:
        params = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        assert validate_arguments({"x": "v"}, params) == []

    def test_missing_required(self) -> None:
        params = {
            "type": "object",
            "properties": {"x": {"type": "string"}, "y": {"type": "string"}},
            "required": ["x", "y"],
        }
        violations = validate_arguments({"x": "v"}, params)
        assert violations.count("missing_required") == 1
        assert "extra_param" not in violations
        assert "type_mismatch" not in violations

    def test_extra_param(self) -> None:
        params = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": [],
        }
        violations = validate_arguments({"x": "v", "y": "extra"}, params)
        assert violations.count("extra_param") == 1

    def test_type_mismatch(self) -> None:
        params = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": [],
        }
        violations = validate_arguments({"x": 1}, params)
        assert violations.count("type_mismatch") == 1

    def test_no_properties_skips_extra_and_type_check(self) -> None:
        params = {"type": "object", "required": ["x"]}
        # No properties → no extra-param check, no type check. Only required.
        assert validate_arguments({"x": "v", "extra": "y"}, params) == []
        assert validate_arguments({}, params) == ["missing_required"]

    def test_non_dict_property_schema_skipped(self) -> None:
        params = {
            "type": "object",
            "properties": {"x": True},
            "required": [],
        }
        assert validate_arguments({"x": "v"}, params) == []

    def test_multiple_violations(self) -> None:
        params = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }
        violations = validate_arguments({"b": 1}, params)
        assert "missing_required" in violations
        assert "extra_param" in violations


class TestHasUndefinedFunctionCalls:
    def test_all_defined(self) -> None:
        s = sample(
            messages=[assistant(tool_calls=[call(name="known")])],
            tools=[func(name="known")],
        )
        assert has_undefined_function_calls(s) is False

    def test_unknown_call(self) -> None:
        s = sample(
            messages=[assistant(tool_calls=[call(name="unknown")])],
            tools=[func(name="known")],
        )
        assert has_undefined_function_calls(s) is True

    def test_no_tool_calls(self) -> None:
        s = sample(
            messages=[assistant(content="text only")],
            tools=[func(name="known")],
        )
        assert has_undefined_function_calls(s) is False

    def test_empty_tools_with_calls(self) -> None:
        s = sample(messages=[assistant(tool_calls=[call(name="anything")])])
        assert has_undefined_function_calls(s) is True

    def test_non_function_tool_ignored(self) -> None:
        # Tools with type != "function" don't count as defined
        s = sample(
            messages=[assistant(tool_calls=[call(name="x")])],
            tools=[{"type": "retrieval", "function": {"name": "x"}}],
        )
        assert has_undefined_function_calls(s) is True

    def test_one_known_one_unknown(self) -> None:
        s = sample(
            messages=[
                assistant(
                    tool_calls=[
                        call(name="known", call_id="1"),
                        call(name="unknown", call_id="2"),
                    ]
                )
            ],
            tools=[func(name="known")],
        )
        assert has_undefined_function_calls(s) is True


class TestHasInvalidArguments:
    def _func_with(self, name: str, **schema_extra: object) -> dict:
        params: dict = {
            "type": "object",
            "properties": schema_extra.get("properties", {}),
            "required": schema_extra.get("required", []),
        }
        return func(name=name, parameters=params)

    def test_valid_arguments_pass(self) -> None:
        s = sample(
            messages=[assistant(tool_calls=[call(name="f", arguments='{"x":"v"}')])],
            tools=[
                self._func_with(
                    "f", properties={"x": {"type": "string"}}, required=["x"]
                )
            ],
        )
        assert has_invalid_arguments(s) is False

    def test_missing_required(self) -> None:
        s = sample(
            messages=[assistant(tool_calls=[call(name="f", arguments="{}")])],
            tools=[
                self._func_with(
                    "f", properties={"x": {"type": "string"}}, required=["x"]
                )
            ],
        )
        assert has_invalid_arguments(s) is True

    def test_extra_param(self) -> None:
        s = sample(
            messages=[assistant(tool_calls=[call(name="f", arguments='{"y":"v"}')])],
            tools=[
                self._func_with("f", properties={"x": {"type": "string"}}, required=[])
            ],
        )
        assert has_invalid_arguments(s) is True

    def test_type_mismatch(self) -> None:
        s = sample(
            messages=[assistant(tool_calls=[call(name="f", arguments='{"x":1}')])],
            tools=[
                self._func_with("f", properties={"x": {"type": "string"}}, required=[])
            ],
        )
        assert has_invalid_arguments(s) is True

    def test_undefined_function_skipped(self) -> None:
        # Calls to undefined functions are NOT flagged here (handled
        # separately by has_undefined_function_calls).
        s = sample(
            messages=[
                assistant(tool_calls=[call(name="unknown", arguments='{"x":1}')])
            ],
            tools=[
                self._func_with(
                    "f", properties={"x": {"type": "string"}}, required=["x"]
                )
            ],
        )
        assert has_invalid_arguments(s) is False

    def test_unparseable_arguments_skipped(self) -> None:
        # Unparseable args are not flagged here (handled separately).
        s = sample(
            messages=[assistant(tool_calls=[call(name="f", arguments="{bad")])],
            tools=[
                self._func_with("f", properties={"x": {"type": "string"}}, required=[])
            ],
        )
        assert has_invalid_arguments(s) is False

    def test_dict_arguments_passed_through(self) -> None:
        # Args already as dict (some loaders pass this through) are checked.
        tc = call(name="f")
        tc["function"]["arguments"] = {"x": 1}
        s = sample(
            messages=[assistant(tool_calls=[tc])],
            tools=[
                self._func_with("f", properties={"x": {"type": "string"}}, required=[])
            ],
        )
        assert has_invalid_arguments(s) is True

    def test_non_dict_parsed_args_flagged(self) -> None:
        # If args parse to a non-dict (list, scalar), flagged as invalid.
        s = sample(
            messages=[assistant(tool_calls=[call(name="f", arguments="[1,2]")])],
            tools=[
                self._func_with("f", properties={"x": {"type": "string"}}, required=[])
            ],
        )
        assert has_invalid_arguments(s) is True

    def test_no_parameters_schema_passes(self) -> None:
        # Tool with no parameters schema: any args are fine.
        s = sample(
            messages=[assistant(tool_calls=[call(name="f", arguments='{"x":"v"}')])],
            tools=[{"type": "function", "function": {"name": "f"}}],
        )
        assert has_invalid_arguments(s) is False
