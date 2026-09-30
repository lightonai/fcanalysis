"""Adversarial contracts for source-independent normalization and row stages."""

from copy import deepcopy
from functools import partial

import pytest
from jsonschema import FormatChecker

from fcanalysis.format import ConversationSample
from fcanalysis.loaders.curation import BoundCallBatch
from fcanalysis.loaders.normalization import (
    Reject,
    json_bytes,
    normalize_arguments,
    normalize_tools,
    parse_json,
    reconcile_tools,
    serialize_result,
)
from fcanalysis.loaders.pipeline import (
    CapabilityChange,
    JobEvent,
    Pipeline,
    RowState,
    link_calls,
    override_system,
    reconcile_definitions,
    remove_reasoning,
    validate_capabilities,
    validate_lifecycle,
    validate_structure,
    validate_termination,
)
from fcanalysis.loaders.schema import (
    check_arguments,
    compile_schema,
    normalize_schema_types,
)


@pytest.mark.parametrize("number", [51090942171709440000, -(2**80)])
def test_json_large_integers_remain_exact_in_arguments_results_and_schemas(number):
    value = {"integer": number, "nested": [number + 1, "É.", 1.0]}
    encoded = json_bytes(value)
    assert parse_json(encoded.decode()) == value
    assert str(number).encode() in encoded
    assert normalize_arguments(value) == (encoded.decode(), value)
    assert serialize_result(value) == encoded.decode()
    validator = compile_schema(
        {
            "type": "object",
            "properties": {"integer": {"type": "integer", "const": number}},
        }
    )
    check_arguments({"integer": number}, validator)
    with pytest.raises(Reject, match="invalid_arguments"):
        check_arguments({"integer": number + 1}, validator)


def test_invalid_unicode_does_not_become_valid_through_json_fallback():
    with pytest.raises(Reject, match="non_json_value"):
        json_bytes({"text": "\ud800"})


def test_schema_aliases_preserve_instance_data_unknown_types_and_raw():
    source = {
        "type": "dict",
        "properties": {
            "count": {"type": "int", "minimum": 1, "maximum": 9},
            "items": {
                "type": "list",
                "items": {"type": "dict", "properties": {"weight": {"type": "float"}}},
            },
            "unknown": {"type": "any"},
            "union": {"type": ["int", "null"]},
        },
        "default": {"type": "int"},
        "enum": [{"type": "int"}],
        "const": {"nested": {"type": "int"}},
        "description": "type=int",
    }
    before = deepcopy(source)
    counts = {}
    result = normalize_schema_types(
        source,
        {"dict": "object", "int": "integer", "list": "array", "float": "number"},
        transforms=counts,
    )
    assert source == before
    assert result["type"] == "object"
    assert result["properties"]["count"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 9,
    }
    assert (
        result["properties"]["items"]["items"]["properties"]["weight"]["type"]
        == "number"
    )
    assert result["properties"]["unknown"] == source["properties"]["unknown"]
    assert result["properties"]["union"] == source["properties"]["union"]
    for key in ("default", "enum", "const", "description"):
        assert result[key] == source[key]
    assert counts == {"source_schema_type_aliases": 5}


@pytest.mark.parametrize(
    "schema",
    [
        {"allOf": [{"type": "int"}]},
        {"$defs": {"x": {"type": "int"}}},
        {"dependentSchemas": {"x": {"type": "int"}}},
        {"dependencies": {"x": {"type": "int"}}},
        {"items": [{"type": "int"}]},
        {"if": {"type": "int"}},
        {"unevaluatedItems": {"type": "int"}},
    ],
)
def test_alias_walker_visits_supported_schema_positions(schema):
    counts = {}
    converted = normalize_schema_types(schema, {"int": "integer"}, transforms=counts)
    assert counts == {"source_schema_type_aliases": 1}
    assert converted != schema


def _tool(name="f", parameters=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Perform an action",
            "parameters": parameters if parameters is not None else {"type": "object"},
        },
    }


def _call(name="f", arguments="{}", **extra):
    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
        **extra,
    }


def _decode_result_echo(message):
    value = parse_json(message["content"])
    if not isinstance(value, dict) or set(value) != {"name", "arguments", "results"}:
        return None
    return value["name"], value["arguments"]


def _echo_result(number, result, **fields):
    return {
        "role": "tool",
        "content": json_bytes(
            {"name": "f", "arguments": {"n": number}, "results": result}
        ).decode(),
        **fields,
    }


def test_complete_result_echo_proves_same_function_pairing_without_changing_content():
    state = RowState(
        _episode(
            calls=[_call(arguments='{"n":1}'), _call(arguments='{"n":2}')],
            results=[_echo_result(1, "one"), _echo_result(2, "two")],
        )
    )
    before = deepcopy(state.sample)
    validate_structure(state)
    link_calls(state, result_echo=_decode_result_echo)
    assert state.sample == before
    assert state.batches == (BoundCallBatch(1, (2, 3), False),)
    override_system(state, override="Updated system")
    validate_structure(state)
    link_calls(state, result_echo=_decode_result_echo)
    assert state.batches == (BoundCallBatch(2, (3, 4), False),)


@pytest.mark.parametrize("parallel", [False, True])
def test_echo_alignment_defaults_on_independently_of_parallel_comparison(parallel):
    state = RowState(
        _episode(
            calls=[_call(arguments='{"n":1}'), _call(arguments='{"n":2}')],
            results=[_echo_result(2, "two"), _echo_result(1, "one")],
        )
    )
    validate_structure(state)
    with pytest.raises(Reject, match="unproven_result_reordering"):
        link_calls(
            state,
            result_echo=_decode_result_echo,
            parallel=parallel,
            align_results=False,
        )
    link_calls(state, result_echo=_decode_result_echo, parallel=parallel)
    assert state.sample.messages[2]["content"] == _echo_result(1, "one")["content"]
    assert state.batches == (BoundCallBatch(1, (2, 3), parallel),)


@pytest.mark.parametrize(
    "numbers, results, reason",
    [
        (
            [1, 1],
            [_echo_result(1, "one"), _echo_result(1, "other")],
            "ambiguous_source_tool_linkage",
        ),
        (
            [1, 2],
            [_echo_result(1, "one"), _echo_result(1, "other")],
            "invalid_source_tool_linkage",
        ),
        (
            [1, 2],
            [_echo_result(1, "one"), _echo_result(3, "other")],
            "invalid_source_tool_linkage",
        ),
        (
            [1, 2],
            [_echo_result(1, "one"), {"role": "tool", "content": '"no echo"'}],
            "invalid_source_tool_linkage",
        ),
        ([1], [_echo_result(1.0, "one")], "invalid_source_tool_linkage"),
    ],
)
def test_partial_duplicate_or_changed_echoes_do_not_prove_pairs(
    numbers, results, reason
):
    state = RowState(
        _episode(
            calls=[_call(arguments=json_bytes({"n": n}).decode()) for n in numbers],
            results=results,
        )
    )
    validate_structure(state)
    with pytest.raises(Reject, match=reason):
        link_calls(state, result_echo=_decode_result_echo)


def test_result_echo_cannot_contradict_explicit_ids_or_names():
    for results in [
        [
            _echo_result(1, "one", tool_call_id="b"),
            _echo_result(2, "two", tool_call_id="a"),
        ],
        [
            _echo_result(1, "one", tool_call_id="a", name="other"),
            _echo_result(2, "two", tool_call_id="b"),
        ],
    ]:
        state = RowState(
            _episode(
                calls=[
                    _call(arguments='{"n":1}', id="a"),
                    _call(arguments='{"n":2}', id="b"),
                ],
                results=results,
            )
        )
        validate_structure(state)
        with pytest.raises(Reject, match="invalid_source_tool_linkage"):
            link_calls(state, result_echo=_decode_result_echo, parallel=True)


def test_name_only_content_echo_survives_final_recheck_without_top_level_names():
    def decode(message):
        value = parse_json(message["content"])
        return value["name"], None

    state = RowState(
        _episode(
            calls=[_call("f"), _call("g")],
            results=[
                {"role": "tool", "content": '{"name":"f","results":"one"}'},
                {"role": "tool", "content": '{"name":"g","results":"two"}'},
            ],
        )
    )
    before = deepcopy(state.sample)
    validate_structure(state)
    link_calls(state, result_echo=decode)
    link_calls(state, result_echo=decode)
    assert state.sample == before
    state.sample.messages[1]["tool_calls"][1]["function"]["name"] = "f"
    with pytest.raises(Reject, match="ambiguous_source_tool_linkage"):
        link_calls(state, result_echo=decode)


@pytest.mark.parametrize("echo_arguments", [None, {}])
def test_explicit_ids_disambiguate_repeated_echo_signatures(echo_arguments):
    state = RowState(
        _episode(
            calls=[_call(id="a"), _call(id="b")],
            results=[
                {
                    "role": "tool",
                    "tool_call_id": "a",
                    "content": json_bytes(
                        {"name": "f", "arguments": echo_arguments, "results": "one"}
                    ).decode(),
                },
                {
                    "role": "tool",
                    "tool_call_id": "b",
                    "content": json_bytes(
                        {"name": "f", "arguments": echo_arguments, "results": "two"}
                    ).decode(),
                },
            ],
        )
    )
    validate_structure(state)
    link_calls(state, result_echo=_decode_result_echo)
    assert state.batches == (BoundCallBatch(1, (2, 3), False),)


@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("name", ["f", "wrong"])
def test_partial_top_level_names_only_supplement_an_independent_binding(
    positional, name
):
    state = RowState(
        _episode(
            calls=[_call("f"), _call("g")],
            results=[
                {"role": "tool", "name": name, "content": "first"},
                {"role": "tool", "content": "second"},
            ],
        )
    )
    validate_structure(state)
    if positional and name == "f":
        link_calls(state, positional=True)
        assert state.sample.messages[2:4] == [
            {"role": "tool", "content": "first"},
            {"role": "tool", "content": "second"},
        ]
    else:
        reason = (
            "invalid_source_tool_linkage"
            if positional
            else "ambiguous_source_tool_linkage"
        )
        with pytest.raises(Reject, match=reason):
            link_calls(state, positional=positional)


@pytest.mark.parametrize("reference", [None, "missing", ""])
def test_all_null_ids_are_absent_but_partial_or_empty_ids_never_fall_back(reference):
    state = RowState(
        _episode(
            calls=[_call(id=None)],
            results=[{"role": "tool", "tool_call_id": reference, "content": "OK"}],
        )
    )
    validate_structure(state)
    if reference is None:
        link_calls(state)
        assert "id" not in state.sample.messages[1]["tool_calls"][0]
        assert state.sample.messages[2] == {"role": "tool", "content": "OK"}
    else:
        with pytest.raises(Reject, match="invalid_source_tool_linkage"):
            link_calls(state, positional=True)


@pytest.mark.parametrize("ids", [False, True])
def test_mixed_full_and_name_only_echoes_require_complete_ids(ids):
    calls = [_call(id="a"), _call(id="b")]
    results = [
        {
            "role": "tool",
            "tool_call_id": "a",
            "content": '{"name":"f","arguments":null,"results":"first"}',
        },
        {
            "role": "tool",
            "tool_call_id": "b",
            "content": '{"name":"f","arguments":{},"results":"second"}',
        },
    ]
    if not ids:
        for call in calls:
            del call["id"]
        for result in results:
            del result["tool_call_id"]
    state = RowState(_episode(calls=calls, results=results))
    validate_structure(state)
    if ids:
        before = [result["content"] for result in results]
        link_calls(state, result_echo=_decode_result_echo)
        assert [m["content"] for m in state.sample.messages[2:4]] == before
    else:
        with pytest.raises(Reject, match="invalid_source_tool_linkage"):
            link_calls(state, positional=True, result_echo=_decode_result_echo)


def _sample(messages=None, tools=None):
    raw = {
        "messages": messages
        if messages is not None
        else [
            {"role": "user", "content": "Please do it"},
            {"role": "assistant", "content": "Done"},
        ],
        "tools": tools if tools is not None else [_tool()],
    }
    return ConversationSample(
        messages=deepcopy(raw["messages"]),
        tools=normalize_tools(raw["tools"]),
        dataset="test",
        sample_id="test",
        raw=raw,
    )


def _episode(calls=None, results=None, tools=None):
    return _sample(
        [
            {"role": "user", "content": "Please do it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": calls if calls is not None else [_call()],
            },
            *(results if results is not None else [{"role": "tool", "content": "OK"}]),
            {"role": "assistant", "content": "Done"},
        ],
        tools=tools,
    )


def _state(sample):
    state = RowState(sample)
    reconcile_definitions(state)
    validate_structure(state)
    return state


@pytest.mark.parametrize(
    "value,reason",
    [
        ('{"x":1,"x":2}', "duplicate_json_key"),
        ('{"x":{"nested":1,"nested":2}}', "duplicate_json_key"),
        ('{"x":NaN}', "nonfinite_json_number"),
        ('{"x":Infinity}', "nonfinite_json_number"),
        ('{"x":-Infinity}', "nonfinite_json_number"),
        ('{"x":1e999}', "nonfinite_json_number"),
        ('{"x":}', "malformed_json"),
        ("{'x':1}", "malformed_json"),
    ],
)
def test_json_rejects_ambiguous_or_non_json_source_values(value, reason):
    with pytest.raises(Reject, match=reason):
        parse_json(value)


@pytest.mark.parametrize(
    "value", [{1: "key"}, {"x": (1, 2)}, {"x": {1}}, {"x": object()}]
)
def test_structured_normalization_rejects_non_json_python_values(value):
    with pytest.raises(Reject, match="non_json_value"):
        json_bytes(value)


@pytest.mark.parametrize("value", ["[]", "null", "1", '"text"', [], None])
def test_arguments_must_be_an_object(value):
    with pytest.raises(Reject, match="non_object_arguments"):
        normalize_arguments(value)


def test_arguments_and_results_preserve_source_string_lexical_content():
    source = '  {"z": [2, 1], "a": "  literal  "}\n'
    serialized, parsed = normalize_arguments(source)
    assert serialized == source
    assert parsed == {"z": [2, 1], "a": "  literal  "}
    assert serialize_result(source) == source
    assert serialize_result({"b": [2, 1], "a": True}) == '{"a":true,"b":[2,1]}'


def test_complete_tool_values_and_absent_optional_fields_are_preserved():
    source = [
        {
            "name": " F ",
            "strict": False,
            "x-extension": {"value": [1, 2]},
            "parameters": {"type": "object", "default": {"description": "literal"}},
        }
    ]
    tools = normalize_tools(source, bare=True)
    assert tools == [{"type": "function", "function": source[0]}]
    assert "description" not in tools[0]["function"]
    tools[0]["function"]["parameters"]["default"]["description"] = "changed"
    assert source[0]["parameters"]["default"]["description"] == "literal"


@pytest.mark.parametrize(
    "source",
    [
        None,
        {},
        [None],
        [{"type": "other", "function": {"name": "f"}}],
        [{"type": "function", "function": {"name": ""}}],
    ],
)
def test_malformed_definition_envelopes_are_explicit_rejections(source):
    with pytest.raises(Reject, match="malformed_tool_definitions"):
        normalize_tools(source)


def test_reconciliation_preserves_first_seen_union_and_case_sensitive_names():
    first, other = _tool("f"), _tool("F")
    repeated = {
        "function": dict(reversed(list(first["function"].items()))),
        "type": "function",
    }
    tools, count = reconcile_tools([first, repeated, other])
    assert tools == [first, other]
    assert tools[0] is first
    assert count == 1


@pytest.mark.parametrize(
    "tools",
    [
        None,
        {},
        [None],
        [{}],
        [{"type": "function"}],
        [{"type": "function", "function": []}],
        [{"type": "function", "function": {"name": []}}],
        [{"type": "function", "function": {"name": ""}}],
    ],
)
def test_shared_reconciliation_rejects_malformed_tools_without_converter_assumptions(
    tools,
):
    with pytest.raises(Reject, match="malformed_tool_definitions"):
        reconcile_tools(tools)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("description", None, "invalid_tool_description"),
        ("description", [], "invalid_tool_description"),
        ("strict", "true", "invalid_tool_strict"),
        ("strict", 1, "invalid_tool_strict"),
    ],
)
def test_model_visible_descriptor_fields_require_supported_values(field, value, reason):
    tool = _tool()
    tool["function"][field] = value
    with pytest.raises(Reject, match=reason):
        reconcile_tools([tool])


@pytest.mark.parametrize("system", [None, "", "Source policy"])
def test_no_tool_no_call_rows_preserve_system_absence_emptiness_and_text(system):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Welcome"},
    ]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    state = _state(_sample(messages, tools=[]))
    link_calls(state)
    validate_capabilities(state)
    validate_termination(state)
    assert state.sample.messages == messages
    assert state.sample.tools == []


@pytest.mark.parametrize(
    "change",
    [
        {"description": "different"},
        {"strict": False},
        {"parameters": {"type": "object", "required": []}},
    ],
)
def test_same_name_definition_differences_never_silently_select_one(change):
    first = _tool()
    second = {"type": "function", "function": {**first["function"], **change}}
    with pytest.raises(Reject, match="conflicting_duplicate_tool_names"):
        reconcile_tools([first, second])


@pytest.mark.parametrize(
    "messages,reason",
    [
        (None, "malformed_messages"),
        ({"role": "user"}, "malformed_messages"),
        ([None], "malformed_message"),
        (["unknown"], "malformed_message"),
        ([], "no_user_message"),
        ([{"role": "assistant", "content": "A"}], "no_user_message"),
        (
            [{"role": "user", "content": "U"}, {"role": "function", "content": "R"}],
            "unknown_role",
        ),
        ([{"role": "user", "content": "U", "tool_calls": []}], "unknown_message_field"),
        ([{"role": "user", "content": ["multimodal"]}], "unsupported_content_shape"),
        (
            [
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": "A", "name": "hidden"},
            ],
            "unknown_message_field",
        ),
        (
            [
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": "A", "reasoning_content": {}},
            ],
            "unsupported_reasoning_shape",
        ),
    ],
)
def test_canonical_structure_reports_unknown_or_malformed_values(messages, reason):
    sample = _sample()
    sample.messages = messages
    with pytest.raises(Reject, match=reason):
        validate_structure(RowState(sample))


@pytest.mark.parametrize(
    "calls,reason",
    [
        (None, "malformed_tool_calls"),
        ([None], "unknown_call_field"),
        ([_call(extra="hidden")], "unknown_call_field"),
        (
            [{"type": "unknown", "function": {"name": "f", "arguments": "{}"}}],
            "malformed_tool_call",
        ),
        ([_call(name="")], "malformed_tool_call"),
        ([{"type": "function", "function": {"name": "f"}}], "malformed_tool_call"),
        ([_call(arguments="[]")], "non_object_arguments"),
    ],
)
def test_canonical_calls_reject_silent_field_or_shape_loss(calls, reason):
    sample = _sample(
        [
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A", "tool_calls": calls},
        ]
    )
    with pytest.raises(Reject, match=reason):
        validate_structure(RowState(sample))


def test_structure_allows_consecutive_assistants_and_parallel_result_messages():
    sample = _episode(
        [_call("a"), _call("b")],
        [{"role": "tool", "content": "A"}, {"role": "tool", "content": "B"}],
    )
    sample.messages.insert(1, {"role": "assistant", "content": "I will act"})
    validate_structure(RowState(sample))


def test_validated_argument_cache_is_reused_and_rebuilt_after_change():
    state = _state(_episode([_call(arguments=' {"x": 1} ')]))
    original = state.parsed_arguments[1, 0]
    validate_structure(state)
    assert state.parsed_arguments[1, 0] is original
    state.sample.messages[1]["tool_calls"][0]["function"]["arguments"] = '{"x":2}'
    validate_structure(state)
    assert state.parsed_arguments[1, 0] == {"x": 2}
    assert (
        state.sample.raw["messages"][1]["tool_calls"][0]["function"]["arguments"]
        == ' {"x": 1} '
    )


def test_id_pairing_reorders_complete_results_and_strips_only_protocol_evidence():
    state = _state(
        _episode(
            [_call("a", id="a"), _call("b", id="b")],
            [
                {"role": "tool", "content": "B", "tool_call_id": "b", "name": "b"},
                {"role": "tool", "content": "A", "tool_call_id": "a", "name": "a"},
            ],
        )
    )
    raw = deepcopy(state.sample.raw)
    link_calls(state, parallel=True)
    assert [message["content"] for message in state.sample.messages[2:4]] == ["A", "B"]
    assert state.batches == (BoundCallBatch(1, (2, 3), True),)
    assert state.transforms["result_batches_reordered"] == 1
    assert state.transforms["source_linkage_fields_removed"] == 6
    assert state.sample.raw == raw
    assert all("id" not in call for call in state.sample.messages[1]["tool_calls"])


@pytest.mark.parametrize(
    "calls,results",
    [
        ([_call(id="a")], [{"role": "tool", "content": "R"}]),
        ([_call()], [{"role": "tool", "content": "R", "tool_call_id": "a"}]),
        ([_call(id="a")], [{"role": "tool", "content": "R", "tool_call_id": "b"}]),
        ([_call(id=1)], [{"role": "tool", "content": "R", "tool_call_id": 1}]),
        (
            [_call(id="a"), _call(id="a")],
            [{"role": "tool", "content": "R", "tool_call_id": "a"}] * 2,
        ),
        (
            [_call("f", id="a")],
            [{"role": "tool", "content": "R", "tool_call_id": "a", "name": "wrong"}],
        ),
        ([_call("f")], [{"role": "tool", "content": "R", "name": "wrong"}]),
    ],
)
def test_contradictory_linkage_cannot_be_rescued_by_positional_policy(calls, results):
    state = _state(_episode(calls, results))
    with pytest.raises(Reject, match="invalid_source_tool_linkage"):
        link_calls(state, positional=True, parallel=True)


def test_unique_result_names_can_prove_reordered_pairing():
    state = _state(
        _episode(
            [_call("a"), _call("b")],
            [
                {"role": "tool", "content": "B", "name": "b"},
                {"role": "tool", "content": "A", "name": "a"},
            ],
        )
    )
    link_calls(state, parallel=True)
    assert [message["content"] for message in state.sample.messages[2:4]] == ["A", "B"]


@pytest.mark.parametrize("evidence", ["id", "name"])
@pytest.mark.parametrize("parallel", [False, True])
def test_alignment_opt_out_is_independent_of_pairing_and_parallel(evidence, parallel):
    calls = [_call("a"), _call("b")]
    results = [{"role": "tool", "content": "B"}, {"role": "tool", "content": "A"}]
    if evidence == "id":
        calls[0]["id"], calls[1]["id"] = "a", "b"
        results[0]["tool_call_id"], results[1]["tool_call_id"] = "b", "a"
    else:
        results[0]["name"], results[1]["name"] = "b", "a"
    state = _state(_episode(calls, results))
    original = deepcopy(state.sample)
    with pytest.raises(Reject, match="unproven_result_reordering"):
        link_calls(state, positional=True, parallel=parallel, align_results=False)
    assert state.sample == original
    assert state.transforms["result_batches_reordered"] == 0
    link_calls(state, parallel=parallel)
    assert [message["content"] for message in state.sample.messages[2:4]] == ["A", "B"]
    assert [c["function"]["name"] for c in state.sample.messages[1]["tool_calls"]] == [
        "a",
        "b",
    ]
    assert state.batches == (BoundCallBatch(1, (2, 3), parallel),)
    assert state.sample.raw == original.raw


@pytest.mark.parametrize("align_results", [False, True])
def test_nonparallel_id_linkage_preserves_already_ordered_results(align_results):
    state = _state(
        _episode(
            [_call("a", id="a"), _call("b", id="b")],
            [
                {"role": "tool", "content": "A", "tool_call_id": "a"},
                {"role": "tool", "content": "B", "tool_call_id": "b"},
            ],
        )
    )
    link_calls(state, parallel=False, align_results=align_results)
    assert [message["content"] for message in state.sample.messages[2:4]] == ["A", "B"]
    assert state.transforms["result_batches_reordered"] == 0


def test_default_alignment_keeps_sequential_batches_and_result_payloads_intact():
    calls = [_call("a", id="a"), _call("b", id="b")]
    results = [
        {"role": "tool", "tool_call_id": "b", "content": '["B", [2, 3]]'},
        {"role": "tool", "tool_call_id": "a", "content": ' {"result": "A"} '},
    ]
    sample = _episode(calls, results)
    sample.messages[-1:-1] = [
        {"role": "assistant", "content": "Next", "tool_calls": [_call("c", id="c")]},
        {"role": "tool", "tool_call_id": "c", "content": "C"},
    ]
    state = _state(sample)
    link_calls(state)
    assert [m["content"] for m in state.sample.messages[2:4]] == [
        ' {"result": "A"} ',
        '["B", [2, 3]]',
    ]
    assert state.sample.messages[4]["tool_calls"][0]["function"]["name"] == "c"
    assert state.sample.messages[5]["content"] == "C"
    assert state.batches == (
        BoundCallBatch(1, (2, 3), False),
        BoundCallBatch(4, (5,), False),
    )


def test_parallel_cardinality_alone_is_not_linkage_evidence():
    state = _state(
        _episode(
            [_call(), _call()],
            [{"role": "tool", "content": "1"}, {"role": "tool", "content": "2"}],
        )
    )
    with pytest.raises(Reject, match="ambiguous_source_tool_linkage"):
        link_calls(state, parallel=True)
    link_calls(state, positional=True, parallel=True)
    assert state.batches[0].result_indices == (2, 3)


@pytest.mark.parametrize(
    "results",
    [[], [{"role": "tool", "content": "1"}, {"role": "tool", "content": "2"}]],
)
def test_missing_and_extra_results_fail_cardinality(results):
    with pytest.raises(Reject, match="unbalanced_cardinality"):
        link_calls(_state(_episode(results=results)), positional=True)


def test_orphan_results_are_rejected_without_naive_role_alternation():
    sample = _sample(
        [
            {"role": "user", "content": "U"},
            {"role": "tool", "content": "R"},
            {"role": "assistant", "content": "A"},
        ]
    )
    with pytest.raises(Reject, match="orphan_tool_result"):
        link_calls(_state(sample))


@pytest.mark.parametrize(
    "schema",
    [
        None,
        [],
        1,
        {"format": []},
        {"format": {}},
        {"$schema": []},
        {"$schema": False},
        {"required": "x"},
        {"type": "unknown"},
        {"properties": {"x": 1}},
        {"minimum": "0"},
    ],
)
def test_invalid_schema_values_are_distinct_from_invalid_arguments(schema):
    with pytest.raises(Reject, match="invalid_tool_schema"):
        compile_schema(schema)


@pytest.mark.parametrize(
    "schema,reason",
    [
        ({"x-constraint": 1}, "unsupported_schema_keyword:x-constraint"),
        (
            {"properties": {"x": {"x-constraint": 1}}},
            "unsupported_schema_keyword:x-constraint",
        ),
        (
            {"format": "unpublished-format"},
            "unsupported_schema_format:unpublished-format",
        ),
        (
            {"$ref": "https://example.com/schema.json"},
            "unsupported_external_schema_reference",
        ),
        ({"$schema": "https://example.com/unknown-draft"}, "unsupported_schema_draft"),
    ],
)
def test_unsupported_schema_constraints_never_claim_argument_validity(schema, reason):
    with pytest.raises(Reject, match=reason):
        compile_schema(schema)


@pytest.mark.parametrize(
    "format_name", ["uri", "date-time", "email", "regex", "unpublished-format"]
)
def test_ambient_optional_format_checkers_cannot_expand_the_supported_contract(
    monkeypatch, format_name
):
    monkeypatch.setitem(FormatChecker.checkers, format_name, (lambda value: True, ()))
    with pytest.raises(Reject, match="unsupported_schema_format:" + format_name):
        compile_schema({"properties": {"field": {"format": format_name}}})


def test_later_global_checker_registration_cannot_weaken_supported_formats(monkeypatch):
    monkeypatch.setitem(FormatChecker.checkers, "ipv4", (lambda value: True, ()))
    with pytest.raises(Reject, match="invalid_arguments"):
        check_arguments(
            {"ip": "999.0.0.0"},
            compile_schema({"properties": {"ip": {"format": "ipv4"}}}),
        )


@pytest.mark.parametrize(
    "format_name,valid,invalid",
    [
        ("date", "2024-02-29", "2025-02-29"),
        ("ipv4", "192.0.2.1", "999.0.0.0"),
        ("ipv6", "2001:db8::1", "2001:db8::z"),
        (
            "uuid",
            "123e4567-e89b-12d3-a456-426614174000",
            "123e4567e89b12d3a456426614174000",
        ),
    ],
)
def test_declared_stdlib_formats_accept_and_reject_supported_values(
    format_name, valid, invalid
):
    validator = compile_schema({"properties": {"field": {"format": format_name}}})
    check_arguments({"field": valid}, validator)
    with pytest.raises(Reject, match="invalid_arguments"):
        check_arguments({"field": invalid}, validator)


@pytest.mark.parametrize(
    "schema,arguments",
    [
        ({"type": "object", "required": ["x"]}, {}),
        ({"properties": {"x": {"type": "integer"}}}, {"x": True}),
        ({"properties": {"x": {"minimum": 2}}}, {"x": 1}),
        ({"properties": {"x": {"enum": ["a", "b"]}}}, {"x": "c"}),
        ({"properties": {"x": {"pattern": "^[0-9]+$"}}}, {"x": "abc"}),
        ({"properties": {"x": {"type": "array", "minItems": 2}}}, {"x": [1]}),
        ({"additionalProperties": False, "properties": {}}, {"x": 1}),
        ({"properties": {"x": {"format": "date"}}}, {"x": "2026-99-99"}),
    ],
)
def test_supported_argument_constraints_are_enforced(schema, arguments):
    with pytest.raises(Reject, match="invalid_arguments"):
        check_arguments(arguments, compile_schema(schema))


def test_local_references_validate_and_unresolved_references_quarantine():
    validator = compile_schema(
        {
            "$defs": {"value": {"type": "integer"}},
            "properties": {"x": {"$ref": "#/$defs/value"}},
        }
    )
    check_arguments({"x": 1}, validator)
    with pytest.raises(Reject, match="invalid_arguments"):
        check_arguments({"x": "one"}, validator)
    with pytest.raises(Reject, match="unresolved_tool_schema"):
        check_arguments({}, compile_schema({"$ref": "#/$defs/absent"}))


def test_schema_defaults_are_never_invented_and_structured_literals_are_opaque():
    schema = {
        "type": "object",
        "properties": {"x": {"default": {"x-constraint": "literal"}}},
    }
    arguments = {}
    check_arguments(arguments, compile_schema(schema))
    assert arguments == {}
    assert schema["properties"]["x"]["default"] == {"x-constraint": "literal"}


def test_definition_absence_does_not_become_an_undefined_call_exception():
    state = _state(_episode(tools=[]))
    with pytest.raises(Reject, match="undefined_function_calls"):
        validate_capabilities(state)


def _discovery(state, index, active):
    message = state.sample.messages[index]
    if message["role"] == "tool" and message.get("content") == "GRANT f":
        return CapabilityChange(tools=(_tool("f"),))
    return None


def test_dynamic_capability_is_available_only_after_its_visible_result():
    state = _state(
        _sample(
            [
                {"role": "user", "content": "Do it"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_call("discover")],
                },
                {"role": "tool", "content": "GRANT f"},
                {"role": "assistant", "content": None, "tool_calls": [_call("f")]},
                {"role": "tool", "content": "OK"},
                {"role": "assistant", "content": "Done"},
            ],
            tools=[_tool("discover")],
        )
    )
    validate_capabilities(state, discovery=_discovery)
    assert [tool["function"]["name"] for tool in state.sample.tools] == ["discover"]
    state.sample.messages[1]["tool_calls"][0]["function"]["name"] = "f"
    with pytest.raises(Reject, match="undefined_function_calls"):
        validate_capabilities(state, discovery=_discovery)


def test_parallel_discovery_result_cannot_grant_a_sibling_call():
    state = _state(
        _episode(
            [_call("discover"), _call("f")],
            [{"role": "tool", "content": "GRANT f"}, {"role": "tool", "content": "OK"}],
            tools=[_tool("discover")],
        )
    )
    with pytest.raises(Reject, match="undefined_function_calls"):
        validate_capabilities(state, discovery=_discovery)


def test_undeclared_discovery_tool_and_unrecognized_result_do_not_grant_capability():
    state = _state(
        _episode(
            [_call("discover")], [{"role": "tool", "content": "GRANT f"}], tools=[]
        )
    )
    with pytest.raises(Reject, match="undefined_function_calls"):
        validate_capabilities(state, discovery=_discovery)
    state = _state(_episode([_call("f")], tools=[_tool("discover")]))
    with pytest.raises(Reject, match="undefined_function_calls"):
        validate_capabilities(state, discovery=_discovery)


def test_dynamic_collision_requires_explicit_protocol_replacement():
    state = _state(_episode())
    changed = _tool("f", {"type": "object", "required": ["x"]})

    def discovery(state, index, active):
        return CapabilityChange(tools=(changed,)) if index == 0 else None

    with pytest.raises(Reject, match="conflicting_dynamic_tool_definition"):
        validate_capabilities(state, discovery=discovery)

    def replace(state, index, active):
        return CapabilityChange(tools=(changed,), replace=True) if index == 0 else None

    with pytest.raises(Reject, match="invalid_arguments"):
        validate_capabilities(state, discovery=replace)


def test_visible_capability_revocation_applies_to_later_calls():
    state = _state(_episode())

    def discovery(state, index, active):
        return CapabilityChange(remove=("f",)) if index == 0 else None

    with pytest.raises(Reject, match="undefined_function_calls"):
        validate_capabilities(state, discovery=discovery)


def test_target_grounding_callback_sees_only_current_capabilities_and_prefix_index():
    state = _state(_episode([_call(arguments='{"token":"hidden"}')]))
    state.sample.raw["secret"] = "hidden"

    def grounding(state, index, active):
        assert set(active) == {"f"}
        for call_index, _ in enumerate(
            state.sample.messages[index].get("tool_calls", [])
        ):
            token = state.parsed_arguments[index, call_index].get("token")
            if token and not any(
                token in (message.get("content") or "")
                for message in state.sample.messages[:index]
            ):
                raise Reject("hidden_scaffolding")

    with pytest.raises(Reject, match="hidden_scaffolding"):
        validate_capabilities(state, target_validator=grounding)
    state.sample.messages[0]["content"] = "Use token hidden"
    validate_capabilities(state, target_validator=grounding)


@pytest.mark.parametrize(
    "content,expected",
    [
        ("A<think>private</think>B", "AB"),
        ("A<reasoning>private</reasoning>B", "AB"),
        (" <think>one</think>\n<reasoning>two</reasoning> Done ", " \n Done "),
        (
            "Thinking and reasoning: Thought, Action, Observation",
            "Thinking and reasoning: Thought, Action, Observation",
        ),
    ],
)
def test_reasoning_removal_preserves_every_outside_character(content, expected):
    state = RowState(
        _sample(
            [
                {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": "private field",
                }
            ]
        )
    )
    raw = deepcopy(state.sample.raw)
    remove_reasoning(state)
    assert state.sample.messages[0]["content"] == expected
    assert "reasoning_content" not in state.sample.messages[0]
    assert state.sample.raw == raw


@pytest.mark.parametrize(
    "content",
    [
        "<think>open",
        "close</think>",
        "<reasoning>open",
        "<think>x</reasoning>",
        "<think><think>x</think></think>",
        "<think><reasoning>x</reasoning></think>",
    ],
)
def test_unmatched_nested_or_mismatched_private_spans_quarantine(content):
    with pytest.raises(Reject, match="ambiguous_reasoning"):
        remove_reasoning(RowState(_sample([{"role": "assistant", "content": content}])))


@pytest.mark.parametrize(
    "content",
    [
        "Use `<think>literal</think>` tags",
        "Use ``<think>literal ` content</think>`` tags",
        "```xml\n<think>literal</think>\n```",
        "~~~xml\n<think>literal</think>\n~~~",
        "````xml\n```\n<think>literal</think>\n````",
        "```xml\n<think>unfinished literal",
        "~~~xml\n<think>literal</think>",
        "    <think>literal</think>\n",
        "\t<reasoning>literal</reasoning>\n",
        "Inline `<think>unfinished literal",
    ],
)
def test_literal_reasoning_tags_in_code_are_preserved(content):
    state = RowState(_sample([{"role": "assistant", "content": content}]))
    remove_reasoning(state)
    assert state.sample.messages[0]["content"] == content


def test_reasoning_like_tools_and_user_tool_content_are_never_removed():
    state = _state(
        _episode(
            [_call("think", '{"text":"<think>literal</think>"}')],
            [{"role": "tool", "content": "<reasoning>visible</reasoning>"}],
            tools=[_tool("think")],
        )
    )
    state.sample.messages[0]["content"] = "Use <think>literal</think>"
    original = deepcopy(state.sample.messages)
    remove_reasoning(state)
    assert state.sample.messages == original


@pytest.mark.parametrize(
    "systems,override,expected",
    [
        ([], None, ["user", "assistant"]),
        ([], "", ["user", "assistant"]),
        ([], "New", ["system", "user", "assistant"]),
        ([{"role": "system", "content": ""}], None, ["system", "user", "assistant"]),
        (
            [{"role": "system", "content": "A"}, {"role": "system", "content": "B"}],
            "",
            ["user", "assistant"],
        ),
        (
            [{"role": "system", "content": "A"}, {"role": "system", "content": "B"}],
            "New",
            ["system", "user", "assistant"],
        ),
    ],
)
def test_system_override_preserves_absence_and_empty_source_states(
    systems, override, expected
):
    sample = _sample(
        [
            *systems,
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ]
    )
    raw = deepcopy(sample.raw)
    state = RowState(sample)
    override_system(state, override=override)
    assert [message["role"] for message in state.sample.messages] == expected
    assert state.sample.raw == raw
    assert state.sample.tools == sample.tools
    if override:
        assert [
            message["content"]
            for message in state.sample.messages
            if message["role"] == "system"
        ] == [override]


def test_system_override_uses_first_prior_position_and_rebuilds_argument_coordinates():
    sample = _episode()
    sample.messages.insert(1, {"role": "system", "content": "A"})
    sample.messages.insert(4, {"role": "system", "content": "B"})
    state = _state(sample)
    override_system(state, override="New")
    assert state.sample.messages[1] == {"role": "system", "content": "New"}
    assert state.parsed_arguments == {}
    validate_structure(state)
    assert (2, 0) in state.parsed_arguments


def test_invalid_source_is_rejected_before_system_override_can_rescue_it():
    sample = _episode([_call("undefined")])
    pipeline = Pipeline(
        [
            ("structure", validate_structure),
            ("capabilities", validate_capabilities),
            ("override", partial(override_system, override="All is permitted")),
        ]
    )
    assert pipeline.process(sample) is None
    assert pipeline.stage_drops == {"capabilities": 1}
    assert pipeline.passed["override"] == 0


@pytest.mark.parametrize(
    "final,reason",
    [
        ({"role": "tool", "content": "R"}, "incomplete_termination"),
        ({"role": "user", "content": "U"}, "incomplete_termination"),
        ({"role": "assistant", "content": ""}, "empty_final_assistant"),
        ({"role": "assistant", "content": " \n "}, "empty_final_assistant"),
        (
            {"role": "assistant", "content": None, "tool_calls": [_call()]},
            "incomplete_termination",
        ),
    ],
)
def test_termination_uses_exact_final_training_view(final, reason):
    with pytest.raises(Reject, match=reason):
        validate_termination(
            RowState(_sample([{"role": "user", "content": "U"}, final]))
        )


@pytest.mark.parametrize(
    "content",
    [
        "<think>private</think>",
        " \n<reasoning>private</reasoning>\t",
        "<think></think>",
    ],
)
def test_native_only_final_is_incomplete_before_and_after_optional_stripping(content):
    state = RowState(
        _sample(
            [
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": content},
            ]
        )
    )
    before = deepcopy(state.sample)
    with pytest.raises(Reject, match="empty_final_assistant"):
        validate_termination(state)
    assert state.sample == before
    remove_reasoning(state)
    with pytest.raises(Reject, match="empty_final_assistant"):
        validate_termination(state)


@pytest.mark.parametrize(
    "content",
    [
        "<think>",
        "<think>\nOkay, let me break down the user's request. They want to book tickets",
        " \r\n<reasoning>Unfinished private content",
        "<think>Literal closing token: `</think>`",
        "<reasoning>\n```xml\n</reasoning>\n```",
        "<think>\n    </think>",
        "<think>nested <think>unresolved",
        "<think>private <reasoning>more private <think>unfinished",
        "<think>complete</think> \n<reasoning>unfinished",
        "<think>a</think><reasoning>b</reasoning><think>unfinished <think>still open",
        "<think>a</think> <think>`</think>`",
        "</think>",
        " \n</reasoning>\t</think> ",
        "<think> </reasoning>",
        "<think><think></think>",
    ],
)
def test_native_only_ambiguous_endpoint_cannot_complete_a_retained_response(content):
    state = RowState(
        _sample(
            [
                {"role": "user", "content": "U"},
                {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": "A separate complete native field.",
                },
            ]
        )
    )
    before = deepcopy(state.sample)
    with pytest.raises(Reject, match="empty_final_assistant"):
        validate_termination(state)
    assert state.sample == before


@pytest.mark.parametrize(
    "content",
    [
        "<think>private</think> Answer.",
        "Answer.<reasoning>private</reasoning>",
        "`<think>literal</think>`",
        "```xml\n<think>literal</think>\n```",
        "    <think>literal</think>",
        "Thought: use the returned result.",
        "`<think>literal opener`",
        "```xml\n<think>literal opener",  # An unfinished code fence is protected.
        "    <reasoning>literal opener",
        "\t<think>literal opener",
        "Answer before an unresolved boundary. <think>private",
        "Answer with a standalone close.</think>",
        "<think>nested <think>unresolved</think>",
        "<think>mismatched</reasoning>",
        "<think>complete</think> Outside answer. <think>unfinished",
        "Outside answer. <think>complete</think><think>unfinished",
        "<think>complete</think> `</think>` <think>unfinished",
        "<think>nested <reasoning>mismatched</think>tail",
        "`</think>`",
        "```xml\n</think>\n```",
        "    </reasoning>",
        r"\</think>",
        "<thinking></thinking>",
    ],
)
def test_termination_inspection_preserves_visible_text_code_and_keep_mode(content):
    state = RowState(_sample([{"role": "assistant", "content": content}]))
    before = deepcopy(state.sample)
    validate_termination(state)
    assert state.sample == before


def test_explicit_action_only_objective_allows_final_call_target():
    state = RowState(
        _sample(
            [
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": None, "tool_calls": [_call()]},
            ]
        )
    )
    validate_termination(state, action_only=True)


@pytest.mark.parametrize(
    "events,reason",
    [
        ([JobEvent("j", "started")], "incomplete_async_job"),
        ([JobEvent("j", "pending")], "incomplete_async_job"),
        ([JobEvent("j", "completed")], "incomplete_async_job"),
        (
            [JobEvent("j", "pending"), JobEvent("j", "summarized")],
            "premature_job_summary",
        ),
        (
            [JobEvent("j", "started"), JobEvent("j", "fetched", True)],
            "premature_job_fetch",
        ),
        ([JobEvent("j", "unknown")], "unknown_job_phase"),
        ([JobEvent("j", "completed", True, grounded=False)], "ungrounded_job_state"),
    ],
)
def test_async_protocol_errors_are_explicit(events, reason):
    with pytest.raises(Reject, match=reason):
        validate_lifecycle(events)


def test_audited_external_status_queries_need_not_complete_the_external_job():
    for phase in ("pending", "completed"):
        validate_lifecycle(
            [JobEvent("external", phase)], require_completion_for="started"
        )
    with pytest.raises(Reject, match="incomplete_async_job"):
        validate_lifecycle(
            [JobEvent("external", "pending"), JobEvent("new", "started")],
            require_completion_for="started",
        )


def test_external_status_exception_never_certifies_missing_payload_or_grounding():
    with pytest.raises(Reject, match="premature_job_summary"):
        validate_lifecycle(
            [JobEvent("external", "completed"), JobEvent("external", "summarized")],
            require_completion_for="started",
        )
    with pytest.raises(Reject, match="ungrounded_job_state"):
        validate_lifecycle(
            [JobEvent("external", "pending", grounded=False)],
            require_completion_for="started",
        )
    with pytest.raises(Reject, match="premature_job_fetch"):
        validate_lifecycle(
            [JobEvent("external", "fetched", payload=True)],
            require_completion_for="started",
        )


def test_external_status_exception_preserves_fetch_policy_for_summaries():
    events = [JobEvent("external", "completed", payload=True)]
    with pytest.raises(Reject, match="premature_job_summary"):
        validate_lifecycle(
            [*events, JobEvent("external", "summarized")],
            require_completion_for="started",
            require_fetch=True,
        )
    validate_lifecycle(
        [
            *events,
            JobEvent("external", "fetched", True),
            JobEvent("external", "summarized"),
        ],
        require_completion_for="started",
        require_fetch=True,
    )


@pytest.mark.parametrize(
    "events",
    [
        [
            JobEvent("j", "started"),
            JobEvent("j", "pending"),
            JobEvent("j", "pending"),
            JobEvent("j", "completed", True),
            JobEvent("j", "summarized"),
        ],
        [JobEvent("j", "pending"), JobEvent("j", "completed", True)],
        [JobEvent("j", "started"), JobEvent("j", "abandoned")],
    ],
)
def test_grounded_polls_retries_and_visible_abandonment_are_preserved(events):
    validate_lifecycle(events)


def test_required_fetch_must_follow_completion_before_summary():
    events = [JobEvent("j", "started"), JobEvent("j", "completed", True)]
    with pytest.raises(Reject, match="incomplete_async_job"):
        validate_lifecycle(events, require_fetch=True)
    with pytest.raises(Reject, match="premature_job_summary"):
        validate_lifecycle([*events, JobEvent("j", "summarized")], require_fetch=True)
    validate_lifecycle(
        [*events, JobEvent("j", "fetched", True), JobEvent("j", "summarized")],
        require_fetch=True,
    )


def test_pipeline_is_streaming_ordered_and_records_exclusive_stage_drops():
    seen = []

    def source_before(state):
        seen.append((state.sample.sample_id, "before"))
        state.transforms["source_transform"] += 1

    def source_after(state):
        seen.append((state.sample.sample_id, "after"))

    pipeline = Pipeline(
        [
            ("source_before", source_before),
            ("structure", validate_structure),
            ("source_after", source_after),
        ]
    )
    good, bad = _sample(), _sample()
    good.sample_id, bad.sample_id = "good", "bad"
    bad.messages[0]["extra"] = "unclassified"
    converted = iter(pipeline(iter([good, bad])))
    assert seen == []
    assert next(converted).sample is good
    assert seen == [("good", "before"), ("good", "after")]
    assert list(converted) == []
    assert pipeline.passed == {"source_before": 2, "structure": 1, "source_after": 1}
    assert pipeline.drops == {"unknown_message_field": 1}
    assert pipeline.transforms == {"source_transform": 2}
    assert seen[-1] == ("bad", "before")


def test_pipeline_does_not_swallow_programming_errors_or_duplicate_stage_names():
    def broken(state):
        raise RuntimeError("implementation bug")

    with pytest.raises(RuntimeError, match="implementation bug"):
        Pipeline([("broken", broken)]).process(_sample())
    with pytest.raises(ValueError, match="unique"):
        Pipeline([("same", broken), ("same", broken)])


def test_explicit_null_strict_is_preserved_and_distinct_from_absence():
    omitted = {"type": "function", "function": {"name": "f"}}
    explicit = {"type": "function", "function": {"name": "f", "strict": None}}
    result, repeats = reconcile_tools([explicit])
    assert result == [explicit] and repeats == 0
    with pytest.raises(Reject, match="conflicting_duplicate_tool_names"):
        reconcile_tools([omitted, explicit])
