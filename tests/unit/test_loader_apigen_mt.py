"""APIGen-MT source conversion and adversarial retained-view contracts."""

from collections import Counter
from copy import deepcopy

import orjson
import pytest
from datasets import Dataset

from fcanalysis.loaders.apigen_mt import (
    APIGenMTConfig,
    DATASET_ID,
    DATA_FILE,
    DATASET_REVISION,
    _RETAIL_POLICY,
    _convert_row,
    _convert_sample,
    _parse_tools,
    _pipeline,
    load,
)
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.pipeline import Reject
from fcanalysis.loaders.normalization import parse_json
from fcanalysis.loaders.source import pinned_file


def _tool(name="lookup", properties=None, required=None, **extra):
    return {
        "name": name,
        "description": "exact source prose",
        "parameters": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
        },
        **extra,
    }


def _msg(role, value):
    return {"from": role, "value": value}


def _call(name="lookup", arguments=None):
    return _msg(
        "function_call",
        orjson.dumps(
            {"name": name, "arguments": arguments if arguments is not None else {}}
        ).decode(),
    )


def _row(conversations=None, tools=None, system="  policy\n  "):
    return {
        "system": system,
        "tools": orjson.dumps(tools or [_tool()]).decode(),
        "conversations": conversations
        if conversations is not None
        else [
            _msg("human", "question"),
            _call(),
            _msg("observation", "result"),
            _msg("gpt", " answer  "),
        ],
    }


def _run(row, filter_config=None):
    pipeline = _pipeline(filter_config or FilterConfig())
    state = pipeline.process(_convert_row(row, 7))
    return state, pipeline


def _load_rows(monkeypatch, rows, **kwargs):
    def fake_load(dataset_id, **load_kwargs):
        assert dataset_id == DATASET_ID
        assert load_kwargs == {
            "name": "dataset",
            "revision": DATASET_REVISION,
            "split": "train",
        }
        return Dataset.from_list(rows)

    monkeypatch.setattr("fcanalysis.loaders.apigen_mt.load_dataset", fake_load)
    return load(**kwargs)


def test_conversion_preserves_full_source_content_and_isolation():
    row = _row(
        tools=[_tool(properties={"nested": {"type": "object"}}, strict=False)],
        conversations=[
            _msg("human", "  question\r\n"),
            _call(arguments={"nested": {"array": [1, " x "]}}),
            _msg("observation", '{ "nested": [1, 2] } '),
            _msg("gpt", "answer  "),
        ],
    )
    original = deepcopy(row)
    state, _ = _run(row)
    assert state is not None
    sample = state.sample
    assert sample.raw == original
    assert sample.messages[0] == {"role": "system", "content": original["system"]}
    assert sample.messages[1]["content"] == "  question\r\n"
    assert sample.messages[3]["content"] == '{ "nested": [1, 2] } '
    assert sample.messages[-1]["content"] == "answer  "
    assert sample.tools == [
        {"type": "function", "function": orjson.loads(row["tools"])[0]}
    ]
    sample.tools[0]["function"]["parameters"]["properties"]["nested"]["x"] = []
    sample.messages[2]["tool_calls"][0]["function"]["arguments"] = "{}"
    assert row == original
    assert sample.raw == original
    assert all(
        "id" not in call
        for message in sample.messages
        for call in message.get("tool_calls", [])
    )
    assert all("tool_call_id" not in message for message in sample.messages)


@pytest.mark.parametrize("system", [None, "", " exact policy "])
def test_system_absence_empty_and_nonempty_are_distinct(system):
    state, _ = _run(_row(system=system))
    assert state is not None
    assert [m for m in state.sample.messages if m["role"] == "system"] == (
        [] if system is None else [{"role": "system", "content": system}]
    )


def test_complete_definition_reconciliation_keeps_first_order_and_fields():
    first = _tool("beta", strict=False)
    duplicate = {
        "parameters": first["parameters"],
        "strict": False,
        "description": first["description"],
        "name": "beta",
    }
    definitions = _parse_tools(
        orjson.dumps([first, duplicate, _tool("alpha"), _tool("Beta")]).decode()
    )
    assert definitions is not None
    assert [t["function"]["name"] for t in definitions] == ["beta", "alpha", "Beta"]
    assert definitions[0]["function"] == first
    assert list(definitions[0]["function"]) == list(first)


@pytest.mark.parametrize(
    "change",
    [
        {"description": "different"},
        {"strict": False},
        {"parameters": {"type": "object", "properties": {}, "required": ["x"]}},
    ],
)
def test_conflicting_definition_is_quarantined(change):
    first = _tool()
    second = {**first, **change}
    assert _parse_tools(orjson.dumps([first, second]).decode()) is None
    assert _convert_sample(None, orjson.dumps([first, second]).decode(), [], 0) is None


@pytest.mark.parametrize(
    "edit",
    [
        lambda row: row.update(extra="model visible?"),
        lambda row: row["conversations"][0].update(reasoning_content="hidden"),
        lambda row: row["conversations"][0].update(**{"from": "developer"}),
        lambda row: row["conversations"][0].update(value={"text": "question"}),
        lambda row: row["conversations"][1].update(value='{"name":"lookup"}'),
        lambda row: row["conversations"][1].update(
            value='{"name":"lookup","arguments":{},"id":"x"}'
        ),
        lambda row: row["conversations"][1].update(
            value='[{"name":"lookup","arguments":{}}]'
        ),
        lambda row: row["conversations"][1].update(
            value='{"name":"lookup","arguments":[]}'
        ),
        lambda row: row.update(tools="{}"),
    ],
)
def test_unknown_or_invalid_source_fields_are_explicit_rejections(edit):
    row = _row()
    edit(row)
    with pytest.raises(Reject):
        _convert_row(row, 0)


def test_serialized_arguments_are_preserved_and_structured_values_serialize_once():
    original = '{ "x" : [1, 2] }'
    row = _row(
        tools=[
            _tool(properties={"x": {"type": "array", "items": {"type": "integer"}}})
        ],
        conversations=[
            _msg("human", "question"),
            _call(arguments=original),
            _msg("observation", "result"),
            _msg("gpt", "done"),
        ],
    )
    state, _ = _run(row)
    assert state is not None
    assert (
        state.sample.messages[2]["tool_calls"][0]["function"]["arguments"] == original
    )


@pytest.mark.parametrize(
    "conversations",
    [
        [_msg("human", "u"), _msg("observation", "orphan"), _msg("gpt", "done")],
        [_msg("human", "u"), _call(), _msg("gpt", "missing result")],
        [
            _msg("human", "u"),
            _call(),
            _msg("observation", "a"),
            _msg("observation", "b"),
            _msg("gpt", "done"),
        ],
        [
            _msg("human", "u"),
            _call(),
            _call(),
            _msg("observation", "a"),
            _msg("observation", "b"),
            _msg("gpt", "done"),
        ],
        [_msg("human", "u"), _call(), _msg("observation", "result")],
        [_msg("human", "u"), _call(), _msg("observation", "result"), _msg("gpt", "")],
    ],
)
def test_orphans_missing_ambiguous_or_incomplete_episodes_are_rejected(conversations):
    state, _ = _run(_row(conversations=conversations))
    assert state is None


def test_consecutive_assistant_and_repeated_valid_calls_remain_unchanged():
    conversations = [_msg("human", "question")]
    for _ in range(3):
        conversations += [_call(), _msg("observation", "Error: temporary failure")]
    conversations += [_msg("gpt", "First response."), _msg("gpt", "Second response.")]
    state, _ = _run(_row(conversations=conversations))
    assert state is not None
    assert len(state.sample.messages) == len(conversations) + 1
    assert state.sample.messages[-2:] == [
        {"role": "assistant", "content": "First response."},
        {"role": "assistant", "content": "Second response."},
    ]


@pytest.mark.parametrize(
    "arguments", ['{"x": {}}', '{"x": [{"n": "wrong"}]}', '{"x": [{}]}']
)
def test_nested_argument_constraints_are_enforced(arguments):
    row = _row(
        tools=[
            _tool(
                properties={
                    "x": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"n": {"type": "integer"}},
                            "required": ["n"],
                        },
                    }
                },
                required=["x"],
            )
        ],
        conversations=[
            _msg("human", "q"),
            _call(arguments=arguments),
            _msg("observation", "r"),
            _msg("gpt", "done"),
        ],
    )
    state, _ = _run(row)
    assert state is None


def test_undefined_call_cannot_be_rescued_by_system_override(monkeypatch):
    row = _row(
        conversations=[
            _msg("human", "q"),
            _call("undeclared"),
            _msg("observation", "r"),
            _msg("gpt", "done"),
        ]
    )
    samples, _ = _load_rows(
        monkeypatch,
        [row],
        filter_config=FilterConfig(system_message_override="replacement"),
    )
    assert samples == []


def test_reasoning_removal_preserves_visible_think_calls_and_all_other_content():
    row = _row(
        tools=[
            _tool(
                "think",
                properties={"thought": {"type": "string"}},
                required=["thought"],
            )
        ],
        conversations=[
            _msg("human", "Please think."),
            _call("think", {"thought": "Reasoning: use this observable tool."}),
            _msg("observation", ""),
            _msg("gpt", "  <think>private</think>Answer.\n"),
        ],
    )
    state, _ = _run(row, FilterConfig(strip_thinking=True))
    assert state is not None
    assert len(state.sample.messages) == 5
    assert state.sample.messages[2]["tool_calls"][0]["function"]["name"] == "think"
    assert state.sample.messages[3] == {"role": "tool", "content": ""}
    assert state.sample.messages[-1]["content"] == "  Answer.\n"
    assert state.sample.raw == row


@pytest.mark.parametrize("final", ["<think>all private</think>", "<reasoning>unclosed"])
def test_reasoning_removal_rechecks_final_boundary(final):
    row = _row()
    row["conversations"][-1]["value"] = final
    state, _ = _run(row, FilterConfig(strip_thinking=True))
    assert state is None


@pytest.mark.parametrize("override", ["", "caller policy"])
def test_system_override_is_post_validation_and_preserves_raw(override):
    row = _row()
    state, _ = _run(row, FilterConfig(system_message_override=override))
    assert state is not None
    assert [m for m in state.sample.messages if m["role"] == "system"] == (
        [] if override == "" else [{"role": "system", "content": override}]
    )
    assert state.sample.raw["system"] == row["system"]


def _id_row(prefix, *, identifier="secret_123", definition_example=False, future=False):
    row = _row(
        tools=[
            _tool(
                "get_user_details",
                properties={
                    "user_id": {
                        "type": "string",
                        "description": identifier
                        if definition_example
                        else "opaque ID",
                    }
                },
            )
        ],
        conversations=[
            *prefix,
            _call("get_user_details", {"user_id": identifier}),
            _msg(
                "observation",
                orjson.dumps({"user_id": identifier}).decode() if future else "{}",
            ),
            _msg("gpt", "done"),
        ],
    )
    return row


@pytest.mark.parametrize(
    "prefix,example,future",
    [
        ([_msg("human", "question")], True, False),
        (
            [_msg("human", "question"), _msg("gpt", "Your ID is secret_123")],
            False,
            False,
        ),
        ([_msg("human", "question")], False, True),
        ([_msg("human", "My ID is secret_1234")], False, False),
    ],
)
def test_schema_examples_assistant_invention_future_results_and_substrings_do_not_ground_ids(
    prefix, example, future
):
    state, _ = _run(_id_row(prefix, definition_example=example, future=future))
    assert state is None


def test_exact_user_identifier_is_grounded():
    state, _ = _run(_id_row([_msg("human", "My ID is secret_123.")]))
    assert state is not None


def test_linked_structured_result_grounds_identifier_but_prose_substring_does_not():
    for result, expected in [
        ('{"user_id":"secret_123"}', True),
        ('{"explanation":"maybe secret_123 later"}', False),
    ]:
        row = _id_row([_msg("human", "question"), _call(), _msg("observation", result)])
        row["tools"] = orjson.dumps([_tool(), *orjson.loads(row["tools"])]).decode()
        state, _ = _run(row)
        assert (state is not None) is expected


def test_documented_order_prefix_is_comparison_only():
    row = _row(
        tools=[_tool("get_order_details", properties={"order_id": {"type": "string"}})],
        conversations=[
            _msg("human", "My order is W1234567."),
            _call("get_order_details", {"order_id": "#W1234567"}),
            _msg("observation", "{}"),
            _msg("gpt", "done"),
        ],
    )
    state, _ = _run(row)
    assert state is not None
    assert (
        state.sample.messages[2]["tool_calls"][0]["function"]["arguments"]
        == '{"order_id":"#W1234567"}'
    )


def _auth_row(result="jane_doe_123", email="jane@example.com"):
    return _row(
        system="# Retail agent policy\n" + _RETAIL_POLICY,
        tools=[
            _tool("find_user_id_by_email", properties={"email": {"type": "string"}}),
            _tool("get_order_details", properties={"order_id": {"type": "string"}}),
        ],
        conversations=[
            _msg("human", "Email JANE@example.com, order W1234567."),
            _call("find_user_id_by_email", {"email": email}),
            _msg("observation", result),
            _call("get_order_details", {"order_id": "#W1234567"}),
            _msg("observation", "{}"),
            _msg("gpt", "done"),
        ],
    )


@pytest.mark.parametrize(
    "result",
    [
        "Error: user not found",
        "Successfully authenticated",
        '{"user_id":"jane_doe_123"}',
    ],
)
def test_failed_or_unreleased_authentication_result_does_not_authorize_protected_calls(
    result,
):
    state, _ = _run(_auth_row(result))
    assert state is None


def test_successful_visible_authentication_and_case_insensitive_email_are_preserved():
    state, _ = _run(_auth_row())
    assert state is not None


def test_authentication_input_must_be_visible_before_lookup():
    state, _ = _run(_auth_row(email="hidden@example.com"))
    assert state is None


def test_final_context_recheck_preserves_default_valid_trajectory():
    row = _auth_row()
    original = _convert_row(row, 7)
    state, pipeline = _run(row)
    assert state is not None
    assert state.sample == original
    assert pipeline.passed["visible_inputs"] == 1
    assert pipeline.passed["final_visible_inputs"] == 1
    assert not pipeline.drops


def test_override_introducing_retail_authentication_requires_visible_lookup():
    row = _row()
    assert _run(row)[0] is not None
    state, pipeline = _run(row, FilterConfig(system_message_override=_RETAIL_POLICY))
    assert state is None
    assert pipeline.passed["visible_inputs"] == 1
    assert pipeline.passed["final_capabilities"] == 1
    assert pipeline.stage_drops == {"final_visible_inputs": 1}
    assert pipeline.drops == {"missing_visible_authentication": 1}
    assert not pipeline.passed["termination"]


@pytest.mark.parametrize("override", ["", "Unrestricted caller policy."])
def test_override_cannot_rescue_source_without_required_authentication(override):
    row = _row(system=_RETAIL_POLICY)
    state, pipeline = _run(row, FilterConfig(system_message_override=override))
    assert state is None
    assert pipeline.stage_drops == {"visible_inputs": 1}
    assert pipeline.drops == {"missing_visible_authentication": 1}
    assert not pipeline.passed["system_override"]


def test_valid_authentication_satisfies_new_retail_override_without_changing_content():
    row = _auth_row()
    row["system"] = "Source policy without a retail authentication requirement."
    original = _convert_row(row, 7)
    state, pipeline = _run(row, FilterConfig(system_message_override=_RETAIL_POLICY))
    assert state is not None
    assert state.sample.messages[0] == {"role": "system", "content": _RETAIL_POLICY}
    assert state.sample.messages[1:] == original.messages[1:]
    assert state.sample.tools == original.tools
    assert state.sample.raw == row
    assert pipeline.passed["final_visible_inputs"] == 1
    assert not pipeline.drops


@pytest.mark.parametrize("requested_user", ["jane_doe_123", "john_smith_456"])
def test_retail_override_rechecks_authenticated_user_consistency(requested_user):
    row = _auth_row()
    row["system"] = "Source policy without a retail authentication requirement."
    row["tools"] = orjson.dumps(
        [
            _tool("find_user_id_by_email", properties={"email": {"type": "string"}}),
            _tool("get_user_details", properties={"user_id": {"type": "string"}}),
        ]
    ).decode()
    row["conversations"][0] = _msg(
        "human", "Email JANE@example.com. The other user ID is john_smith_456."
    )
    row["conversations"][3] = _call("get_user_details", {"user_id": requested_user})
    # Both IDs are grounded, but only one belongs to the authenticated user.
    assert _run(row)[0] is not None
    state, pipeline = _run(row, FilterConfig(system_message_override=_RETAIL_POLICY))
    assert pipeline.passed["visible_inputs"] == 1
    if requested_user == "jane_doe_123":
        assert state is not None
        assert pipeline.passed["final_visible_inputs"] == 1
    else:
        assert state is None
        assert pipeline.stage_drops == {"final_visible_inputs": 1}
        assert pipeline.drops == {"conflicting_authenticated_user": 1}


@pytest.mark.parametrize(
    "config",
    [
        APIGenMTConfig(strip_think_tool=True),
        APIGenMTConfig(drop_repeated_tool_call_streaks=True),
        APIGenMTConfig(drop_error_recovery_loops=True),
        APIGenMTConfig(drop_consecutive_assistant=True),
    ],
)
def test_unsafe_legacy_transforms_fail_explicitly(config):
    with pytest.raises(ValueError):
        load(dataset_config=config)


def test_corpus_order_and_exclusive_counts(monkeypatch):
    bad = _row()
    bad["conversations"] = [
        _msg("human", "question"),
        _msg("observation", "orphan"),
        _msg("gpt", "done"),
    ]
    samples, report = _load_rows(monkeypatch, [bad, _row()])
    assert [s.sample_id for s in samples] == [1]
    assert report.raw_count == report.stage1_count == 2
    assert report.final_count == report.filtered_count == len(samples) == 1
    assert sum(report.filter_drop_reasons.values()) == 1


@pytest.mark.e2e
def test_pinned_raw_census_and_complete_retained_contract():
    from datasets import load_dataset

    ds = load_dataset(
        DATASET_ID, name="dataset", revision=DATASET_REVISION, split="train"
    )
    roles = Counter()
    endings = Counter()
    systems = Counter()
    argument_types = Counter()
    raw_rows = parse_json(
        pinned_file(DATASET_ID, DATASET_REVISION, DATA_FILE).read_text()
    )
    assert len(raw_rows) == len(ds)
    for index, row in enumerate(ds):
        assert row == raw_rows[index]
        assert set(row) == {"system", "tools", "conversations"}
        systems[row["system"]] += 1
        assert all(
            set(t) == {"name", "description", "parameters"}
            for t in orjson.loads(row["tools"])
        )
        for i, message in enumerate(row["conversations"]):
            assert set(message) == {"from", "value"}
            roles[message["from"]] += 1
            if message["from"] == "function_call":
                call = orjson.loads(message["value"])
                assert set(call) == {"name", "arguments"}
                argument_types[type(call["arguments"]).__name__] += 1
                assert row["conversations"][i + 1]["from"] == "observation"
        endings[row["conversations"][-1]["from"]] += 1
    assert len(ds) == 5000
    assert roles == {
        "human": 24229,
        "gpt": 24172,
        "function_call": 21955,
        "observation": 21955,
    }
    assert endings == {"gpt": 4904, "observation": 96}
    assert sorted(systems.values()) == [1589, 3411]
    assert argument_types == {"dict": 16929, "str": 5026}
    samples, report = load()
    assert report.raw_count == report.stage1_count == 5000
    assert report.final_count == len(samples)
    assert 5000 == len(samples) + sum(report.filter_drop_reasons.values())
    for sample in samples:
        assert sample.messages[-1]["role"] == "assistant"
        assert sample.messages[-1]["content"].strip()
        assert sample.raw == ds[sample.sample_id]
        assert sample.tools == [
            {"type": "function", "function": tool}
            for tool in orjson.loads(sample.raw["tools"])
        ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("tools", '[{"name":"lookup","name":"different","parameters":{}}]'),
        ("tools", '[{"name":"lookup","parameters":{"minimum":NaN}}]'),
    ],
)
def test_ambiguous_or_nonfinite_json_is_rejected(field, value):
    row = _row()
    row[field] = value
    with pytest.raises(Reject):
        _convert_row(row, 0)


def test_unknown_model_visible_definition_fields_are_not_discarded():
    row = _row(tools=[_tool(unclassified_source_field="visible contract")])
    with pytest.raises(Reject, match="unknown_source_definition_fields"):
        _convert_row(row, 0)
    assert "unclassified_source_field" in row["tools"]


def test_load_checks_batch_size_before_source_access():
    with pytest.raises(ValueError, match="batch_size"):
        load(batch_size=0)


def test_curation_selects_one_original_row_and_reports_post_filter_counts(monkeypatch):
    first = _row()
    second = deepcopy(first)
    first["conversations"][-1]["value"] = "A much longer final assistant response."
    second["conversations"][-1]["value"] = "Done."
    samples, report = _load_rows(monkeypatch, [first, second])
    assert report.filtered_count == 2
    assert report.final_count == 1
    assert [sample.sample_id for sample in samples] == [1]
    assert samples[0].raw == second
    assert samples[0].messages[-1]["content"] == "Done."
    assert samples[0].annotations == {}


def test_curation_can_be_disabled_without_disabling_row_validation(monkeypatch):
    from fcanalysis.loaders.curation import CurationConfig

    first = _row()
    second = deepcopy(first)
    samples, report = _load_rows(
        monkeypatch, [first, second], curation_config=CurationConfig(max_level=None)
    )
    assert report.final_count == report.filtered_count == 2
    assert [sample.sample_id for sample in samples] == [0, 1]
    assert report.dataset_config_transform_counts["curation"][0]["stages"] == {}


@pytest.mark.parametrize("identifier", [""])
def test_empty_opaque_argument_is_not_grounded_by_empty_text(identifier):
    state, _ = _run(
        _id_row([_msg("human", "I have not supplied an ID.")], identifier=identifier)
    )
    assert state is None


def test_empty_authentication_argument_is_not_an_inferred_default():
    state, _ = _run(_auth_row(email=""))
    assert state is None


def test_missing_optional_system_field_preserves_absent_source_state():
    row = _row()
    del row["system"]
    state, _ = _run(row)
    assert state is not None
    assert not any(message["role"] == "system" for message in state.sample.messages)
    assert "system" not in state.sample.raw


def test_unreleased_list_of_systems_is_rejected_instead_of_merged():
    row = _row(system=["one policy", "another policy"])
    with pytest.raises(Reject, match="invalid_source_system"):
        _convert_row(row, 0)
