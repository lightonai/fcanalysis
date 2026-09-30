"""Adversarial complete-trajectory contracts shared by both Nemotron sources."""

from copy import deepcopy
import json
from typing import Any

import pytest

from fcanalysis.loaders import nemotron_agentic_v1 as v1, nemotron_agentic_v2 as v2
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.curation import CurationConfig
from fcanalysis.loaders.normalization import Reject


def tool(name="f", parameters=None, **extras):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": parameters if parameters is not None else {"type": "object"},
            **extras,
        },
    }


def call(name="f", arguments=None, id="a"):
    return {
        "type": "function",
        "id": id,
        "function": {
            "name": name,
            "arguments": "{}" if arguments is None else arguments,
        },
    }


def row(*, messages=None, tools=None, uuid="row"):
    return {
        "uuid": uuid,
        "metadata": {"uuid": uuid},
        "messages": messages
        if messages is not None
        else [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "", "tool_calls": [call()]},
            {"role": "tool", "content": '{"result":"ok"}', "tool_call_id": "a"},
            {"role": "assistant", "content": "Answer"},
        ],
        "tools": tools if tools is not None else [tool()],
    }


def process(loader, raw, filters=None, split="interactive_agent"):
    stages = loader._pipeline(filters or FilterConfig(), split)
    state = stages.process(loader._convert_row(raw, split))
    return state, stages


@pytest.fixture(params=[v1, v2], ids=["v1", "v2"])
def loader(request):
    return request.param


def test_complete_sequential_and_raw_fidelity(loader):
    raw = row()
    original = deepcopy(raw)
    raw["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{  "x":1 }'
    raw["tools"][0]["function"]["parameters"]["properties"] = {"x": {"type": "integer"}}
    original = deepcopy(raw)
    state, _ = process(loader, raw)
    assert state is not None
    assert (
        state.sample.messages[1]["tool_calls"][0]["function"]["arguments"]
        == '{  "x":1 }'
    )
    assert state.sample.messages[2] == {"role": "tool", "content": '{"result":"ok"}'}
    assert "id" not in state.sample.messages[1]["tool_calls"][0]
    state.sample.tools[0]["function"]["parameters"]["properties"]["x"]["type"] = (
        "string"
    )
    assert raw == original
    assert state.sample.raw is raw


@pytest.mark.parametrize(
    "systems",
    [
        [],
        [{"role": "system", "content": ""}],
        [
            {"role": "system", "content": "Policy"},
            {"role": "system", "content": "Second"},
        ],
    ],
)
def test_system_absence_emptiness_and_multiplicity(loader, systems):
    raw = row()
    raw["messages"] = systems + raw["messages"]
    state, _ = process(loader, raw)
    assert state is not None
    assert [m for m in state.sample.messages if m["role"] == "system"] == systems


@pytest.mark.parametrize("override", ["", "New policy"])
def test_system_override_follows_validation(loader, override):
    raw = row()
    raw["messages"].insert(0, {"role": "system", "content": "Old"})
    state, _ = process(loader, raw, FilterConfig(system_message_override=override))
    assert state is not None
    assert [m["content"] for m in state.sample.messages if m["role"] == "system"] == (
        [override] if override else []
    )
    assert raw["messages"][0]["content"] == "Old"


def test_full_schema_legacy_union_dedup_and_raw_isolation(loader):
    t = tool(
        parameters={
            "type": "object",
            "properties": {"x": {"type": "string"}, "y": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        required=["y"],
        strict=True,
        description="exact  description",
    )
    equivalent = deepcopy(t)
    del equivalent["function"]["required"]
    equivalent["function"]["parameters"]["required"] = ["x", "y"]
    raw = row(tools=[t, equivalent])
    raw["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{"x":"v","y":1}'
    state, stages = process(loader, raw)
    assert state is not None
    assert state.sample.tools == [equivalent]
    assert stages.transforms["duplicate_tool_definitions_removed"] == 1
    assert t["function"]["required"] == ["y"]


@pytest.mark.parametrize(
    "change,reason",
    [
        (
            lambda r: r["tools"].append(tool(description="different")),
            "conflicting_duplicate_tool_names",
        ),
        (
            lambda r: r["messages"][1]["tool_calls"][0]["function"].update(
                name="future"
            ),
            "undefined_function_calls",
        ),
        (
            lambda r: r["messages"][1]["tool_calls"][0]["function"].update(
                arguments="[]"
            ),
            "non_object_arguments",
        ),
        (
            lambda r: r["messages"][1]["tool_calls"][0]["function"].update(
                arguments='{"x":1,"x":2}'
            ),
            "duplicate_json_key",
        ),
        (
            lambda r: r["tools"][0]["function"]["parameters"].update(required=["x"]),
            "invalid_arguments",
        ),
        (
            lambda r: r["tools"][0]["function"]["parameters"].update(format="email"),
            "unsupported_schema_format:email",
        ),
        (
            lambda r: r["tools"][0]["function"]["parameters"].update(unknown=True),
            "unsupported_schema_keyword:unknown",
        ),
        (lambda r: r["messages"].pop(2), "unbalanced_cardinality"),
        (
            lambda r: r["messages"].insert(
                3, {"role": "tool", "content": "extra", "tool_call_id": "b"}
            ),
            "unbalanced_cardinality",
        ),
        (
            lambda r: r["messages"].insert(0, {"role": "tool", "content": "orphan"}),
            "orphan_tool_result",
        ),
        (
            lambda r: r["messages"][2].update(tool_call_id="different"),
            "invalid_source_tool_linkage",
        ),
        (
            lambda r: r["messages"][2].update(name="different"),
            "invalid_source_tool_linkage",
        ),
        (lambda r: r["messages"].pop(), "incomplete_termination"),
        (lambda r: r["messages"][-1].update(content=" "), "empty_final_assistant"),
    ],
)
def test_rejections_are_required_even_without_legacy_flags(loader, change, reason):
    raw = row()
    change(raw)
    state, stages = process(loader, raw)
    assert state is None
    assert stages.drops == {reason: 1}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["messages"][0].update(unknown="visible"),
        lambda r: r["messages"][0].update(role="developer"),
        lambda r: r["tools"][0].update(unknown="visible"),
        lambda r: r["tools"][0]["function"].update(unknown="visible"),
        lambda r: r["tools"][0]["function"].update(required="x"),
    ],
)
def test_unknown_or_ambiguous_source_values_reject(loader, mutate):
    raw = row()
    mutate(raw)
    with pytest.raises(Reject):
        loader._convert_row(raw, "interactive_agent")


def test_reasoning_is_representation_based_and_does_not_strip_visible_think_tool(
    loader,
):
    raw = row(tools=[tool("think")])
    raw["messages"][1]["tool_calls"][0]["function"]["name"] = "think"
    raw["messages"][1]["reasoning_content"] = "private"
    raw["messages"][-1]["content"] = (
        "A<think>private</think>B <reasoning>private2</reasoning> C ` <think>literal</think> `"
    )
    raw["messages"][-1]["reasoning_content"] = "private3"
    retained, _ = process(loader, raw)
    assert retained.sample.messages[-1]["content"] == raw["messages"][-1]["content"]
    stripped, _ = process(loader, raw, FilterConfig(strip_thinking=True))
    assert stripped is not None
    assert stripped.sample.messages[-1]["content"] == "AB  C ` <think>literal</think> `"
    assert stripped.sample.messages[1]["tool_calls"][0]["function"]["name"] == "think"
    assert all("reasoning_content" not in m for m in stripped.sample.messages)
    assert raw["messages"][1]["reasoning_content"] == "private"


@pytest.mark.parametrize(
    "answer,reason",
    [
        ("<think>private</think>", "empty_final_assistant"),
        ("<think>unfinished", "ambiguous_reasoning"),
    ],
)
def test_final_reasoning_transform_can_reject(loader, answer, reason):
    raw = row()
    raw["messages"][-1]["content"] = answer
    state, stages = process(loader, raw, FilterConfig(strip_thinking=True))
    assert state is None and stages.drops == {reason: 1}


def test_consecutive_assistants_and_visible_failure_recovery_are_valid(loader):
    raw = row()
    raw["messages"].insert(1, {"role": "assistant", "content": "I will try this."})
    raw["messages"][3]["content"] = '{"error":"please retry"}'
    raw["messages"][4:4] = [deepcopy(raw["messages"][2]), deepcopy(raw["messages"][3])]
    raw["messages"][4]["tool_calls"][0]["id"] = "retry"
    raw["messages"][5]["tool_call_id"] = "retry"
    assert process(loader, raw)[0] is not None


def credential_row(value="123456"):
    raw = row(
        tools=[
            tool(
                "authenticate_user",
                parameters={
                    "type": "object",
                    "properties": {"verification_code": {"type": "string"}},
                    "required": ["verification_code"],
                },
            )
        ]
    )
    raw["messages"][1]["tool_calls"][0]["function"].update(
        name="authenticate_user", arguments=json.dumps({"verification_code": value})
    )
    return raw


def test_hidden_code_future_echo_and_assistant_reasoning_do_not_ground(loader):
    raw = credential_row()
    raw["metadata"]["code"] = "123456"
    raw["messages"][1]["content"] = "Use 123456"
    raw["messages"][1]["reasoning_content"] = "The code is 123456"
    raw["messages"][2]["content"] = '{"code":"123456"}'
    raw["tools"][0]["function"]["description"] = "For example 123456"
    state, stages = process(loader, raw)
    assert state is None and stages.drops == {"ungrounded_credential": 1}


def test_visible_user_and_prior_result_ground_codes(loader):
    raw = credential_row()
    raw["messages"][0]["content"] = "My code: 123456."
    assert process(loader, raw)[0] is not None
    raw = credential_row()
    raw["tools"].insert(0, tool("get_code"))
    raw["messages"][1:1] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [call("get_code", id="prior")],
        },
        {"role": "tool", "tool_call_id": "prior", "content": '{"code":"123456"}'},
    ]
    assert process(loader, raw)[0] is not None


def test_final_system_override_rechecks_grounding_and_cannot_rescue(loader):
    raw = credential_row()
    raw["messages"].insert(
        0, {"role": "system", "content": "Verification code: 123456."}
    )
    assert process(loader, raw)[0] is not None
    state, stages = process(loader, raw, FilterConfig(system_message_override=""))
    assert state is None and stages.stage_drops == {"final_visible_inputs": 1}
    raw["messages"].pop(0)
    state, stages = process(
        loader, raw, FilterConfig(system_message_override="Verification code: 123456.")
    )
    assert state is None and stages.stage_drops == {"visible_inputs": 1}


def test_new_password_is_not_misclassified_as_existing_secret(loader):
    raw = row(tools=[tool("create_user_account")])
    raw["messages"][1]["tool_calls"][0]["function"].update(
        name="create_user_account", arguments='{"password":"generated value"}'
    )
    assert process(loader, raw)[0] is not None


def test_streaming_counts_malformed_rows_curation_scope_and_final_count(
    loader, monkeypatch, tmp_path
):
    raw = row()
    plain = row(
        messages=[
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    )
    bad = deepcopy(raw)
    bad["messages"].pop(2)
    serialized = [
        json.dumps(raw),
        json.dumps(raw),
        "{bad",
        "",
        json.dumps(bad),
        json.dumps(plain),
    ]
    monkeypatch.setattr(loader, "_source_lines", lambda split: iter(serialized))
    config_cls = (
        loader.NemotronAgenticV1Config
        if loader is v1
        else loader.NemotronAgenticV2Config
    )
    config = config_cls(splits=("interactive_agent",))
    values, report = loader.load(
        config, curation_config=CurationConfig(temporary_directory=tmp_path)
    )
    assert len(values) == report.final_count == 2
    assert report.raw_count == 6 and report.stage1_count == 4
    assert report.filtered_count == 3 and report.stage1_drop_reasons == {
        "malformed_json": 2
    }
    assert report.filter_drop_reasons == {"unbalanced_cardinality": 1}
    assert not list(tmp_path.iterdir())


def test_v1_anonymous_singleton_valid_parallel_and_contradiction_quarantined():
    raw = row()
    del raw["messages"][2]["tool_call_id"]
    state, _ = process(v1, raw, split="tool_calling")
    assert state is not None
    raw["messages"][1]["tool_calls"].append(call(id="b"))
    raw["messages"].insert(3, {"role": "tool", "content": "second"})
    state, stages = process(v1, raw, split="tool_calling")
    assert state is None and stages.drops == {"ambiguous_source_tool_linkage": 1}
    state, stages = process(v1, row(), split="tool_calling")
    assert state is None and stages.drops == {
        "unexpected_v1_anonymous_result_linkage": 1
    }


@pytest.mark.parametrize("anonymous", [False, True])
def test_reused_call_ids_across_distinct_batches_reject(loader, anonymous):
    if anonymous and loader is v2:
        return
    raw = row()
    raw["messages"][-1:-1] = deepcopy(raw["messages"][1:3])
    if anonymous:
        for message in raw["messages"]:
            if message["role"] == "tool":
                message.pop("tool_call_id")
    state, stages = process(
        loader, raw, split="tool_calling" if anonymous else "interactive_agent"
    )
    assert state is None and stages.drops == {"invalid_source_tool_linkage": 1}


def test_curation_levels_are_cumulative_and_keep_original_winner(
    loader, monkeypatch, tmp_path
):
    first = row(uuid="first")
    first["messages"][-1]["content"] = "\nDone\n"
    exact = deepcopy(first)
    exact["metadata"]["uuid"] = "exact"
    whitespace = row(uuid="whitespace")
    whitespace["messages"][-1]["content"] = "Done"
    short = row(uuid="short")
    short["messages"][-1]["content"] = "OK"
    monkeypatch.setattr(
        loader,
        "_source_lines",
        lambda split: map(json.dumps, [first, exact, whitespace, short]),
    )
    config_cls = (
        loader.NemotronAgenticV1Config
        if loader is v1
        else loader.NemotronAgenticV2Config
    )
    values, report = loader.load(
        config_cls(splits=("interactive_agent",)),
        curation_config=CurationConfig(temporary_directory=tmp_path),
    )
    assert len(values) == report.final_count == 1
    assert values[0].sample_id == "interactive_agent_short"
    assert values[0].raw == short
    stats = report.dataset_config_transform_counts["curation_by_subset"][
        "interactive_agent"
    ][0]
    assert [
        stats["stages"][level]["removed_samples"]
        for level in ("level_1", "level_1_5", "level_2")
    ] == [1, 1, 1]
    assert set(stats["audit"]) == {"level_3", "level_4", "level_5"}
    assert not list(tmp_path.iterdir())


def test_closing_stream_early_does_not_claim_final_count(loader, monkeypatch, tmp_path):
    records = [row(uuid=str(i)) for i in range(3)]
    for i, record in enumerate(records):
        record["messages"][0]["content"] += str(i)
    monkeypatch.setattr(loader, "_source_lines", lambda split: map(json.dumps, records))
    config_cls = (
        loader.NemotronAgenticV1Config
        if loader is v1
        else loader.NemotronAgenticV2Config
    )
    stream, report = loader.iter_load(
        config_cls(splits=("interactive_agent",)),
        curation_config=CurationConfig(temporary_directory=tmp_path),
    )
    assert next(stream).sample_id == "interactive_agent_0"
    stream.close()
    assert report.final_count is None
    assert not list(tmp_path.iterdir())


def test_v1_structured_result_values_preserved():
    for value in ({"a": [1, False, None]}, [1, "x"], False, None):
        raw = row()
        raw["messages"][2] = {"role": "tool", "content": value}
        state, _ = process(v1, raw, split="tool_calling")
        assert state is not None
        assert json.loads(state.sample.messages[2]["content"]) == value
        assert raw["messages"][2]["content"] == value


def test_unsafe_config_requests_fail_without_loading():
    with pytest.raises(ValueError):
        v1.iter_load(v1.NemotronAgenticV1Config(drop_empty_system=True))
    with pytest.raises(ValueError):
        v2.iter_load(v2.NemotronAgenticV2Config(interactive_agent_domain_cap=10))
    with pytest.raises(ValueError):
        v1.iter_load(v1.NemotronAgenticV1Config(splits=("search",)))
    with pytest.raises(ValueError):
        v2.iter_load(v2.NemotronAgenticV2Config(splits=("search", "search")))


@pytest.mark.parametrize(
    "name,field",
    [
        ("sending_sms_otp_custom_otp", "otp"),
        ("create_verification_code", "verification_code"),
        ("create_access_token", "access_token"),
    ],
)
def test_new_credentials_are_not_inferred_from_slot_names(loader, name, field):
    raw = row(
        tools=[
            tool(
                name,
                parameters={
                    "type": "object",
                    "properties": {
                        field: {"type": "string", "description": "New value to create"}
                    },
                },
            )
        ]
    )
    raw["messages"][1]["tool_calls"][0]["function"].update(
        name=name, arguments=json.dumps({field: "new code"})
    )
    assert process(loader, raw)[0] is not None


@pytest.mark.parametrize("evidence", [None, 123456, "123456", True])
def test_numeric_otp_requires_prior_scalar_evidence_and_preserves_source(
    loader, evidence
):
    raw = row(tools=[tool("f"), tool("submit_code_telegram_submitcode_get")])
    raw["messages"][2]["content"] = json.dumps({"otp": evidence})
    raw["messages"][-1:-1] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                call("submit_code_telegram_submitcode_get", '{"otp":123456}', id="otp")
            ],
        },
        {"role": "tool", "tool_call_id": "otp", "content": "verified"},
    ]
    original = deepcopy(raw)
    state, stages = process(loader, raw)
    if evidence is None or evidence is True:
        assert state is None and stages.drops == {"ungrounded_credential": 1}
    else:
        assert state is not None
        assert state.sample.messages[-3]["tool_calls"][0]["function"]["arguments"] == (
            '{"otp":123456}'
        )
    assert raw == original


def test_numeric_otp_can_be_grounded_by_exact_prior_user_text(loader):
    raw = row(tools=[tool("submit_code_telegram_submitcode_get")])
    raw["messages"][0]["content"] = "Use OTP 123456."
    raw["messages"][1]["tool_calls"][0]["function"].update(
        name="submit_code_telegram_submitcode_get", arguments='{"otp":123456}'
    )
    assert process(loader, raw)[0] is not None
    raw["messages"][0]["content"] = "Use OTP 1234567."
    state, stages = process(loader, raw)
    assert state is None and stages.drops == {"ungrounded_credential": 1}


@pytest.mark.parametrize("visible", ["123456.0", "-123456", "1,123456"])
def test_numeric_credential_is_not_a_substring_of_a_different_number(loader, visible):
    raw = credential_row()
    raw["messages"][0]["content"] = f"The value I have is {visible}."
    state, stages = process(loader, raw)
    assert state is None and stages.drops == {"ungrounded_credential": 1}


@pytest.mark.parametrize(
    "field,visible",
    [
        ("ssn_last4", "XXX-XX-1234"),
        ("phone_last4", "555-1234"),
        ("tax_id_last4", "XXXX-1234"),
        ("last_four_card", "****-****-****-1234"),
    ],
)
def test_audited_last_four_credentials_preserve_formatted_visible_identifiers(
    loader, field, visible
):
    raw = row(tools=[tool("authenticate_user")])
    raw["messages"][0]["content"] = f"My identifier is {visible}."
    raw["messages"][1]["tool_calls"][0]["function"].update(
        name="authenticate_user", arguments=json.dumps({"credentials": {field: "1234"}})
    )
    assert process(loader, raw)[0] is not None
    raw["messages"][0]["content"] = "The value is -1234."
    assert process(loader, raw)[0] is None
    raw["messages"][0]["content"] = "The value is 1234.0."
    assert process(loader, raw)[0] is None


def test_last_four_method_is_required_and_does_not_ground_an_unrelated_otp(loader):
    raw = row(
        tools=[
            tool(
                "authenticate_member",
                parameters={
                    "type": "object",
                    "properties": {
                        "verification_method": {"enum": ["phone_digits", "access_pin"]},
                        "verification_code": {"type": "string"},
                    },
                },
            )
        ]
    )
    raw["messages"][0]["content"] = "My phone is 555-1234."
    function = raw["messages"][1]["tool_calls"][0]["function"]
    function.update(
        name="authenticate_member",
        arguments='{"verification_method":"phone_digits","verification_code":"1234"}',
    )
    assert process(loader, raw)[0] is not None
    function["arguments"] = (
        '{"verification_method":"access_pin","verification_code":"1234"}'
    )
    assert process(loader, raw)[0] is None
    raw["tools"] = [tool("authenticate_user")]
    function.update(
        name="authenticate_user",
        arguments='{"credentials":{"phone_last4":"1234","otp":"1234"}}',
    )
    assert process(loader, raw)[0] is None


def calendar_row(*, name="get_available_rooms", field="date", target="2025-09-24"):
    raw = row(tools=[tool(name)])
    raw["messages"][0]["content"] = "Please check availability for tomorrow."
    raw["messages"][1]["tool_calls"][0]["function"].update(
        name=name, arguments=json.dumps({field: target})
    )
    return raw


@pytest.mark.parametrize("strip", [False, True])
def test_relative_calendar_consumption_cannot_invent_a_clock(loader, strip):
    raw = calendar_row()
    raw["metadata"]["today"] = "2025-09-23"
    raw["messages"][1]["reasoning_content"] = "Today is 2025-09-23."
    raw["messages"][1]["content"] = "Today is September 23, 2025."
    raw["tools"][0]["function"]["description"] = "Example current date: 2025-09-23."
    raw["messages"][2]["content"] = '{"current_date":"2025-09-23"}'
    state, stages = process(loader, raw, FilterConfig(strip_thinking=strip))
    assert state is None and stages.drops == {"ungrounded_temporal_reference": 1}


@pytest.mark.parametrize(
    "evidence",
    [
        "Tomorrow, September 24, 2025, please.",
        "Tomorrow, 24th September 2025, please.",
        "Tomorrow, 2025-09-24, please.",
        "Today is September 23, 2025. Book tomorrow.",
        "Current date: Sept. 23, 2025. Book tomorrow.",
    ],
)
def test_explicit_target_dates_and_labeled_current_dates_are_visible(loader, evidence):
    raw = calendar_row()
    raw["messages"][0]["content"] = evidence
    original = deepcopy(raw)
    state, _ = process(loader, raw)
    assert state is not None and raw == original


@pytest.mark.parametrize(
    "historical",
    [
        "D-Day was June 6, 1944.",
        "Today is unknown. D-Day was June 6, 1944.",
        "Current date: unavailable; D-Day was June 6, 1944.",
    ],
)
def test_unrelated_historical_dates_do_not_become_current_clocks(loader, historical):
    raw = calendar_row()
    raw["messages"].insert(0, {"role": "system", "content": historical})
    state, stages = process(loader, raw)
    assert state is None and stages.drops == {"ungrounded_temporal_reference": 1}


@pytest.mark.parametrize(
    "user_text",
    [
        "Please hold today's rate for Nov 15–22, 2025.",
        "Next month means October 2025; check October 17 to October 19.",
        "The certificate arrives tomorrow. Rent from June 15th to July 5th.",
        "Book tomorrow, 09/24/2025.",
        "Book tomorrow, 24.09.2025.",
        "Book tomorrow, September 24.",
    ],
)
def test_unresolved_explicit_user_dates_do_not_prove_missing_clock(loader, user_text):
    raw = calendar_row()
    raw["messages"][0]["content"] = user_text
    original = deepcopy(raw)
    assert process(loader, raw)[0] is not None
    assert raw == original


def test_user_historical_full_date_does_not_disable_missing_clock_check(loader):
    raw = calendar_row()
    raw["messages"][0]["content"] = (
        "D-Day was June 6, 1944. Check room availability for tomorrow under $325.00."
    )
    state, stages = process(loader, raw)
    assert state is None and stages.drops == {"ungrounded_temporal_reference": 1}


def test_temporal_policy_does_not_delete_unrelated_historical_fact_queries(loader):
    raw = calendar_row(name="historical_exchange_rates", target="1944-06-06")
    raw["messages"][0]["content"] = (
        "Today I'm studying D-Day. Query the historical exchange rate for that day."
    )
    assert process(loader, raw)[0] is not None
    raw = calendar_row(name="create_test_record")
    raw["messages"][0]["content"] = (
        "Tomorrow we demonstrate generated sample records. Make up a date value."
    )
    assert process(loader, raw)[0] is not None


def test_temporal_gate_abstains_after_unclassified_visible_results(loader):
    raw = calendar_row()
    raw["tools"].append(tool("lookup_reservation"))
    raw["messages"][1:1] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [call("lookup_reservation", id="lookup")],
        },
        {"role": "tool", "tool_call_id": "lookup", "content": "Visible backend state"},
    ]
    assert process(loader, raw)[0] is not None


def test_final_system_override_rechecks_temporal_evidence(loader):
    raw = calendar_row()
    raw["messages"].insert(0, {"role": "system", "content": "Current date: 2025-09-23"})
    assert process(loader, raw)[0] is not None
    state, stages = process(loader, raw, FilterConfig(system_message_override=""))
    assert state is None and stages.stage_drops == {"final_temporal_inputs": 1}


def test_exact_12306_consuming_definition_and_visible_reasoning_boundary(loader):
    parameters = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "format": "date",
                "description": "出发日期 格式：YYYY-MM-DD",
            },
            "fromCity": {"type": "string", "description": "出发城市"},
            "toCity": {"type": "string", "description": "到达城市"},
        },
    }
    raw = calendar_row(name="search", target="2025-08-18")
    raw["tools"] = [
        tool("search", parameters=parameters, description="查询12306火车票")
    ]
    raw["messages"][0]["content"] = "Travel from Beijing to Shanghai next Monday."
    state, stages = process(loader, raw)
    assert state is None and stages.drops == {"ungrounded_temporal_reference": 1}
    raw["tools"][0]["function"]["description"] = "Search historical records"
    assert process(loader, raw)[0] is not None


def test_source_call_index_is_assembly_metadata_not_batch_linkage(loader):
    raw = row()
    raw["messages"][1]["tool_calls"][0]["index"] = -1 if loader is v2 else 3
    state, _ = process(loader, raw)
    assert state is not None
    assert "index" not in state.sample.messages[1]["tool_calls"][0]
    assert raw["messages"][1]["tool_calls"][0]["index"] == (-1 if loader is v2 else 3)


def async_row(*, include_start=True, status="succeeded", payload=True):
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Search places."}]
    if include_start:
        messages += [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call("get_serp_async", id="start")],
            },
            {
                "role": "tool",
                "tool_call_id": "start",
                "content": '{"task_id":"task-1","status":"queued"}',
            },
        ]
    else:
        messages[0]["content"] = "Check status of task-1."
    result = {"task_id": "task-1", "status": status}
    if payload:
        result["result"] = ["found result"]
    messages += [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                call("gettaskresult_free_of_use", '{"task_id":"task-1"}', id="poll")
            ],
        },
        {"role": "tool", "tool_call_id": "poll", "content": json.dumps(result)},
        {"role": "assistant", "content": "Here is the task status."},
    ]
    return row(
        messages=messages,
        tools=[tool("get_serp_async"), tool("gettaskresult_free_of_use")],
    )


def test_async_start_poll_completion_and_external_status_query(loader):
    assert process(loader, async_row())[0] is not None
    assert (
        process(
            loader, async_row(include_start=False, status="pending", payload=False)
        )[0]
        is not None
    )


@pytest.mark.parametrize("status,payload", [("pending", False), ("succeeded", False)])
def test_started_job_cannot_finish_pending_or_without_payload(loader, status, payload):
    state, stages = process(loader, async_row(status=status, payload=payload))
    assert state is None and stages.drops == {"incomplete_async_job": 1}


def test_async_missing_identifier_not_rescued_by_future_result(loader):
    raw = async_row(include_start=False)
    raw["messages"][0]["content"] = "Check my task."
    state, stages = process(loader, raw)
    assert state is None and stages.drops == {"ungrounded_job_identifier": 1}


def test_async_poll_result_must_match_bound_job(loader):
    raw = async_row()
    raw["messages"][-2]["content"] = (
        '{"task_id":"other","status":"succeeded","result":[1]}'
    )
    state, stages = process(loader, raw)
    assert state is None and stages.drops == {"mismatched_job_identifier": 1}


def test_async_retry_preserves_pending_observation_and_grounded_completion(loader):
    raw = async_row()
    retry = deepcopy(raw["messages"][-3:-1])
    retry[1]["content"] = '{"task_id":"task-1","status":"pending"}'
    retry[0]["tool_calls"][0]["id"] = "retry"
    retry[1]["tool_call_id"] = "retry"
    raw["messages"][-3:-3] = retry
    state, _ = process(loader, raw)
    assert state is not None
    assert len(state.sample.messages) == len(raw["messages"])


def test_job_grounding_final_override_recheck(loader):
    raw = async_row(include_start=False)
    raw["messages"][0]["content"] = "Check my task."
    raw["messages"].insert(0, {"role": "system", "content": "Active task: task-1."})
    state, stages = process(loader, raw, FilterConfig(system_message_override=""))
    assert state is None and stages.stage_drops == {"final_async_lifecycle": 1}


def test_future_static_definition_is_not_invented_from_discovery_text(loader):
    raw = row(tools=[tool("list_tools")])
    raw["messages"][1]["tool_calls"][0]["function"]["name"] = "list_tools"
    raw["messages"][2]["content"] = '{"tools":[{"name":"hidden"}]}'
    raw["messages"][-1:-1] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [call("hidden", id="hidden")],
        },
        {"role": "tool", "tool_call_id": "hidden", "content": "done"},
    ]
    state, stages = process(loader, raw)
    assert state is None and stages.drops == {"undefined_function_calls": 1}


def described_credential_row(name, field, description, value, *, default=None):
    schema = {"type": "string", "description": description}
    if default is not None:
        schema["default"] = default
    raw = row(
        tools=[tool(name, parameters={"type": "object", "properties": {field: schema}})]
    )
    raw["messages"][1]["tool_calls"][0]["function"].update(
        name=name, arguments=json.dumps({field: value})
    )
    return raw


@pytest.mark.parametrize(
    "name,field,description",
    [
        (
            "mutual_funds",
            "x_mashape_key",
            "API authentication key for accessing financial data services",
        ),
        (
            "translate_to_old_english",
            "x_funtranslations_api_secret",
            "API Key for accessing the FunTranslations Old English Translator.",
        ),
        (
            "current_weather",
            "apikey",
            "The apikey neeeded to have access to the OpenWeatherMap API.",
        ),
        (
            "landscapeProducts",
            "serpapi_key",
            "Parameter defines the SerpApi private key to use.",
        ),
        (
            "recieve_list",
            "login_password",
            "The password(POP3) to login to the email server.",
        ),
        (
            "authenticate_user",
            "value",
            "Authentication value (email code, SMS code, password)",
        ),
        (
            "authenticate_customer",
            "authentication_token",
            "Verification token (PIN/password)",
        ),
        ("authorize_parent", "auth_data", "Authentication token/payload"),
        (
            "on_line",
            "channel_token",
            "LINEチャンネルアクセストークンを指定します。API認証に必要なシークレットキーです。",
        ),
    ],
)
@pytest.mark.parametrize("strip", [False, True])
def test_descriptor_audit_existing_credentials_need_prior_visible_values(
    loader, name, field, description, strip
):
    raw = described_credential_row(name, field, description, "absent-secret")
    raw["messages"][1]["reasoning_content"] = "Use absent-secret."
    raw["messages"][2]["content"] = '{"token":"absent-secret"}'
    original = deepcopy(raw)
    state, stages = process(loader, raw, FilterConfig(strip_thinking=strip))
    assert state is None and stages.drops == {"ungrounded_credential": 1}
    assert raw == original
    raw["messages"][0]["content"] = "Use my credential absent-secret."
    state, _ = process(loader, raw, FilterConfig(strip_thinking=strip))
    assert state is not None
    assert state.sample.messages[1]["tool_calls"][0]["function"][
        "arguments"
    ] == json.dumps({field: "absent-secret"})


def test_credential_signature_is_finite_and_does_not_infer_new_slot_semantics(loader):
    raw = described_credential_row(
        "create_test_key",
        "apikey",
        "The apikey neeeded to have access to the OpenWeatherMap API.",
        "new-secret",
    )
    assert process(loader, raw)[0] is not None
    raw = described_credential_row(
        "current_weather",
        "apikey",
        "Newly generated key for a demonstration.",
        "new-secret",
    )
    assert process(loader, raw)[0] is not None


@pytest.mark.parametrize(
    "name,field,description,value,default",
    [
        (
            "create_user_account",
            "password",
            "The password for the user account",
            "new-password",
            None,
        ),
        (
            "shorten_a_url",
            "password",
            "Optional password required to access the shortened URL. If provided, this password must be used during URL retrieval. No password is required by default.",
            "new-password",
            "",
        ),
        (
            "generate_jwt_token",
            "secret_key",
            "The secret key used for signing the token",
            "generated-key",
            None,
        ),
        (
            "cash_flow_statement",
            "apikey",
            "Your API key from https://fmpcloud.io/register.",
            "demo",
            "demo",
        ),
        (
            "get_vehicule_info",
            "token",
            "The token for API authentication. Defaults to 'TokenDemoRapidapi'.",
            "TokenDemoRapidapi",
            "TokenDemoRapidapi",
        ),
        (
            "calling_api_method",
            "oauth_signature",
            "HMAC-SHA1 signature for request authentication",
            "derived-signature",
            None,
        ),
    ],
)
def test_generated_derived_and_ambiguous_default_contracts_remain_outside_extension(
    loader, name, field, description, value, default
):
    raw = described_credential_row(name, field, description, value, default=default)
    assert process(loader, raw)[0] is not None


def test_explicit_placeholder_default_cannot_supply_authentication(loader):
    raw = described_credential_row(
        "get_amazon_search_results",
        "api_key",
        "API key for authenticating with the Amazon data scraper service",
        "YOUR_API_KEY",
        default="YOUR_API_KEY",
    )
    state, stages = process(loader, raw)
    assert state is None and stages.drops == {"ungrounded_credential": 1}
    raw["messages"][0]["content"] = "For this explicit test, use YOUR_API_KEY."
    assert process(loader, raw)[0] is not None


def test_composite_credentials_with_reformatted_dob_are_not_newly_rejected(loader):
    raw = row(
        tools=[
            tool(
                "authenticate_applicant",
                parameters={
                    "type": "object",
                    "properties": {
                        "credentials": {
                            "type": "object",
                            "description": "Authentication credentials object containing either member_id/dob/ssn_last4 OR full_name/email/zip_code",
                        }
                    },
                },
            )
        ]
    )
    raw["messages"][0]["content"] = (
        "Member ID MEM-3456, DOB March 15, 1988, last four SSN digits 5678."
    )
    raw["messages"][1]["tool_calls"][0]["function"].update(
        name="authenticate_applicant",
        arguments='{"credentials":{"member_id":"MEM-3456","dob":"1988-03-15","ssn_last4":"5678"}}',
    )
    assert process(loader, raw)[0] is not None


@pytest.mark.parametrize("strip", [False, True])
def test_documented_bearer_scheme_uses_prior_bare_token_without_changing_content(
    loader, strip
):
    raw = described_credential_row(
        "sentiment_results",
        "authorization",
        "Bearer token for API authentication and access control",
        "Bearer BROOKLYN34Miami",
    )
    raw["messages"][0]["content"] = "Use my auth token 'BROOKLYN34Miami'."
    original = deepcopy(raw)
    state, _ = process(loader, raw, FilterConfig(strip_thinking=strip))
    assert state is not None and raw == original
    assert (
        state.sample.messages[1]["tool_calls"][0]["function"]["arguments"]
        == '{"authorization": "Bearer BROOKLYN34Miami"}'
    )
    raw["messages"][0]["content"] = "The token is BROOKLYN34MiamiExtra."
    state, stages = process(loader, raw)
    assert state is None and stages.drops == {"ungrounded_credential": 1}


def test_bearer_evidence_is_rechecked_after_system_override(loader):
    raw = described_credential_row(
        "sentiment_results",
        "authorization",
        "Bearer token for API authentication and access control",
        "Bearer token-secret",
    )
    raw["messages"].insert(0, {"role": "system", "content": "Token: token-secret"})
    assert process(loader, raw)[0] is not None
    state, stages = process(
        loader, raw, FilterConfig(system_message_override="Replacement")
    )
    assert state is None and stages.drops == {"ungrounded_credential": 1}


@pytest.mark.parametrize(
    "value,visible",
    [("38927", "3-8-9-2-7"), ("123456", "1 - 2 - 3 - 4 - 5 - 6"), ("0012", "0-0-1-2")],
)
def test_exact_otp_consumers_preserve_spelled_individual_digits(loader, value, visible):
    raw = described_credential_row(
        "verify_authentication_code",
        "code",
        "Verification code provided by user",
        value,
    )
    raw["messages"][0]["content"] = f"The code I received is {visible}."
    original = deepcopy(raw)
    assert process(loader, raw)[0] is not None
    assert raw == original


@pytest.mark.parametrize(
    "visible",
    [
        "-1-2-3-4-5-6",
        "+1-2-3-4-5-6",
        "0-1-2-3-4-5-6",
        "1-2-3-4-5-6-7",
        "1-2-3-4-5-6.0",
        "1-2-3-4-5-6x",
        "123456.0",
        "-123456",
    ],
)
def test_spelled_otp_evidence_does_not_match_another_numeric_value(loader, visible):
    raw = described_credential_row(
        "verify_authentication_code",
        "code",
        "Verification code provided by user",
        "123456",
    )
    raw["messages"][0]["content"] = f"The value is {visible}."
    state, stages = process(loader, raw)
    assert state is None and stages.drops == {"ungrounded_credential": 1}


def test_spelled_digits_are_not_password_or_api_key_evidence(loader):
    raw = described_credential_row(
        "recieve_list",
        "login_password",
        "The password(POP3) to login to the email server.",
        "123456",
    )
    raw["messages"][0]["content"] = "The value is 1-2-3-4-5-6."
    assert process(loader, raw)[0] is None


def test_audited_ambiguous_code_label_contract_is_held(loader):
    raw = described_credential_row(
        "validate_verification_code", "code", "6-digit verification code", "123456"
    )
    raw["tools"][0]["function"]["parameters"]["properties"]["code"]["pattern"] = (
        "^[0-9]{6}$"
    )
    raw["messages"][0]["content"] = "[code-123456] – verification complete."
    assert process(loader, raw)[0] is not None


def test_undocumented_bearer_construction_holds_its_consuming_contract(loader):
    raw = described_credential_row(
        "deleteaccount",
        "token",
        "The authentication token of the user.",
        "Bearer user-token",
    )
    raw["messages"][0]["content"] = "My token is user-token."
    assert process(loader, raw)[0] is not None


@pytest.mark.parametrize(
    "visible",
    [
        "0 - 1 - 2 - 3 - 4 - 5 - 6",
        "0- 1-2-3-4-5-6",
        "- 1-2-3-4-5-6",
        "+ 1-2-3-4-5-6",
        "-0 - 1-2-3-4-5-6",
        "12 - 3-4-5-6",
    ],
)
def test_spelled_otp_does_not_restart_inside_a_spaced_longer_run(loader, visible):
    raw = described_credential_row(
        "verify_authentication_code",
        "code",
        "Verification code provided by user",
        "123456",
    )
    raw["messages"][0]["content"] = f"The value is {visible}."
    state, stages = process(loader, raw)
    assert state is None and stages.drops == {"ungrounded_credential": 1}


@pytest.mark.parametrize(
    "name,description,value",
    [
        (
            "document_getdocumentlist",
            "Access token or API key used to authenticate the request and verify user permissions. Format depends on the authentication system (e.g., Bearer token, API key string)",
            "Basic dXNlcjpwYXNzd29yZA==",
        ),
        (
            "gettaxratebyzip",
            "API key or bearer token for authenticating with the tax rate service. Format: 'Bearer <token>' or 'ApiKey <key>' depending on service requirements.",
            "ApiKey tax-key",
        ),
    ],
)
def test_other_composed_auth_header_contracts_are_held(
    loader, name, description, value
):
    raw = described_credential_row(name, "authorization", description, value)
    raw["messages"][0]["content"] = (
        "My username is user, password is password, and API key is tax-key."
    )
    assert process(loader, raw)[0] is not None


def test_inactive_alternative_authentication_slot_is_held(loader):
    raw = described_credential_row(
        "downloaddocument",
        "authorization1",
        "PandaDoc API key with appropriate document access permissions. Required when no bearer token is provided in the 'authorization' parameter. Format: 'YOUR_API_KEY'",
        "Bearer auth_code_7890",
    )
    parameters = raw["tools"][0]["function"]["parameters"]
    parameters["properties"]["authorization"] = {
        "type": "string",
        "description": "Bearer token for alternative authentication. Format: 'Bearer <access_token>'. Takes precedence over authorization1 when provided.",
        "default": "",
    }
    arguments = {
        "authorization": "Bearer auth_code_7890",
        "authorization1": "Bearer auth_code_7890",
    }
    raw["messages"][1]["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments)
    raw["messages"][0]["content"] = "My token is auth_code_7890."
    original = deepcopy(raw)
    assert process(loader, raw)[0] is not None
    assert raw == original


@pytest.mark.parametrize(
    "name,field,description,value",
    [
        (
            "all_locations",
            "api_token",
            "Authentication token required to access the API. Must be included in request headers as 'Authorization: Bearer <token>'",
            "Authorization: Bearer my-token",
        ),
        (
            "media",
            "authorization",
            "Authentication token for API access. Must be a valid Bearer token formatted as 'Authorization: Bearer <your_api_key>'",
            "Authorization: Bearer my-token",
        ),
        (
            "followings_that_don_t_follow_you_back",
            "authorization",
            "Authentication token required for accessing social media account data. Format: 'Bearer <token>' for OAuth2 or 'API_KEY=<value>' for API key authentication",
            "API_KEY=my-token",
        ),
    ],
)
def test_whole_or_alternative_authentication_headers_are_held(
    loader, name, field, description, value
):
    raw = described_credential_row(name, field, description, value)
    raw["messages"][0]["content"] = "My token is my-token."
    assert process(loader, raw)[0] is not None
