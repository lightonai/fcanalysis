"""Adversarial contracts for the pinned UltraData Tool-Use adapter."""

from collections import Counter
from copy import deepcopy
import json

import pytest

from fcanalysis.loaders import ultradata_tool_use as loader
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.curation import CurationConfig
from fcanalysis.loaders.normalization import Reject


def tool(name="lookup"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Look up a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def assistant(content="", *, calls=None, reasoning="Reason exactly."):
    out = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if calls is not None:
        out["tool_calls"] = calls
    return out


def call(city="Paris", *, name="lookup", id=None):
    out = {"type": "function", "function": {"name": name, "arguments": {"city": city}}}
    if id is not None:
        out["id"] = id
    return out


def row(*, source="toolmind_graphsyn"):
    return {
        "uuid": "row_1",
        "source": source,
        "domain": "Tool_Use",
        "tools": [tool()],
        "messages": [
            {"role": "user", "content": "Paris, please."},
            assistant(calls=[call()]),
            {"role": "tool", "content": '[{"value":18},{"value":24}]'},
            assistant("Exact Answer.  KEEP Capitals!\n"),
        ],
    }


def run(raw, *, strip=False, override=None):
    return loader._pipeline(
        FilterConfig(strip_thinking=strip, system_message_override=override)
    ).process(loader._convert_row(raw, Counter()))


@pytest.mark.parametrize("strip", [False, True])
def test_lossless_payloads_native_reasoning_and_raw_isolation(strip):
    raw = row()
    raw["messages"][-1]["content"] = (
        "<think>private</think>Exact Answer.  KEEP Capitals!\n"
    )
    before = deepcopy(raw)
    state = run(raw, strip=strip)
    assert state is not None
    assert raw == before == state.sample.raw
    assert state.sample.messages[2]["content"] == '[{"value":18},{"value":24}]'
    assert state.sample.messages[-1]["content"] == (
        "Exact Answer.  KEEP Capitals!\n" if strip else raw["messages"][-1]["content"]
    )
    assert ("reasoning_content" in state.sample.messages[-1]) is not strip
    state.sample.tools[0]["function"]["parameters"]["required"].append("new")
    state.sample.messages[0]["content"] = "Changed"
    assert raw == before


@pytest.mark.parametrize("strip", [False, True])
def test_visible_think_tool_is_preserved(strip):
    raw = row()
    raw["tools"] = [tool("think")]
    raw["messages"][1]["tool_calls"] = [call(name="think")]
    state = run(raw, strip=strip)
    assert state is not None
    assert state.sample.tools == raw["tools"]
    assert state.sample.messages[1]["tool_calls"][0]["function"]["name"] == "think"
    assert state.sample.messages[2] == raw["messages"][2]


@pytest.mark.parametrize("ending", ["call", "tool", "user", "native_only"])
def test_never_manufactures_an_endpoint(ending):
    raw = row()
    if ending == "call":
        raw["messages"] = raw["messages"][:2]
    if ending == "tool":
        raw["messages"].pop()
    if ending == "user":
        raw["messages"].append({"role": "user", "content": "Thanks!\n[FINISHED]"})
    if ending == "native_only":
        raw["messages"][-1]["content"] = "<think>only reasoning</think>"
    assert run(raw) is None


def test_anonymous_multicall_cannot_use_array_length_or_plausibility():
    raw = row()
    raw["messages"][1]["tool_calls"].append(call("Rome"))
    assert run(raw) is None
    raw["messages"].insert(3, {"role": "tool", "content": "24"})
    assert run(raw) is None


def test_complete_ids_align_results_but_do_not_authorize_pair_permutation():
    raw = row(source="spider_bird_merged_traj")
    raw["messages"][1]["tool_calls"] = [call(id="p"), call("Rome", id="r")]
    raw["messages"][2:3] = [
        {"role": "tool", "tool_call_id": "r", "content": "24"},
        {"role": "tool", "tool_call_id": "p", "content": "18"},
    ]
    state = run(raw)
    assert state is not None
    assert [m["content"] for m in state.sample.messages[2:4]] == ["18", "24"]
    assert not state.batches[0].parallel
    assert all("tool_call_id" not in m for m in state.sample.messages)
    assert all("id" not in c for c in state.sample.messages[1]["tool_calls"])
    pipe = loader._pipeline(FilterConfig(align_results=False))
    assert pipe.process(loader._convert_row(raw, Counter())) is None
    assert pipe.drops == {"unproven_result_reordering": 1}
    raw["messages"][2]["tool_call_id"] = "unknown"
    assert run(raw) is None


@pytest.mark.parametrize("override", ["", "New policy", "Authenticate first"])
def test_system_override_cannot_erase_or_add_source_context(override):
    raw = row()
    raw["messages"].insert(
        0,
        {
            "role": "system",
            "content": "Current time is 2026-01-01. Remember: answer in French.",
        },
    )
    assert run(raw, override=override) is None
    assert run(raw, override=raw["messages"][0]["content"]) is not None
    raw["messages"].pop(0)
    if override:
        assert run(raw, override=override) is None


def test_explicit_empty_reasoning_is_not_filled_from_another_row(monkeypatch):
    donor = row()
    donor["uuid"] = "donor"
    donor["messages"] = donor["messages"][:2]
    complete = row()
    complete["messages"][1]["reasoning_content"] = ""
    monkeypatch.setattr(
        loader,
        "jsonl_lines",
        lambda *a: iter([json.dumps(donor), json.dumps(complete)]),
    )
    samples, report = loader.load(filter_config=FilterConfig())
    assert len(samples) == 1
    assert samples[0].messages[1]["reasoning_content"] == ""
    assert report.filter_drop_reasons == {"unbalanced_cardinality": 1}


def test_source_epoch_is_not_inferred_from_a_later_static_catalog():
    raw = row()
    raw["messages"][0]["content"] = (
        "I have updated some more functions you can choose from. What about now?"
    )
    assert run(raw) is None


def test_shared_legacy_projection_preserves_values_and_annotations():
    raw = row()
    fn = raw["tools"][0]["function"]
    fn["arguments"] = fn.pop("parameters")["properties"]
    fn["arguments"]["city"]["type"] = "str"
    fn["arguments"]["city"]["default"] = "  KEEP  "
    fn["required"] = ["city"]
    raw["tools"][0]["output_description"] = "Exact output."
    sample = loader._convert_row(raw, Counter())
    expected = sample.tools[0]["function"]["parameters"]
    assert expected == {
        "type": "object",
        "properties": {"city": {"type": "string", "default": "  KEEP  "}},
        "required": ["city"],
    }
    assert sample.tools[0]["output_description"] == "Exact output."
    assert raw["tools"][0]["function"]["arguments"]["city"]["type"] == "str"


def test_loss_metadata_preserves_greeting():
    raw = row(source="areal_tau2")
    raw["messages"].insert(0, {**assistant("Hello", reasoning=""), "loss": False})
    state = run(raw)
    assert state is not None
    assert state.sample.messages[0]["content"] == "Hello"
    assert "loss" not in state.sample.messages[0]
    assert state.sample.raw["messages"][0]["loss"] is False


def test_scoped_disk_curation_and_source_selection(monkeypatch, tmp_path):
    a = row()
    b = deepcopy(a)
    b["uuid"] = "duplicate"
    c = deepcopy(a)
    c["uuid"] = "other-source"
    c["source"] = "synth_search_20260609"
    monkeypatch.setattr(
        loader, "jsonl_lines", lambda *args: iter(map(json.dumps, [a, b, c]))
    )
    config = CurationConfig(temporary_directory=tmp_path)
    samples, report = loader.load(filter_config=FilterConfig(), curation_config=config)
    assert [s.sample_id for s in samples] == ["row_1", "other-source"]
    assert report.raw_count == 3 and report.final_count == 2
    assert not list(tmp_path.iterdir())
    samples, report = loader.load(
        loader.UltraDataToolUseConfig(sources=("synth_search_20260609",)),
        FilterConfig(),
        curation_config=config,
    )
    assert [s.sample_id for s in samples] == ["other-source"]
    assert report.stage1_drop_reasons == {"unselected_source": 2}


@pytest.mark.parametrize("sources", [(), ("wrong",), ("simia_retail", "simia_retail")])
def test_invalid_source_selection(sources):
    with pytest.raises(ValueError):
        loader.UltraDataToolUseConfig(sources=sources)


def test_unknown_fields_and_missing_reasoning_fail_closed():
    raw = row()
    raw["messages"][1]["hidden_context"] = "state"
    with pytest.raises(Reject, match="unknown_source_message_shape"):
        loader._convert_row(raw, Counter())
    raw = row()
    del raw["messages"][1]["reasoning_content"]
    with pytest.raises(Reject, match="unsupported_source_reasoning"):
        loader._convert_row(raw, Counter())


def retail_row():
    def definition(name, field):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {
                    "type": "object",
                    "properties": {field: {"type": "string"}},
                    "required": [field],
                },
            },
        }

    def invoke(name, field, value):
        return assistant(
            calls=[
                {
                    "type": "function",
                    "function": {"name": name, "arguments": {field: value}},
                }
            ]
        )

    raw = row(source="simia_retail")
    raw["tools"] = [
        definition("find_user_id_by_email", "email"),
        definition("get_user_details", "user_id"),
    ]
    raw["messages"] = [
        {
            "role": "system",
            "content": "# Retail agent policy\nAuthenticate the user first.",
        },
        {"role": "user", "content": "My email is alex@example.com."},
        invoke("find_user_id_by_email", "email", "alex@example.com"),
        {"role": "tool", "content": '{"result":"alex_123"}'},
        invoke("get_user_details", "user_id", "alex_123"),
        {"role": "tool", "content": '{"user_id":"alex_123"}'},
        assistant("Here are your details."),
    ]
    return raw


@pytest.mark.parametrize("envelope", ["result", "user_id"])
def test_audited_authentication_result_envelopes(envelope):
    raw = retail_row()
    raw["messages"][3]["content"] = json.dumps({envelope: "alex_123"})
    assert run(raw) is not None


@pytest.mark.parametrize(
    "mutation", ["hidden_email", "future_id", "failed_auth", "wrong_user", "no_lookup"]
)
def test_authentication_requires_prior_source_evidence(mutation):
    raw = retail_row()
    if mutation == "hidden_email":
        raw["messages"][1]["content"] = "Look up my account."
        raw["messages"][2]["reasoning_content"] = "My hidden email is alex@example.com."
    if mutation == "future_id":
        raw["messages"][4]["tool_calls"][0]["function"]["arguments"]["user_id"] = (
            "future_456"
        )
        raw["messages"][5]["content"] = '{"user_id":"future_456"}'
    if mutation == "failed_auth":
        raw["messages"][3]["content"] = '{"error":"User not found"}'
    if mutation == "wrong_user":
        raw["messages"][1]["content"] += " Also bob_456."
        raw["messages"][4]["tool_calls"][0]["function"]["arguments"]["user_id"] = (
            "bob_456"
        )
    if mutation == "no_lookup":
        raw["messages"][1]["content"] += " My user ID is alex_123."
        del raw["messages"][2:4]
    assert run(raw) is None


def test_source_text_call_exclusivity_is_enforced():
    raw = retail_row()
    raw["messages"][0]["content"] += "\nYou cannot do both at the same time."
    raw["messages"][2]["content"] = "I will look that up."
    assert run(raw) is None


def test_unverified_banking_discovery_does_not_become_an_accepted_runtime():
    assert run(row(source=loader.SOURCES[-1])) is None


@pytest.mark.parametrize(
    "failure",
    [
        "Error: user not found",
        "USER_NOT_FOUND",
        "User not found for email alex@example.com",
    ],
)
def test_failed_lookup_can_be_followed_by_visible_successful_retry(failure):
    raw = retail_row()
    attempt = deepcopy(raw["messages"][2:4])
    attempt[1]["content"] = json.dumps({"result": failure})
    raw["messages"][2:2] = attempt
    assert run(raw) is not None


def test_authentication_errors_do_not_supply_identifier_evidence():
    raw = retail_row()
    raw["messages"][3]["content"] = '{"error":"alex_123"}'
    assert run(raw) is None


@pytest.mark.parametrize("strip", [False, True])
def test_malformed_inline_native_export_is_excluded_in_both_modes(strip):
    raw = row()
    raw["messages"][-1]["content"] = "Internal reasoning.</think>Public answer."
    assert run(raw, strip=strip) is None
    raw["messages"][-1]["content"] = "Literal closing tag: `</think>`."
    assert run(raw, strip=strip) is not None
