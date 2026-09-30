"""ToolMind reconstruction is exact, scoped, and source-faithful."""

from copy import deepcopy
from typing import Any
import json

import pytest

from fcanalysis.loaders import toolmind
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.curation import CurationConfig
from fcanalysis.loaders.normalization import Reject
from fcanalysis.loaders.pipeline import (
    RowState,
    reconcile_definitions,
    validate_structure,
    validate_capabilities,
)

NO_CURATION = CurationConfig(max_level=None, audit=False)
SOURCE = toolmind.SOURCES[0]


def definition(name="lookup"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Visible operation",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        },
    }


def family():
    initial = [{"role": "user", "content": "Find one"}]
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "lookup", "arguments": {"x": 1}}}],
    }
    first = dict(call, content="<think>native call</think>")
    last = {"role": "assistant", "content": "<think>native answer</think>Done"}
    return [
        {"tools": [definition()], "conversations": initial + [first]},
        {
            "tools": [definition()],
            "conversations": initial
            + [call, {"role": "tool", "content": {"found": [1]}}, last],
        },
    ]


def run(monkeypatch, rows, filters=None, sources=None):
    monkeypatch.setattr(
        toolmind,
        "jsonl_lines",
        lambda *a, **kw: (json.dumps(r) for r in deepcopy(rows)),
    )
    return toolmind.load(
        toolmind.ToolMindConfig(sources=sources or [SOURCE]),
        filters or FilterConfig(),
        curation_config=NO_CURATION,
    )


def test_restores_exact_content_and_preserves_selected_raw(monkeypatch):
    rows = family()
    saved = deepcopy(rows)
    samples, report = run(monkeypatch, rows)
    assert len(samples) == report.final_count == 1
    assert samples[0].raw == saved[1]
    assert samples[0].messages[1]["content"] == "<think>native call</think>"
    assert samples[0].messages[-1]["content"] == "<think>native answer</think>Done"
    assert rows == saved
    assert (
        report.dataset_config_transform_counts["reconstruction"][SOURCE][
            "restored_messages"
        ]
        == 1
    )
    samples[0].messages[1]["tool_calls"][0]["function"]["arguments"] = "{}"
    samples[0].tools[0]["function"]["parameters"]["properties"]["x"]["type"] = "string"
    assert samples[0].raw == saved[1]


def test_stripping_happens_after_required_reconstruction(monkeypatch):
    samples, _ = run(monkeypatch, family(), FilterConfig(strip_thinking=True))
    assert samples[0].messages[1]["content"] == ""
    assert samples[0].messages[-1]["content"] == "Done"
    assert samples[0].raw == family()[1]
    samples, report = run(monkeypatch, family()[1:], FilterConfig(strip_thinking=True))
    assert not samples
    assert (
        report.dataset_config_transform_counts["reconstruction"][SOURCE][
            "missing_donor_rows"
        ]
        == 1
    )


def test_conflicts_do_not_choose_adjacent_or_first_donor(monkeypatch):
    first, last = family()
    alt = deepcopy(first)
    alt["conversations"][-1]["content"] = "<think>other reasoning</think>"
    samples, report = run(monkeypatch, [first, last, alt])
    assert not samples
    assert (
        report.dataset_config_transform_counts["reconstruction"][SOURCE][
            "ambiguous_donor_rows"
        ]
        == 1
    )


def test_first_question_is_not_family_proof(monkeypatch):
    first, last = family()
    last["tools"][0]["function"]["description"] = "Other complete environment"
    samples, _ = run(monkeypatch, [first, last])
    assert not samples


def test_toolmind_separator_differences_abstain(monkeypatch):
    first, last = family()
    last["conversations"][1]["content"] = " "
    samples, report = run(monkeypatch, [first, last])
    assert not samples
    assert (
        report.dataset_config_transform_counts["reconstruction"][SOURCE][
            "missing_donor_rows"
        ]
        == 1
    )
    assert (
        toolmind._project_message(
            {"role": "assistant", "content": "<think>r</think>\n\nA"}
        )["content"]
        == "\n\nA"
    )


def test_source_scopes_never_share_donors(monkeypatch):
    first, last = family()

    def lines(dataset, revision, files):
        return iter([json.dumps(first if files[0] == SOURCE else last)])

    monkeypatch.setattr(toolmind, "jsonl_lines", lines)
    samples, _ = toolmind.load(
        toolmind.ToolMindConfig(sources=[SOURCE, toolmind.SOURCES[1]]),
        FilterConfig(),
        curation_config=NO_CURATION,
    )
    assert not samples


@pytest.mark.parametrize(
    "content",
    ["<think>unclosed", "<think>a<think>b</think>", "<think>a<reasoning>b</think>"],
)
def test_inseparable_native_source_boundaries_quarantine(content):
    with pytest.raises(Reject, match="inseparable_source_reasoning"):
        toolmind._project_message({"role": "assistant", "content": content})


def test_literal_reasoning_like_prose_is_not_a_projection():
    msg = {"role": "assistant", "content": "Use `<think>` in this code example."}
    assert toolmind._project_message(msg) == msg


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(extra="unknown"),
        lambda row: row["conversations"][-1].update(reasoning_content="unknown"),
        lambda row: row["conversations"][-1].update(content="No native anchor"),
        lambda row: row["conversations"][0].update(role="unknown"),
    ],
)
def test_unknown_source_schema_is_explicit(mutation):
    raw = family()[0]
    mutation(raw)
    with pytest.raises(Reject):
        toolmind._prefix_record(raw, 0, SOURCE)


def test_complete_definition_preserves_constraints_and_values():
    raw = definition()
    raw["function"]["strict"] = True
    p = raw["function"]["parameters"]
    p.update(additionalProperties=False)
    p["properties"]["x"].update(
        type="int", minimum=1, default={"type": "dict", "nested": [1]}
    )
    normalized = toolmind._normalize_tool(raw)
    assert normalized["function"]["strict"] is True
    assert normalized["function"]["parameters"]["additionalProperties"] is False
    assert normalized["function"]["parameters"]["properties"]["x"] == {
        "type": "integer",
        "minimum": 1,
        "default": {"type": "dict", "nested": [1]},
    }
    assert raw["function"]["parameters"]["properties"]["x"]["type"] == "int"


def test_legacy_argument_map_does_not_infer_required_or_nullability():
    raw = {
        "type": "function",
        "function": {"name": "f", "arguments": {"x": {"type": "str", "default": "a"}}},
    }
    function = toolmind._normalize_tool(raw)["function"]
    assert function["parameters"] == {
        "type": "object",
        "properties": {"x": {"type": "string", "default": "a"}},
    }
    assert "description" not in function


def test_legacy_required_moves_without_erasing_conflicts():
    raw = definition()
    raw["function"]["required"] = ["x"]
    assert "required" not in toolmind._normalize_tool(raw)["function"]
    raw["function"]["required"] = []
    with pytest.raises(Reject, match="conflicting_legacy_required"):
        toolmind._normalize_tool(raw)


def test_null_required_placeholder_and_double_wrapper():
    raw = definition()
    raw["function"]["required"] = None
    nested = {"type": "function", "function": raw}
    assert toolmind._normalize_tool(nested) == definition()


def test_unknown_type_is_not_guessed_string():
    raw = definition()
    raw["function"]["parameters"]["properties"]["x"]["type"] = "MysteryObject"
    normalized = toolmind._normalize_tool(raw)
    assert (
        normalized["function"]["parameters"]["properties"]["x"]["type"]
        == "MysteryObject"
    )


def test_known_outer_graph_fields_are_preserved_without_repositioning():
    raw = definition()
    raw["input_description"] = "Graph inputs"
    assert toolmind._normalize_tool(raw) == raw
    assert "input_description" not in toolmind._normalize_tool(raw)["function"]


def test_null_schema_is_not_repaired():
    raw = family()[0]
    raw["tools"][0]["function"]["parameters"] = None
    state = RowState(toolmind._convert_row(raw, 0))
    validate_structure(state)
    with pytest.raises(Reject):
        validate_capabilities(state)
    assert state.sample.tools[0]["function"]["parameters"] is None


@pytest.mark.parametrize("arguments", [None, [], 3, "null"])
def test_null_and_nonobject_arguments_are_not_changed_to_empty_object(arguments):
    raw = family()[0]
    raw["conversations"][-1]["tool_calls"][0]["function"]["arguments"] = arguments
    with pytest.raises(Reject, match="non_object_arguments"):
        toolmind._convert_row(raw, 0)


@pytest.mark.parametrize("alter", ["missing", "orphan", "parallel"])
def test_call_result_protocol_failures(monkeypatch, alter):
    first, last = family()
    if alter == "missing":
        last["conversations"].pop(2)
    elif alter == "orphan":
        last["conversations"].insert(3, {"role": "tool", "content": "extra"})
    else:
        for raw in (first, last):
            raw["conversations"][1]["tool_calls"].append(
                {"function": {"name": "lookup", "arguments": {"x": 2}}}
            )
        last["conversations"].insert(3, {"role": "tool", "content": "second"})
    samples, report = run(monkeypatch, [first, last])
    assert not samples
    assert report.filter_drop_reasons


def test_visible_think_tool_survives_native_stripping(monkeypatch):
    rows = family()
    for raw in rows:
        raw["tools"][0]["function"]["name"] = "think"
        raw["conversations"][1]["tool_calls"][0]["function"]["name"] = "think"
    samples, _ = run(monkeypatch, rows, FilterConfig(strip_thinking=True))
    assert samples[0].tools[0]["function"]["name"] == "think"
    assert samples[0].messages[1]["tool_calls"][0]["function"]["name"] == "think"


@pytest.mark.parametrize(
    "field",
    ["strip_think_tool", "merge_split_assistant", "drop_consecutive_text_assistant"],
)
def test_unsafe_legacy_transforms_are_explicit_errors(field):
    config = toolmind.ToolMindConfig()
    setattr(config, field, True)
    with pytest.raises(ValueError, match="preserves visible"):
        toolmind.iter_load(config)


@pytest.mark.parametrize("strip", [True, False])
def test_reasoning_only_final_is_not_a_complete_answer(monkeypatch, strip):
    raw = {
        "tools": [],
        "conversations": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "<think>only reasoning</think>"},
        ],
    }
    samples, report = run(monkeypatch, [raw], FilterConfig(strip_thinking=strip))
    assert not samples
    assert report.filter_drop_reasons == {"empty_final_assistant": 1}


def test_reconciliation_retains_first_exact_complete_definition():
    raw = family()[0]
    raw["tools"].append(deepcopy(raw["tools"][0]))
    state = RowState(toolmind._convert_row(raw, 0))
    reconcile_definitions(state)
    assert len(state.sample.tools) == 1
    assert state.transforms["duplicate_tool_definitions_removed"] == 1


def test_default_removes_native_reasoning(monkeypatch):
    monkeypatch.setattr(
        toolmind, "jsonl_lines", lambda *a, **kw: (json.dumps(r) for r in family())
    )
    samples, _ = toolmind.load(
        toolmind.ToolMindConfig(sources=[SOURCE]), curation_config=NO_CURATION
    )
    assert samples[0].messages[-1]["content"] == "Done"


def test_function_response_and_graph_annotations_stay_at_source_positions():
    raw = definition()
    raw["input_description"] = "Input description"
    raw["output_description"] = "Output description"
    raw["functionality"] = {"explanation": "visible source annotation"}
    raw["output_structure"] = {"shape": [1, 2]}
    raw["function"]["response"] = {
        "type": "dict",
        "properties": {"found": {"type": "string"}},
    }
    converted = toolmind._normalize_tool(raw)
    assert converted == raw
    converted["output_structure"]["shape"].append(3)
    assert raw["output_structure"]["shape"] == [1, 2]


def test_stable_sample_id_under_subset_order(monkeypatch):
    def lines(dataset, revision, files):
        return (json.dumps(r) for r in family())

    monkeypatch.setattr(toolmind, "jsonl_lines", lines)
    first, second = toolmind.SOURCES[:2]
    ab, _ = toolmind.load(
        toolmind.ToolMindConfig(sources=[first, second]),
        FilterConfig(),
        curation_config=NO_CURATION,
    )
    ba, _ = toolmind.load(
        toolmind.ToolMindConfig(sources=[second, first]),
        FilterConfig(),
        curation_config=NO_CURATION,
    )
    assert [s.sample_id for s in ab] == [f"{first}:1", f"{second}:1"]
    assert [s.sample_id for s in ba] == [f"{second}:1", f"{first}:1"]
    assert all(s.raw == family()[1] for s in ab + ba)


def test_new_password_hash_does_not_require_existing_secret(monkeypatch):
    rows = family()
    for raw in rows:
        raw["tools"][0]["function"]["name"] = "generate_password_hash"
        raw["tools"][0]["function"]["parameters"] = {
            "type": "object",
            "properties": {"password": {"type": "string"}},
        }
        raw["conversations"][1]["tool_calls"][0]["function"] = {
            "name": "generate_password_hash",
            "arguments": {"password": "newly-generated-value"},
        }
    samples, _ = run(monkeypatch, rows)
    assert len(samples) == 1


def test_multiple_systems_absence_and_empty_content_are_preserved():
    raw = family()[0]
    raw["conversations"][0:0] = [
        {"role": "system"},
        {"role": "system", "content": ""},
        {"role": "system", "content": "Policy"},
    ]
    sample = toolmind._convert_row(raw, 0)
    assert sample.messages[:3] == raw["conversations"][:3]


def test_sequential_calls_and_context_only_assistant_remain_separate(monkeypatch):
    one, two = family()
    three = deepcopy(two)
    three["conversations"][-1]["content"] = "Done"
    three["conversations"].append(
        {"role": "assistant", "content": "<think>after context</think>All complete"}
    )
    samples, report = run(monkeypatch, [one, two, three])
    assert len(samples) == 1
    assert [m["role"] for m in samples[0].messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "assistant",
    ]
    assert samples[0].messages[-2]["content"] == "<think>native answer</think>Done"
    assert (
        report.dataset_config_transform_counts["reconstruction"][SOURCE][
            "removed_prefix_rows"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "reasoning",
    [
        "<think>Discuss `</think>` literally.</think>Answer",
        "<think>Example:\n```xml\n</think>\n```\n</think>Answer",
        "<think>Example:\n    </think>\n\n</think>Answer",
    ],
)
def test_native_projection_shields_literal_closing_tags(reasoning):
    projected = toolmind._project_message({"role": "assistant", "content": reasoning})
    assert projected["content"] == "Answer"


@pytest.mark.parametrize(
    "content",
    [
        "<think>r</think>Answer</think>",
        "<think>r</think><think>second</think>Answer",
        "<think>r</think>Answer<reasoning>unclosed",
    ],
)
def test_source_projection_rejects_ambiguous_extra_native_spans(content):
    with pytest.raises(Reject, match="inseparable_source_reasoning"):
        toolmind._project_message({"role": "assistant", "content": content})


def test_legacy_parameters_property_map_preserves_explicit_required():
    raw = {
        "type": "function",
        "function": {
            "name": "navigate_to_coordinates",
            "parameters": {
                "latitude": {"type": "number", "description": "Latitude"},
                "longitude": {"type": "number", "description": "Longitude"},
            },
            "required": ["latitude", "longitude"],
        },
    }
    normalized = toolmind._normalize_tool(raw)
    assert normalized["function"]["parameters"] == {
        "type": "object",
        "properties": raw["function"]["parameters"],
        "required": ["latitude", "longitude"],
    }


def test_legacy_parameters_property_map_does_not_infer_required():
    raw = {
        "type": "function",
        "function": {
            "name": "getcompanies",
            "parameters": {"page": {"type": "int", "default": "1"}},
        },
    }
    normalized = toolmind._normalize_tool(raw)
    assert normalized["function"]["parameters"] == {
        "type": "object",
        "properties": {"page": {"type": "integer", "default": "1"}},
    }


def test_real_schema_keywords_are_not_interpreted_as_flat_parameter_names():
    raw = {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {"additionalProperties": {"type": "string"}},
        },
    }
    assert toolmind._normalize_tool(raw) == raw


def test_flat_argument_names_type_properties_required_are_preserved():
    properties = {
        "type": {"type": "str"},
        "properties": {"type": "dict"},
        "required": {"type": "bool"},
    }
    raw = {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": properties,
            "required": ["type", "properties", "required"],
        },
    }
    params = toolmind._normalize_tool(raw)["function"]["parameters"]
    assert list(params["properties"]) == ["type", "properties", "required"]
    assert params["required"] == ["type", "properties", "required"]
    assert params["properties"] == {
        "type": {"type": "string"},
        "properties": {"type": "object"},
        "required": {"type": "boolean"},
    }


def test_schema_defaults_and_const_payloads_are_never_parameter_maps():
    for key in ("default", "const"):
        raw = {
            "type": "function",
            "function": {"name": "f", "parameters": {key: {"type": "str"}}},
        }
        assert toolmind._normalize_tool(raw) == raw


def test_flat_definition_name_collision_still_rejects():
    left: dict[str, Any] = {
        "type": "function",
        "function": {"name": "f", "parameters": {"x": {"type": "str"}}},
    }
    right = deepcopy(left)
    right["function"]["parameters"]["x"]["enum"] = ["restricted"]
    state = RowState(
        toolmind._convert_row(
            {
                "tools": [left, right],
                "conversations": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "<think>r</think>A"},
                ],
            },
            0,
        )
    )
    with pytest.raises(Reject, match="conflicting_duplicate_tool_names"):
        reconcile_definitions(state)


def test_flat_items_argument_is_proved_by_outer_required():
    raw = {
        "type": "function",
        "function": {
            "name": "count_items",
            "parameters": {
                "items": {"type": "array", "description": "List of items to count"}
            },
            "required": ["items"],
        },
    }
    params = toolmind._normalize_tool(raw)["function"]["parameters"]
    assert params == {
        "type": "object",
        "properties": raw["function"]["parameters"],
        "required": ["items"],
    }


@pytest.mark.parametrize(
    ("name", "field", "description", "existing"),
    [
        ("lookup", "apikey", "The API key for authentication", True),
        ("lookup", "authorization", "Authorization token for the API request", True),
        ("verify_user", "password", "The password", True),
        ("create_user", "password", "The password", False),
        ("login", "password", "The password of the user", True),
        ("create_account", "password", "The password of the user", False),
        ("deleteDatabase", "password", "The password to access the database", True),
        ("createDatabase", "password", "The password to access the database", False),
        ("Linkedin Contacts", "key", "Use this key for testing.", True),
        ("unrelated", "key", "Use this key for testing.", False),
        ("Send Custom Voice OTP", "otp", "Custom 4-digit OTP code to be sent", False),
    ],
)
def test_finite_credential_contract_distinguishes_consumption(
    name, field, description, existing
):
    from fcanalysis.loaders._toolmind_credentials import credential_requires_evidence

    function = {
        "parameters": {
            "type": "object",
            "properties": {field: {"description": description}},
        }
    }
    assert credential_requires_evidence(name, field, function, "new-value") is existing


def test_unrelated_default_is_not_a_credential_but_released_public_constant_is():
    from fcanalysis.loaders._toolmind_credentials import credential_requires_evidence

    field = "apikey"
    function = {
        "parameters": {
            "properties": {
                field: {
                    "description": "API key for authentication. Defaults to 'demo'.",
                    "default": "arbitrary-example",
                }
            }
        }
    }
    assert not credential_requires_evidence("mfs_list", field, function, "demo")
    assert credential_requires_evidence(
        "mfs_list", field, function, "arbitrary-example"
    )


@pytest.mark.parametrize("strip", [True, False])
@pytest.mark.parametrize(
    "evidence", ["user", "system", "removed_system", "native", "future", "none"]
)
def test_notification_token_uses_visible_evidence_after_reconstruction(
    monkeypatch, strip, evidence
):
    rows = family()
    for row in rows:
        function = row["tools"][0]["function"]
        function["name"] = "Receive Notification"
        function["parameters"] = {
            "type": "object",
            "properties": {
                "apiTokenInstance": {
                    "type": "string",
                    "description": "The API token used to authenticate the request.",
                }
            },
            "required": ["apiTokenInstance"],
        }
        call = row["conversations"][1]["tool_calls"][0]["function"]
        call.update(
            name="Receive Notification", arguments={"apiTokenInstance": "secret"}
        )
        if evidence == "user":
            row["conversations"][0]["content"] = "Use the API token secret."
    if evidence == "native":
        rows[0]["conversations"][-1]["content"] = "<think>Use secret.</think>"
    if evidence == "future":
        rows[-1]["conversations"][2]["content"] = {"token": "secret"}
    if evidence in {"system", "removed_system"}:
        for row in rows:
            row["conversations"].insert(
                0, {"role": "system", "content": "The configured API token is secret."}
            )
    saved = deepcopy(rows)
    samples, report = run(
        monkeypatch,
        rows,
        FilterConfig(
            strip_thinking=strip,
            system_message_override="" if evidence == "removed_system" else None,
        ),
    )
    assert rows == saved
    if evidence in {"user", "system"}:
        assert len(samples) == 1
        assert samples[0].raw == saved[-1]
    else:
        assert not samples
        assert report.filter_drop_reasons["ungrounded_credential_argument"] == 1


@pytest.mark.parametrize("strip", [True, False])
@pytest.mark.parametrize(
    ("substituted", "same_contract", "provided"),
    [
        (False, True, False),
        (False, True, True),
        (True, True, True),
        (False, False, False),
    ],
)
def test_http_template_is_checked_without_rewriting_other_urls(
    monkeypatch, strip, substituted, same_contract, provided
):
    template = "https://api.example.com/v1/users/create?apiKey={YOUR-API-KEY}"
    description = (
        "For submitting information to create a new user in a database, modify this "
        "template URL with your actual API key where indicated: " + template
    )
    value = template.replace("{YOUR-API-KEY}", "secret") if substituted else template
    rows = family()
    for row in rows:
        function = row["tools"][0]["function"]
        function["name"] = "http.post"
        function["parameters"] = {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": description
                    if same_contract
                    else "Store literal URL text",
                    "default": template,
                }
            },
            "required": ["url"],
        }
        row["conversations"][1]["tool_calls"][0]["function"].update(
            name="http.post", arguments={"url": value}
        )
        if provided:
            row["conversations"][0]["content"] = (
                "Create the account using API key secret."
            )
    saved = deepcopy(rows)
    samples, report = run(monkeypatch, rows, FilterConfig(strip_thinking=strip))
    assert rows == saved
    if same_contract and not substituted:
        assert not samples
        assert report.filter_drop_reasons["unresolved_source_api_key_placeholder"] == 1
    else:
        assert len(samples) == 1
        assert samples[0].raw == saved[-1]
        arguments = json.loads(
            samples[0].messages[1]["tool_calls"][0]["function"]["arguments"]
        )
        assert arguments["url"] == value


@pytest.mark.parametrize(
    "evidence", ["user", "tool_value", "assistant", "tool_key", "future"]
)
def test_actual_credential_gate_uses_only_prior_eligible_channels(
    monkeypatch, evidence
):
    rows = family()
    for raw in rows:
        fn = raw["tools"][0]["function"]
        fn["parameters"] = {
            "type": "object",
            "properties": {
                "apikey": {
                    "type": "string",
                    "description": "The API key for authentication",
                }
            },
        }
        raw["conversations"][1]["tool_calls"][0]["function"]["arguments"] = {
            "apikey": "exact-secret"
        }
        raw["conversations"][0]["content"] = (
            "Use exact-secret" if evidence == "user" else "Look up an item"
        )
        if evidence == "assistant":
            # Assistant reasoning is never evidence, including exact reconstruction.
            raw["conversations"][1]["content"] = raw["conversations"][1][
                "content"
            ].replace("native call", "exact-secret")
        if evidence in {"tool_value", "tool_key"}:
            raw["conversations"].insert(
                0,
                {
                    "role": "tool",
                    "content": {"key": "exact-secret"}
                    if evidence == "tool_value"
                    else {"exact-secret": None},
                },
            )
    if evidence in {"tool_value", "tool_key"}:
        # Isolate evidence policy: an orphan result is mechanically rejected by
        # the full loader, tested elsewhere. This view represents a prior result.
        sample = toolmind._convert_row(rows[-1], 1)
        state = RowState(sample)
        state.parsed_arguments[2, 0] = {"apikey": "exact-secret"}
        if evidence == "tool_value":
            toolmind._validate_visible_context(state)
        else:
            with pytest.raises(Reject, match="ungrounded_credential_argument"):
                toolmind._validate_visible_context(state)
        return
    if evidence == "future":
        rows[-1]["conversations"][2]["content"] = {"key": "exact-secret"}
    samples, report = run(monkeypatch, rows)
    if evidence == "user":
        assert len(samples) == 1
        assert samples[0].raw == rows[-1]
    else:
        assert not samples
        assert report.filter_drop_reasons["ungrounded_credential_argument"] == 1


def test_bundle_conversion_preserves_complete_entries_and_raw():
    from fcanalysis.loaders._toolmind_results import BUTTON, TOOLACE

    for source_file, entry in [
        (BUTTON, {"name": "lookup", "arguments": {"x": 1}, "results": {"found": [1]}}),
        (TOOLACE, {"name": "lookup", "results": {"found": [1]}}),
    ]:
        raw = family()[-1]
        raw["conversations"][2]["content"] = json.dumps([entry])
        saved = deepcopy(raw)
        sample = toolmind._convert_row(raw, 1, source_file=source_file)
        assert json.loads(sample.messages[2]["content"]) == entry
        assert sample.raw == saved == raw
        assert sample.messages[2].keys() == {"role", "content"}
    ordinary = toolmind._convert_row(raw, 1, source_file=SOURCE)
    assert json.loads(ordinary.messages[2]["content"]) == [entry]


@pytest.mark.parametrize(
    "value",
    [[], {}, [{"name": "lookup", "results": 1, "extra": "keep"}], [{"name": "lookup"}]],
)
def test_bundle_conversion_rejects_unreleased_shape(value):
    from fcanalysis.loaders._toolmind_results import TOOLACE, expand_result

    with pytest.raises(Reject, match="invalid_source_result_bundle"):
        expand_result(value, source_file=TOOLACE)


def test_result_projection_excludes_echoed_arguments_and_names():
    from fcanalysis.loaders._toolmind_results import BUTTON, result_values

    entry = {
        "name": "not-state",
        "arguments": {"key": "invented"},
        "results": {"code": 123456},
    }
    assert list(result_values(entry, source_file=BUTTON)) == ["123456"]
    assert list(result_values(entry, source_file=SOURCE)) == [
        "not-state",
        "invented",
        "123456",
    ]


def bundle_family(*, arguments=True, repeated_name=False):
    rows = [deepcopy(raw) for raw in family()]
    for raw in rows:
        calls = raw["conversations"][1]["tool_calls"]
        second = deepcopy(calls[0])
        second["function"]["arguments"] = {"x": 2}
        if not repeated_name:
            second["function"]["name"] = "other"
            raw["tools"].append(definition("other"))
        calls.append(second)
    entries = []
    for call in rows[-1]["conversations"][1]["tool_calls"]:
        fn = call["function"]
        entry = {"name": fn["name"], "results": {"found": fn["arguments"]["x"]}}
        if arguments:
            entry["arguments"] = deepcopy(fn["arguments"])
        entries.append(entry)
    rows[-1]["conversations"][2]["content"] = json.dumps(entries)
    return rows


@pytest.mark.parametrize("strip", [False, True])
@pytest.mark.parametrize("arguments", [False, True])
def test_exact_bundle_linkage_survives_reconstruction_and_final_recheck(
    monkeypatch, strip, arguments
):
    from fcanalysis.loaders._toolmind_results import BUTTON, TOOLACE

    source_file = BUTTON if arguments else TOOLACE
    rows = bundle_family(arguments=arguments, repeated_name=arguments)
    samples, report = run(
        monkeypatch, rows, FilterConfig(strip_thinking=strip), [source_file]
    )
    assert len(samples) == 1
    assert samples[0].raw == rows[-1]
    assert [m["role"] for m in samples[0].messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert [json.loads(m["content"])["results"] for m in samples[0].messages[2:4]] == [
        {"found": 1},
        {"found": 2},
    ]
    assert (
        report.dataset_config_transform_counts["transformations"][
            "source_result_bundles_expanded"
        ]
        == 1
    )
    assert (
        report.dataset_config_transform_counts["pipeline_passed"]["final_linkage"] == 1
    )


@pytest.mark.parametrize(
    "issue",
    [
        "argument_conflict",
        "duplicate_echo",
        "reorder",
        "name_conflict",
        "boolean_conflict",
    ],
)
@pytest.mark.parametrize("align_results", [False, True])
def test_bundle_alignment_never_relaxes_correspondence(
    monkeypatch, issue, align_results
):
    from fcanalysis.loaders._toolmind_results import BUTTON

    rows = bundle_family(arguments=True, repeated_name=True)
    entries = json.loads(rows[-1]["conversations"][2]["content"])
    if issue == "argument_conflict":
        entries[0]["arguments"]["x"] = 99
    elif issue == "boolean_conflict":
        entries[0]["arguments"]["x"] = True
    elif issue == "duplicate_echo":
        entries[1] = deepcopy(entries[0])
    elif issue == "reorder":
        entries.reverse()
    else:
        entries[0]["name"] = "unrelated"
    rows[-1]["conversations"][2]["content"] = json.dumps(entries)
    samples, report = run(
        monkeypatch, rows, FilterConfig(align_results=align_results), sources=[BUTTON]
    )
    if issue == "reorder" and align_results:
        assert len(samples) == 1
        assert samples[0].raw == rows[-1]
        assert [json.loads(m["content"]) for m in samples[0].messages[2:4]] == list(
            reversed(entries)
        )
        assert (
            report.dataset_config_transform_counts["pipeline_passed"]["final_linkage"]
            == 1
        )
        return
    assert not samples
    reason = (
        "unproven_result_reordering"
        if issue == "reorder"
        else "invalid_source_tool_linkage"
    )
    assert report.filter_drop_reasons[reason] == 1


def test_name_only_bundle_cannot_disambiguate_same_name_calls(monkeypatch):
    from fcanalysis.loaders._toolmind_results import TOOLACE

    rows = bundle_family(arguments=False, repeated_name=True)
    samples, report = run(monkeypatch, rows, sources=[TOOLACE])
    assert not samples
    assert report.filter_drop_reasons["ambiguous_source_tool_linkage"] == 1


def test_bundle_grammar_does_not_apply_to_other_toolmind_sources(monkeypatch):
    samples, report = run(monkeypatch, bundle_family(), sources=[SOURCE])
    assert not samples
    assert report.filter_drop_reasons["unbalanced_cardinality"] == 2


@pytest.mark.parametrize("in_results", [False, True])
def test_bundle_echoed_arguments_do_not_supply_external_credentials(in_results):
    from fcanalysis.format import ConversationSample
    from fcanalysis.loaders._toolmind_results import BUTTON

    entry = {
        "name": "echo",
        "arguments": {"text": "invented-secret"},
        "results": {"value": "invented-secret" if in_results else "ok"},
    }
    state = RowState(
        ConversationSample(
            dataset=toolmind.DATASET_ID,
            sample_id=0,
            raw={},
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "consumer",
                        "parameters": {
                            "properties": {
                                "apikey": {
                                    "description": "The API key for authentication"
                                }
                            }
                        },
                    },
                }
            ],
            messages=[
                {"role": "tool", "content": json.dumps(entry)},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "consumer",
                                "arguments": '{"apikey":"invented-secret"}',
                            }
                        }
                    ],
                },
            ],
        )
    )
    state.parsed_arguments[1, 0] = {"apikey": "invented-secret"}
    if in_results:
        toolmind._validate_visible_context(state, source_file=BUTTON)
    else:
        with pytest.raises(Reject, match="ungrounded_credential_argument"):
            toolmind._validate_visible_context(state, source_file=BUTTON)


def calendar_family(user="What is the weather tomorrow at 3 PM?", *, system=None):
    rows = family()
    for raw in rows:
        function = raw["tools"][0]["function"]
        function.update(
            name="weather.fetchCurrentTemperature",
            description="Retrieves the current temperature for a specified location and time.",
            parameters={
                "type": "object",
                "properties": {
                    "dateTime": {
                        "type": "string",
                        "description": "Example: 2023-10-05 14:00",
                    }
                },
                "required": ["dateTime"],
            },
        )
        raw["conversations"][0]["content"] = user
        call = raw["conversations"][1]["tool_calls"][0]["function"]
        call.update(name=function["name"], arguments={"dateTime": "2023-10-06 15:00"})
        if system is not None:
            raw["conversations"].insert(0, {"role": "system", "content": system})
    return rows


@pytest.mark.parametrize("strip", [True, False])
def test_calendar_missing_clock_excludes_schema_and_native_examples(monkeypatch, strip):
    rows = calendar_family()
    rows[0]["conversations"][-1]["content"] = (
        "<think>Today is 2023-10-05, so tomorrow is October 6.</think>"
    )
    before = deepcopy(rows)
    samples, report = run(monkeypatch, rows, FilterConfig(strip_thinking=strip))
    assert not samples
    assert report.filter_drop_reasons["ungrounded_temporal_reference"] == 1
    assert rows == before


@pytest.mark.parametrize(
    ("user", "system"),
    [
        ("Weather tomorrow at 3 PM?", "Current date: 2023-10-05"),
        ("Weather tomorrow, October 6, 2023, at 3 PM?", None),
        ("Weather next week on October 6?", None),
        ("Weather next week, October 6–8, 2023?", None),
        ("Weather tomorrow (06/10/2023)?", None),
    ],
)
def test_calendar_explicit_clock_and_unsupported_dates_preserved(
    monkeypatch, user, system
):
    rows = calendar_family(user, system=system)
    samples, _ = run(monkeypatch, rows)
    assert len(samples) == 1
    assert samples[0].raw == rows[-1]


def test_calendar_gate_rechecks_after_system_override(monkeypatch):
    rows = calendar_family(system="Current date: 2023-10-05")
    samples, report = run(
        monkeypatch, rows, FilterConfig(system_message_override="You are helpful.")
    )
    assert not samples
    assert report.filter_drop_reasons["ungrounded_temporal_reference"] == 1


def test_calendar_exact_source_and_definition_contract(monkeypatch):
    rows = calendar_family()
    samples, _ = run(monkeypatch, rows, sources=[toolmind.SOURCES[-1]])
    assert len(samples) == 1
    for raw in rows:
        raw["tools"][0]["function"]["description"] = (
            "Generate fictional weather records."
        )
    samples, _ = run(monkeypatch, rows)
    assert len(samples) == 1


def test_calendar_month_query_preserves_explicit_month_and_year(monkeypatch):
    rows = calendar_family("Find cheap flights next month, November 2023.")
    name = "Calendar of Prices for a Month"
    for raw in rows:
        function = raw["tools"][0]["function"]
        function.update(
            name=name,
            description="Returns the prices for each day of a month, grouped together by the number of transfers, for a given origin and destination.",
            parameters={
                "type": "object",
                "properties": {
                    "month": {
                        "type": "string",
                        "description": "The beginning of the month in the YYYY-MM-DD format",
                    },
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                },
                "required": ["origin", "destination"],
            },
        )
        raw["conversations"][1]["tool_calls"][0]["function"].update(
            name=name,
            arguments={"month": "2023-11-01", "origin": "LED", "destination": "MOW"},
        )
    samples, _ = run(monkeypatch, rows)
    assert len(samples) == 1
    assert samples[0].raw == rows[-1]
