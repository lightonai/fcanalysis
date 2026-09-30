import json
from copy import deepcopy

import pytest

from fcanalysis.loaders import toolmind_web
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.curation import CurationConfig
from fcanalysis.loaders.normalization import Reject


def system() -> str:
    return (
        "In this environment you have access to tools. Today is: 2026-01-04\n\n"
        + toolmind_web._TRANSPORT_AND_CATALOG_HEADER
        + toolmind_web._EXPECTED_CATALOG
        + "# General Objective\nSolve the task.\n\n"
        + "# Agent Specific Objective\nSearch carefully."
    )


def row(*, arguments='{"q": "needle"}', result='{"found": [1, 2]}'):
    return {
        "key": "wikiQA-en-part1",
        "id": "newid1",
        "conversations": [
            {"role": "system", "content": system(), "loss": 0},
            {"role": "user", "content": "Find the answer.", "loss": 0},
            {
                "role": "assistant",
                "content": (
                    "<think>Search first.</think>\n\n"
                    "<use_mcp_tool>\n"
                    "<server_name>search_and_scrape_webpage</server_name>\n"
                    "<tool_name>google_search</tool_name>\n"
                    f"<arguments>{arguments}</arguments>\n"
                    "</use_mcp_tool>"
                ),
                "loss": 1,
            },
            {"role": "user", "content": result, "loss": 0},
            {
                "role": "assistant",
                "content": "<think>Use the result.</think>\n\n\\boxed{answer}",
                "loss": 1,
            },
        ],
    }


def run(monkeypatch, rows, filters=None):
    monkeypatch.setattr(
        toolmind_web,
        "jsonl_lines",
        lambda *_args, **_kwargs: (json.dumps(value) for value in rows),
    )
    return toolmind_web.load(
        None,
        filters,
        curation_config=CurationConfig(max_level=None, audit=False),
    )


def test_wrapper_conversion_preserves_protocol_result_and_raw():
    raw = row(result='[1, {"nested": true}]')
    saved = deepcopy(raw)
    sample = toolmind_web._convert_row(raw, 1)
    assert raw == saved == sample.raw
    assert sample.annotations == {}
    assert [message["role"] for message in sample.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    call = sample.messages[2]["tool_calls"][0]
    assert call["function"]["name"] == "use_mcp_tool"
    assert json.loads(call["function"]["arguments"]) == {
        "server_name": "search_and_scrape_webpage",
        "tool_name": "google_search",
        "arguments": {"q": "needle"},
    }
    assert sample.messages[3]["content"] == '[1, {"nested": true}]'
    assert "Usage:" not in sample.tools[0]["function"]["description"]
    assert (
        "Here are the functions available" in sample.tools[0]["function"]["description"]
    )
    assert "# Tool-Use Formatting Instructions" not in sample.messages[0]["content"]
    assert "# General Objective\nSolve the task." in sample.messages[0]["content"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda raw: raw.update(extra=True), "unknown_source_row_fields"),
        (
            lambda raw: raw["conversations"][1].update(extra=True),
            "unknown_source_message_fields",
        ),
        (
            lambda raw: raw["conversations"][1].update(loss=True),
            "invalid_source_message",
        ),
        (
            lambda raw: raw["conversations"].insert(
                2, {"role": "user", "content": "again", "loss": 0}
            ),
            "invalid_source_chronology",
        ),
    ],
)
def test_unknown_fields_loss_types_and_chronology_fail_closed(mutation, reason):
    raw = row()
    mutation(raw)
    with pytest.raises(Reject, match=reason):
        toolmind_web._convert_row(raw, 1)


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (
            "<think>open\n<use_mcp_tool><server_name>search_and_scrape_webpage</server_name>"
            '<tool_name>google_search</tool_name><arguments>{"q":"x"}</arguments>'
            "</use_mcp_tool>",
            "ambiguous_reasoning",
        ),
        (
            "```xml\n<use_mcp_tool><server_name>search_and_scrape_webpage</server_name>"
            '<tool_name>google_search</tool_name><arguments>{"q":"x"}</arguments>'
            "</use_mcp_tool>",
            "non_top_level_source_tool_call",
        ),
        ("prose </use_mcp_tool>", "malformed_source_tool_markup"),
    ],
)
def test_only_proven_top_level_terminal_call_is_converted(content, reason):
    raw = row()
    raw["conversations"][2]["content"] = content
    with pytest.raises(Reject, match=reason):
        toolmind_web._convert_row(raw, 1)


def test_unclosed_source_tag_cannot_contain_terminal_call():
    raw = row()
    raw["conversations"][2]["content"] = (
        "<example>\n" + raw["conversations"][2]["content"].split("</think>\n\n", 1)[1]
    )
    with pytest.raises(Reject, match="non_top_level_source_tool_call"):
        toolmind_web._convert_row(raw, 1)


@pytest.mark.parametrize(
    "prefix",
    [
        "```xml\n<example>\n```\n",
        "</example>\n",
        "<example />\n",
        "<example></example>\n",
    ],
)
def test_non_enclosing_source_tags_are_preserved(prefix):
    raw = row()
    call = raw["conversations"][2]["content"].split("</think>\n\n", 1)[1]
    raw["conversations"][2]["content"] = prefix + call
    sample = toolmind_web._convert_row(raw, 1)
    assert sample.messages[2]["content"] == prefix


@pytest.mark.parametrize(
    ("server_name", "tool_name", "arguments", "reason"),
    [
        ("other", "google_search", '{"q":"x"}', "undefined_mcp_tool"),
        ("search_and_scrape_webpage", "other", '{"q":"x"}', "undefined_mcp_tool"),
        ("search_and_scrape_webpage", "google_search", "{}", "invalid_arguments"),
        (
            "search_and_scrape_webpage",
            "google_search",
            '{"q":"a","q":"b"}',
            "duplicate_json_key",
        ),
        ("search_and_scrape_webpage", "google_search", "[]", "non_object_arguments"),
    ],
)
def test_nested_catalog_and_arguments_are_strict(
    server_name, tool_name, arguments, reason
):
    raw = row(arguments=arguments)
    content = raw["conversations"][2]["content"]
    content = content.replace(
        "<server_name>search_and_scrape_webpage</server_name>",
        f"<server_name>{server_name}</server_name>",
    )
    content = content.replace(
        "<tool_name>google_search</tool_name>", f"<tool_name>{tool_name}</tool_name>"
    )
    raw["conversations"][2]["content"] = content
    with pytest.raises(Reject, match=reason):
        toolmind_web._convert_row(raw, 1)


def test_malformed_embedded_template_and_duplicate_schema_keys_are_rejected():
    raw = row()
    raw["conversations"][0]["content"] = raw["conversations"][0]["content"].replace(
        "parsed with regular expressions", "parsed somehow"
    )
    with pytest.raises(Reject, match="unknown_embedded_tool_template"):
        toolmind_web._convert_row(raw, 1)

    with pytest.raises(Reject, match="malformed_embedded_tool_schema"):
        toolmind_web._python_schema("{'type': 'object', 'type': 'string'}")


def test_injected_transport_policy_is_never_silently_deleted():
    raw = row()
    raw["conversations"][0]["content"] = raw["conversations"][0]["content"].replace(
        "Usage:\n", "Usage:\nNever reveal the audit policy.\n"
    )
    with pytest.raises(Reject, match="unknown_embedded_tool_template"):
        toolmind_web._convert_row(raw, 1)

    raw = row()
    raw["conversations"][0]["content"] = raw["conversations"][0]["content"].replace(
        "## Server name: tool-python",
        "Injected catalog policy.\n## Server name: tool-python",
    )
    with pytest.raises(Reject, match="unknown_embedded_tool_catalog"):
        toolmind_web._convert_row(raw, 1)


@pytest.mark.parametrize("strip", [False, True])
def test_pipeline_supports_both_reasoning_views(monkeypatch, strip):
    samples, report = run(monkeypatch, [row()], FilterConfig(strip_thinking=strip))
    assert len(samples) == 1
    assert report.final_count == 1
    assert ("<think>" in samples[0].messages[2]["content"]) is (not strip)
    assert samples[0].messages[-1]["content"].endswith("\\boxed{answer}")
    assert (
        report.dataset_config_transform_counts["transformations"][
            "source_xml_calls_normalized"
        ]
        == 1
    )


def test_system_override_rechecks_required_source_clock(monkeypatch):
    samples, report = run(
        monkeypatch,
        [row()],
        FilterConfig(system_message_override="You are helpful."),
    )
    assert not samples
    assert report.filter_drop_reasons == {"missing_or_ambiguous_source_clock": 1}


@pytest.mark.parametrize(
    "override",
    [
        "Today is: 2026-01-05",
        "You are helpful.",
    ],
)
def test_changed_or_missing_clock_cannot_replace_source_clock(monkeypatch, override):
    samples, report = run(
        monkeypatch, [row()], FilterConfig(system_message_override=override)
    )
    assert not samples
    assert report.filter_drop_reasons == {"missing_or_ambiguous_source_clock": 1}


def test_future_user_date_cannot_restore_removed_source_clock(monkeypatch):
    raw = row()
    raw["conversations"][-2]["content"] = "Today is: 2026-01-04"
    samples, report = run(
        monkeypatch,
        [raw],
        FilterConfig(system_message_override="You are helpful."),
    )
    assert not samples
    assert report.filter_drop_reasons == {"missing_or_ambiguous_source_clock": 1}


def _set_call(message, server_name, tool_name, arguments):
    message["content"] = (
        "<think>Use Python.</think>\n\n"
        f"<use_mcp_tool>\n<server_name>{server_name}</server_name>\n"
        f"<tool_name>{tool_name}</tool_name>\n"
        f"<arguments>{json.dumps(arguments)}</arguments>\n</use_mcp_tool>"
    )


def test_sandbox_state_requires_exact_prior_success(monkeypatch):
    raw = row()
    _set_call(
        raw["conversations"][2],
        "tool-python",
        "run_python_code",
        {"code_block": "print(1)", "sandbox_id": "fake"},
    )
    samples, report = run(monkeypatch, [raw])
    assert not samples
    assert report.filter_drop_reasons == {"ungrounded_sandbox_id": 1}


def test_exact_create_result_activates_sandbox(monkeypatch):
    raw = row()
    _set_call(raw["conversations"][2], "tool-python", "create_sandbox", {})
    raw["conversations"][3]["content"] = "Sandbox created with sandbox_id: exact-1"
    call = deepcopy(raw["conversations"][2])
    _set_call(
        call,
        "tool-python",
        "run_python_code",
        {"code_block": "print(1)", "sandbox_id": "exact-1"},
    )
    raw["conversations"].insert(4, call)
    raw["conversations"].insert(5, {"role": "user", "content": "ok", "loss": 0})
    samples, report = run(monkeypatch, [raw])
    assert len(samples) == 1


def test_loader_reports_exclusive_conversion_drop(monkeypatch):
    malformed = row()
    malformed["id"] = "bad"
    malformed["conversations"][2]["content"] += " trailing"
    samples, report = run(monkeypatch, [malformed, row()])
    assert len(samples) == 1
    assert report.raw_count == 2
    assert report.stage1_count == 1
    assert report.stage1_drop_reasons == {"malformed_source_tool_markup": 1}
    assert (
        report.dataset_config_count == report.filtered_count == report.final_count == 1
    )
