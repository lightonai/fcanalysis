"""Independently project retained ToolMind-Web rows from a second source pass.

This checks every retained source field in the canonical view, source-row
isolation, chronology and exact text preservation in both reasoning modes.
It deliberately does not reuse the adapter's call/template parsers. Strict
JSON serialization and native-reasoning boundaries remain shared primitives;
this is not an independent reimplementation of rejection or deduplication.
"""

import json

import pytest

from fcanalysis.loaders import toolmind_web
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.normalization import json_bytes, parse_json
from fcanalysis.loaders.pipeline import native_reasoning_spans
from fcanalysis.loaders.source import jsonl_lines


def _outside_reasoning(content: str) -> str:
    parts = []
    end = 0
    for start, stop in native_reasoning_spans(content):
        parts.append(content[end:start])
        end = stop
    parts.append(content[end:])
    return "".join(parts)


def _source_tool(system: str) -> tuple[str, dict]:
    marker = "# Tool-Use Formatting Instructions \n"
    objective = "# General Objective\n"
    prefix, _, rest = system.partition(marker)
    section, _, suffix = rest.partition(objective)
    catalog_marker = "Here are the functions available in JSONSchema format:\n\n\n"
    framing, _, catalog = section.partition(catalog_marker)
    semantic_start = framing.index("The Model Context Protocol (MCP) connects")
    description = framing[semantic_start:].split("\n\nParameters:", 1)[0]
    parameter_text = framing.split("\n\nParameters:\n", 1)[1].split("\n\nUsage:", 1)[0]
    properties = {}
    for line in parameter_text.splitlines():
        key, description_line = line.removeprefix("- ").split(": (required) ", 1)
        properties[key] = {
            "type": "object" if key == "arguments" else "string",
            "description": description_line,
        }
    assert list(properties) == ["server_name", "tool_name", "arguments"]
    tool = {
        "type": "function",
        "function": {
            "name": "use_mcp_tool",
            "description": description + "\n\n" + catalog_marker + catalog,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
        },
    }
    return prefix + objective + suffix, tool


def _project(raw: dict, sample, *, strip: bool) -> None:
    source = raw["conversations"]
    system, tool = _source_tool(source[0]["content"])
    assert sample.tools == [tool]
    assert sample.dataset == toolmind_web.DATASET_ID
    assert sample.sample_id == raw["id"]
    assert not sample.annotations
    assert sample.raw == raw
    assert len(sample.messages) == len(source)
    pending = False
    for index, (original, actual) in enumerate(
        zip(source, sample.messages, strict=True)
    ):
        role, content = original["role"], original["content"]
        expected = {"role": role, "content": content}
        if index == 0:
            expected["content"] = system
        elif pending:
            assert role == "user"
            expected["role"] = "tool"
            pending = False
        elif role == "assistant":
            if actual.get("tool_calls"):
                text, opener, call_text = content.rpartition("<use_mcp_tool>")
                assert opener
                outer = {}
                for key in ("server_name", "tool_name"):
                    _, start, remainder = call_text.partition(f"<{key}>")
                    value, stop, call_text = remainder.partition(f"</{key}>")
                    assert start and stop
                    outer[key] = value
                _, start, arguments_text = call_text.partition("<arguments>")
                assert start
                arguments_text = arguments_text.lstrip()
                _, end = json.JSONDecoder().raw_decode(arguments_text)
                outer["arguments"] = parse_json(arguments_text[:end])
                assert arguments_text[end:].strip().startswith("</arguments>")
                assert (
                    arguments_text[end:].strip().removeprefix("</arguments>").strip()
                    == "</use_mcp_tool>"
                )
                expected["content"] = text
                expected["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": "use_mcp_tool",
                            "arguments": json_bytes(outer).decode(),
                        },
                    }
                ]
                pending = True
            if strip:
                expected["content"] = _outside_reasoning(expected["content"])
        assert actual == expected, (raw["id"], index)
    assert not pending


@pytest.mark.e2e
@pytest.mark.parametrize("strip", [False, True], ids=["keep", "strip"])
def test_full_retained_source_projection(strip: bool) -> None:
    samples, report = toolmind_web.iter_load(
        filter_config=FilterConfig(strip_thinking=strip)
    )
    lines = jsonl_lines(
        toolmind_web.DATASET_ID, toolmind_web.DATASET_REVISION, [toolmind_web.SOURCE]
    )
    physical = retained = 0
    try:
        for sample in samples:
            for line in lines:
                physical += 1
                raw = parse_json(line)
                if raw["id"] == sample.sample_id:
                    break
            else:
                pytest.fail(
                    "Retained sample was missing or moved backwards in the source"
                )
            _project(raw, sample, strip=strip)
            retained += 1
        physical += sum(1 for _ in lines)
    finally:
        samples.close()
        close = getattr(lines, "close", None)
        if close is not None:
            close()
    assert physical == report.raw_count == 5624
    assert retained == report.final_count
    assert retained > 0
