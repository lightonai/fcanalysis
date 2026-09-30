"""Second-source-pass projection of every retained UltraData canonical field.

Does not call the adapter's converter or legacy schema conversion functions.
Strict JSON bytes and native tag boundaries remain shared primitives. This
proves preservation for retained rows, not the correctness of every exclusion,
producer history, semantic label or deduplication choice.
"""

from copy import deepcopy
from dataclasses import asdict

import pytest

from fcanalysis.loaders import ultradata_tool_use as loader
from fcanalysis.loaders.normalization import json_bytes, parse_json
from fcanalysis.loaders.pipeline import native_reasoning_spans
from fcanalysis.loaders.source import jsonl_lines
from tests.matrix import ULTRADATA_SPECS
from tests.tools.fixture_io import read_hash, read_report
from tests.tools.hash_jsonl import hash_samples


def outside_native(text):
    spans = native_reasoning_spans(text)
    out = []
    start = 0
    for a, b in spans:
        out.append(text[start:a])
        start = b
    out.append(text[start:])
    return "".join(out)


def project_schema(schema):
    if not isinstance(schema, dict):
        return schema
    result = deepcopy(schema)
    aliases = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
    }
    if isinstance(result.get("type"), str):
        result["type"] = aliases.get(result["type"], result["type"])
    for key, value in result.items():
        if key in (
            "properties",
            "patternProperties",
            "$defs",
            "definitions",
            "dependentSchemas",
            "dependencies",
        ) and isinstance(value, dict):
            result[key] = {name: project_schema(child) for name, child in value.items()}
        elif key in ("allOf", "anyOf", "oneOf", "prefixItems") and isinstance(
            value, list
        ):
            result[key] = [project_schema(child) for child in value]
        elif key in (
            "items",
            "additionalItems",
            "additionalProperties",
            "contains",
            "not",
            "if",
            "then",
            "else",
            "propertyNames",
            "unevaluatedItems",
            "unevaluatedProperties",
        ):
            result[key] = (
                [project_schema(child) for child in value]
                if isinstance(value, list)
                else project_schema(value)
            )
    return result


def project_tools(raw):
    projected = []
    for tool in deepcopy(raw["tools"]):
        if raw["source"] in (
            "toolmind_graphsyn",
            "synth_memory_20260608",
            "synth_search_20260609",
        ):
            f = tool["function"]
            if set(f) == {"type", "function"}:
                assert f["type"] == "function"
                tool["function"] = f = f["function"]
            if "arguments" in f:
                assert "parameters" not in f
                f["parameters"] = {"type": "object", "properties": f.pop("arguments")}
            p = f.get("parameters")
            if (
                isinstance(p, dict)
                and p
                and all(
                    isinstance(v, dict)
                    and (
                        isinstance(v.get("type"), (str, list))
                        or isinstance(v.get("description"), str)
                    )
                    for v in p.values()
                )
            ):
                ambiguous = {
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
                names = f.get("required")
                explicit = (
                    isinstance(names, list)
                    and bool(names)
                    and all(isinstance(n, str) and n in p for n in names)
                )
                if not (set(p) & ambiguous) or explicit:
                    f["parameters"] = {"type": "object", "properties": p}
            if "parameters" in f:
                f["parameters"] = project_schema(f["parameters"])
            if "required" in f:
                required = f.pop("required")
                if required is not None:
                    p = f["parameters"]
                    assert "required" not in p or p["required"] == required
                    p["required"] = required
        if tool not in projected:
            projected.append(tool)
    return projected


def project(raw, *, strip):
    messages = deepcopy(raw["messages"])
    for i, message in enumerate(messages):
        message.pop("loss", None)
        if message["role"] != "assistant":
            continue
        calls = message.get("tool_calls", [])
        if calls:
            block = messages[i + 1 : i + 1 + len(calls)]
            assert len(block) == len(calls) and all(m["role"] == "tool" for m in block)
            if any("id" in c for c in calls):
                by_id = {r["tool_call_id"]: r for r in block}
                assert len(by_id) == len(calls)
                block = [by_id[c["id"]] for c in calls]
                messages[i + 1 : i + 1 + len(calls)] = block
            else:
                assert len(calls) == 1
            for result in block:
                result.pop("tool_call_id", None)
            for call in calls:
                call.pop("id", None)
                call["function"]["arguments"] = json_bytes(
                    call["function"]["arguments"]
                ).decode()
        if strip:
            message.pop("reasoning_content")
            message["content"] = outside_native(message["content"])
    return messages, project_tools(raw)


@pytest.mark.e2e
@pytest.mark.parametrize("strip", [False, True], ids=["keep", "strip"])
def test_every_retained_row_matches_independent_source_projection(strip):
    spec = next(s for s in ULTRADATA_SPECS if s.filter_config.strip_thinking == strip)
    samples, report = loader.iter_load(
        spec.dataset_config, spec.filter_config, **spec.extra_kwargs
    )
    source = jsonl_lines(loader.DATASET_ID, loader.DATASET_REVISION, loader.FILES)
    physical = retained = 0

    def observed():
        nonlocal physical, retained
        for sample in samples:
            for text in source:
                physical += 1
                raw = parse_json(text)
                assert raw["uuid"] == f"Tool_Use_{physical:06d}"
                if raw["uuid"] == sample.sample_id:
                    break
            else:
                pytest.fail("Missing or out-of-order source row")
            messages, tools = project(raw, strip=strip)
            assert sample.dataset == loader.DATASET_ID
            assert sample.messages == messages, raw["uuid"]
            assert sample.tools == tools, raw["uuid"]
            assert sample.raw == raw
            assert sample.annotations == {}
            retained += 1
            yield sample

    try:
        actual_hash = hash_samples(observed())
        for text in source:
            physical += 1
            assert parse_json(text)["uuid"] == f"Tool_Use_{physical:06d}"
    finally:
        samples.close()
        close = getattr(source, "close", None)
        if close is not None:
            close()
    assert physical == report.raw_count == 82760
    assert retained == report.final_count
    assert retained > 0
    assert actual_hash == read_hash(spec.loader, spec.config_id)
    assert asdict(report) == read_report(spec.loader, spec.config_id)
