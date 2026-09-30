"""Synthetic contracts for the experimental Turnstile converter."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from fcanalysis.loaders import turnstile
from fcanalysis.loaders.base import FilterConfig, apply_filters


def _definition(name: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Use {name}.",
        "parameters": parameters or {"type": "dict", "properties": {}},
        "metadata": {"domain": "synthetic", "top_related_apis": [name]},
    }


def _row(interaction: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "interaction_template_name": "SYNTHETIC_PARALLEL",
        "api_names": ["z_used"],
        "distractors": ["a_other"],
        "interaction": interaction
        if interaction is not None
        else [
            {"SYSTEM": "The date is 2026-01-01."},
            {"USER": "Please perform the operation."},
            {"THINKING": "I should use the exposed operation."},
            {"API_CALL": "z_used()"},
            {"API_OBS": {"ok": True}},
            {"ASST": "Done."},
        ],
    }


def _convert(row: Any, definitions: Any = None):
    if definitions is None:
        definitions = {name: _definition(name) for name in ("a_other", "z_used")}
    tools, errors = turnstile._prepare_definitions(definitions)
    return turnstile._convert_row(row, 7, tools, errors)


def _write_source(tmp_path: Path, rows: list[Any], definitions: Any = None) -> Path:
    if definitions is None:
        definitions = {name: _definition(name) for name in ("a_other", "z_used")}
    (tmp_path / "api_definitions.json").write_text(json.dumps(definitions))
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return data_path


def test_preserves_model_context_without_source_control_fields() -> None:
    row = _row()
    before = deepcopy(row)
    sample = _convert(row)
    assert sample.raw is row
    assert row == before
    assert sample.sample_id == 7
    assert sample.dataset == turnstile.DATASET_ID
    assert sample.annotations == {}
    assert [tool["function"]["name"] for tool in sample.tools] == ["a_other", "z_used"]
    assert all(
        set(tool["function"]) == {"name", "description", "parameters"}
        for tool in sample.tools
    )
    assert sample.messages[0] == {
        "role": "system",
        "content": "The date is 2026-01-01.",
    }
    assert (
        sample.messages[2]["reasoning_content"] == "I should use the exposed operation."
    )
    assert sample.messages[3] == {
        "role": "tool",
        "name": "z_used",
        "content": '{"ok":true}',
    }


def test_tool_order_is_independent_of_target_membership() -> None:
    first = _row()
    second = deepcopy(first)
    second["api_names"], second["distractors"] = (
        second["distractors"],
        second["api_names"],
    )
    first["distractors"] += ["a_other", "z_used"]
    assert _convert(first).tools == _convert(second).tools


def test_no_call_preserves_distractors_and_reasoning_on_final() -> None:
    row = _row(
        [
            {"USER": "Explain a poem."},
            {"THINKING": "No API applies."},
            {"ASST": "Here is an explanation."},
        ]
    )
    row["api_names"] = []
    sample = _convert(row)
    assert len(sample.tools) == 1
    assert sample.messages[-1]["reasoning_content"] == "No API applies."
    assert all("tool_calls" not in message for message in sample.messages)
    filtered, drops = apply_filters([sample], FilterConfig(strip_thinking=True))
    assert drops == {}
    assert "reasoning_content" not in filtered[0].messages[-1]
    assert row["interaction"][1] == {"THINKING": "No API applies."}


def test_parallel_template_never_reorders_serial_source_calls() -> None:
    sample = _convert(
        _row(
            [
                {"USER": "Do two operations."},
                {"THINKING": "Start with the first."},
                {"API_CALL": "z_used(x=1)"},
                {"API_OBS": {"value": 2}},
                {"API_CALL": "z_used(x=2)"},
                {"API_OBS": "source string observation"},
                {"ASST": "Both completed."},
            ]
        )
    )
    assert [message["role"] for message in sample.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert [
        len(message["tool_calls"])
        for message in sample.messages
        if "tool_calls" in message
    ] == [1, 1]
    assert "reasoning_content" in sample.messages[1]
    assert "reasoning_content" not in sample.messages[3]
    assert sample.messages[4]["content"] == "source string observation"


def test_preformatted_json_observation_text_is_preserved_exactly() -> None:
    observation = '{\n  "result": "#008000", "escaped": "line\\nnext"\n}'
    sample = _convert(
        _row(
            [
                {"API_CALL": "z_used()"},
                {"API_OBS": observation},
                {"ASST": "Done."},
            ]
        )
    )
    assert sample.messages[1]["content"] == observation


def test_literal_argument_conversion_preserves_json_types() -> None:
    call = turnstile._parse_call(
        "f(text='é', count=-2, amount=1.5, enabled=True, value=None, rows=[{'type': 'dict', 'valid': False}])"
    )
    assert json.loads(call["function"]["arguments"]) == {
        "text": "é",
        "count": -2,
        "amount": 1.5,
        "enabled": True,
        "value": None,
        "rows": [{"type": "dict", "valid": False}],
    }


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("f(", "invalid_call_syntax"),
        (None, "invalid_call_syntax"),
        ("f(1)", "unsupported_call_expression"),
        ("obj.f(x=1)", "unsupported_call_expression"),
        ("f(*[1])", "unsupported_call_expression"),
        ("f(**{'x': 1})", "unsupported_call_expression"),
        ("[f(x=1)]", "unsupported_call_expression"),
        ("f(x=1, x=2)", "duplicate_call_keyword"),
        ("f(x=some_variable)", "nonliteral_argument"),
        ("f(x=1 + 2)", "nonliteral_argument"),
        ("f(x=[n for n in [1]])", "nonliteral_argument"),
        ("f(x=(1, 2))", "non_json_argument"),
        ("f(x={1, 2})", "non_json_argument"),
        ("f(x=b'bytes')", "non_json_argument"),
        ("f(x={1: 'not a JSON key'})", "non_json_argument"),
        ("f(x={'nested': [1e999]})", "non_json_argument"),
        ("f(x=1j)", "non_json_argument"),
    ],
)
def test_rejects_nonliteral_or_non_json_calls(expression: Any, reason: str) -> None:
    with pytest.raises(turnstile._ConversionError, match=f"^{reason}$"):
        turnstile._parse_call(expression)


def test_malicious_call_argument_is_never_executed(tmp_path: Path) -> None:
    target = tmp_path / "must-not-exist"
    expression = (
        f"f(x=__import__('pathlib').Path({str(target)!r}).write_text('executed'))"
    )
    with pytest.raises(turnstile._ConversionError, match="nonliteral_argument"):
        turnstile._parse_call(expression)
    assert not target.exists()


def test_schema_alias_normalization_does_not_rewrite_property_names_or_values() -> None:
    schema = {
        "type": "dict",
        "properties": {
            "type": {"type": "string", "enum": ["dict", "float"], "default": "dict"},
            "payload": {
                "type": "array",
                "items": {
                    "type": "dict",
                    "properties": {"x": {"type": "float", "required": True}},
                },
            },
            "choice": {"anyOf": [{"type": "float"}, {"type": "null"}]},
        },
        "default": {"type": "dict"},
        "example_extension": {"type": "float"},
    }
    before = deepcopy(schema)
    result = turnstile._normalize_schema(schema)
    assert schema == before
    assert result["type"] == "object"
    assert result["properties"]["type"] == schema["properties"]["type"]
    assert result["properties"]["payload"]["items"]["properties"]["x"] == {
        "type": "number",
        "required": True,
    }
    assert result["properties"]["choice"]["anyOf"][0]["type"] == "number"
    assert result["default"] == {"type": "dict"}
    assert result["example_extension"] == {"type": "float"}


def test_schema_aliases_inside_definitions_and_union_types() -> None:
    schema = {
        "$defs": {"type": {"type": ["float", "null"]}},
        "additionalProperties": {"type": "dict"},
        "prefixItems": [{"type": "float"}],
        "dependencies": {"x": ["type"], "y": {"type": "dict"}},
        "dependentSchemas": {"x": {"type": "dict"}},
    }
    result = turnstile._normalize_schema(schema)
    assert result["$defs"]["type"]["type"] == ["number", "null"]
    assert result["additionalProperties"]["type"] == "object"
    assert result["prefixItems"] == [{"type": "number"}]
    assert result["dependencies"] == {"x": ["type"], "y": {"type": "object"}}
    assert result["dependentSchemas"] == {"x": {"type": "object"}}


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "mystery"},
        {"type": []},
        {"type": 7},
        {"type": "dict", "properties": {"type": {"type": "mystery"}}},
    ],
)
def test_unknown_schema_type_is_reported_without_guessing(schema: Any) -> None:
    definitions = {name: _definition(name) for name in ("a_other", "z_used")}
    definitions["z_used"]["parameters"] = schema
    with pytest.raises(turnstile._ConversionError, match="unsupported_schema_type"):
        _convert(_row(), definitions)


def test_unused_bad_definition_does_not_drop_other_tools() -> None:
    definitions = {name: _definition(name) for name in ("a_other", "z_used")}
    definitions["unused"] = _definition("unused", {"type": "unknown"})
    assert len(_convert(_row(), definitions).tools) == 2


@pytest.mark.parametrize(
    ("interaction", "reason"),
    [
        (
            [{"THINKING": "one"}, {"THINKING": "two"}, {"ASST": "done"}],
            "duplicate_reasoning",
        ),
        ([{"USER": "u"}, {"THINKING": "pending"}], "dangling_reasoning"),
        ([{"THINKING": "pending"}, {"USER": "new request"}], "ambiguous_reasoning"),
        ([{"API_CALL": "z_used()"}, {"THINKING": "pending"}], "ambiguous_reasoning"),
        ([{"API_CALL": "z_used()"}], "unbalanced_call_sequence"),
        ([{"API_CALL": "z_used()"}, {"ASST": "done"}], "unbalanced_call_sequence"),
        ([{"API_OBS": {}}], "orphan_observation"),
        ([{"API_CALL": "z_used()"}, {"API_OBS": (1, 2)}], "invalid_observation"),
        ([{"USER": "a", "ASST": "b"}], "malformed_role"),
        ([{}], "malformed_role"),
        ([{"ALIEN": "a"}], "unknown_role"),
        ([{"USER": {"text": "a"}}], "invalid_message_content"),
        ([{"THINKING": None}], "invalid_message_content"),
    ],
)
def test_ambiguous_reasoning_and_malformed_message_sequences_drop(
    interaction: Any, reason: str
) -> None:
    with pytest.raises(turnstile._ConversionError, match=f"^{reason}$"):
        _convert(_row(interaction))


def test_missing_distractor_definition_drops_instead_of_narrowing_context() -> None:
    with pytest.raises(turnstile._ConversionError, match="missing_tool_definition"):
        _convert(_row(), {"z_used": _definition("z_used")})


def test_definition_name_must_match_lookup_key() -> None:
    with pytest.raises(turnstile._ConversionError, match="malformed_tool_definition"):
        _convert(
            _row(),
            {"z_used": _definition("z_used"), "a_other": _definition("different")},
        )


def test_load_reports_first_failures_and_preserves_source_row_positions(
    tmp_path: Path,
) -> None:
    missing = _row()
    missing["distractors"] = ["missing"]
    path = _write_source(tmp_path, [_row(), missing, _row()])
    with path.open("a") as source:
        source.write("not json\n")
    samples, report = turnstile.load(path=path)
    assert [sample.sample_id for sample in samples] == [0, 2]
    assert report.raw_count == 4
    assert report.stage1_count == 2
    assert report.stage1_drop_reasons == {
        "missing_tool_definition": 1,
        "invalid_row_json": 1,
    }
    assert report.filtered_count is None
    assert report.dataset_config_count is None


def test_undefined_call_is_visible_and_universal_filter_is_opt_in(
    tmp_path: Path,
) -> None:
    row = _row()
    row["interaction"][3] = {"API_CALL": "not_exposed(x=1)"}
    _write_source(tmp_path, [row])
    samples, report = turnstile.load(path=tmp_path)
    assert len(samples) == 1
    assert report.stage1_issue_counts == {"undefined_function_calls": 1}
    filtered, filtered_report = turnstile.load(
        FilterConfig(require_defined_functions=True, strip_thinking=True),
        path=tmp_path,
    )
    assert filtered == []
    assert filtered_report.filtered_count == 0
    assert filtered_report.filter_drop_reasons == {"undefined_function_calls": 1}


def test_downloads_both_files_from_exact_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_source(tmp_path, [_row()])
    requests = []

    def download(repo_id: str, filename: str, **kwargs: Any) -> str:
        requests.append((repo_id, filename, kwargs))
        return str(tmp_path / filename)

    monkeypatch.setattr(turnstile, "hf_hub_download", download)
    samples, _ = turnstile.load()
    assert len(samples) == 1
    assert {filename for _, filename, _ in requests} == {
        "data.jsonl",
        "api_definitions.json",
    }
    assert all(repo_id == turnstile.DATASET_ID for repo_id, _, _ in requests)
    assert all(
        kwargs == {"repo_type": "dataset", "revision": turnstile.DATASET_REVISION}
        for _, _, kwargs in requests
    )


def test_explicit_local_api_definitions_path(tmp_path: Path) -> None:
    path = _write_source(tmp_path, [_row()])
    definitions_path = tmp_path / "custom-definitions.json"
    (tmp_path / "api_definitions.json").rename(definitions_path)
    samples, _ = turnstile.load(path=path, api_definitions_path=definitions_path)
    assert len(samples) == 1


def test_nonfinite_source_json_is_reported(tmp_path: Path) -> None:
    row = _row()
    row["interaction"][4] = {"API_OBS": {"value": float("nan")}}
    _write_source(tmp_path, [row])
    samples, report = turnstile.load(path=tmp_path)
    assert samples == []
    assert report.stage1_drop_reasons == {"non_json_row_value": 1}
