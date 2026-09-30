"""TOUCAN's source contract and composed production pipeline."""

from collections import Counter
from copy import deepcopy
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fcanalysis.loaders import toucan
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.curation import CurationConfig
from fcanalysis.loaders.normalization import Reject


def tool(name="echo", parameters=None, **fields):
    function = {
        "name": name,
        "parameters": parameters
        if parameters is not None
        else {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    }
    function.update(fields)
    return {"type": "function", "function": function}


def assistant(text="done", **fields):
    return {"role": "assistant", "content": text, **fields}


def call(name="echo", arguments=None, **fields):
    return assistant(
        "",
        function_call={
            "name": name,
            "arguments": json.dumps(
                arguments if arguments is not None else {"value": "hi"}
            ),
            **fields,
        },
    )


def result(name="echo", content="hi", **fields):
    return {"role": "function", "content": content, "name": name, **fields}


def row(messages=None, tools=None, **fields):
    return {
        "uuid": "source-id",
        "subset_name": "single-turn-original",
        "messages": json.dumps(
            messages
            if messages is not None
            else [{"role": "user", "content": "Say hi."}, call(), result(), assistant()]
        ),
        "available_tools": json.dumps(tools if tools is not None else [tool()]),
        **fields,
    }


def run(raw, *, strip=False, override=None, dataset_config=None):
    sample, issues = toucan._convert_sample(raw, "OSS")
    pipeline = toucan._pipeline(
        FilterConfig(strip_thinking=strip, system_message_override=override),
        dataset_config,
    )
    return pipeline.process(sample), pipeline, issues


def template(tools, kind="kimi"):
    if kind == "kimi":
        return (
            toucan._KIMI_TOOL_TEMPLATE_PREFIX
            + json.dumps(tools)
            + toucan._KIMI_TOOL_TEMPLATE_SUFFIX
        )
    return (
        toucan._XML_TOOL_TEMPLATE_PREFIX
        + "\n".join(json.dumps(t) for t in tools)
        + toucan._XML_TOOL_TEMPLATE_SUFFIX
    )


def test_source_split_reconstruction_preserves_every_character_and_raw():
    raw = row(
        [
            {"role": "user", "content": "hello"},
            assistant("", reasoning_content="  think\n"),
            assistant("  before \n"),
            call(call_id="producer"),
            result(content='{"ok": true}'),
            assistant(" after  "),
        ]
    )
    original = deepcopy(raw)
    state, pipeline, _ = run(raw)
    assert state is not None
    assert state.sample.raw == original
    assert state.sample.sample_id == "source-id"
    assert state.sample.messages[1]["content"] == "  before \n"
    assert state.sample.messages[1]["reasoning_content"] == "  think\n"
    assert state.sample.messages[-1]["content"] == " after  "
    assert "id" not in state.sample.messages[1]["tool_calls"][0]
    assert "name" not in state.sample.messages[2]
    assert pipeline.transforms["producer_call_ids_removed"] == 1


def test_structured_values_are_owned_without_raw_aliases():
    raw = row()
    raw["messages"] = json.loads(raw["messages"])
    raw["messages"][1]["function_call"]["arguments"] = {"value": "hi"}
    raw["available_tools"] = [tool()]
    original = deepcopy(raw)
    state, _, _ = run(raw)
    assert state
    state.sample.tools[0]["function"]["parameters"]["properties"]["value"]["type"] = (
        "integer"
    )
    state.sample.messages[1]["tool_calls"][0]["function"]["arguments"] = "{}"
    assert raw == original


@pytest.mark.parametrize("kind", ["kimi", "xml"])
def test_template_extracts_only_exact_span_and_reconciles_all_sources(kind):
    first, second = tool(), tool("other")
    raw = row(
        [
            {
                "role": "system",
                "content": " first\n" + template([first, second], kind) + "\nlast ",
            },
            {"role": "user", "content": "hi"},
            assistant(),
        ],
        [first],
    )
    state, pipeline, issues = run(raw)
    assert state
    assert state.sample.messages[0]["content"] == " first\n\nlast "
    assert state.sample.tools == [first, second]
    assert issues["system_available_tool_context_mismatch"] == 1
    assert pipeline.transforms["duplicate_tool_definitions_removed"] == 1


@pytest.mark.parametrize("systems", [[], [""], ["one", "", "two"]])
def test_system_absence_emptiness_multiplicity_and_override(systems):
    raw = row(
        [
            *({"role": "system", "content": s} for s in systems),
            {"role": "user", "content": "hi"},
            assistant(),
        ]
    )
    state, _, _ = run(raw)
    assert state
    assert [
        m["content"] for m in state.sample.messages if m["role"] == "system"
    ] == systems
    removed, _, _ = run(raw, override="")
    assert removed and not any(m["role"] == "system" for m in removed.sample.messages)
    replaced, _, _ = run(raw, override="new")
    assert replaced and replaced.sample.messages[0] == {
        "role": "system",
        "content": "new",
    }


def test_empty_assistant_is_not_silently_lost():
    state, pipeline, _ = run(row([{"role": "user", "content": "hi"}, assistant("")]))
    assert state is None and pipeline.drops == {"empty_final_assistant": 1}


@pytest.mark.parametrize(
    "content",
    [
        "<tools>unrecognized</tools>",
        toucan._KIMI_TOOL_TEMPLATE_PREFIX
        + "not json"
        + toucan._KIMI_TOOL_TEMPLATE_SUFFIX,
    ],
)
def test_malformed_or_unfamiliar_embedded_context_quarantines(content):
    with pytest.raises(Reject, match="malformed_embedded_tool_template"):
        run(
            row(
                [
                    {"role": "system", "content": content},
                    {"role": "user", "content": "hi"},
                    assistant(),
                ]
            )
        )


def test_later_definition_epoch_cannot_leak_backward():
    with pytest.raises(Reject, match="unsupported_later_tool_epoch"):
        run(
            row(
                [
                    {"role": "user", "content": "hi"},
                    {"role": "system", "content": template([tool()])},
                    assistant(),
                ]
            )
        )


def test_root_annotation_compatibility_is_one_sided_and_constraint_preserving():
    visible = tool(parameters={"type": "object", "properties": {}, "required": []})
    source = tool(parameters={"type": "object", "properties": {}, "title": "Inputs"})
    transforms = Counter()
    sample, _ = toucan._convert_sample(
        row(
            [
                {"role": "system", "content": template([visible])},
                {"role": "user", "content": "hi"},
                assistant(),
            ],
            [source],
        ),
        "Kimi-K2",
        transforms,
    )
    assert sample.tools == [visible]
    assert transforms["cross_source_schema_encodings_reconciled"] == 1
    # Same-source annotations remain model-visible differences.
    state, pipeline, _ = run(row(tools=[visible, source]))
    assert state is None and "conflicting_duplicate_tool_names" in pipeline.drops


@pytest.mark.parametrize(
    "constraint,value",
    [
        ("additionalProperties", False),
        ("$schema", "http://json-schema.org/draft-07/schema#"),
        ("$defs", {"x": {"type": "string"}}),
        ("maxProperties", 0),
    ],
)
def test_cross_source_projection_never_erases_constraints(constraint, value):
    visible = tool(parameters={"type": "object", "properties": {}, "required": []})
    source = deepcopy(visible)
    source["function"]["parameters"][constraint] = value
    state, pipeline, _ = run(
        row(
            [
                {"role": "system", "content": template([visible])},
                {"role": "user", "content": "hi"},
                assistant(),
            ],
            [source],
        )
    )
    assert state is None and pipeline.drops == {"conflicting_duplicate_tool_names": 1}


def test_property_names_and_data_values_are_not_schema_keywords():
    visible = tool(
        parameters={
            "type": "object",
            "properties": {"description": {"type": "string", "enum": ["a", "b"]}},
            "required": [],
        }
    )
    source = deepcopy(visible)
    source["function"]["parameters"]["properties"]["description"]["enum"] = ["b", "a"]
    assert not toucan._cross_source_compatible(visible, source)
    source = deepcopy(visible)
    source["function"]["parameters"]["properties"]["description"]["default"] = "secret"
    assert not toucan._cross_source_compatible(visible, source)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda r: r.update(extra="hidden"), "unknown_source_row_field"),
        (
            lambda r: r["messages"][0].update(hidden_context="x"),
            "unknown_source_message_field",
        ),
        (
            lambda r: r["messages"][1]["function_call"].update(extra=True),
            "unknown_source_call_field",
        ),
        (lambda r: r["messages"][0].update(role="human"), "unknown_role"),
        (
            lambda r: r["messages"][0].update(content=[{"text": "hi"}]),
            "unsupported_content_shape",
        ),
        (
            lambda r: r["available_tools"][0]["function"].update(secret=True),
            "unknown_function_definition_field",
        ),
        (
            lambda r: r["available_tools"][0].update(hidden=True),
            "unknown_tool_envelope_field",
        ),
    ],
)
def test_unknown_potential_context_is_explicitly_rejected(mutation, reason):
    raw = row()
    raw["messages"] = json.loads(raw["messages"])
    raw["available_tools"] = json.loads(raw["available_tools"])
    mutation(raw)
    with pytest.raises(Reject, match=reason):
        run(raw)


def test_duplicate_json_keys_are_not_silently_last_wins():
    raw = row()
    raw["messages"] = '[{"role":"user","role":"assistant","content":"hi"}]'
    with pytest.raises(Reject, match="duplicate_json_key"):
        run(raw)


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda m: (m.pop(2), m[-1].update(role="user")), "unbalanced_cardinality"),
        (lambda m: m.insert(1, result()), "orphan_tool_result"),
        (lambda m: m[2].update(name="other"), "invalid_source_tool_linkage"),
        (
            lambda m: m[1]["function_call"].update(name="other"),
            "invalid_source_tool_linkage",
        ),
        (
            lambda m: m[1]["function_call"].update(arguments="[]"),
            "non_object_arguments",
        ),
        (
            lambda m: m[1]["function_call"].update(arguments='{"value": 2}'),
            "invalid_arguments",
        ),
    ],
)
def test_shared_and_source_validation_cannot_be_disabled(mutate, reason):
    raw = row()
    raw["messages"] = json.loads(raw["messages"])
    mutate(raw["messages"])
    state, pipeline, _ = run(
        raw, dataset_config=toucan.ToucanConfig(drop_invalid_source_tool_linkage=False)
    )
    assert state is None and pipeline.drops == {reason: 1}


def test_parallel_batch_order_preserved_and_echo_swap_rejected():
    raw = row(
        [
            {"role": "user", "content": "hi"},
            call("a", call_id="one"),
            call("b", call_id="two"),
            result("a", "A"),
            result("b", "B"),
            assistant(),
        ],
        [tool("a"), tool("b")],
    )
    state, _, _ = run(raw)
    assert state
    assert [t["function"]["name"] for t in state.sample.messages[1]["tool_calls"]] == [
        "a",
        "b",
    ]
    assert len(state.batches) == 1 and not state.batches[0].parallel
    raw["messages"] = json.loads(raw["messages"])
    raw["messages"][3:5] = raw["messages"][3:5][::-1]
    state, pipeline, _ = run(raw)
    assert state is None and pipeline.drops == {"invalid_source_tool_linkage": 1}


def test_same_name_multiplicity_and_sequential_batches_stay_distinct():
    raw = row(
        [
            {"role": "user", "content": "hi"},
            call(),
            call(),
            result(content="first"),
            result(content="second"),
            call(),
            result(content="third"),
            assistant(),
        ]
    )
    state, _, _ = run(raw)
    assert state and len(state.batches) == 2
    assert [
        len(m.get("tool_calls", []))
        for m in state.sample.messages
        if m["role"] == "assistant"
    ] == [2, 1, 0]
    assert [m["content"] for m in state.sample.messages if m["role"] == "tool"] == [
        "first",
        "second",
        "third",
    ]


def test_anonymous_balanced_results_use_audited_position_without_permutation():
    raw = row(
        [
            {"role": "user", "content": "hi"},
            call("a"),
            call("b"),
            result("", "A"),
            result("", "B"),
            assistant(),
        ],
        [tool("a"), tool("b")],
    )
    state, _, _ = run(raw)
    assert state and not state.batches[0].parallel


def test_duplicate_producer_ids_rejected():
    with pytest.raises(Reject, match="invalid_producer_call_id"):
        run(
            row(
                [
                    {"role": "user", "content": "hi"},
                    call(call_id="same"),
                    call(call_id="same"),
                    result(),
                    result(),
                    assistant(),
                ]
            )
        )


@pytest.mark.parametrize(
    "schema,reason",
    [
        ({"type": "object", "optional": True}, "unsupported_schema_keyword:optional"),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "string", "format": "uri"}},
            },
            "unsupported_schema_format:uri",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
            None,
        ),
        (
            {"type": "object", "properties": {}, "additionalProperties": False},
            "invalid_arguments",
        ),
        ({"type": "object", "$ref": "#/$defs/missing"}, "unresolved_tool_schema"),
    ],
)
def test_supported_schema_never_silently_ignores_constraints(schema, reason):
    state, pipeline, _ = run(row(tools=[tool(parameters=schema)]))
    if reason:
        assert state is None and pipeline.drops == {reason: 1}
    else:
        assert state


def test_native_reasoning_removal_keeps_visible_thought_tool_and_react():
    raw = row(
        [
            {"role": "user", "content": "<think>visible user</think>"},
            assistant("", reasoning_content="native"),
            assistant("<think>private</think>Thought: visible.\nAction: use tool."),
            call("think", {"value": "<think>argument</think>"}),
            result("think", "Observation: keep <think>result</think>"),
            assistant("<reasoning>private</reasoning>answer"),
        ],
        [tool("think")],
    )
    state, _, _ = run(raw, strip=True)
    assert state
    assert state.sample.messages[1]["content"] == "Thought: visible.\nAction: use tool."
    assert "reasoning_content" not in state.sample.messages[1]
    assert (
        state.sample.messages[2]["content"] == "Observation: keep <think>result</think>"
    )
    assert state.sample.messages[-1]["content"] == "answer"
    assert (
        "<think>argument</think>"
        in state.sample.messages[1]["tool_calls"][0]["function"]["arguments"]
    )
    retained, _, _ = run(raw, strip=False)
    assert retained and retained.sample.messages[1]["reasoning_content"] == "native"


@pytest.mark.parametrize(
    "text,reason",
    [
        ("<think>only</think>", "empty_final_assistant"),
        ("<think>unfinished", "ambiguous_reasoning"),
        ("</think>unmatched", "ambiguous_reasoning"),
        ("<think><reasoning>x</reasoning></think>", "ambiguous_reasoning"),
    ],
)
def test_reasoning_termination_uses_final_view(text, reason):
    state, pipeline, _ = run(
        row([{"role": "user", "content": "hi"}, assistant(text)]), strip=True
    )
    assert state is None and pipeline.drops == {reason: 1}


@pytest.mark.parametrize(
    "text",
    [
        "Use `<think>x</think>` literally.",
        "```xml\n<tool_call>\n<reasoning>documentation</reasoning>\n```",
        "Thought: careful. Action: explain. Observation: text.",
    ],
)
def test_literal_protocol_and_reasoning_documentation_survive(text):
    state, _, _ = run(
        row([{"role": "user", "content": "hi"}, assistant(text)]), strip=True
    )
    assert state and state.sample.messages[-1]["content"] == text


@pytest.mark.parametrize(
    "text,reason",
    [
        ("[ERROR: generation failed]", "producer_error_target"),
        ("[UNEXPECTED_ERROR: oops]", "producer_error_target"),
        ('<tool_call>{"name":"echo"}</tool_call>', "assistant_protocol_residue"),
        ("<|im_end|>", "assistant_protocol_residue"),
    ],
)
def test_exact_producer_failures_and_protocol_residue_quarantine(text, reason):
    state, pipeline, _ = run(row([{"role": "user", "content": "hi"}, assistant(text)]))
    assert state is None and pipeline.drops == {reason: 1}


def test_recoverable_tool_errors_and_retries_survive():
    state, _, _ = run(
        row(
            [
                {"role": "user", "content": "hi"},
                call(),
                result(content='{"error":"retry"}'),
                assistant("Tool failed; retry."),
                call(),
                result(),
                assistant("Recovered."),
            ]
        )
    )
    assert state and len(state.batches) == 2


def test_legitimate_irrelevant_no_call_answers_survive_default_policy():
    raw = row(
        [{"role": "user", "content": "Say hi"}, assistant("hi")],
        [],
        subset_name="irrelevant",
        response_quality_assessment="",
    )
    state, _, _ = run(raw)
    assert state and not state.batches


def test_quality_is_opt_in_audit_policy():
    raw = row(
        [{"role": "user", "content": "hi"}, assistant()],
        [],
        response_quality_assessment="",
    )
    assert run(raw)[0]
    state, pipeline, _ = run(
        raw, dataset_config=toucan.ToucanConfig(drop_low_quality=True)
    )
    assert state is None and pipeline.drops == {"low_quality": 1}


def write_rows(root, config, rows):
    (root / config).mkdir(exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows), root / config / "train-00000-of-00001.parquet"
    )


def test_full_loader_curates_inside_teacher_subset_scope_and_reports(tmp_path):
    a = row(uuid="a")
    b = row(uuid="b")
    c = row(uuid="c", subset_name="irrelevant")
    write_rows(tmp_path, "OSS", [a, b, c])
    write_rows(tmp_path, "Kimi-K2", [row(uuid="d")])
    samples, report = toucan.load(
        path=tmp_path,
        configs=("OSS", "Kimi-K2"),
        curation_config=CurationConfig(max_level="level_2"),
    )
    assert [s.sample_id for s in samples] == ["a", "c", "d"]
    assert report.raw_count == report.stage1_count == report.filtered_count == 4
    assert report.final_count == 3
    assert len(report.dataset_config_transform_counts["curation"]) == 3
    assert all(not s.annotations for s in samples)


def test_curation_level_two_keeps_fewest_characters_without_splicing(tmp_path):
    verbose = row(uuid="long")
    short = row(uuid="short")
    verbose["messages"] = json.loads(verbose["messages"])
    verbose["messages"][-1]["content"] = "A much longer response."
    verbose["messages"] = json.dumps(verbose["messages"])
    write_rows(tmp_path, "OSS", [verbose, short])
    samples, report = toucan.load(path=tmp_path, configs=("OSS",))
    assert len(samples) == 1 and samples[0].sample_id == "short"
    assert samples[0].raw == short and report.final_count == 1


def test_disabled_curation_stream_and_close_has_no_stale_final_count(tmp_path):
    write_rows(tmp_path, "OSS", [row(), row(uuid="two")])
    stream, report = toucan.iter_load(
        path=tmp_path,
        configs=("OSS",),
        curation_config=CurationConfig(max_level=None, audit=False),
    )
    next(stream)
    stream.close()
    assert report.final_count is None


def test_scaffold_removal_is_explicit_error_before_io(tmp_path):
    with pytest.raises(ValueError, match="scaffold"):
        toucan.iter_load(toucan.ToucanConfig(strip_scaffold_tools=True), path=tmp_path)


@pytest.mark.parametrize("configs", [("SFT",), (), ("OSS", "OSS"), ("unknown",)])
def test_unsupported_configs_are_explicit(configs, tmp_path):
    with pytest.raises(ValueError):
        toucan.iter_load(path=tmp_path, configs=configs)


def test_complete_optional_definition_fields_and_parameter_absence_preserved():
    definition = {
        "type": "function",
        "function": {"name": "echo", "description": "  exact prose\n", "strict": False},
    }
    state, _, _ = run(row(tools=[definition]))
    assert state and state.sample.tools == [definition]
    missing = {"type": "function", "function": {"name": "echo"}}
    state, pipeline, _ = run(row(tools=[definition, missing]))
    assert state is None and pipeline.drops == {"conflicting_duplicate_tool_names": 1}


def test_loader_default_removes_only_native_reasoning(tmp_path):
    raw = row(
        [{"role": "user", "content": "hi"}, assistant("<think>native</think> visible")]
    )
    write_rows(tmp_path, "OSS", [raw])
    samples, report = toucan.load(path=tmp_path, configs=("OSS",))
    assert samples[0].messages[-1]["content"] == " visible"
    assert report.strip_thinking_applied
    retained, report = toucan.load(
        filter_config=FilterConfig(strip_thinking=False),
        path=tmp_path,
        configs=("OSS",),
    )
    assert retained[0].messages[-1]["content"] == "<think>native</think> visible"
    assert not report.strip_thinking_applied


def test_two_independent_text_assistants_are_not_merged():
    raw = row([{"role": "user", "content": "hi"}, assistant("one"), assistant("two")])
    state, _, _ = run(raw)
    assert state
    assert [
        m["content"] for m in state.sample.messages if m["role"] == "assistant"
    ] == ["one", "two"]


@pytest.mark.parametrize(
    "prefix",
    [
        [assistant("one"), assistant("two"), call()],
        [call(), assistant("trailing")],
        [call(), assistant("", reasoning_content="later"), call()],
    ],
)
def test_ambiguous_assistant_call_run_is_not_guessed(prefix):
    with pytest.raises(Reject, match="ambiguous_source_assistant_run"):
        run(row([{"role": "user", "content": "hi"}, *prefix, result(), assistant()]))


def test_available_only_capability_is_not_promoted_into_verified_context():
    raw = row(
        [
            {"role": "system", "content": template([tool()])},
            {"role": "user", "content": "hi"},
            assistant(),
        ],
        [tool(), tool("hidden")],
    )
    with pytest.raises(Reject, match="unexposed_available_tool_definition"):
        run(raw)


def test_optional_quality_uses_nested_scores_and_fractional_tool_coverage():
    raw = row(
        question_quality_assessment=json.dumps(
            {
                "question_quality": {"score": 5, "reasoning": "audit only"},
                "scenario_realism": {"score": 5},
            }
        ),
        response_quality_assessment=json.dumps(
            {
                "completeness": {"score": 4},
                "conciseness": {"score": 4},
                "desired_tools_used_percentage": 1.0,
            }
        ),
    )
    config = toucan.ToucanConfig(drop_low_quality=True, require_full_tool_use=True)
    state, _, _ = run(raw, dataset_config=config)
    assert state
    raw["response_quality_assessment"] = json.dumps(
        {
            "completeness": {"score": 4},
            "conciseness": {"score": 4},
            "desired_tools_used_percentage": 0.5,
        }
    )
    assert not run(raw, dataset_config=config)[0]
    raw["subset_name"] = "irrelevant"
    raw["response_quality_assessment"] = ""
    assert run(raw, dataset_config=config)[0]


def test_complete_reasoning_like_call_with_continuation_argument_is_preserved():
    # A tool's own next-step boolean is observable application data, not the
    # loader's asynchronous or private-reasoning termination protocol.
    definition = tool(
        "sequentialthinking",
        {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "nextThoughtNeeded": {"type": "boolean"},
            },
            "required": ["thought", "nextThoughtNeeded"],
        },
    )
    raw = row(
        [
            {"role": "user", "content": "Explain"},
            call(
                "sequentialthinking", {"thought": "visible", "nextThoughtNeeded": True}
            ),
            result(
                "sequentialthinking", '{"thoughtNumber":1,"thoughtHistoryLength":1}'
            ),
            assistant("Answer."),
        ],
        [definition],
    )
    state, _, _ = run(raw, strip=True)
    assert state and state.sample.messages[1]["tool_calls"]


def test_definition_variation_across_rows_is_not_a_global_name_collision(tmp_path):
    a = row(uuid="first", tools=[tool(description="first visible contract")])
    b = row(uuid="second", tools=[tool(description="second visible contract")])
    write_rows(tmp_path, "OSS", [a, b])
    samples, _ = toucan.load(path=tmp_path, configs=("OSS",))
    assert [sample.sample_id for sample in samples] == ["first", "second"]


def test_early_consumer_close_explicitly_closes_source_parquet(tmp_path, monkeypatch):
    write_rows(tmp_path, "OSS", [row(), row(uuid="two")])
    opened = []
    parquet_file = pq.ParquetFile

    def tracked_file(path):
        source = parquet_file(path)
        opened.append(source)
        return source

    class FirstOnly:
        reports = {}

        def __init__(self, inputs, **kwargs):
            self.inputs = inputs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __iter__(self):
            yield next(self.inputs).sample

    monkeypatch.setattr(toucan.pq, "ParquetFile", tracked_file)
    monkeypatch.setattr(toucan, "curate", FirstOnly)
    stream, report = toucan.iter_load(path=tmp_path, configs=("OSS",))
    next(stream)
    assert len(opened) == 1 and not opened[0].closed
    stream.close()
    assert opened[0].closed and report.final_count is None


@pytest.mark.parametrize("strip", [False, True])
def test_native_reasoning_only_final_is_incomplete_in_both_modes(strip):
    raw = row(
        [
            {"role": "user", "content": "Answer the question."},
            assistant("<think>Private analysis only.</think>"),
        ]
    )
    state, pipeline, _ = run(raw, strip=strip)
    assert state is None and pipeline.drops == {"empty_final_assistant": 1}
