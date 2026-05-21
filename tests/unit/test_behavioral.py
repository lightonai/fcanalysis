from fcanalysis.behavioral import (
    BiasReport,
    SampleBehavioralAnalysis,
    SamplePattern,
    SemanticLayerInput,
    TurnAnalysis,
    _classify_param_types,
    analyze_dataset_behavior,
    analyze_sample_behavior,
    compute_bias_report,
    prepare_semantic_layer_inputs,
    print_bias_report,
)
from fcanalysis.format import ConversationSample


def _sample(
    messages: list[dict], tools: list[dict] | None = None
) -> ConversationSample:
    return ConversationSample(
        messages=messages,
        tools=tools or [],
        dataset="test",
        sample_id="s1",
        raw={},
    )


def _tool(name: str, params: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": params or {"type": "object", "properties": {}},
        },
    }


def _tc(name: str = "f", args: str = "{}") -> dict:
    return {"type": "function", "function": {"name": name, "arguments": args}}


class TestClassifyParamTypes:
    def test_no_tools(self) -> None:
        assert _classify_param_types([]) == "no_tools"

    def test_no_params(self) -> None:
        assert _classify_param_types([_tool("f", {})]) == "no_params"

    def test_numeric_only(self) -> None:
        params = {"properties": {"n": {"type": "integer"}}}
        assert _classify_param_types([_tool("f", params)]) == "numeric_only"

    def test_string_only(self) -> None:
        params = {"properties": {"s": {"type": "string"}}}
        assert _classify_param_types([_tool("f", params)]) == "string_only"

    def test_array_or_nested(self) -> None:
        params = {"properties": {"a": {"type": "array"}}}
        assert _classify_param_types([_tool("f", params)]) == "array_or_nested"

    def test_mixed(self) -> None:
        params = {"properties": {"s": {"type": "string"}, "n": {"type": "integer"}}}
        assert _classify_param_types([_tool("f", params)]) == "mixed"

    def test_anyof_array_wins(self) -> None:
        params = {
            "properties": {
                "x": {"anyOf": [{"type": "string"}, {"type": "array"}]},
            }
        }
        assert _classify_param_types([_tool("f", params)]) == "array_or_nested"

    def test_anyof_numeric(self) -> None:
        params = {
            "properties": {
                "x": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            }
        }
        assert _classify_param_types([_tool("f", params)]) == "numeric_only"

    def test_type_list_union(self) -> None:
        params = {"properties": {"x": {"type": ["string", "integer"]}}}
        assert _classify_param_types([_tool("f", params)]) == "numeric_only"


class TestAnalyzeSampleBehavior:
    def test_returns_full_dataclass(self) -> None:
        s = _sample(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [_tc()]},
                {"role": "tool", "content": "r"},
                {"role": "assistant", "content": "done"},
            ],
            tools=[_tool("f")],
        )
        a = analyze_sample_behavior(s)
        assert isinstance(a, SampleBehavioralAnalysis)
        assert a.sample_id == "s1"
        assert a.dataset == "test"
        assert a.num_turns == 1
        assert a.is_single_turn is True
        assert a.is_multi_turn is False
        assert a.sample_pattern == SamplePattern.NORMAL_FC
        assert a.num_fc_turns == 1
        assert a.num_no_fc_turns == 0
        assert a.num_tools_available == 1
        assert a.total_tool_calls == 1

    def test_never_call(self) -> None:
        s = _sample(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "I cannot help."},
            ]
        )
        a = analyze_sample_behavior(s)
        assert a.sample_pattern == SamplePattern.NEVER_CALL

    def test_no_turns(self) -> None:
        a = analyze_sample_behavior(_sample([]))
        assert a.sample_pattern == SamplePattern.NO_TURNS
        assert a.num_turns == 0

    def test_multi_turn_prefix_sums(self) -> None:
        s = _sample(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "abc"},
                {"role": "assistant", "tool_calls": [_tc()]},
                {"role": "tool", "content": "r1"},
                {"role": "assistant", "content": "ans"},
                {"role": "user", "content": "xyzw"},
                {"role": "assistant", "tool_calls": [_tc()]},
                {"role": "tool", "content": "r2"},
                {"role": "assistant", "content": "ans2"},
            ]
        )
        a = analyze_sample_behavior(s)
        assert a.is_multi_turn is True
        # Second turn's conv_chars_before should be 3 (abc) + len("") for
        # assistant + len("r1") + len("ans") = 3 + 0 + 2 + 3 = 8
        assert a.turns[1].conversation_length_chars_before == 8

    def test_turn_analysis_fields(self) -> None:
        s = _sample(
            [
                {"role": "user", "content": "hi there"},
                {"role": "assistant", "tool_calls": [_tc("foo")]},
                {"role": "tool", "content": "r"},
            ]
        )
        a = analyze_sample_behavior(s)
        t = a.turns[0]
        assert isinstance(t, TurnAnalysis)
        assert t.turn_index == 0
        assert t.has_fc is True
        assert t.user_message_length_chars == 8
        assert t.user_message_length_words == 2
        assert t.function_names == ["foo"]


class TestAnalyzeDatasetBehavior:
    def test_returns_list_of_analyses(self) -> None:
        samples = [
            _sample(
                [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            )
            for _ in range(3)
        ]
        result = analyze_dataset_behavior(samples)
        assert len(result) == 3
        assert all(isinstance(a, SampleBehavioralAnalysis) for a in result)


class TestComputeBiasReport:
    def test_returns_full_report(self) -> None:
        samples = [
            _sample(
                [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "tool_calls": [_tc()]},
                    {"role": "tool", "content": "r"},
                ],
                tools=[_tool("f")],
            ),
            _sample(
                [
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "I can't"},
                ]
            ),
        ]
        analyses = analyze_dataset_behavior(samples)
        report = compute_bias_report(analyses)
        assert isinstance(report, BiasReport)
        assert report.total_samples == 2
        assert report.fc_turns == 1
        assert report.no_fc_turns == 1
        # Pattern counts cover every SamplePattern value
        assert set(report.pattern_counts.keys()) == {p.value for p in SamplePattern}

    def test_empty(self) -> None:
        report = compute_bias_report([])
        assert report.total_samples == 0
        assert report.total_turns == 0


class TestPrepareSemanticLayerInputs:
    def test_only_returns_samples_with_no_fc(self) -> None:
        samples = [
            _sample(
                [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "tool_calls": [_tc()]},
                    {"role": "tool", "content": "r"},
                ]
            ),
            _sample(
                [
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "no"},
                ]
            ),
        ]
        samples[1].sample_id = "s2"
        analyses = analyze_dataset_behavior(samples)
        inputs = prepare_semantic_layer_inputs(samples, analyses)
        assert len(inputs) == 1
        assert isinstance(inputs[0], SemanticLayerInput)
        assert inputs[0].sample_id == "s2"
        assert inputs[0].no_fc_turn_indices == [0]


class TestPrintBiasReport:
    def test_returns_string_with_expected_sections(self) -> None:
        samples = [
            _sample(
                [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "tool_calls": [_tc()]},
                    {"role": "tool", "content": "r"},
                ]
            )
        ]
        analyses = analyze_dataset_behavior(samples)
        out = print_bias_report(compute_bias_report(analyses))
        assert isinstance(out, str)
        for section in (
            "Sample-Level Patterns",
            "Turn-Level Counts",
            "FC Calling Pattern Diversity",
            "Semantic Layer Handoff",
        ):
            assert section in out
