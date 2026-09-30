"""Unit tests for fcanalysis.core: turn identification and classification.

Covers TurnPattern, ToolCallStep, RealTurn, identify_real_turns,
classify_turn_pattern, and analyze_sample. Tests use raw message lists
(not ConversationSample) because that is what core.py consumes.
"""

from fcanalysis.core import (
    RealTurn,
    SampleAnalysis,
    ToolCallStep,
    TurnPattern,
    analyze_sample,
    classify_turn_pattern,
    identify_real_turns,
)

from tests.helpers import assistant, call, system, tool_response, user


class TestTurnPatternEnum:
    def test_str_values(self) -> None:
        assert TurnPattern.NO_CALLS.value == "no_calls"
        assert TurnPattern.SINGLE_CALL.value == "single_call"
        assert TurnPattern.SEQUENTIAL.value == "sequential"
        assert TurnPattern.PARALLEL.value == "parallel"
        assert TurnPattern.HYBRID.value == "hybrid"

    def test_is_str_subclass(self) -> None:
        # Used as both Enum and string at call sites.
        assert isinstance(TurnPattern.NO_CALLS, str)


class TestClassifyTurnPattern:
    def test_empty_steps(self) -> None:
        assert classify_turn_pattern([]) == TurnPattern.NO_CALLS

    def test_single_call(self) -> None:
        steps = [
            ToolCallStep(num_tool_calls=1, assistant_message_idx=1, function_names=[])
        ]
        assert classify_turn_pattern(steps) == TurnPattern.SINGLE_CALL

    def test_single_step_parallel(self) -> None:
        steps = [
            ToolCallStep(num_tool_calls=3, assistant_message_idx=1, function_names=[])
        ]
        assert classify_turn_pattern(steps) == TurnPattern.PARALLEL

    def test_sequential(self) -> None:
        steps = [
            ToolCallStep(num_tool_calls=1, assistant_message_idx=1, function_names=[]),
            ToolCallStep(num_tool_calls=1, assistant_message_idx=3, function_names=[]),
        ]
        assert classify_turn_pattern(steps) == TurnPattern.SEQUENTIAL

    def test_hybrid_when_any_step_parallel(self) -> None:
        steps = [
            ToolCallStep(num_tool_calls=1, assistant_message_idx=1, function_names=[]),
            ToolCallStep(num_tool_calls=2, assistant_message_idx=3, function_names=[]),
        ]
        assert classify_turn_pattern(steps) == TurnPattern.HYBRID


class TestRealTurnMethods:
    def test_total_tool_calls_aggregates(self) -> None:
        rt = RealTurn(
            user_message_idx=0,
            steps=[
                ToolCallStep(
                    num_tool_calls=2, assistant_message_idx=1, function_names=[]
                ),
                ToolCallStep(
                    num_tool_calls=3, assistant_message_idx=3, function_names=[]
                ),
            ],
            pattern=TurnPattern.HYBRID,
        )
        assert rt.total_tool_calls() == 5

    def test_total_tool_calls_reflects_current_steps(self) -> None:
        rt = RealTurn(
            user_message_idx=0,
            steps=[
                ToolCallStep(
                    num_tool_calls=1, assistant_message_idx=1, function_names=[]
                )
            ],
            pattern=TurnPattern.SINGLE_CALL,
        )
        assert rt.total_tool_calls() == 1
        rt.steps.append(
            ToolCallStep(num_tool_calls=99, assistant_message_idx=2, function_names=[])
        )
        assert rt.total_tool_calls() == 100

    def test_num_steps(self) -> None:
        rt = RealTurn(
            user_message_idx=0,
            steps=[
                ToolCallStep(
                    num_tool_calls=1, assistant_message_idx=1, function_names=[]
                ),
                ToolCallStep(
                    num_tool_calls=1, assistant_message_idx=2, function_names=[]
                ),
            ],
            pattern=TurnPattern.SEQUENTIAL,
        )
        assert rt.num_steps() == 2

    def test_all_function_names_flattens(self) -> None:
        rt = RealTurn(
            user_message_idx=0,
            steps=[
                ToolCallStep(
                    num_tool_calls=2,
                    assistant_message_idx=1,
                    function_names=["a", "b"],
                ),
                ToolCallStep(
                    num_tool_calls=1,
                    assistant_message_idx=3,
                    function_names=["c"],
                ),
            ],
            pattern=TurnPattern.HYBRID,
        )
        assert rt.all_function_names() == ["a", "b", "c"]


class TestIdentifyRealTurns:
    def test_empty(self) -> None:
        assert identify_real_turns([]) == []

    def test_skips_system_messages(self) -> None:
        msgs = [system("you are an agent"), user("hi"), assistant(content="hello")]
        turns = identify_real_turns(msgs)
        assert len(turns) == 1
        assert turns[0].pattern == TurnPattern.NO_CALLS
        assert turns[0].user_message_idx == 1

    def test_skips_empty_user_content(self) -> None:
        msgs = [
            {"role": "user", "content": "  "},
            user("real"),
            assistant(content="hello"),
        ]
        turns = identify_real_turns(msgs)
        assert len(turns) == 1
        assert turns[0].user_message_idx == 1

    def test_skips_user_with_no_content_key(self) -> None:
        msgs = [{"role": "user"}, user("real"), assistant(content="hi")]
        turns = identify_real_turns(msgs)
        assert len(turns) == 1
        assert turns[0].user_message_idx == 1

    def test_no_call_turn(self) -> None:
        msgs = [user("hi"), assistant(content="hello")]
        turns = identify_real_turns(msgs)
        assert len(turns) == 1
        assert turns[0].pattern == TurnPattern.NO_CALLS
        assert turns[0].steps == []

    def test_single_call_turn(self) -> None:
        msgs = [
            user("hi"),
            assistant(tool_calls=[call()]),
            tool_response(),
            assistant(content="done"),
        ]
        turns = identify_real_turns(msgs)
        assert len(turns) == 1
        assert turns[0].pattern == TurnPattern.SINGLE_CALL
        assert turns[0].num_steps() == 1
        assert turns[0].steps[0].num_tool_calls == 1

    def test_parallel_call_turn(self) -> None:
        msgs = [
            user("hi"),
            assistant(
                tool_calls=[
                    call(call_id="1"),
                    call(name="g", call_id="2"),
                ]
            ),
            tool_response(tool_call_id="1"),
            tool_response(content="r2", tool_call_id="2"),
        ]
        turns = identify_real_turns(msgs)
        assert len(turns) == 1
        assert turns[0].pattern == TurnPattern.PARALLEL
        assert turns[0].steps[0].num_tool_calls == 2

    def test_sequential_call_turn(self) -> None:
        msgs = [
            user("hi"),
            assistant(tool_calls=[call(call_id="1")]),
            tool_response(tool_call_id="1"),
            assistant(tool_calls=[call(name="g", call_id="2")]),
            tool_response(tool_call_id="2"),
        ]
        turns = identify_real_turns(msgs)
        assert len(turns) == 1
        assert turns[0].pattern == TurnPattern.SEQUENTIAL
        assert turns[0].num_steps() == 2

    def test_hybrid_turn(self) -> None:
        msgs = [
            user("hi"),
            assistant(tool_calls=[call(call_id="1")]),
            tool_response(tool_call_id="1"),
            assistant(
                tool_calls=[
                    call(name="g", call_id="2"),
                    call(name="h", call_id="3"),
                ]
            ),
            tool_response(tool_call_id="2"),
            tool_response(content="r3", tool_call_id="3"),
        ]
        turns = identify_real_turns(msgs)
        assert len(turns) == 1
        assert turns[0].pattern == TurnPattern.HYBRID

    def test_multiple_real_turns(self) -> None:
        msgs = [
            user("first"),
            assistant(tool_calls=[call(call_id="1")]),
            tool_response(tool_call_id="1"),
            user("second"),
            assistant(content="reply"),
            user("third"),
            assistant(
                tool_calls=[
                    call(call_id="2"),
                    call(name="g", call_id="3"),
                ]
            ),
            tool_response(tool_call_id="2"),
            tool_response(content="r3", tool_call_id="3"),
        ]
        turns = identify_real_turns(msgs)
        assert len(turns) == 3
        assert [t.pattern for t in turns] == [
            TurnPattern.SINGLE_CALL,
            TurnPattern.NO_CALLS,
            TurnPattern.PARALLEL,
        ]

    def test_extract_function_names(self) -> None:
        msgs = [
            user("hi"),
            assistant(
                tool_calls=[
                    call(name="alpha", call_id="1"),
                    call(name="beta", call_id="2"),
                ]
            ),
            tool_response(tool_call_id="1"),
            tool_response(content="r2", tool_call_id="2"),
        ]
        turns = identify_real_turns(msgs, extract_function_names=True)
        assert turns[0].steps[0].function_names == ["alpha", "beta"]

    def test_function_names_empty_by_default(self) -> None:
        msgs = [
            user("hi"),
            assistant(tool_calls=[call(name="alpha")]),
            tool_response(),
        ]
        turns = identify_real_turns(msgs)
        assert turns[0].steps[0].function_names == []


class TestAnalyzeSample:
    def test_empty_messages(self) -> None:
        result = analyze_sample([])
        assert result.num_real_turns == 0
        assert result.total_tool_calls == 0
        assert result.turn_patterns == []
        assert result.all_turns is None

    def test_no_call_only(self) -> None:
        msgs = [user("hi"), assistant(content="hello")]
        result = analyze_sample(msgs)
        assert result.num_real_turns == 1
        assert result.num_no_call_turns == 1
        assert result.total_tool_calls == 0
        assert result.max_steps_in_any_turn == 0
        assert result.max_parallel_calls_in_any_step == 0

    def test_aggregates_counts(self) -> None:
        msgs = [
            user("a"),
            assistant(tool_calls=[call(call_id="1")]),
            tool_response(tool_call_id="1"),
            user("b"),
            assistant(
                tool_calls=[
                    call(call_id="2"),
                    call(name="g", call_id="3"),
                ]
            ),
            tool_response(tool_call_id="2"),
            tool_response(content="r3", tool_call_id="3"),
            user("c"),
            assistant(content="text"),
        ]
        result = analyze_sample(msgs)
        assert result.num_real_turns == 3
        assert result.num_single_call_turns == 1
        assert result.num_parallel_turns == 1
        assert result.num_no_call_turns == 1
        assert result.num_sequential_turns == 0
        assert result.num_hybrid_turns == 0
        assert result.total_tool_calls == 3
        assert result.max_parallel_calls_in_any_step == 2
        assert result.max_steps_in_any_turn == 1

    def test_max_steps_in_any_turn(self) -> None:
        msgs = [
            user("a"),
            assistant(tool_calls=[call(call_id="1")]),
            tool_response(tool_call_id="1"),
            assistant(tool_calls=[call(name="g", call_id="2")]),
            tool_response(tool_call_id="2"),
            assistant(tool_calls=[call(name="h", call_id="3")]),
            tool_response(tool_call_id="3"),
        ]
        result = analyze_sample(msgs)
        assert result.num_sequential_turns == 1
        assert result.max_steps_in_any_turn == 3

    def test_all_turns_only_when_function_names_extracted(self) -> None:
        msgs = [user("a"), assistant(tool_calls=[call(name="f")]), tool_response()]
        without = analyze_sample(msgs)
        assert without.all_turns is None
        with_names = analyze_sample(msgs, extract_function_names=True)
        assert with_names.all_turns is not None
        assert len(with_names.all_turns) == 1
        assert with_names.all_turns[0].steps[0].function_names == ["f"]

    def test_return_type(self) -> None:
        result = analyze_sample([user("hi")])
        assert isinstance(result, SampleAnalysis)
