from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class TurnPattern(StrEnum):
    NO_CALLS = "no_calls"
    SINGLE_CALL = "single_call"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


@dataclass(slots=True)
class ToolCallStep:
    num_tool_calls: int
    assistant_message_idx: int
    function_names: list[str]


@dataclass(slots=True)
class RealTurn:
    user_message_idx: int
    steps: list[ToolCallStep]
    pattern: TurnPattern

    def total_tool_calls(self) -> int:
        return sum(step.num_tool_calls for step in self.steps)

    def num_steps(self) -> int:
        return len(self.steps)

    def all_function_names(self) -> list[str]:
        return [fn for step in self.steps for fn in step.function_names]


@dataclass(slots=True)
class SampleAnalysis:
    num_real_turns: int
    turn_patterns: list[TurnPattern]
    total_tool_calls: int
    num_single_call_turns: int
    num_sequential_turns: int
    num_parallel_turns: int
    num_hybrid_turns: int
    num_no_call_turns: int
    max_steps_in_any_turn: int
    max_parallel_calls_in_any_step: int
    all_turns: list[RealTurn] | None = None


def _is_qualifying_user(msg: dict[str, Any]) -> bool:
    return msg["role"] == "user" and bool((msg.get("content") or "").strip())


def identify_real_turns(
    messages: list[dict[str, Any]],
    extract_function_names: bool = False,
) -> list[RealTurn]:
    turns: list[RealTurn] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if not _is_qualifying_user(msg):
            i += 1
            continue

        user_idx = i
        steps: list[ToolCallStep] = []
        i += 1

        while i < len(messages):
            msg = messages[i]

            if _is_qualifying_user(msg):
                break

            if msg["role"] == "assistant" and msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
                function_names = (
                    [tc["function"]["name"] for tc in tool_calls]
                    if extract_function_names
                    else []
                )
                steps.append(
                    ToolCallStep(
                        num_tool_calls=len(tool_calls),
                        assistant_message_idx=i,
                        function_names=function_names,
                    )
                )
                i += 1
                while i < len(messages) and messages[i]["role"] == "tool":
                    i += 1
            else:
                i += 1

        turns.append(
            RealTurn(
                user_message_idx=user_idx,
                steps=steps,
                pattern=classify_turn_pattern(steps),
            )
        )

    return turns


def classify_turn_pattern(steps: list[ToolCallStep]) -> TurnPattern:
    match steps:
        case []:
            return TurnPattern.NO_CALLS
        case [single] if single.num_tool_calls == 1:
            return TurnPattern.SINGLE_CALL
        case [_]:
            return TurnPattern.PARALLEL
        case _ if any(step.num_tool_calls > 1 for step in steps):
            return TurnPattern.HYBRID
        case _:
            return TurnPattern.SEQUENTIAL


def analyze_sample(
    messages: list[dict[str, Any]],
    extract_function_names: bool = False,
) -> SampleAnalysis:
    turns = identify_real_turns(messages, extract_function_names=extract_function_names)
    turn_patterns = [t.pattern for t in turns]
    pcounts = Counter(turn_patterns)

    return SampleAnalysis(
        num_real_turns=len(turns),
        turn_patterns=turn_patterns,
        total_tool_calls=sum(t.total_tool_calls() for t in turns),
        num_single_call_turns=pcounts[TurnPattern.SINGLE_CALL],
        num_sequential_turns=pcounts[TurnPattern.SEQUENTIAL],
        num_parallel_turns=pcounts[TurnPattern.PARALLEL],
        num_hybrid_turns=pcounts[TurnPattern.HYBRID],
        num_no_call_turns=pcounts[TurnPattern.NO_CALLS],
        max_steps_in_any_turn=max((t.num_steps() for t in turns), default=0),
        max_parallel_calls_in_any_step=max(
            (step.num_tool_calls for t in turns for step in t.steps),
            default=0,
        ),
        all_turns=turns if extract_function_names else None,
    )
