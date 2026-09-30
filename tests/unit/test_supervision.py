from copy import deepcopy
import json

import pytest

from fcanalysis.core import analyze_sample
from fcanalysis.supervision import SupervisionSummary, count_bin, describe_supervision
from tests.helpers import assistant, call, func, sample, system, tool_response, user


def call_step(width=1):
    return assistant(tool_calls=[call() for _ in range(width)])


def test_parallel_width_is_not_sequential_depth():
    sequential = describe_supervision(
        sample([user("task")] + [call_step(), tool_response()] * 10)
    )
    parallel = describe_supervision(
        sample([user("task"), call_step(10), tool_response()])
    )
    assert sequential.turns[0].pattern == "sequential"
    assert parallel.turns[0].pattern == "parallel"
    assert sequential.turns[0].tool_calls == parallel.turns[0].tool_calls == 10
    assert sequential.episodes[0].call_decisions == 10
    assert sequential.episodes[0].continuations_with_feedback == 9
    assert parallel.episodes[0].call_decisions == 1
    assert parallel.episodes[0].max_parallel_width == 10


@pytest.mark.parametrize(
    "interrupt", [assistant("prose"), user(""), user("  "), system("new")]
)
def test_strict_episodes_differ_from_real_turn_depth(interrupt):
    result = describe_supervision(
        sample(
            [
                user("task"),
                call_step(),
                tool_response(),
                interrupt,
                call_step(),
            ]
        )
    )
    assert len(result.turns) == 1
    assert result.turns[0].call_decisions == 2
    assert len(result.episodes) == 2
    assert all(e.continuations_with_feedback == 0 for e in result.episodes)


def test_feedback_and_adjacent_calls_are_not_conflated():
    result = describe_supervision(
        sample(
            [
                user("task"),
                call_step(),
                call_step(2),
                tool_response(),
                tool_response(),
                call_step(),
            ]
        )
    )
    (episode,) = result.episodes
    assert episode.call_decisions == 3 and episode.tool_calls == 4
    assert episode.continuations_with_feedback == 1
    assert episode.continuations_without_feedback == 1
    assert result.turns[0].pattern == "hybrid"


def test_initial_assistants_real_turns_names_and_no_mutation():
    canonical = sample(
        [
            call_step(2),
            user("first"),
            call_step(),
            assistant("done"),
            user("second"),
            assistant("no call"),
        ],
        tools=[func(), func(), func("g")],
        sample_id="42",
    )
    original = deepcopy(canonical)
    result = describe_supervision(canonical)
    assert result.calls_before_first_user == 2
    assert result.distinct_used_tools == 1 and result.distinct_declared_tools == 2
    assert result.targets[0].real_turn_index is None
    assert result.targets[-1].real_turn_index == 1
    assert [t.no_call_decisions for t in result.turns] == [1, 1]
    assert [t.pattern for t in result.turns] == ["single_call", "no_calls"]
    assert result.sample_id == "42" and canonical == original
    assert json.loads(json.dumps(result.as_dict()))["sample_id"] == "42"


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [assistant("initial")],
        [user("hi")],
        [user("hi"), assistant("hello")],
        [
            call_step(),
            user("task"),
            call_step(2),
            tool_response(),
            call_step(),
            user("next"),
        ],
    ],
)
def test_counts_reconcile_with_core_and_episode_partition(messages):
    result = describe_supervision(sample(messages))
    core = analyze_sample(messages)
    assert len(result.turns) == core.num_real_turns
    assert sum(t.tool_calls for t in result.turns) == core.total_tool_calls
    total = sum(t.tool_calls for t in result.targets)
    assert total == core.total_tool_calls + result.calls_before_first_user
    assert total == sum(e.tool_calls for e in result.episodes)
    assert sum(e.call_decisions for e in result.episodes) == sum(
        t.tool_calls > 0 for t in result.targets
    )
    assert len({t.target_index for t in result.targets}) == len(result.targets)
    for episode in result.episodes:
        assert (
            episode.continuations_with_feedback + episode.continuations_without_feedback
            == episode.call_decisions - 1
        )


def test_summary_units_empty_cells_and_source_denominators():
    summary = SupervisionSummary()
    empty = summary.as_dict()
    assert empty["totals"]["conversations"] == 0
    assert empty["histograms"]["real_turns"]["minimum"] is None
    assert empty["training_exposure"] is None
    for canonical in (
        sample([user("task"), call_step(3)], dataset="a"),
        sample(
            [user("task"), assistant("done"), user("more"), call_step()], dataset="b"
        ),
        sample([], dataset="b"),
    ):
        summary.add(describe_supervision(canonical))
    result = summary.as_dict()
    assert result["totals"] == {
        "conversations": 3,
        "real_turns": 3,
        "assistant_decisions": 3,
        "call_decisions": 2,
        "tool_calls": 4,
        "episodes": 2,
        "tool_result_messages": 0,
        "calls_before_first_user": 0,
    }
    assert result["histograms"]["calls_per_decision"]["total"] == 4
    assert result["histograms"]["real_turns"]["bins"]["0"] == 1
    assert result["by_dataset"]["b"]["totals"]["conversations"] == 2
    assert sum(result["patterns"]["counts"].values()) == 3
    assert sum(cell["count"] for cell in result["depth_x_width"]["counts"]) == 3
    for histogram in result["histograms"].values():
        assert sum(histogram["bins"].values()) == histogram["observations"]
    json.dumps(result)


@pytest.mark.parametrize(
    "value,label",
    [(0, "0"), (2, "2"), (3, "3-4"), (4, "3-4"), (32, "17-32"), (33, "33+")],
)
def test_count_bins(value, label):
    assert count_bin(value) == label


@pytest.mark.parametrize("value", [-1, 1.0, True])
def test_invalid_counts_fail(value):
    with pytest.raises(ValueError):
        count_bin(value)
