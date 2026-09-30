"""Structural supervision features, without token or semantic-quality guesses.

Input is canonical loader output. This module neither curates nor mutates it.
Real turns reuse ``core``; strict call episodes additionally end at *any* user,
system, or no-call assistant message. An observed tool result is feedback, not
proof of a correct result, dependency, recovery, or successful execution.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from .core import TurnPattern, identify_real_turns
from .format import ConversationSample


@dataclass(frozen=True, slots=True)
class TargetFeatures:
    target_index: int
    real_turn_index: int | None
    tool_calls: int


@dataclass(frozen=True, slots=True)
class TurnFeatures:
    user_message_index: int
    pattern: TurnPattern
    call_decisions: int
    tool_calls: int
    max_parallel_width: int
    no_call_decisions: int


@dataclass(frozen=True, slots=True)
class EpisodeFeatures:
    first_target_index: int
    last_target_index: int
    call_decisions: int
    tool_calls: int
    max_parallel_width: int
    continuations_with_feedback: int
    continuations_without_feedback: int


@dataclass(frozen=True, slots=True)
class SupervisionFeatures:
    dataset: str
    sample_id: str | int
    turns: tuple[TurnFeatures, ...]
    targets: tuple[TargetFeatures, ...]
    episodes: tuple[EpisodeFeatures, ...]
    tool_result_messages: int
    distinct_declared_tools: int
    distinct_used_tools: int
    calls_before_first_user: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_supervision(sample: ConversationSample) -> SupervisionFeatures:
    """Return numeric values and categories at their original observational unit.

    Initial assistants are included as candidates, even if a training renderer
    cannot admit them. Empty user messages are not real turns but still interrupt
    strict episodes. Prose accompanying a tool call remains one call decision.
    Declared/used counts concern distinct names, not schema complexity or quality.
    """
    real_turns = identify_real_turns(sample.messages, extract_function_names=True)
    turn_starts = {t.user_message_idx: i for i, t in enumerate(real_turns)}
    turn_index = None
    targets = []
    episodes = []
    active: list[tuple[int, int, bool]] = []
    feedback = False
    used_names = set()
    no_calls: Counter[int] = Counter()
    tool_results = pre_user_calls = 0

    def finish_episode() -> None:
        if active:
            episodes.append(
                EpisodeFeatures(
                    first_target_index=active[0][0],
                    last_target_index=active[-1][0],
                    call_decisions=len(active),
                    tool_calls=sum(width for _, width, _ in active),
                    max_parallel_width=max(width for _, width, _ in active),
                    continuations_with_feedback=sum(fb for _, _, fb in active[1:]),
                    continuations_without_feedback=sum(
                        not fb for _, _, fb in active[1:]
                    ),
                )
            )
            active.clear()

    for index, message in enumerate(sample.messages):
        if index in turn_starts:
            turn_index = turn_starts[index]
        role = message["role"]
        calls = (message.get("tool_calls") or []) if role == "assistant" else []
        if role == "assistant":
            targets.append(TargetFeatures(index, turn_index, len(calls)))
            used_names.update(call["function"]["name"] for call in calls)
            if turn_index is None:
                pre_user_calls += len(calls)
            elif not calls:
                no_calls[turn_index] += 1
        if role == "tool":
            tool_results += 1
            feedback = bool(active)
        elif calls:
            active.append((index, len(calls), feedback))
            feedback = False
        else:
            finish_episode()
            feedback = False
    finish_episode()
    turns = tuple(
        TurnFeatures(
            user_message_index=turn.user_message_idx,
            pattern=turn.pattern,
            call_decisions=turn.num_steps(),
            tool_calls=turn.total_tool_calls(),
            max_parallel_width=max((s.num_tool_calls for s in turn.steps), default=0),
            no_call_decisions=no_calls[i],
        )
        for i, turn in enumerate(real_turns)
    )
    return SupervisionFeatures(
        dataset=sample.dataset,
        sample_id=sample.sample_id,
        turns=turns,
        targets=tuple(targets),
        episodes=tuple(episodes),
        tool_result_messages=tool_results,
        distinct_declared_tools=len({t["function"]["name"] for t in sample.tools}),
        distinct_used_tools=len(used_names),
        calls_before_first_user=pre_user_calls,
    )


COUNT_BINS = ("0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33+")


def count_bin(value: int) -> str:
    """Fixed inclusive integer bins; original row values remain unbinned."""
    if type(value) is not int or value < 0:
        raise ValueError("count must be a nonnegative integer")
    for upper, label in zip((0, 1, 2, 4, 8, 16, 32), COUNT_BINS, strict=False):
        if value <= upper:
            return label
    return "33+"


@dataclass
class _Histogram:
    unit: str
    bins: Counter[str] = field(
        default_factory=lambda: Counter(dict.fromkeys(COUNT_BINS, 0))
    )
    observations: int = 0
    total: int = 0
    minimum: int | None = None
    maximum: int | None = None

    def add(self, value: int) -> None:
        self.bins[count_bin(value)] += 1
        self.observations += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "bins": dict(self.bins),
            "observations": self.observations,
            "total": self.total,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


class SupervisionSummary:
    """Streaming counts with explicit units; memory O(number of source names).

    Retains neither transcripts nor per-example rows. The caller can separately
    stream ``features.as_dict()`` to retain values for later subset selection.
    Input records are counted as supplied, without deduplication or population
    weighting. In particular, a balanced reservoir is not a natural mixture.
    """

    def __init__(self) -> None:
        self.totals: Counter[str] = Counter(
            dict.fromkeys(
                (
                    "conversations",
                    "real_turns",
                    "assistant_decisions",
                    "call_decisions",
                    "tool_calls",
                    "episodes",
                    "tool_result_messages",
                    "calls_before_first_user",
                ),
                0,
            )
        )
        self.histograms = {
            name: _Histogram(unit)
            for name, unit in (
                ("real_turns", "conversation"),
                ("distinct_declared_tools", "conversation"),
                ("distinct_used_tools", "conversation"),
                ("calls_per_decision", "assistant_decision"),
                ("call_decisions_per_turn", "real_turn"),
                ("parallel_width_per_turn", "real_turn"),
                ("call_decisions_per_episode", "episode"),
            )
        }
        self.patterns: Counter[str] = Counter(dict.fromkeys(TurnPattern, 0))
        self.depth_width: Counter[tuple[str, str]] = Counter()
        self.by_dataset: dict[str, SupervisionSummary] = {}

    def add(self, features: SupervisionFeatures, *, per_dataset: bool = True) -> None:
        self.totals.update(
            {
                "conversations": 1,
                "real_turns": len(features.turns),
                "assistant_decisions": len(features.targets),
                "call_decisions": sum(t.tool_calls > 0 for t in features.targets),
                "tool_calls": sum(t.tool_calls for t in features.targets),
                "episodes": len(features.episodes),
                "tool_result_messages": features.tool_result_messages,
                "calls_before_first_user": features.calls_before_first_user,
            }
        )
        for name, value in (
            ("real_turns", len(features.turns)),
            ("distinct_declared_tools", features.distinct_declared_tools),
            ("distinct_used_tools", features.distinct_used_tools),
        ):
            self.histograms[name].add(value)
        for target in features.targets:
            self.histograms["calls_per_decision"].add(target.tool_calls)
        for turn in features.turns:
            self.patterns[turn.pattern] += 1
            self.histograms["call_decisions_per_turn"].add(turn.call_decisions)
            self.histograms["parallel_width_per_turn"].add(turn.max_parallel_width)
            self.depth_width[
                count_bin(turn.call_decisions), count_bin(turn.max_parallel_width)
            ] += 1
        for episode in features.episodes:
            self.histograms["call_decisions_per_episode"].add(episode.call_decisions)
        if per_dataset:
            if features.dataset not in self.by_dataset:
                self.by_dataset[features.dataset] = SupervisionSummary()
            self.by_dataset[features.dataset].add(features, per_dataset=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "totals": dict(self.totals),
            "histograms": {
                name: hist.as_dict() for name, hist in self.histograms.items()
            },
            "patterns": {"unit": "real_turn", "counts": dict(self.patterns)},
            "depth_x_width": {
                "unit": "real_turn",
                "unlisted_cells": 0,
                "counts": [
                    {"depth": depth, "width": width, "count": count}
                    for (depth, width), count in sorted(self.depth_width.items())
                ],
            },
            "training_exposure": None,
            "semantic_quality": None,
            "by_dataset": {
                name: sub.as_dict() for name, sub in sorted(self.by_dataset.items())
            },
        }
