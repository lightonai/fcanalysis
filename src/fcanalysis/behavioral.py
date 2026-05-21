from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import md5
from typing import Any

from ._gc import gc_disabled
from .core import RealTurn, SampleAnalysis, TurnPattern, identify_real_turns
from .format import ConversationSample
from .statistics import compute_basic_stats, compute_percentiles


_NUMERIC_TYPES = frozenset(("integer", "number", "float", "int"))
_ARRAY_TYPES = frozenset(("array", "object", "dict", "list"))


class SamplePattern(StrEnum):
    NORMAL_FC = "normal_fc"
    NEVER_CALL = "never_call"
    MIXED = "mixed"
    NO_TURNS = "no_turns"


@dataclass(slots=True)
class TurnAnalysis:
    turn_index: int
    user_message_idx: int

    has_fc: bool
    turn_pattern: TurnPattern
    num_tool_calls: int
    function_names: list[str]

    user_message_length_chars: int
    user_message_length_words: int

    # num_tools_available is sample-static; ConversationSample.tools is a
    # flat list defined once. Repeated on each turn for downstream join-free
    # consumption in compute_bias_report.
    num_tools_available: int
    conversation_length_messages_before: int
    conversation_length_chars_before: int
    total_turns_in_sample: int


@dataclass(slots=True)
class SampleBehavioralAnalysis:
    sample_id: str | int
    dataset: str

    num_turns: int
    sample_pattern: SamplePattern
    is_single_turn: bool
    is_multi_turn: bool

    turns: list[TurnAnalysis]

    num_fc_turns: int
    num_no_fc_turns: int
    fc_turn_indices: list[int]
    no_fc_turn_indices: list[int]

    total_messages: int
    num_tools_available: int
    total_tool_calls: int

    system_prompt_hash: str
    param_type_profile: str


def _get_user_message_at(messages: list[dict[str, Any]], idx: int) -> str:
    if idx < len(messages) and messages[idx]["role"] == "user":
        return messages[idx].get("content") or ""
    return ""


def _count_messages_before(messages: list[dict[str, Any]], idx: int) -> int:
    return sum(1 for i in range(idx) if messages[i]["role"] != "system")


def _count_chars_before(messages: list[dict[str, Any]], idx: int) -> int:
    return sum(
        len(messages[i].get("content") or "")
        for i in range(idx)
        if messages[i]["role"] != "system"
    )


def _get_system_prompt_hash(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg["role"] == "system":
            return md5((msg.get("content") or "").encode()).hexdigest()
    return md5(b"").hexdigest()


def _classify_param_types(tools: list[dict[str, Any]]) -> str:
    # Inlines what would otherwise be _classify_property +
    # _classify_single_type to avoid ~300k function calls on xlam-60k.
    # Priority among categories: array_or_nested > numeric > string;
    # early-exit on array_or_nested.
    if not tools:
        return "no_tools"

    has_numeric = False
    has_string = False
    has_any_param = False

    for tool in tools:
        func = tool.get("function", {})
        params = func.get("parameters") or {}
        properties = params.get("properties", {})
        for prop_def in properties.values():
            has_any_param = True
            if not isinstance(prop_def, dict):
                has_string = True
                continue

            any_of = prop_def.get("anyOf")
            if any_of is not None:
                prop_has_numeric = False
                for member in any_of:
                    if isinstance(member, dict):
                        mtype = member.get("type", "")
                        if mtype in _ARRAY_TYPES:
                            return "array_or_nested"
                        if mtype in _NUMERIC_TYPES:
                            prop_has_numeric = True
                if prop_has_numeric:
                    has_numeric = True
                else:
                    has_string = True
                continue

            ptype = prop_def.get("type", "")
            # JSON Schema allows type: ["string", "integer"] as a union.
            if isinstance(ptype, list):
                prop_has_numeric = False
                for member_type in ptype:
                    if member_type in _ARRAY_TYPES:
                        return "array_or_nested"
                    if member_type in _NUMERIC_TYPES:
                        prop_has_numeric = True
                if prop_has_numeric:
                    has_numeric = True
                else:
                    has_string = True
            elif ptype in _ARRAY_TYPES:
                return "array_or_nested"
            elif ptype in _NUMERIC_TYPES:
                has_numeric = True
            else:
                has_string = True

    if not has_any_param:
        return "no_params"
    if has_numeric and has_string:
        return "mixed"
    if has_numeric:
        return "numeric_only"
    if has_string:
        return "string_only"
    return "no_params"


def _classify_sample_pattern(turns: list[TurnAnalysis]) -> SamplePattern:
    if not turns:
        return SamplePattern.NO_TURNS
    has_any_fc = any(t.has_fc for t in turns)
    has_any_no_fc = any(not t.has_fc for t in turns)
    if not has_any_no_fc:
        return SamplePattern.NORMAL_FC
    if not has_any_fc:
        return SamplePattern.NEVER_CALL
    return SamplePattern.MIXED


def _build_turn_analysis(
    turn_idx: int,
    turn: RealTurn,
    user_msg: str,
    msgs_before: int,
    chars_before: int,
    num_real_turns: int,
    num_tools: int,
) -> TurnAnalysis:
    return TurnAnalysis(
        turn_index=turn_idx,
        user_message_idx=turn.user_message_idx,
        has_fc=turn.pattern != TurnPattern.NO_CALLS,
        turn_pattern=turn.pattern,
        num_tool_calls=turn.total_tool_calls(),
        function_names=turn.all_function_names(),
        user_message_length_chars=len(user_msg),
        user_message_length_words=len(user_msg.split()) if user_msg else 0,
        num_tools_available=num_tools,
        conversation_length_messages_before=msgs_before,
        conversation_length_chars_before=chars_before,
        total_turns_in_sample=num_real_turns,
    )


def analyze_sample_behavior(
    sample: ConversationSample,
    precomputed_turns: list[RealTurn] | None = None,
    precomputed_total_tool_calls: int | None = None,
) -> SampleBehavioralAnalysis:
    """Structural per-sample analysis (turn pattern, FC coverage, user-msg lengths, system-prompt hash); `precomputed_*` reuses upstream values to avoid recomputation."""
    messages = sample.messages
    num_tools = len(sample.tools) if sample.tools else 0

    real_turns = (
        precomputed_turns
        if precomputed_turns is not None
        else identify_real_turns(messages, extract_function_names=True)
    )
    num_real_turns = len(real_turns)

    # Single-turn samples (majority of xlam-60k) skip the prefix-sum buildup.
    # Multi-turn samples use prefix sums to avoid O(turns * messages) scans.
    turn_analyses: list[TurnAnalysis] = []
    if num_real_turns <= 1:
        for turn_idx, turn in enumerate(real_turns):
            user_msg = _get_user_message_at(messages, turn.user_message_idx)
            turn_analyses.append(
                _build_turn_analysis(
                    turn_idx,
                    turn,
                    user_msg,
                    _count_messages_before(messages, turn.user_message_idx),
                    _count_chars_before(messages, turn.user_message_idx),
                    num_real_turns,
                    num_tools,
                )
            )
    else:
        n_msgs = len(messages)
        prefix_msgs = [0] * (n_msgs + 1)
        prefix_chars = [0] * (n_msgs + 1)
        for i in range(n_msgs):
            m = messages[i]
            is_non_system = m["role"] != "system"
            prefix_msgs[i + 1] = prefix_msgs[i] + (1 if is_non_system else 0)
            prefix_chars[i + 1] = prefix_chars[i] + (
                len(m.get("content") or "") if is_non_system else 0
            )

        for turn_idx, turn in enumerate(real_turns):
            user_msg = _get_user_message_at(messages, turn.user_message_idx)
            idx = turn.user_message_idx
            turn_analyses.append(
                _build_turn_analysis(
                    turn_idx,
                    turn,
                    user_msg,
                    prefix_msgs[idx],
                    prefix_chars[idx],
                    num_real_turns,
                    num_tools,
                )
            )

    sample_pattern = _classify_sample_pattern(turn_analyses)
    fc_indices = [t.turn_index for t in turn_analyses if t.has_fc]
    no_fc_indices = [t.turn_index for t in turn_analyses if not t.has_fc]

    total_msgs = sum(1 for m in messages if m["role"] != "system")

    return SampleBehavioralAnalysis(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        num_turns=len(turn_analyses),
        sample_pattern=sample_pattern,
        is_single_turn=len(turn_analyses) == 1,
        is_multi_turn=len(turn_analyses) > 1,
        turns=turn_analyses,
        num_fc_turns=len(fc_indices),
        num_no_fc_turns=len(no_fc_indices),
        fc_turn_indices=fc_indices,
        no_fc_turn_indices=no_fc_indices,
        total_messages=total_msgs,
        num_tools_available=num_tools,
        total_tool_calls=(
            precomputed_total_tool_calls
            if precomputed_total_tool_calls is not None
            else sum(t.num_tool_calls for t in turn_analyses)
        ),
        system_prompt_hash=_get_system_prompt_hash(messages),
        param_type_profile=_classify_param_types(sample.tools),
    )


def analyze_dataset_behavior(
    samples: list[ConversationSample],
    analyses: list[SampleAnalysis] | None = None,
) -> list[SampleBehavioralAnalysis]:
    # Generation-0 GC scans dominate when accumulating millions of nested
    # objects; one collection at exit restores correctness.
    with gc_disabled():
        if analyses is not None:
            return [
                analyze_sample_behavior(s, a.all_turns, a.total_tool_calls)
                for s, a in zip(samples, analyses, strict=True)
            ]
        return [analyze_sample_behavior(s) for s in samples]


@dataclass(slots=True)
class BiasReport:
    total_samples: int
    pattern_counts: dict[str, int]
    pattern_percentages: dict[str, float]

    single_turn_count: int
    multi_turn_count: int

    total_turns: int
    fc_turns: int
    no_fc_turns: int

    fc_turn_pattern_counts: dict[str, int]

    fc_turn_user_msg_lengths: list[int]
    no_fc_turn_user_msg_lengths: list[int]

    fc_turn_positions: list[int]
    no_fc_turn_positions: list[int]

    fc_turn_tool_counts: list[int]
    no_fc_turn_tool_counts: list[int]

    fc_turn_conv_chars_before: list[int]
    no_fc_turn_conv_chars_before: list[int]

    system_prompt_unique_count: int
    system_prompt_distribution: dict[str, int]

    param_type_distribution: dict[str, int]

    no_fc_turn_indices_per_sample: dict[str | int, list[int]]


def compute_bias_report(
    analyses: list[SampleBehavioralAnalysis],
) -> BiasReport:
    """Dataset-level bias aggregates: sample-pattern distribution, FC-vs-no-FC user-msg length disparities, param-type spread, system-prompt diversity."""
    pattern_counter = Counter(a.sample_pattern for a in analyses)
    pattern_counts = {p.value: pattern_counter[p] for p in SamplePattern}
    total = len(analyses)
    pattern_pcts = {
        k: (v / total * 100) if total > 0 else 0.0 for k, v in pattern_counts.items()
    }

    fc_lengths: list[int] = []
    no_fc_lengths: list[int] = []
    fc_positions: list[int] = []
    no_fc_positions: list[int] = []
    fc_tool_counts: list[int] = []
    no_fc_tool_counts: list[int] = []
    fc_conv_chars: list[int] = []
    no_fc_conv_chars: list[int] = []
    fc_pattern_counter: Counter[str] = Counter()
    no_fc_indices_map: dict[str | int, list[int]] = {}
    system_prompt_counter: Counter[str] = Counter()
    param_type_counter: Counter[str] = Counter()

    total_turns = 0
    fc_turn_count = 0
    no_fc_turn_count = 0
    single_turn_count = 0
    multi_turn_count = 0

    for a in analyses:
        system_prompt_counter[a.system_prompt_hash] += 1
        param_type_counter[a.param_type_profile] += 1
        if a.is_single_turn:
            single_turn_count += 1
        elif a.is_multi_turn:
            multi_turn_count += 1

        for t in a.turns:
            total_turns += 1
            if t.has_fc:
                fc_turn_count += 1
                fc_lengths.append(t.user_message_length_chars)
                fc_positions.append(t.turn_index)
                fc_tool_counts.append(t.num_tools_available)
                fc_conv_chars.append(t.conversation_length_chars_before)
                fc_pattern_counter[t.turn_pattern.value] += 1
            else:
                no_fc_turn_count += 1
                no_fc_lengths.append(t.user_message_length_chars)
                no_fc_positions.append(t.turn_index)
                no_fc_tool_counts.append(t.num_tools_available)
                no_fc_conv_chars.append(t.conversation_length_chars_before)

        if a.no_fc_turn_indices:
            no_fc_indices_map[a.sample_id] = a.no_fc_turn_indices

    return BiasReport(
        total_samples=total,
        pattern_counts=pattern_counts,
        pattern_percentages=pattern_pcts,
        single_turn_count=single_turn_count,
        multi_turn_count=multi_turn_count,
        total_turns=total_turns,
        fc_turns=fc_turn_count,
        no_fc_turns=no_fc_turn_count,
        fc_turn_pattern_counts=dict(fc_pattern_counter),
        fc_turn_user_msg_lengths=fc_lengths,
        no_fc_turn_user_msg_lengths=no_fc_lengths,
        fc_turn_positions=fc_positions,
        no_fc_turn_positions=no_fc_positions,
        fc_turn_tool_counts=fc_tool_counts,
        no_fc_turn_tool_counts=no_fc_tool_counts,
        fc_turn_conv_chars_before=fc_conv_chars,
        no_fc_turn_conv_chars_before=no_fc_conv_chars,
        system_prompt_unique_count=len(system_prompt_counter),
        system_prompt_distribution=dict(system_prompt_counter),
        param_type_distribution=dict(param_type_counter),
        no_fc_turn_indices_per_sample=no_fc_indices_map,
    )


@dataclass(slots=True)
class SemanticLayerInput:
    sample_id: str | int
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    no_fc_turn_indices: list[int]
    no_fc_user_message_indices: list[int]


def prepare_semantic_layer_inputs(
    samples: list[ConversationSample],
    analyses: list[SampleBehavioralAnalysis],
) -> list[SemanticLayerInput]:
    """Build per-sample inputs for the LLM-driven semantic classifier, one per sample with at least one no-FC turn to classify."""
    sample_map = {s.sample_id: s for s in samples}
    inputs: list[SemanticLayerInput] = []
    for a in analyses:
        if not a.no_fc_turn_indices:
            continue
        sample = sample_map[a.sample_id]
        no_fc_msg_indices = [t.user_message_idx for t in a.turns if not t.has_fc]
        inputs.append(
            SemanticLayerInput(
                sample_id=a.sample_id,
                messages=sample.messages,
                tools=sample.tools,
                no_fc_turn_indices=a.no_fc_turn_indices,
                no_fc_user_message_indices=no_fc_msg_indices,
            )
        )
    return inputs


def print_bias_report(report: BiasReport) -> str:
    lines: list[str] = []

    def _section(title: str) -> None:
        lines.append("")
        lines.append(f"=== {title} ===")

    def _stats_line(label: str, data: Sequence[int | float]) -> None:
        if not data:
            lines.append(f"  {label}: (no data)")
            return
        s = compute_basic_stats(data, include_std=True)
        p = compute_percentiles(data, [25, 75, 90])

        # compute_percentiles returns None values when len(data) < 2; render
        # as "N/A" rather than the bare "None" literal.
        def _fmt(v: float | None) -> str:
            return "N/A" if v is None else f"{v}"

        lines.append(
            f"  {label}: n={len(data)}, "
            f"mean={s['mean']:.1f}, median={s['median']:.1f}, "
            f"std={s.get('std_dev', 0):.1f}, "
            f"p25={_fmt(p['percentile_25'])}, "
            f"p75={_fmt(p['percentile_75'])}, "
            f"p90={_fmt(p['percentile_90'])}"
        )

    _section("Sample-Level Patterns")
    lines.append(f"  Total samples: {report.total_samples}")
    if report.total_samples > 0:
        lines.append(
            f"  Single-turn: {report.single_turn_count} "
            f"({report.single_turn_count / report.total_samples * 100:.1f}%)"
        )
        lines.append(
            f"  Multi-turn: {report.multi_turn_count} "
            f"({report.multi_turn_count / report.total_samples * 100:.1f}%)"
        )
    else:
        lines.append("  Single-turn: 0")
        lines.append("  Multi-turn: 0")
    for pattern, count in report.pattern_counts.items():
        lines.append(
            f"  {pattern}: {count} ({report.pattern_percentages[pattern]:.1f}%)"
        )

    _section("Turn-Level Counts")
    lines.append(f"  Total turns: {report.total_turns}")
    if report.total_turns > 0:
        lines.append(
            f"  FC turns: {report.fc_turns} "
            f"({report.fc_turns / report.total_turns * 100:.1f}%)"
        )
        lines.append(
            f"  No-FC turns: {report.no_fc_turns} "
            f"({report.no_fc_turns / report.total_turns * 100:.1f}%)"
        )
    else:
        lines.append("  FC turns: 0")
        lines.append("  No-FC turns: 0")

    _section("FC Calling Pattern Diversity")
    for pattern, count in sorted(report.fc_turn_pattern_counts.items()):
        pct = count / report.fc_turns * 100 if report.fc_turns > 0 else 0
        lines.append(f"  {pattern}: {count} ({pct:.1f}%)")

    _section("User Message Length (chars): FC vs No-FC Turns")
    _stats_line("FC turns", report.fc_turn_user_msg_lengths)
    _stats_line("No-FC turns", report.no_fc_turn_user_msg_lengths)

    _section("Turn Position: FC vs No-FC Turns")
    _stats_line("FC turn positions", report.fc_turn_positions)
    _stats_line("No-FC turn positions", report.no_fc_turn_positions)

    _section("Tools Available: FC vs No-FC Turns")
    _stats_line("FC turns", report.fc_turn_tool_counts)
    _stats_line("No-FC turns", report.no_fc_turn_tool_counts)

    _section("Conversation Context Before Turn (chars): FC vs No-FC")
    _stats_line("FC turns", report.fc_turn_conv_chars_before)
    _stats_line("No-FC turns", report.no_fc_turn_conv_chars_before)

    _section("System Prompt Diversity")
    lines.append(f"  Unique system prompts: {report.system_prompt_unique_count}")
    for h, count in sorted(
        report.system_prompt_distribution.items(), key=lambda x: -x[1]
    )[:5]:
        pct = count / report.total_samples * 100 if report.total_samples > 0 else 0
        lines.append(f"  {h[:12]}...: {count} ({pct:.1f}%)")

    _section("Parameter Type Diversity")
    for profile, count in sorted(
        report.param_type_distribution.items(), key=lambda x: -x[1]
    ):
        pct = count / report.total_samples * 100 if report.total_samples > 0 else 0
        lines.append(f"  {profile}: {count} ({pct:.1f}%)")

    _section("Semantic Layer Handoff")
    lines.append(
        f"  Samples with no-FC turns: {len(report.no_fc_turn_indices_per_sample)}"
    )
    total_no_fc_to_classify = sum(
        len(v) for v in report.no_fc_turn_indices_per_sample.values()
    )
    lines.append(f"  Total no-FC turns to classify: {total_no_fc_to_classify}")

    return "\n".join(lines)
