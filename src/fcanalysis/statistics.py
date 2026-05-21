from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import orjson

from .core import RealTurn, SampleAnalysis, TurnPattern, identify_real_turns
from .validation import validate_arguments


def compute_basic_stats(
    data: Sequence[int | float], include_std: bool = False
) -> dict[str, float | None]:
    if not data:
        result: dict[str, float | None] = {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
        if include_std:
            result["std_dev"] = None
        return result

    arr = np.asarray(data, dtype=np.float64)
    result = {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "min": np.min(arr),
        "max": np.max(arr),
    }
    if include_std:
        result["std_dev"] = np.std(arr) if len(arr) > 1 else 0.0
    return result


def compute_percentiles(
    data: Sequence[int | float], percentiles: list[int]
) -> dict[str, float | None]:
    if len(data) < 2:
        return {f"percentile_{p}": None for p in percentiles}
    arr = np.asarray(data, dtype=np.float64)
    values = np.percentile(arr, percentiles)
    return {f"percentile_{p}": v for p, v in zip(percentiles, values, strict=True)}


def compute_dataset_overview(
    total_unfiltered: int,
    total_filtered: int,
    filter_description: str,
) -> dict[str, Any]:
    samples_filtered_out = total_unfiltered - total_filtered
    filter_percentage = (
        (samples_filtered_out / total_unfiltered * 100) if total_unfiltered > 0 else 0
    )
    return {
        "total_unfiltered": total_unfiltered,
        "samples_filtered_out": samples_filtered_out,
        "filter_percentage": filter_percentage,
        "total_filtered": total_filtered,
        "filter_description": filter_description,
    }


def compute_token_length_statistics(token_lengths: list[int]) -> dict[str, Any]:
    if not token_lengths:
        return {}
    return {
        **compute_basic_stats(token_lengths, include_std=True),
        **compute_percentiles(token_lengths, [25, 75, 90, 95, 99]),
        "distribution": dict(Counter(token_lengths)),
    }


def _pct(count: int, total: int) -> float:
    return count / total * 100 if total else 0.0


def _count_dict(count: int, total: int) -> dict[str, int | float]:
    return {"count": count, "percentage": _pct(count, total)}


def _resolve_turns(
    messages: list[dict[str, Any]],
    analysis: SampleAnalysis | None,
    extract_function_names: bool,
) -> list[RealTurn]:
    if analysis is not None and analysis.all_turns is not None:
        return analysis.all_turns
    return identify_real_turns(messages, extract_function_names=extract_function_names)


def _partition_by_turn_count(
    analyses: list[SampleAnalysis],
) -> tuple[list[SampleAnalysis], list[SampleAnalysis]]:
    single_turn: list[SampleAnalysis] = []
    multi_turn: list[SampleAnalysis] = []
    for a in analyses:
        if a.num_real_turns == 1:
            single_turn.append(a)
        elif a.num_real_turns > 1:
            multi_turn.append(a)
    return single_turn, multi_turn


def _collect_function_names(
    messages_list: list[list[dict[str, Any]]],
) -> tuple[list[str], list[int]]:
    all_names: list[str] = []
    unique_per_sample: list[int] = []
    for messages in messages_list:
        sample_functions: set[str] = set()
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    all_names.append(name)
                    sample_functions.add(name)
        unique_per_sample.append(len(sample_functions))
    return all_names, unique_per_sample


def compute_function_diversity(
    messages_list: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    all_names, unique_per_sample = _collect_function_names(messages_list)
    function_counter = Counter(all_names)
    return {
        "total_unique_functions": len(function_counter),
        "top_20_functions": function_counter.most_common(20),
        "unique_per_sample": {
            **compute_basic_stats(unique_per_sample),
            **compute_percentiles(unique_per_sample, [25, 75, 90, 95, 99]),
            "distribution": dict(Counter(unique_per_sample))
            if unique_per_sample
            else {},
        },
    }


def compute_function_calling_patterns(
    analyses: list[SampleAnalysis],
) -> dict[str, Any]:
    total_calls_per_sample = [a.total_tool_calls for a in analyses]
    if total_calls_per_sample:
        calls_stats = {
            **compute_basic_stats(total_calls_per_sample),
            **compute_percentiles(total_calls_per_sample, [25, 75, 90, 95, 99]),
        }
    else:
        calls_stats = {}

    all_turn_patterns = [p for a in analyses for p in a.turn_patterns]
    total_turns = len(all_turn_patterns)
    pattern_counter = Counter(all_turn_patterns)

    def _pattern_entry(pattern: TurnPattern) -> dict[str, int | float]:
        return _count_dict(pattern_counter[pattern], total_turns)

    turn_level = {
        "total_turns": total_turns,
        "single_call": _pattern_entry(TurnPattern.SINGLE_CALL),
        "sequential": _pattern_entry(TurnPattern.SEQUENTIAL),
        "parallel": _pattern_entry(TurnPattern.PARALLEL),
        "hybrid": _pattern_entry(TurnPattern.HYBRID),
        "no_calls": _pattern_entry(TurnPattern.NO_CALLS),
    }

    # Single pass replaces 10 separate scans (5 _sample_with + 5 _sample_only).
    short_names = ("single_call", "sequential", "parallel", "hybrid", "no_call")
    n_attrs = len(short_names)
    with_counts = [0] * n_attrs
    only_counts = [0] * n_attrs
    total_samples = len(analyses)

    for a in analyses:
        vals = (
            a.num_single_call_turns,
            a.num_sequential_turns,
            a.num_parallel_turns,
            a.num_hybrid_turns,
            a.num_no_call_turns,
        )
        nonzero = [i for i, v in enumerate(vals) if v > 0]
        for i in nonzero:
            with_counts[i] += 1
        if len(nonzero) == 1:
            only_counts[nonzero[0]] += 1

    sample_level: dict[str, Any] = {}
    for i, short in enumerate(short_names):
        sample_level[f"samples_with_{short}"] = _count_dict(
            with_counts[i], total_samples
        )
        sample_level[f"samples_only_{short}"] = _count_dict(
            only_counts[i], total_samples
        )
    samples_mixed = total_samples - sum(only_counts)
    sample_level["samples_mixed_patterns"] = _count_dict(samples_mixed, total_samples)

    max_parallel = [
        a.max_parallel_calls_in_any_step
        for a in analyses
        if a.max_parallel_calls_in_any_step > 1
    ]
    parallel_stats = (
        {
            **compute_basic_stats(max_parallel),
            **compute_percentiles(max_parallel, [75, 90, 99]),
        }
        if max_parallel
        else {}
    )

    turns_with_calls = [
        a.num_single_call_turns
        + a.num_sequential_turns
        + a.num_parallel_turns
        + a.num_hybrid_turns
        for a in analyses
        if a.total_tool_calls > 0
    ]
    turns_stats = (
        {
            **compute_basic_stats(turns_with_calls),
            **compute_percentiles(turns_with_calls, [75, 90, 99]),
        }
        if turns_with_calls
        else {}
    )

    return {
        "total_calls_per_sample": calls_stats,
        "turn_level": turn_level,
        "sample_level": sample_level,
        "max_parallel_calls": parallel_stats,
        "turns_with_function_calls": turns_stats,
    }


def compute_parallel_function_diversity(
    messages_list: list[list[dict[str, Any]]],
    analyses: list[SampleAnalysis] | None = None,
) -> dict[str, Any]:
    same_function_fanout_count = 0
    multi_function_bundle_steps: list[dict[str, int]] = []
    unique_functions_distribution: list[int] = []
    pure_count = 0
    hybrid_bundles: list[dict[str, int]] = []

    for idx, messages in enumerate(messages_list):
        analysis = analyses[idx] if analyses is not None else None
        turns = _resolve_turns(messages, analysis, extract_function_names=True)
        for turn in turns:
            for step in turn.steps:
                if step.num_tool_calls < 2 or not step.function_names:
                    continue
                unique_functions = set(step.function_names)
                num_unique = len(unique_functions)
                unique_functions_distribution.append(num_unique)

                if num_unique == 1:
                    same_function_fanout_count += 1
                    continue

                function_counter = Counter(step.function_names)
                max_fanout = max(function_counter.values())
                functions_with_fanout = sum(
                    1 for count in function_counter.values() if count > 1
                )
                multi_function_bundle_steps.append(
                    {"num_functions": num_unique, "max_fanout": max_fanout}
                )
                if max_fanout == 1:
                    pure_count += 1
                else:
                    hybrid_bundles.append(
                        {
                            "max_fanout": max_fanout,
                            "functions_with_fanout": functions_with_fanout,
                        }
                    )

    total_parallel_steps = same_function_fanout_count + len(multi_function_bundle_steps)
    if total_parallel_steps == 0:
        return {
            "total_parallel_steps": 0,
            "same_function_fanout": {"count": 0, "percentage": 0},
            "multi_function_bundles": {"count": 0, "percentage": 0},
        }

    total_bundles = len(multi_function_bundle_steps)
    hybrid_count = len(hybrid_bundles)
    bundle_patterns = {
        "total_bundles": total_bundles,
        "pure_bundles": _count_dict(pure_count, total_bundles),
        "hybrid_bundles": {
            **_count_dict(hybrid_count, total_bundles),
            "max_fanout_distribution": dict(
                Counter(b["max_fanout"] for b in hybrid_bundles)
            ),
            "functions_with_fanout_distribution": dict(
                Counter(b["functions_with_fanout"] for b in hybrid_bundles)
            ),
        },
    }

    return {
        "total_parallel_steps": total_parallel_steps,
        "same_function_fanout": {
            "count": same_function_fanout_count,
            "percentage": same_function_fanout_count / total_parallel_steps * 100,
        },
        "multi_function_bundles": {
            "count": len(multi_function_bundle_steps),
            "percentage": len(multi_function_bundle_steps) / total_parallel_steps * 100,
        },
        "unique_functions_per_step": {
            "distribution": dict(Counter(unique_functions_distribution))
        },
        "bundle_patterns": bundle_patterns,
    }


def compute_conversation_turn_structure(
    analyses: list[SampleAnalysis],
    messages_list: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    turns_per_sample = [a.num_real_turns for a in analyses]
    if turns_per_sample:
        turns_stats = {
            **compute_basic_stats(turns_per_sample),
            **compute_percentiles(turns_per_sample, [75, 90, 99]),
            "distribution": dict(Counter(turns_per_sample)),
        }
    else:
        turns_stats = {}

    total_messages: list[int] = []
    user_messages: list[int] = []
    assistant_messages: list[int] = []
    tool_messages: list[int] = []
    for messages in messages_list:
        counts = Counter(m["role"] for m in messages)
        n_user = counts.get("user", 0)
        n_assistant = counts.get("assistant", 0)
        n_tool = counts.get("tool", 0)
        total_messages.append(n_user + n_assistant + n_tool)
        user_messages.append(n_user)
        assistant_messages.append(n_assistant)
        tool_messages.append(n_tool)

    def _message_stats(values: list[int]) -> dict[str, float | None]:
        if not values:
            return {}
        basic = compute_basic_stats(values)
        pcts = compute_percentiles(values, [75, 90])
        return {
            "mean": basic["mean"],
            "median": basic["median"],
            "percentile_75": pcts["percentile_75"],
            "percentile_90": pcts["percentile_90"],
        }

    return {
        "real_turns_per_sample": turns_stats,
        "message_breakdown": {
            "total_messages": _message_stats(total_messages),
            "user_messages": _message_stats(user_messages),
            "assistant_messages": _message_stats(assistant_messages),
            "tool_messages": _message_stats(tool_messages),
        },
    }


def compute_termination_supervision(
    messages_list: list[list[dict[str, Any]]],
    analyses: list[SampleAnalysis] | None = None,
) -> dict[str, Any]:
    turns_with_final_answer = 0
    turns_without_final_answer = 0
    tool_loop_lengths: list[int] = []

    for idx, messages in enumerate(messages_list):
        analysis = analyses[idx] if analyses is not None else None
        turns = _resolve_turns(messages, analysis, extract_function_names=False)
        for turn in turns:
            if turn.total_tool_calls() == 0:
                continue

            has_final_answer = False
            if turn.steps:
                i = turn.steps[-1].assistant_message_idx + 1
                while i < len(messages) and messages[i]["role"] == "tool":
                    i += 1
                has_final_answer = (
                    i < len(messages) and messages[i]["role"] == "assistant"
                )

            if has_final_answer:
                turns_with_final_answer += 1
            else:
                turns_without_final_answer += 1
            tool_loop_lengths.append(turn.num_steps())

    total_tool_using_turns = turns_with_final_answer + turns_without_final_answer
    loop_stats = (
        {
            **compute_basic_stats(tool_loop_lengths),
            **compute_percentiles(tool_loop_lengths, [75, 90, 95]),
            "distribution": dict(Counter(tool_loop_lengths)),
        }
        if tool_loop_lengths
        else {}
    )

    return {
        "total_tool_using_turns": total_tool_using_turns,
        "turns_with_final_answer": _count_dict(
            turns_with_final_answer, total_tool_using_turns
        ),
        "turns_without_final_answer": _count_dict(
            turns_without_final_answer, total_tool_using_turns
        ),
        "tool_loop_length": loop_stats,
    }


def compute_abstention_supervision(
    messages_list: list[list[dict[str, Any]]],
    tools_list: list[list[dict[str, Any]]],
    analyses: list[SampleAnalysis] | None = None,
) -> dict[str, Any]:
    zero_call_samples = 0
    zero_call_with_tools = 0

    for idx, (messages, tools) in enumerate(
        zip(messages_list, tools_list, strict=True)
    ):
        if analyses is not None:
            has_calls = analyses[idx].total_tool_calls > 0
        else:
            has_calls = any(
                m.get("role") == "assistant" and m.get("tool_calls") for m in messages
            )
        if has_calls:
            continue
        zero_call_samples += 1
        if tools:
            zero_call_with_tools += 1

    return {
        "zero_call_samples": {
            "total": zero_call_samples,
            "percentage": _pct(zero_call_samples, len(messages_list)),
            "with_tools_defined": zero_call_with_tools,
            "with_tools_percentage": _pct(zero_call_with_tools, zero_call_samples),
        },
    }


def _has_complete_fc_turn(
    turns: list[RealTurn], messages: list[dict[str, Any]]
) -> bool:
    for turn in turns:
        if turn.total_tool_calls() == 0 or not turn.steps:
            continue
        i = turn.steps[-1].assistant_message_idx + 1
        num_responses = 0
        while i < len(messages) and messages[i]["role"] == "tool":
            num_responses += 1
            i += 1
        if (
            num_responses > 0
            and i < len(messages)
            and messages[i]["role"] == "assistant"
        ):
            return True
    return False


def compute_fc_coverage(
    analyses: list[SampleAnalysis],
    messages_list: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    total = len(analyses)
    with_calls = 0
    complete = 0
    for analysis, messages in zip(analyses, messages_list, strict=True):
        if analysis.total_tool_calls > 0:
            with_calls += 1
        turns = _resolve_turns(messages, analysis, extract_function_names=False)
        if _has_complete_fc_turn(turns, messages):
            complete += 1
    return {
        "samples_with_tool_calls": _count_dict(with_calls, total),
        "complete_fc_samples": _count_dict(complete, total),
    }


def _collect_tool_schemas(
    tools: list[dict[str, Any]],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    defined: set[str] = set()
    schema_map: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name")
        if name is None:
            continue
        defined.add(name)
        params = func.get("parameters")
        if isinstance(params, dict):
            schema_map[name] = params
    return defined, schema_map


def compute_tool_call_validation(
    messages_list: list[list[dict[str, Any]]],
    tools_list: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    undefined_calls = 0
    undefined_call_samples = 0
    hallucinated_names: set[str] = set()
    validated_calls = 0
    calls_with_violations = 0
    violation_samples = 0
    violation_counts: Counter[str] = Counter()
    any_violation_samples = 0

    for messages, tools in zip(messages_list, tools_list, strict=True):
        defined, schema_map = _collect_tool_schemas(tools)
        sample_has_undefined = False
        sample_has_violation = False

        for msg in messages:
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                if fn_name not in defined:
                    undefined_calls += 1
                    hallucinated_names.add(fn_name)
                    sample_has_undefined = True
                    continue
                if fn_name not in schema_map:
                    continue

                args_raw = tc["function"].get("arguments", "")
                if isinstance(args_raw, str):
                    try:
                        parsed = orjson.loads(args_raw)
                    except ValueError:
                        # json.JSONDecodeError is a ValueError subclass; one
                        # except clause covers both.
                        validated_calls += 1
                        calls_with_violations += 1
                        violation_counts["unparseable_arguments"] += 1
                        sample_has_violation = True
                        continue
                else:
                    parsed = args_raw

                validated_calls += 1
                if not isinstance(parsed, dict):
                    calls_with_violations += 1
                    violation_counts["arguments_not_object"] += 1
                    sample_has_violation = True
                    continue

                violations = validate_arguments(parsed, schema_map[fn_name])
                if violations:
                    calls_with_violations += 1
                    sample_has_violation = True
                    for v in violations:
                        violation_counts[v] += 1

        if sample_has_undefined:
            undefined_call_samples += 1
        if sample_has_violation:
            violation_samples += 1
        if sample_has_undefined or sample_has_violation:
            any_violation_samples += 1

    n = len(messages_list)
    return {
        "undefined_function_calls": {
            "total_calls": undefined_calls,
            "affected_samples": undefined_call_samples,
            "percentage_of_samples": _pct(undefined_call_samples, n),
            "hallucinated_names": sorted(hallucinated_names),
            "unique_hallucinated_count": len(hallucinated_names),
        },
        "argument_validation": {
            "total_calls_validated": validated_calls,
            "calls_with_violations": calls_with_violations,
            "affected_samples": violation_samples,
            "percentage_of_samples": _pct(violation_samples, n),
            "violation_breakdown": dict(violation_counts),
        },
        "samples_with_any_violation": any_violation_samples,
        "percentage_with_any_violation": _pct(any_violation_samples, n),
    }


def aggregate_statistics(
    analyses: list[SampleAnalysis],
    messages_list: list[list[dict[str, Any]]],
    tools_list: list[list[dict[str, Any]]],
    token_lengths: list[int] | None = None,
    dataset_overview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every per-aspect `compute_*` and return the nested dict consumed by `reporter.print_full_report`."""
    stats: dict[str, Any] = {"total_samples": len(analyses)}
    if dataset_overview:
        stats["dataset_overview"] = dataset_overview
    if token_lengths:
        stats["token_length_distribution"] = compute_token_length_statistics(
            token_lengths
        )
    stats["fc_coverage"] = compute_fc_coverage(analyses, messages_list)
    stats["function_diversity"] = compute_function_diversity(messages_list)
    stats["tool_call_validation"] = compute_tool_call_validation(
        messages_list, tools_list
    )
    stats["function_calling_patterns"] = compute_function_calling_patterns(analyses)
    stats["parallel_function_diversity"] = compute_parallel_function_diversity(
        messages_list, analyses
    )
    stats["conversation_turn_structure"] = compute_conversation_turn_structure(
        analyses, messages_list
    )
    stats["termination_supervision"] = compute_termination_supervision(
        messages_list, analyses
    )
    stats["abstention_supervision"] = compute_abstention_supervision(
        messages_list, tools_list, analyses
    )
    return stats


def compute_single_vs_multi_turn_distribution(
    analyses: list[SampleAnalysis],
) -> dict[str, Any]:
    total = len(analyses)
    n_single = sum(1 for a in analyses if a.num_real_turns == 1)
    n_multi = sum(1 for a in analyses if a.num_real_turns > 1)
    return {
        "total_samples": total,
        "single_turn_samples": _count_dict(n_single, total),
        "multi_turn_samples": _count_dict(n_multi, total),
    }


def compute_multi_turn_distribution(
    analyses: list[SampleAnalysis],
) -> dict[str, Any]:
    multi_turn = [a for a in analyses if a.num_real_turns > 1]
    if not multi_turn:
        return {
            "total_multi_turn_samples": 0,
            "turns_per_sample": {},
            "distribution": {},
        }
    turns_per_sample = [a.num_real_turns for a in multi_turn]
    return {
        "total_multi_turn_samples": len(multi_turn),
        "turns_per_sample": {
            **compute_basic_stats(turns_per_sample),
            **compute_percentiles(turns_per_sample, [75, 90, 95]),
        },
        "distribution": dict(Counter(turns_per_sample)),
    }


def compute_pattern_distribution_single_turn(
    analyses: list[SampleAnalysis],
    *,
    single_turn: list[SampleAnalysis] | None = None,
) -> dict[str, Any]:
    if single_turn is None:
        single_turn, _ = _partition_by_turn_count(analyses)
    total = len(single_turn)
    if total == 0:
        return {"total_single_turn_samples": 0}

    def _count(attr: str) -> dict[str, int | float]:
        return _count_dict(sum(1 for a in single_turn if getattr(a, attr) == 1), total)

    return {
        "total_single_turn_samples": total,
        "no_calls_samples": _count("num_no_call_turns"),
        "single_call_samples": _count("num_single_call_turns"),
        "parallel_samples": _count("num_parallel_turns"),
        "sequential_samples": _count("num_sequential_turns"),
        "hybrid_samples": _count("num_hybrid_turns"),
    }


def compute_pattern_distribution_multi_turn(
    analyses: list[SampleAnalysis],
    *,
    multi_turn: list[SampleAnalysis] | None = None,
) -> dict[str, Any]:
    if multi_turn is None:
        _, multi_turn = _partition_by_turn_count(analyses)
    total = len(multi_turn)
    if total == 0:
        return {"total_multi_turn_samples": 0}

    def _containing(attr: str) -> dict[str, int | float]:
        return _count_dict(sum(1 for a in multi_turn if getattr(a, attr) > 0), total)

    return {
        "total_multi_turn_samples": total,
        "samples_containing_no_calls_turn": _containing("num_no_call_turns"),
        "samples_containing_single_call_turn": _containing("num_single_call_turns"),
        "samples_containing_sequential_turn": _containing("num_sequential_turns"),
        "samples_containing_parallel_turn": _containing("num_parallel_turns"),
        "samples_containing_hybrid_turn": _containing("num_hybrid_turns"),
    }


def compute_turn_level_statistics_by_sample_type(
    analyses: list[SampleAnalysis],
    *,
    single_turn: list[SampleAnalysis] | None = None,
    multi_turn: list[SampleAnalysis] | None = None,
) -> dict[str, Any]:
    if single_turn is None or multi_turn is None:
        single_turn, multi_turn = _partition_by_turn_count(analyses)
    single_patterns = [p for a in single_turn for p in a.turn_patterns]
    multi_patterns = [p for a in multi_turn for p in a.turn_patterns]
    all_patterns = [p for a in analyses for p in a.turn_patterns]

    def _pattern_counts(patterns: list[TurnPattern]) -> dict[str, Any]:
        total = len(patterns)
        c = Counter(patterns)
        return {
            "total_turns": total,
            "no_calls_turns": _count_dict(c[TurnPattern.NO_CALLS], total),
            "single_call_turns": _count_dict(c[TurnPattern.SINGLE_CALL], total),
            "sequential_turns": _count_dict(c[TurnPattern.SEQUENTIAL], total),
            "parallel_turns": _count_dict(c[TurnPattern.PARALLEL], total),
            "hybrid_turns": _count_dict(c[TurnPattern.HYBRID], total),
        }

    return {
        "single_turn_samples_turns": _pattern_counts(single_patterns),
        "multi_turn_samples_turns": _pattern_counts(multi_patterns),
        "all_samples_turns": _pattern_counts(all_patterns),
    }


def _analyze_parallel_diversity(
    indices: list[int],
    analyses: list[SampleAnalysis],
    messages_list: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    same_function_count = 0
    different_function_count = 0
    parallel_call_counts: list[int] = []
    unique_function_counts: list[int] = []

    for idx in indices:
        turns = _resolve_turns(
            messages_list[idx], analyses[idx], extract_function_names=True
        )
        for turn in turns:
            for step in turn.steps:
                if step.num_tool_calls < 2:
                    continue
                parallel_call_counts.append(step.num_tool_calls)
                if not step.function_names:
                    continue
                unique = set(step.function_names)
                unique_function_counts.append(len(unique))
                if len(unique) == 1:
                    same_function_count += 1
                else:
                    different_function_count += 1

    total = same_function_count + different_function_count
    result: dict[str, Any] = {
        "total_parallel_steps": total,
        "same_function_parallel_steps": _count_dict(same_function_count, total),
        "different_function_parallel_steps": _count_dict(
            different_function_count, total
        ),
    }
    if parallel_call_counts:
        result["calls_per_parallel_step"] = {
            **compute_basic_stats(parallel_call_counts),
            "distribution": dict(Counter(parallel_call_counts)),
        }
    if unique_function_counts:
        result["unique_functions_per_parallel_step"] = {
            **compute_basic_stats(unique_function_counts),
            "distribution": dict(Counter(unique_function_counts)),
        }
    return result


def compute_parallel_function_diversity_by_sample_type(
    analyses: list[SampleAnalysis],
    messages_list: list[list[dict[str, Any]]],
    *,
    single_idx: list[int] | None = None,
    multi_idx: list[int] | None = None,
) -> dict[str, Any]:
    if single_idx is None:
        single_idx = [i for i, a in enumerate(analyses) if a.num_real_turns == 1]
    if multi_idx is None:
        multi_idx = [i for i, a in enumerate(analyses) if a.num_real_turns > 1]
    return {
        "all_samples": _analyze_parallel_diversity(
            list(range(len(analyses))), analyses, messages_list
        ),
        "single_turn_samples": _analyze_parallel_diversity(
            single_idx, analyses, messages_list
        ),
        "multi_turn_samples": _analyze_parallel_diversity(
            multi_idx, analyses, messages_list
        ),
    }


def _unique_functions_for_indices(
    indices: list[int],
    messages_list: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    unique_per_sample: list[int] = []
    for idx in indices:
        sample_functions: set[str] = set()
        for msg in messages_list[idx]:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    sample_functions.add(tc["function"]["name"])
        unique_per_sample.append(len(sample_functions))
    if not unique_per_sample:
        return {}
    return {
        **compute_basic_stats(unique_per_sample),
        **compute_percentiles(unique_per_sample, [75, 90, 95]),
        "distribution": dict(Counter(unique_per_sample)),
    }


def compute_unique_functions_per_sample_by_type(
    analyses: list[SampleAnalysis],
    messages_list: list[list[dict[str, Any]]],
    *,
    single_idx: list[int] | None = None,
    multi_idx: list[int] | None = None,
) -> dict[str, Any]:
    if single_idx is None:
        single_idx = [i for i, a in enumerate(analyses) if a.num_real_turns == 1]
    if multi_idx is None:
        multi_idx = [i for i, a in enumerate(analyses) if a.num_real_turns > 1]
    return {
        "all_samples": _unique_functions_for_indices(
            list(range(len(analyses))), messages_list
        ),
        "single_turn_samples": _unique_functions_for_indices(single_idx, messages_list),
        "multi_turn_samples": _unique_functions_for_indices(multi_idx, messages_list),
    }


def compute_hybrid_turn_analysis(
    analyses: list[SampleAnalysis],
    messages_list: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    parallel_steps_per_hybrid_turn: list[int] = []
    for idx, analysis in enumerate(analyses):
        if analysis.num_hybrid_turns == 0:
            continue
        turns = _resolve_turns(
            messages_list[idx], analysis, extract_function_names=True
        )
        for turn in turns:
            if turn.pattern == TurnPattern.HYBRID:
                parallel_steps = sum(
                    1 for step in turn.steps if step.num_tool_calls >= 2
                )
                parallel_steps_per_hybrid_turn.append(parallel_steps)
    if not parallel_steps_per_hybrid_turn:
        return {"total_hybrid_turns": 0}
    return {
        "total_hybrid_turns": len(parallel_steps_per_hybrid_turn),
        "parallel_steps_per_hybrid_turn": {
            **compute_basic_stats(parallel_steps_per_hybrid_turn),
            "distribution": dict(Counter(parallel_steps_per_hybrid_turn)),
        },
    }


def aggregate_enhanced_statistics(
    analyses: list[SampleAnalysis],
    messages_list: list[list[dict[str, Any]]],
    token_lengths: list[int] | None = None,
    dataset_overview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extended cuts (per-sample-type breakdowns, hybrid-turn analysis) that complement `aggregate_statistics`; merge both dicts to get the full picture."""
    stats: dict[str, Any] = {}
    if dataset_overview:
        stats["dataset_overview"] = dataset_overview

    # Partition once and thread through to consumers that would otherwise
    # recompute it. On million-sample inputs this avoids 3 redundant passes.
    single_turn, multi_turn = _partition_by_turn_count(analyses)
    single_idx = [i for i, a in enumerate(analyses) if a.num_real_turns == 1]
    multi_idx = [i for i, a in enumerate(analyses) if a.num_real_turns > 1]

    stats["single_vs_multi_turn"] = compute_single_vs_multi_turn_distribution(analyses)
    stats["multi_turn_distribution"] = compute_multi_turn_distribution(analyses)
    stats["pattern_distribution_single_turn"] = (
        compute_pattern_distribution_single_turn(analyses, single_turn=single_turn)
    )
    stats["pattern_distribution_multi_turn"] = compute_pattern_distribution_multi_turn(
        analyses, multi_turn=multi_turn
    )
    stats["turn_level_by_sample_type"] = compute_turn_level_statistics_by_sample_type(
        analyses, single_turn=single_turn, multi_turn=multi_turn
    )
    stats["parallel_function_diversity_by_type"] = (
        compute_parallel_function_diversity_by_sample_type(
            analyses, messages_list, single_idx=single_idx, multi_idx=multi_idx
        )
    )
    stats["unique_functions_per_sample_by_type"] = (
        compute_unique_functions_per_sample_by_type(
            analyses, messages_list, single_idx=single_idx, multi_idx=multi_idx
        )
    )
    stats["hybrid_turn_analysis"] = compute_hybrid_turn_analysis(
        analyses, messages_list
    )

    if token_lengths:
        stats["token_length_statistics"] = {
            **compute_basic_stats(token_lengths, include_std=True),
            **compute_percentiles(token_lengths, [25, 75, 90, 95, 99]),
        }

    all_names = [
        tc["function"]["name"]
        for messages in messages_list
        for msg in messages
        if msg.get("role") == "assistant" and msg.get("tool_calls")
        for tc in msg["tool_calls"]
    ]
    function_counter = Counter(all_names)
    stats["total_unique_functions"] = len(function_counter)
    stats["top_20_functions"] = function_counter.most_common(20)
    return stats
