import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TextIO

import numpy as np

from .behavioral import analyze_dataset_behavior, compute_bias_report
from .core import analyze_sample
from .statistics import (
    compute_fc_coverage,
    compute_function_calling_patterns,
    compute_function_diversity,
    compute_parallel_function_diversity,
    compute_single_vs_multi_turn_distribution,
    compute_termination_supervision,
    compute_tool_call_validation,
)


def _fmt_count_pct(count: int, pct: float, decimals: int = 1) -> str:
    return f"{count:,} ({pct:.{decimals}f}%)"


class _MarkdownBuilder:
    """Accumulator for markdown lines with table/section helpers."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def section(self, title: str) -> None:
        self._lines.append("")
        self._lines.append(f"### {title}")
        self._lines.append("")

    def line(self, text: str = "") -> None:
        self._lines.append(text)

    def kv_header(self) -> None:
        self._lines.append("| Metric | Value |")
        self._lines.append("|--------|-------|")

    def kv_row(self, metric: str, value: str) -> None:
        self._lines.append(f"| {metric} | {value} |")

    def table(self, header: list[str], rows: Iterable[list[str]]) -> None:
        self._lines.append("| " + " | ".join(header) + " |")
        self._lines.append("|" + "|".join("---" for _ in header) + "|")
        for row in rows:
            self._lines.append("| " + " | ".join(row) + " |")

    def render(self) -> str:
        return "\n".join(self._lines)


def print_dataset_overview(stats: dict[str, Any], file: TextIO | None = None) -> None:
    out = file or sys.stdout
    if "dataset_overview" not in stats:
        return

    overview = stats["dataset_overview"]
    print("DATASET OVERVIEW", file=out)
    print(
        f"Total samples (unfiltered): {overview.get('total_unfiltered', 'N/A'):,}",
        file=out,
    )
    print(
        f"Samples filtered out: {overview.get('samples_filtered_out', 0):,} "
        f"({overview.get('filter_percentage', 0):.2f}%)",
        file=out,
    )
    print(
        f"Final dataset size: "
        f"{overview.get('total_filtered', stats['total_samples']):,} samples",
        file=out,
    )
    if "filter_description" in overview:
        print(f"Filter criteria: {overview['filter_description']}", file=out)
    print(file=out)


def print_token_length_distribution(
    stats: dict[str, Any], file: TextIO | None = None
) -> None:
    out = file or sys.stdout
    if "token_length_distribution" not in stats:
        return

    tld = stats["token_length_distribution"]
    print("TOKEN LENGTH DISTRIBUTION", file=out)
    print(f"Mean: {tld['mean']:.0f} tokens", file=out)
    print(f"Median: {tld['median']} tokens", file=out)
    print(f"Std Dev: {tld['std_dev']:.0f} tokens", file=out)
    print(f"Min: {tld['min']} tokens", file=out)
    print(f"Max: {tld['max']} tokens", file=out)
    for p in (25, 75, 90, 95, 99):
        print(f"{p}th percentile: {tld[f'percentile_{p}']} tokens", file=out)
    print(file=out)


def print_function_diversity(stats: dict[str, Any], file: TextIO | None = None) -> None:
    out = file or sys.stdout
    if "function_diversity" not in stats:
        return

    fd = stats["function_diversity"]
    print("FUNCTION DIVERSITY", file=out)
    print(f"Total unique functions: {fd['total_unique_functions']:,}", file=out)
    print(file=out)

    print("Top 20 Most Frequent Functions:", file=out)
    for i, (func_name, count) in enumerate(fd["top_20_functions"], 1):
        print(f"  {i:2d}. {func_name}: {count:,} calls", file=out)
    print(file=out)

    ups = fd["unique_per_sample"]
    print("Unique Functions Per Sample:", file=out)
    print(f"  Mean: {ups['mean']:.2f} functions", file=out)
    print(f"  Median: {ups['median']:.0f} functions", file=out)
    print(f"  Min: {ups['min']} functions", file=out)
    print(f"  Max: {ups['max']} functions", file=out)
    for p in (25, 75, 90, 95, 99):
        print(f"  {p}th percentile: {ups[f'percentile_{p}']} functions", file=out)
    print(file=out)

    print("Distribution Breakdown:", file=out)
    dist = ups["distribution"]
    total_samples = sum(dist.values())
    for num_funcs in sorted(dist.keys())[:10]:
        count = dist[num_funcs]
        pct = count / total_samples * 100
        print(
            f"  {num_funcs} functions: {count:,} samples ({pct:.2f}%)",
            file=out,
        )
    print(file=out)


def print_function_calling_patterns(
    stats: dict[str, Any], file: TextIO | None = None
) -> None:
    out = file or sys.stdout
    if "function_calling_patterns" not in stats:
        return

    fcp = stats["function_calling_patterns"]
    print("FUNCTION CALLING PATTERNS", file=out)

    if "total_calls_per_sample" in fcp and fcp["total_calls_per_sample"]:
        tcps = fcp["total_calls_per_sample"]
        print("Total Function Calls Per Sample:", file=out)
        print(f"  Mean: {tcps['mean']:.2f} calls", file=out)
        print(f"  Median: {tcps['median']:.0f} calls", file=out)
        print(f"  Min: {tcps['min']} calls", file=out)
        print(f"  Max: {tcps['max']} calls", file=out)
        for p in (25, 75, 90, 95, 99):
            print(f"  {p}th percentile: {tcps[f'percentile_{p}']} calls", file=out)
        print(file=out)

    tl = fcp["turn_level"]
    print("Turn-Level Pattern Distribution:", file=out)
    print(f"  Total turns across all samples: {tl['total_turns']:,}", file=out)
    for pattern in ("single_call", "parallel", "sequential", "hybrid", "no_calls"):
        entry = tl[pattern]
        print(
            f"  {pattern.replace('_', ' ').title()} turns: "
            f"{entry['count']:,} ({entry['percentage']:.2f}%)",
            file=out,
        )
    print(file=out)

    sl = fcp["sample_level"]
    print("Sample-Level Pattern Classification:", file=out)
    print("  Samples with ONLY one pattern (mutually exclusive):", file=out)
    for pattern in ("single_call", "parallel", "sequential", "hybrid", "no_call"):
        key = f"samples_only_{pattern}"
        if key in sl:
            entry = sl[key]
            print(
                f"    Only {pattern}: {entry['count']:,} ({entry['percentage']:.2f}%)",
                file=out,
            )
    if "samples_mixed_patterns" in sl:
        entry = sl["samples_mixed_patterns"]
        print(
            f"    Mixed patterns: {entry['count']:,} ({entry['percentage']:.2f}%)",
            file=out,
        )
    print(file=out)

    if "max_parallel_calls" in fcp and fcp["max_parallel_calls"]:
        mpc = fcp["max_parallel_calls"]
        print("Max Parallel Calls in Any Single Turn:", file=out)
        print(f"  Mean: {mpc['mean']:.2f} calls", file=out)
        print(f"  Median: {mpc['median']:.0f} calls", file=out)
        print(f"  Min: {mpc['min']} calls", file=out)
        print(f"  Max: {mpc['max']} calls", file=out)
        for p in (75, 90, 99):
            print(f"  {p}th percentile: {mpc[f'percentile_{p}']} calls", file=out)
        print(file=out)

    if "turns_with_function_calls" in fcp and fcp["turns_with_function_calls"]:
        twfc = fcp["turns_with_function_calls"]
        print(
            "Number of Turns with Function Calls (among samples with calls):",
            file=out,
        )
        print(f"  Mean: {twfc['mean']:.2f} turns", file=out)
        print(f"  Median: {twfc['median']:.0f} turns", file=out)
        print(f"  Min: {twfc['min']} turns", file=out)
        print(f"  Max: {twfc['max']} turns", file=out)
        for p in (75, 90, 99):
            print(f"  {p}th percentile: {twfc[f'percentile_{p}']} turns", file=out)
        print(file=out)


def print_parallel_function_diversity(
    stats: dict[str, Any], file: TextIO | None = None
) -> None:
    out = file or sys.stdout
    if "parallel_function_diversity" not in stats:
        return

    pfd = stats["parallel_function_diversity"]
    print("PARALLEL FUNCTION DIVERSITY", file=out)
    print(f"Total parallel steps: {pfd['total_parallel_steps']:,}", file=out)
    print(file=out)

    print("Main Results:", file=out)
    print(
        f"  Same-function fanout: {pfd['same_function_fanout']['count']:,} "
        f"({pfd['same_function_fanout']['percentage']:.2f}%)",
        file=out,
    )
    print(
        f"  Multi-function bundles: {pfd['multi_function_bundles']['count']:,} "
        f"({pfd['multi_function_bundles']['percentage']:.2f}%)",
        file=out,
    )
    print(file=out)

    if "unique_functions_per_step" in pfd:
        dist = pfd["unique_functions_per_step"]["distribution"]
        print("Distribution of Unique Functions per Parallel Step:", file=out)
        total_steps = sum(dist.values())
        for num_funcs in sorted(dist.keys())[:10]:
            count = dist[num_funcs]
            pct = count / total_steps * 100
            interpretation = (
                "Same-function fanout"
                if num_funcs == 1
                else f"{num_funcs} different functions"
            )
            print(
                f"  {num_funcs} unique: {count:,} steps ({pct:.2f}%) "
                f"({interpretation})",
                file=out,
            )
        print(file=out)

    if "bundle_patterns" in pfd and pfd["bundle_patterns"]["total_bundles"] > 0:
        bp = pfd["bundle_patterns"]
        print("Pure vs Hybrid Multi-Function Bundles:", file=out)
        print(f"  Total bundles: {bp['total_bundles']:,}", file=out)
        print(
            f"  Pure bundles (each function once): "
            f"{bp['pure_bundles']['count']:,} ({bp['pure_bundles']['percentage']:.2f}%)",
            file=out,
        )
        print(
            f"  Hybrid bundles (at least one with fanout): "
            f"{bp['hybrid_bundles']['count']:,} "
            f"({bp['hybrid_bundles']['percentage']:.2f}%)",
            file=out,
        )
        print(file=out)

        if bp["hybrid_bundles"]["count"] > 0:
            hb = bp["hybrid_bundles"]
            total_hybrid = bp["hybrid_bundles"]["count"]
            print("Hybrid Bundle Characteristics:", file=out)

            print("  Max fanout distribution:", file=out)
            for fanout in sorted(hb["max_fanout_distribution"].keys())[:5]:
                count = hb["max_fanout_distribution"][fanout]
                pct = count / total_hybrid * 100
                print(f"    Max fanout {fanout}: {count:,} ({pct:.2f}%)", file=out)

            print("  Functions with fanout:", file=out)
            for num_funcs in sorted(hb["functions_with_fanout_distribution"].keys())[
                :5
            ]:
                count = hb["functions_with_fanout_distribution"][num_funcs]
                pct = count / total_hybrid * 100
                print(f"    {num_funcs} functions: {count:,} ({pct:.2f}%)", file=out)
            print(file=out)


def print_conversation_turn_structure(
    stats: dict[str, Any], file: TextIO | None = None
) -> None:
    out = file or sys.stdout
    if "conversation_turn_structure" not in stats:
        return

    cts = stats["conversation_turn_structure"]
    print("CONVERSATION TURN STRUCTURE", file=out)

    if "real_turns_per_sample" in cts and cts["real_turns_per_sample"]:
        rtps = cts["real_turns_per_sample"]
        print("Real Turns Per Sample:", file=out)
        print(f"  Mean: {rtps['mean']:.2f} turns", file=out)
        print(f"  Median: {rtps['median']:.0f} turns", file=out)
        print(f"  Min: {rtps['min']} turns", file=out)
        print(f"  Max: {rtps['max']} turns", file=out)
        for p in (75, 90, 99):
            print(f"  {p}th percentile: {rtps[f'percentile_{p}']} turns", file=out)
        print(file=out)

        if "distribution" in rtps:
            print("Distribution of Real Turns:", file=out)
            dist = rtps["distribution"]
            total_samples = sum(dist.values())
            for num_turns in sorted(dist.keys())[:10]:
                count = dist[num_turns]
                pct = count / total_samples * 100
                print(
                    f"  {num_turns} turns: {count:,} samples ({pct:.2f}%)",
                    file=out,
                )
            print(file=out)

    if "message_breakdown" in cts:
        mb = cts["message_breakdown"]
        print("Message Breakdown Per Sample:", file=out)
        for key, label in [
            ("total_messages", "Total messages"),
            ("user_messages", "User messages"),
            ("assistant_messages", "Assistant messages"),
            ("tool_messages", "Tool messages"),
        ]:
            if key in mb and mb[key]:
                entry = mb[key]
                print(
                    f"  {label}: Mean {entry['mean']:.1f}, "
                    f"Median {entry['median']:.0f}",
                    file=out,
                )
        print(file=out)


def print_termination_supervision(
    stats: dict[str, Any], file: TextIO | None = None
) -> None:
    out = file or sys.stdout
    if "termination_supervision" not in stats:
        return

    ts = stats["termination_supervision"]
    print("TERMINATION SUPERVISION QUALITY", file=out)

    print("Final Answer After Tool Use:", file=out)
    print(f"  Total tool-using turns: {ts['total_tool_using_turns']:,}", file=out)
    print(
        f"  Turns with final answer: "
        f"{ts['turns_with_final_answer']['count']:,} "
        f"({ts['turns_with_final_answer']['percentage']:.2f}%)",
        file=out,
    )
    print(
        f"  Turns without final answer: "
        f"{ts['turns_without_final_answer']['count']:,} "
        f"({ts['turns_without_final_answer']['percentage']:.2f}%)",
        file=out,
    )
    print(file=out)

    if "tool_loop_length" in ts and ts["tool_loop_length"]:
        tll = ts["tool_loop_length"]
        print("Tool Loop Length Distribution:", file=out)
        print(f"  Mean: {tll['mean']:.2f} cycles", file=out)
        print(f"  Median: {tll['median']:.0f} cycles", file=out)
        print(f"  Min: {tll['min']} cycles", file=out)
        print(f"  Max: {tll['max']} cycles", file=out)
        for p in (75, 90, 95):
            print(f"  {p}th percentile: {tll[f'percentile_{p}']} cycles", file=out)
        print(file=out)

        if "distribution" in tll:
            print("Distribution:", file=out)
            dist = tll["distribution"]
            total = sum(dist.values())
            for cycles in sorted(dist.keys())[:10]:
                count = dist[cycles]
                pct = count / total * 100
                print(
                    f"  {cycles} cycles: {count:,} turns ({pct:.2f}%)",
                    file=out,
                )
            print(file=out)


def print_abstention_supervision(
    stats: dict[str, Any], file: TextIO | None = None
) -> None:
    out = file or sys.stdout
    if "abstention_supervision" not in stats:
        return

    abs_sup = stats["abstention_supervision"]
    print("ABSTENTION / IRRELEVANCE SUPERVISION QUALITY", file=out)

    zcs = abs_sup["zero_call_samples"]
    print("Zero Function Call Samples:", file=out)
    print(
        f"  Total: {zcs['total']:,} ({zcs['percentage']:.2f}% of dataset)",
        file=out,
    )
    print(
        f"  With tools defined: {zcs['with_tools_defined']:,} "
        f"({zcs['with_tools_percentage']:.2f}%)",
        file=out,
    )
    print(file=out)


def print_tool_call_validation(
    stats: dict[str, Any], file: TextIO | None = None
) -> None:
    out = file or sys.stdout
    if "tool_call_validation" not in stats:
        return

    tcv = stats["tool_call_validation"]
    print("TOOL CALL VALIDATION", file=out)

    ufc = tcv["undefined_function_calls"]
    print("Undefined Function Calls:", file=out)
    print(f"  Total: {ufc['total_calls']:,}", file=out)
    print(
        f"  Affected samples: {ufc['affected_samples']:,} "
        f"({ufc['percentage_of_samples']:.2f}%)",
        file=out,
    )
    print(
        f"  Unique hallucinated names: {ufc['unique_hallucinated_count']:,}", file=out
    )
    if ufc["hallucinated_names"]:
        names = ufc["hallucinated_names"][:20]
        print(f"  Names: {', '.join(names)}", file=out)
        if ufc["unique_hallucinated_count"] > 20:
            print(f"  ... and {ufc['unique_hallucinated_count'] - 20} more", file=out)
    print(file=out)

    av = tcv["argument_validation"]
    print("Argument Schema Violations:", file=out)
    print(f"  Calls validated: {av['total_calls_validated']:,}", file=out)
    print(f"  Calls with violations: {av['calls_with_violations']:,}", file=out)
    print(
        f"  Affected samples: {av['affected_samples']:,} "
        f"({av['percentage_of_samples']:.2f}%)",
        file=out,
    )
    if av["violation_breakdown"]:
        print("  Violation breakdown:", file=out)
        for kind, count in sorted(av["violation_breakdown"].items()):
            print(f"    {kind}: {count:,}", file=out)
    print(file=out)

    print(
        f"Samples with any validation issue: "
        f"{tcv['samples_with_any_violation']:,} "
        f"({tcv['percentage_with_any_violation']:.2f}%)",
        file=out,
    )
    print(file=out)


type _SectionFn = Callable[[dict[str, Any], TextIO | None], None]


_REPORT_SECTIONS: tuple[_SectionFn, ...] = (
    print_dataset_overview,
    print_token_length_distribution,
    print_function_diversity,
    print_tool_call_validation,
    print_function_calling_patterns,
    print_parallel_function_diversity,
    print_conversation_turn_structure,
    print_termination_supervision,
    print_abstention_supervision,
)


def print_full_report(stats: dict[str, Any], file: TextIO | None = None) -> None:
    out = file or sys.stdout
    print("FUNCTION CALLING DATASET ANALYSIS", file=out)
    print(file=out)
    for render in _REPORT_SECTIONS:
        render(stats, out)


@dataclass(slots=True)
class _MetricsContext:
    """Precomputed values consumed by the markdown section renderers."""

    report: Any
    bias: Any
    n: int
    fc: dict[str, Any]
    fcp: dict[str, Any]
    fd: dict[str, Any]
    pfd: dict[str, Any]
    ts: dict[str, Any]
    tcv: dict[str, Any]
    svmt: dict[str, Any]
    total_calls: int
    calls_arr: np.ndarray
    tools_arr: np.ndarray
    fn_defined: set[str]
    fn_called: set[str]
    fc_lengths: list[int]
    no_fc_lengths: list[int]
    sys_msg_count: int


def _prepare_metrics_context(
    samples: list[Any],
    report: Any,
    *,
    analyses: list[Any] | None,
    beh_analyses: list[Any] | None,
    bias: Any | None,
    stats: dict[str, Any] | None,
) -> _MetricsContext:
    messages_list = [s.messages for s in samples]
    tools_list = [s.tools for s in samples]

    if analyses is None:
        analyses = [
            analyze_sample(m, extract_function_names=True) for m in messages_list
        ]
    if beh_analyses is None:
        beh_analyses = analyze_dataset_behavior(samples, analyses)
    if bias is None:
        bias = compute_bias_report(beh_analyses)

    if stats is not None:
        fc = stats["fc_coverage"]
        fcp = stats["function_calling_patterns"]
        fd = stats["function_diversity"]
        pfd = stats["parallel_function_diversity"]
        ts = stats["termination_supervision"]
        tcv = stats["tool_call_validation"]
    else:
        fc = compute_fc_coverage(analyses, messages_list)
        fcp = compute_function_calling_patterns(analyses)
        fd = compute_function_diversity(messages_list)
        pfd = compute_parallel_function_diversity(messages_list, analyses)
        ts = compute_termination_supervision(messages_list, analyses)
        tcv = compute_tool_call_validation(messages_list, tools_list)

    fn_defined: set[str] = set()
    fn_called: set[str] = set()
    for t_list in tools_list:
        for t in t_list:
            fn_defined.add(t["function"]["name"])
    for msgs in messages_list:
        for m in msgs:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn_called.add(tc["function"]["name"])

    sys_msg_count = sum(
        1 for msgs in messages_list if any(m["role"] == "system" for m in msgs)
    )

    return _MetricsContext(
        report=report,
        bias=bias,
        n=len(samples),
        fc=fc,
        fcp=fcp,
        fd=fd,
        pfd=pfd,
        ts=ts,
        tcv=tcv,
        svmt=compute_single_vs_multi_turn_distribution(analyses),
        total_calls=sum(a.total_tool_calls for a in analyses),
        calls_arr=np.array([a.total_tool_calls for a in analyses], dtype=np.float64),
        tools_arr=np.array([len(t) for t in tools_list], dtype=np.float64),
        fn_defined=fn_defined,
        fn_called=fn_called,
        fc_lengths=bias.fc_turn_user_msg_lengths,
        no_fc_lengths=bias.no_fc_turn_user_msg_lengths,
        sys_msg_count=sys_msg_count,
    )


def _render_summary(md: _MarkdownBuilder, ctx: _MetricsContext) -> None:
    md.section("Summary")
    md.kv_header()

    report = ctx.report
    md.kv_row("Total raw samples", f"{report.raw_count:,}")
    md.kv_row("Stage 1 converted", f"{report.stage1_count:,}")
    if report.dataset_config_count is not None:
        md.kv_row("After dataset-specific config", f"{report.dataset_config_count:,}")
    if report.filtered_count is not None:
        md.kv_row("After universal filters", f"{report.filtered_count:,}")

    wc = ctx.fc["samples_with_tool_calls"]
    md.kv_row("Samples with tool calls", _fmt_count_pct(wc["count"], wc["percentage"]))
    cfc = ctx.fc["complete_fc_samples"]
    md.kv_row(
        "Complete FC samples (call + response + answer)",
        _fmt_count_pct(cfc["count"], cfc["percentage"]),
    )
    no_tc_count = ctx.n - wc["count"]
    no_tc_pct = no_tc_count / ctx.n * 100 if ctx.n else 0
    md.kv_row("No-tool-call samples", _fmt_count_pct(no_tc_count, no_tc_pct))

    st = ctx.svmt["single_turn_samples"]
    mt = ctx.svmt["multi_turn_samples"]
    md.kv_row("Single-turn", _fmt_count_pct(st["count"], st["percentage"]))
    md.kv_row("Multi-turn", _fmt_count_pct(mt["count"], mt["percentage"]))

    pat = ctx.bias.pattern_counts
    md.kv_row(
        "Sample patterns: normal_fc / never_call / mixed",
        " / ".join(f"{pat.get(p, 0):,}" for p in ("normal_fc", "never_call", "mixed")),
    )

    md.kv_row("Total tool calls", f"{ctx.total_calls:,}")
    md.kv_row(
        "Calls per sample: mean / median / max",
        f"{np.mean(ctx.calls_arr):.2f} / {np.median(ctx.calls_arr):.0f} "
        f"/ {np.max(ctx.calls_arr):.0f}",
    )

    tl = ctx.fcp["turn_level"]
    md.kv_row(
        "Turn-level: single / parallel / sequential / hybrid",
        " / ".join(
            f"{tl[p]['count']:,}"
            for p in ("single_call", "parallel", "sequential", "hybrid")
        ),
    )
    md.kv_row("Parallel calling rate", f"{tl['parallel']['percentage']:.1f}%")

    md.kv_row("Unique functions called", f"{ctx.fd['total_unique_functions']:,}")
    md.kv_row("Unique functions defined", f"{len(ctx.fn_defined):,}")

    ufc = ctx.tcv["undefined_function_calls"]
    md.kv_row(
        "Calls to undefined functions",
        f"{ufc['total_calls']:,} ({ufc['affected_samples']:,} samples)",
    )
    av = ctx.tcv["argument_validation"]
    md.kv_row(
        "Calls with argument violations",
        f"{av['calls_with_violations']:,} ({av['affected_samples']:,} samples)",
    )

    md.kv_row(
        "Tools per sample: mean / median / max",
        f"{np.mean(ctx.tools_arr):.1f} / {np.median(ctx.tools_arr):.0f} "
        f"/ {np.max(ctx.tools_arr):.0f}",
    )

    md.kv_row(
        "Termination completeness",
        f"{ctx.ts['turns_with_final_answer']['percentage']:.1f}%",
    )

    if ctx.fc_lengths:
        a = np.array(ctx.fc_lengths, dtype=np.float64)
        md.kv_row(
            "User msg length, FC turns: mean / median / std",
            f"{np.mean(a):.1f} / {np.median(a):.0f} / {np.std(a):.1f}",
        )
    if ctx.no_fc_lengths:
        a = np.array(ctx.no_fc_lengths, dtype=np.float64)
        md.kv_row(
            "User msg length, no-FC turns: mean / median / std",
            f"{np.mean(a):.1f} / {np.median(a):.0f} / {np.std(a):.1f}",
        )

    if ctx.sys_msg_count == 0:
        md.kv_row("System prompt diversity", "0 (no system messages)")
    else:
        # _get_system_prompt_hash hashes b"" for samples without a system
        # message, so the bias report's unique count includes that empty
        # hash when not all samples have system messages. Subtract it.
        unique = ctx.bias.system_prompt_unique_count
        if ctx.sys_msg_count < ctx.n:
            unique -= 1
        md.kv_row("System prompt diversity", f"{unique} unique")

    ptd = ctx.bias.param_type_distribution
    total_s = sum(ptd.values())
    if total_s > 0:
        md.kv_row(
            "Param types: mixed / string / array / numeric",
            " / ".join(
                f"{ptd.get(p, 0) / total_s * 100:.1f}%"
                for p in ("mixed", "string_only", "array_or_nested", "numeric_only")
            ),
        )


def _render_calls_per_sample(md: _MarkdownBuilder, ctx: _MetricsContext) -> None:
    md.section("Function Calls Per Sample")
    md.line("| Calls | Samples | % |")
    md.line("|-------|---------|---|")
    calls_dist = Counter(int(c) for c in ctx.calls_arr)
    sorted_keys = sorted(calls_dist.keys())
    for k in sorted_keys:
        if k <= 6:
            md.line(f"| {k} | {calls_dist[k]:,} | {calls_dist[k] / ctx.n * 100:.1f}% |")
        elif k == min(kk for kk in sorted_keys if kk >= 7):
            sevenplus = sum(v for kk, v in calls_dist.items() if kk >= 7)
            md.line(f"| 7+ | {sevenplus:,} | {sevenplus / ctx.n * 100:.1f}% |")


def _render_tools_per_sample(md: _MarkdownBuilder, ctx: _MetricsContext) -> None:
    md.section("Tools Per Sample")
    md.line("| Tools | Samples | % |")
    md.line("|-------|---------|---|")
    tools_dist = Counter(int(t) for t in ctx.tools_arr)
    for k in sorted(tools_dist.keys()):
        md.line(f"| {k} | {tools_dist[k]:,} | {tools_dist[k] / ctx.n * 100:.1f}% |")


def _render_turn_level_patterns(md: _MarkdownBuilder, ctx: _MetricsContext) -> None:
    md.section("Turn-Level Calling Patterns")
    md.line("| Pattern | Turns | % |")
    md.line("|---------|-------|---|")
    tl = ctx.fcp["turn_level"]
    for pat_name in ("single_call", "parallel", "sequential", "hybrid", "no_calls"):
        e = tl[pat_name]
        label = pat_name.replace("_", " ").title()
        md.line(f"| {label} | {e['count']:,} | {e['percentage']:.1f}% |")


def _render_parallel_diversity(md: _MarkdownBuilder, ctx: _MetricsContext) -> None:
    if ctx.pfd["total_parallel_steps"] == 0:
        return
    md.section("Parallel Call Diversity")
    md.kv_header()
    md.kv_row("Total parallel steps", f"{ctx.pfd['total_parallel_steps']:,}")
    sf = ctx.pfd["same_function_fanout"]
    md.kv_row("Same-function fanout", _fmt_count_pct(sf["count"], sf["percentage"]))
    mf = ctx.pfd["multi_function_bundles"]
    md.kv_row("Multi-function bundles", _fmt_count_pct(mf["count"], mf["percentage"]))

    bp = ctx.pfd.get("bundle_patterns")
    if bp and bp["total_bundles"] > 0:
        pb = bp["pure_bundles"]
        md.kv_row(
            "Pure bundles (each function once)",
            f"{pb['count']:,} ({pb['percentage']:.1f}% of bundles)",
        )
        hb = bp["hybrid_bundles"]
        md.kv_row(
            "Hybrid bundles (some function repeated)",
            f"{hb['count']:,} ({hb['percentage']:.1f}% of bundles)",
        )


def _render_unique_funcs_per_step(md: _MarkdownBuilder, ctx: _MetricsContext) -> None:
    ufps_block = ctx.pfd.get("unique_functions_per_step")
    if not ufps_block:
        return
    ufps = ufps_block["distribution"]
    if not ufps:
        return
    md.section("Unique Functions Per Parallel Step")
    md.line("| Unique functions | Steps | % |")
    md.line("|------------------|-------|---|")
    total_steps = sum(ufps.values())
    for k in sorted(ufps.keys()):
        label = "1 (same-function fanout)" if k == 1 else str(k)
        md.line(f"| {label} | {ufps[k]:,} | {ufps[k] / total_steps * 100:.1f}% |")


def _render_sample_patterns(md: _MarkdownBuilder, ctx: _MetricsContext) -> None:
    md.section("Sample-Level Patterns")
    md.line("| Pattern | Samples | % |")
    md.line("|---------|---------|---|")
    for p in ("normal_fc", "never_call", "mixed"):
        c = ctx.bias.pattern_counts.get(p, 0)
        pct = ctx.bias.pattern_percentages.get(p, 0.0)
        md.line(f"| {p} | {c:,} | {pct:.1f}% |")


def _render_function_diversity(md: _MarkdownBuilder, ctx: _MetricsContext) -> None:
    md.section("Function Diversity")
    md.line("")
    defined_not_called = len(ctx.fn_defined - ctx.fn_called)
    called_not_defined = len(ctx.fn_called - ctx.fn_defined)
    line = (
        f"{len(ctx.fn_called):,} unique functions called across all samples. "
        f"{len(ctx.fn_defined):,} unique functions defined"
    )
    parts: list[str] = []
    if defined_not_called:
        parts.append(f"{defined_not_called:,} defined but never called")
    if called_not_defined:
        parts.append(f"{called_not_defined:,} called but not defined")
    line += f" ({'; '.join(parts)})." if parts else "."
    md.line(line)

    md.line("")
    top_10 = ctx.fd["top_20_functions"][:10]
    md.line(
        "**Top 10 functions**: "
        + ", ".join(f"{name} ({count:,})" for name, count in top_10)
        + "."
    )

    ups = ctx.fd["unique_per_sample"]
    md.line("")
    md.line(
        f"Unique functions per sample: mean {ups['mean']:.2f}, "
        f"median {ups['median']:.0f}, max {ups['max']:.0f}."
    )
    ups_dist = ups["distribution"]
    total_ups = sum(ups_dist.values())
    dist_parts = [
        f"{ups_dist[k] / total_ups * 100:.1f}% use {k}"
        for k in sorted(ups_dist.keys())[:5]
    ]
    if dist_parts:
        md.line(f"{', '.join(dist_parts)}.")


def _render_tool_call_validation(md: _MarkdownBuilder, ctx: _MetricsContext) -> None:
    if ctx.tcv["samples_with_any_violation"] == 0:
        return
    md.section("Tool Call Validation")
    md.kv_header()

    ufc = ctx.tcv["undefined_function_calls"]
    av = ctx.tcv["argument_validation"]
    md.kv_row("Calls to undefined functions", f"{ufc['total_calls']:,}")
    md.kv_row(
        "Samples with undefined calls",
        _fmt_count_pct(ufc["affected_samples"], ufc["percentage_of_samples"]),
    )
    md.kv_row("Unique hallucinated names", f"{ufc['unique_hallucinated_count']:,}")
    md.kv_row("Calls validated against schema", f"{av['total_calls_validated']:,}")
    md.kv_row("Calls with argument violations", f"{av['calls_with_violations']:,}")
    md.kv_row(
        "Samples with argument violations",
        _fmt_count_pct(av["affected_samples"], av["percentage_of_samples"]),
    )
    for kind in sorted(av["violation_breakdown"]):
        md.kv_row(f"  {kind}", f"{av['violation_breakdown'][kind]:,}")

    if ufc["hallucinated_names"]:
        md.line("")
        names = ufc["hallucinated_names"][:20]
        names_str = ", ".join(f"`{name}`" for name in names)
        if ufc["unique_hallucinated_count"] > 20:
            names_str += f", ... ({ufc['unique_hallucinated_count']:,} total)"
        md.line(f"**Hallucinated function names**: {names_str}.")


def _render_param_type_diversity(md: _MarkdownBuilder, ctx: _MetricsContext) -> None:
    ptd = ctx.bias.param_type_distribution
    total_s = sum(ptd.values())
    if total_s == 0:
        return
    md.section("Parameter Type Diversity")
    md.line("| Profile | Samples | % |")
    md.line("|---------|---------|---|")
    for p in sorted(ptd.keys(), key=lambda x: -ptd[x]):
        c = ptd[p]
        md.line(f"| {p} | {c:,} | {c / total_s * 100:.1f}% |")


type _MetricsSectionFn = Callable[[_MarkdownBuilder, _MetricsContext], None]


_METRICS_SECTIONS: tuple[_MetricsSectionFn, ...] = (
    _render_summary,
    _render_calls_per_sample,
    _render_tools_per_sample,
    _render_turn_level_patterns,
    _render_parallel_diversity,
    _render_unique_funcs_per_step,
    _render_sample_patterns,
    _render_function_diversity,
    _render_tool_call_validation,
    _render_param_type_diversity,
)


def render_metrics_markdown(
    samples: list[Any],
    report: Any,
    *,
    analyses: list[Any] | None = None,
    beh_analyses: list[Any] | None = None,
    bias: Any | None = None,
    stats: dict[str, Any] | None = None,
) -> str:
    """Render the auto-generated programmatic-metrics table as markdown.

    Each precomputed input (`analyses`, `beh_analyses`, `bias`, `stats`)
    is reused when provided to skip redundant recomputation. The output
    is the same markdown the consumer would get from a from-scratch call.
    """
    ctx = _prepare_metrics_context(
        samples,
        report,
        analyses=analyses,
        beh_analyses=beh_analyses,
        bias=bias,
        stats=stats,
    )
    md = _MarkdownBuilder()
    for render in _METRICS_SECTIONS:
        render(md, ctx)
    return md.render()


def save_json_report(stats: dict[str, Any], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)


def save_text_report(stats: dict[str, Any], output_path: str) -> None:
    with open(output_path, "w") as f:
        print_full_report(stats, f)
