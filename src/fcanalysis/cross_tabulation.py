import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .behavioral import SampleBehavioralAnalysis
from .statistics import compute_basic_stats, compute_percentiles


type SemanticResults = dict[str | int, dict[int, dict[str, Any]]]
"""LLM-classified no-FC turns, keyed by sample_id then turn_index."""


_LENGTH_BINS: tuple[tuple[int, int | float], ...] = (
    (0, 50),
    (50, 100),
    (100, 150),
    (150, 250),
    (250, 500),
    (500, float("inf")),
)


def load_semantic_results(
    path: str | Path,
) -> SemanticResults:
    results: SemanticResults = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "error" in row:
                continue
            results[row["sample_id"]] = {
                c["turn_index"]: c for c in row.get("classifications", [])
            }
    return results


def _stats_summary(data: Sequence[int | float]) -> dict[str, Any]:
    if not data:
        return {"n": 0}
    s = compute_basic_stats(data, include_std=True)
    p = compute_percentiles(data, [25, 50, 75, 90])
    return {
        "n": len(data),
        "mean": round(s["mean"], 1) if s["mean"] is not None else None,
        "median": s["median"],
        "std": round(std, 1) if (std := s.get("std_dev")) is not None else None,
        "p25": p.get("percentile_25"),
        "p75": p.get("percentile_75"),
        "p90": p.get("percentile_90"),
    }


def _bin_label(lo: int, hi: float) -> str:
    return f"{lo}-{hi}" if hi != float("inf") else f"{lo}+"


def _bin_lengths(lengths: list[int]) -> dict[str, int]:
    binned = {_bin_label(lo, hi): 0 for lo, hi in _LENGTH_BINS}
    for length in lengths:
        for lo, hi in _LENGTH_BINS:
            if lo <= length < hi:
                binned[_bin_label(lo, hi)] += 1
                break
    return binned


def cross_tabulate(
    analyses: list[SampleBehavioralAnalysis],
    semantic: SemanticResults,
) -> dict[str, Any]:
    """Join behavioral analyses with `{sample_id: {turn_index: {category, justified, ...}}}` semantic results; aggregates per-category metrics (user-msg length, turn position, tool count, system-prompt diversity)."""
    cat_lengths: dict[str, list[int]] = defaultdict(list)
    cat_positions: dict[str, list[int]] = defaultdict(list)
    cat_tool_counts: dict[str, list[int]] = defaultdict(list)
    cat_sys_hashes: dict[str, list[str]] = defaultdict(list)

    fc_lengths: list[int] = []
    fc_positions: list[int] = []

    justified_lengths: list[int] = []
    unjustified_lengths: list[int] = []
    justified_positions: list[int] = []
    unjustified_positions: list[int] = []
    justified_tool_counts: list[int] = []
    unjustified_tool_counts: list[int] = []

    matched_turns = 0
    unmatched_turns = 0

    for a in analyses:
        sem = semantic.get(a.sample_id)
        for t in a.turns:
            if t.has_fc:
                fc_lengths.append(t.user_message_length_chars)
                fc_positions.append(t.turn_index)
                continue
            cls = sem.get(t.turn_index) if sem is not None else None
            if cls is None:
                unmatched_turns += 1
                continue

            matched_turns += 1
            cat = cls.get("category", "")
            cat_lengths[cat].append(t.user_message_length_chars)
            cat_positions[cat].append(t.turn_index)
            cat_tool_counts[cat].append(t.num_tools_available)
            cat_sys_hashes[cat].append(a.system_prompt_hash)

            if cls.get("justified"):
                justified_lengths.append(t.user_message_length_chars)
                justified_positions.append(t.turn_index)
                justified_tool_counts.append(t.num_tools_available)
            else:
                unjustified_lengths.append(t.user_message_length_chars)
                unjustified_positions.append(t.turn_index)
                unjustified_tool_counts.append(t.num_tools_available)

    total_no_fc = matched_turns + unmatched_turns
    total_classified = matched_turns

    report: dict[str, Any] = {
        "coverage": {
            "matched_no_fc_turns": matched_turns,
            "unmatched_no_fc_turns": unmatched_turns,
            "coverage_pct": round(matched_turns / total_no_fc * 100, 1)
            if total_no_fc > 0
            else 0,
        },
        "category_counts": {
            cat: len(lengths)
            for cat, lengths in sorted(cat_lengths.items(), key=lambda x: -len(x[1]))
        },
        "justified_vs_unjustified": {
            "total_classified": total_classified,
            "justified": len(justified_lengths),
            "unjustified": len(unjustified_lengths),
            "unjustified_pct": round(
                len(unjustified_lengths) / total_classified * 100, 2
            )
            if total_classified > 0
            else 0,
        },
        "length_by_category": {
            cat: _stats_summary(cat_lengths[cat]) for cat in sorted(cat_lengths)
        }
        | {"_FC_TURNS": _stats_summary(fc_lengths)},
        "length_justified_vs_unjustified": {
            "justified": _stats_summary(justified_lengths),
            "unjustified": _stats_summary(unjustified_lengths),
            "fc_turns": _stats_summary(fc_lengths),
        },
        "position_by_category": {
            cat: _stats_summary(cat_positions[cat]) for cat in sorted(cat_positions)
        },
        "position_justified_vs_unjustified": {
            "justified": _stats_summary(justified_positions),
            "unjustified": _stats_summary(unjustified_positions),
        },
        "tool_count_by_category": {
            cat: _stats_summary(cat_tool_counts[cat]) for cat in sorted(cat_tool_counts)
        },
        "tool_count_justified_vs_unjustified": {
            "justified": _stats_summary(justified_tool_counts),
            "unjustified": _stats_summary(unjustified_tool_counts),
        },
        "system_prompt_by_category": {
            cat: {
                "unique_prompts": len(counter := Counter(cat_sys_hashes[cat])),
                "top_3": counter.most_common(3),
            }
            for cat in sorted(cat_sys_hashes)
        },
    }

    justified_binned = _bin_lengths(justified_lengths)
    unjustified_binned = _bin_lengths(unjustified_lengths)
    bin_report: dict[str, dict[str, Any]] = {}
    for lo, hi in _LENGTH_BINS:
        label = _bin_label(lo, hi)
        n_just = justified_binned[label]
        n_unjust = unjustified_binned[label]
        total = n_just + n_unjust
        bin_report[label] = {
            "total": total,
            "unjustified": n_unjust,
            "unjustified_pct": round(n_unjust / total * 100, 1) if total > 0 else 0,
        }
    report["unjustified_rate_by_length_bin"] = bin_report
    return report


def print_cross_tab_report(report: dict[str, Any]) -> str:
    lines: list[str] = []

    def _section(title: str) -> None:
        lines.append("")
        lines.append(f"=== {title} ===")

    def _stats_line(label: str, s: dict[str, Any]) -> None:
        if s.get("n", 0) == 0:
            lines.append(f"  {label}: (no data)")
            return
        lines.append(
            f"  {label}: n={s['n']}, "
            f"mean={s['mean']}, median={s['median']}, "
            f"std={s['std']}, p25={s['p25']}, p75={s['p75']}, p90={s['p90']}"
        )

    _section("Coverage")
    cov = report["coverage"]
    lines.append(
        f"  Matched: {cov['matched_no_fc_turns']}, "
        f"Unmatched: {cov['unmatched_no_fc_turns']}, "
        f"Coverage: {cov['coverage_pct']}%"
    )

    _section("Semantic Category Distribution")
    for cat, count in report["category_counts"].items():
        lines.append(f"  {cat}: {count}")

    _section("Justified vs Unjustified")
    jvu = report["justified_vs_unjustified"]
    lines.append(
        f"  Total: {jvu['total_classified']}, "
        f"Justified: {jvu['justified']}, "
        f"Unjustified: {jvu['unjustified']} ({jvu['unjustified_pct']}%)"
    )

    _section("User Message Length by Category")
    for cat, s in report["length_by_category"].items():
        _stats_line(cat, s)

    _section("User Message Length: Justified vs Unjustified vs FC")
    for label, s in report["length_justified_vs_unjustified"].items():
        _stats_line(label, s)

    _section("Unjustified Rate by Length Bin")
    for label, d in report["unjustified_rate_by_length_bin"].items():
        lines.append(
            f"  {label}: {d['unjustified']}/{d['total']} "
            f"({d['unjustified_pct']}% unjustified)"
        )

    _section("Turn Position by Category")
    for cat, s in report["position_by_category"].items():
        _stats_line(cat, s)

    _section("Turn Position: Justified vs Unjustified")
    for label, s in report["position_justified_vs_unjustified"].items():
        _stats_line(label, s)

    _section("Tool Count by Category")
    for cat, s in report["tool_count_by_category"].items():
        _stats_line(cat, s)

    _section("Tool Count: Justified vs Unjustified")
    for label, s in report["tool_count_justified_vs_unjustified"].items():
        _stats_line(label, s)

    _section("System Prompt Diversity by Category")
    for cat, d in report["system_prompt_by_category"].items():
        top = ", ".join(f"{h[:8]}({c})" for h, c in d["top_3"])
        lines.append(f"  {cat}: {d['unique_prompts']} unique, top: {top}")

    return "\n".join(lines)
