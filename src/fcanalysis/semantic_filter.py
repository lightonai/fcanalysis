from collections import Counter
from dataclasses import dataclass

from .behavioral import (
    SampleBehavioralAnalysis,
    SamplePattern,
    analyze_sample_behavior,
)
from .core import TurnPattern, identify_real_turns
from .cross_tabulation import SemanticResults
from .format import ConversationSample


_UNJUSTIFIED_CATEGORIES = frozenset(
    {
        "ANTI_MANUAL_SOLVE",
        "ANTI_UNJUSTIFIED_REFUSAL",
        "ANTI_PRESSURE_CAVE",
        "OTHER_UNJUSTIFIED",
    }
)


@dataclass(slots=True, frozen=True)
class SemanticFilterResult:
    input_samples: int
    output_samples: int
    removed_samples: int
    fc_only_samples: int
    classified_samples: int
    unclassified_samples: int
    samples_flagged_by_category: dict[str, int]
    exclude_categories: frozenset[str]

    @property
    def removal_rate(self) -> float:
        return self.removed_samples / self.input_samples if self.input_samples else 0.0


def _sample_has_nofc_turns(sample: ConversationSample) -> bool:
    return any(
        t.pattern == TurnPattern.NO_CALLS for t in identify_real_turns(sample.messages)
    )


def filter_by_categories(
    samples: list[ConversationSample],
    semantic: SemanticResults,
    exclude_categories: set[str],
) -> tuple[list[ConversationSample], SemanticFilterResult]:
    """Drop samples whose semantic results contain any classification in `exclude_categories`; samples missing from `semantic` are kept conservatively."""
    kept: list[ConversationSample] = []
    fc_only = 0
    classified = 0
    unclassified = 0
    flagged_cats: Counter[str] = Counter()

    for sample in samples:
        sem = semantic.get(sample.sample_id)

        if sem is None:
            # Classifier failed on this sample. Conservative: keep.
            if _sample_has_nofc_turns(sample):
                unclassified += 1
            else:
                fc_only += 1
            kept.append(sample)
            continue

        if not sem:
            fc_only += 1
            kept.append(sample)
            continue

        classified += 1
        matched = {
            cat
            for turn_cls in sem.values()
            if (cat := turn_cls.get("category", "")) in exclude_categories
        }
        if matched:
            for cat in matched:
                flagged_cats[cat] += 1
        else:
            kept.append(sample)

    return kept, SemanticFilterResult(
        input_samples=len(samples),
        output_samples=len(kept),
        removed_samples=len(samples) - len(kept),
        fc_only_samples=fc_only,
        classified_samples=classified,
        unclassified_samples=unclassified,
        samples_flagged_by_category=dict(flagged_cats),
        exclude_categories=frozenset(exclude_categories),
    )


def filter_ams(
    samples: list[ConversationSample],
    semantic: SemanticResults,
) -> tuple[list[ConversationSample], SemanticFilterResult]:
    """Drop samples with any turn classified ANTI_MANUAL_SOLVE (AMS); thin wrapper around `filter_by_categories`."""
    return filter_by_categories(samples, semantic, {"ANTI_MANUAL_SOLVE"})


@dataclass(slots=True)
class QualitySummary:
    dataset: str
    total_samples: int

    pattern_counts: dict[str, int]

    total_classified_turns: int
    ams_turns: int
    ams_turn_rate: float
    category_turn_counts: dict[str, int]

    classified_samples: int
    ams_samples: int
    ams_sample_rate: float

    justified_turns: int
    unjustified_turns: int
    unjustified_turn_rate: float


def compute_quality_summary(
    samples: list[ConversationSample],
    semantic: SemanticResults,
    dataset_name: str = "",
    analyses: list[SampleBehavioralAnalysis] | None = None,
) -> QualitySummary:
    if analyses is None:
        analyses = [analyze_sample_behavior(s) for s in samples]

    pattern_counts: Counter[str] = Counter(a.sample_pattern.value for a in analyses)

    total_classified_turns = 0
    ams_turns = 0
    justified_turns = 0
    unjustified_turns = 0
    category_turn_counts: Counter[str] = Counter()
    classified_sample_count = 0
    ams_ids: set[str | int] = set()

    for a in analyses:
        sem = semantic.get(a.sample_id)
        if not sem:
            continue
        classified_sample_count += 1
        sample_has_ams = False
        for turn_cls in sem.values():
            cat = turn_cls.get("category", "")
            if not cat:
                # Turn was emitted but classifier failed to assign a category;
                # ignore for justified/unjustified tallies (would otherwise be
                # silently counted as justified).
                continue
            total_classified_turns += 1
            category_turn_counts[cat] += 1
            if cat == "ANTI_MANUAL_SOLVE":
                ams_turns += 1
                sample_has_ams = True
            if cat in _UNJUSTIFIED_CATEGORIES:
                unjustified_turns += 1
            else:
                justified_turns += 1
        if sample_has_ams:
            ams_ids.add(a.sample_id)

    return QualitySummary(
        dataset=dataset_name,
        total_samples=len(samples),
        pattern_counts=dict(pattern_counts),
        total_classified_turns=total_classified_turns,
        ams_turns=ams_turns,
        ams_turn_rate=ams_turns / total_classified_turns
        if total_classified_turns
        else 0.0,
        category_turn_counts=dict(category_turn_counts.most_common()),
        classified_samples=classified_sample_count,
        ams_samples=len(ams_ids),
        ams_sample_rate=len(ams_ids) / classified_sample_count
        if classified_sample_count
        else 0.0,
        justified_turns=justified_turns,
        unjustified_turns=unjustified_turns,
        unjustified_turn_rate=unjustified_turns / total_classified_turns
        if total_classified_turns
        else 0.0,
    )


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "n/a"


def format_quality_summary(qs: QualitySummary) -> str:
    lines: list[str] = [
        f"=== Quality Summary: {qs.dataset or '(unnamed)'} ===",
        f"Total samples: {qs.total_samples:,}",
        "",
        "--- Sample patterns ---",
    ]
    for pattern in (
        SamplePattern.NORMAL_FC,
        SamplePattern.MIXED,
        SamplePattern.NEVER_CALL,
        SamplePattern.NO_TURNS,
    ):
        c = qs.pattern_counts.get(pattern.value, 0)
        lines.append(f"  {pattern.value:15s}: {c:>7,} ({_pct(c, qs.total_samples)})")

    lines.extend(
        [
            "",
            "--- Semantic: sample-level ---",
            f"  Classified samples: {qs.classified_samples:,}",
            f"  AMS-flagged samples: {qs.ams_samples:,} "
            f"({_pct(qs.ams_samples, qs.classified_samples)} of classified)",
            "",
            "--- Semantic: turn-level ---",
            f"  Classified turns: {qs.total_classified_turns:,}",
            f"  AMS turns: {qs.ams_turns:,} "
            f"({_pct(qs.ams_turns, qs.total_classified_turns)})",
            f"  Justified turns: {qs.justified_turns:,} "
            f"({_pct(qs.justified_turns, qs.total_classified_turns)})",
            f"  Unjustified turns: {qs.unjustified_turns:,} "
            f"({_pct(qs.unjustified_turns, qs.total_classified_turns)})",
            "",
            "--- Category distribution (turns) ---",
        ]
    )
    for cat, count in qs.category_turn_counts.items():
        lines.append(
            f"  {cat:30s}: {count:>6,} ({_pct(count, qs.total_classified_turns)})"
        )

    return "\n".join(lines)


def format_filter_result(result: SemanticFilterResult) -> str:
    lines = [
        f"Semantic filter: {result.input_samples:,} → {result.output_samples:,} "
        f"({result.removed_samples:,} removed, {result.removal_rate:.1%})",
        f"  FC-only (no no-FC turns): {result.fc_only_samples:,}",
        f"  Classified: {result.classified_samples:,}",
        f"  Unclassified (kept): {result.unclassified_samples:,}",
        f"  Excluded categories: {sorted(result.exclude_categories)}",
    ]
    if result.samples_flagged_by_category:
        lines.append("  Flagged by category:")
        for cat, count in sorted(
            result.samples_flagged_by_category.items(), key=lambda x: -x[1]
        ):
            lines.append(f"    {cat}: {count:,}")
    return "\n".join(lines)
