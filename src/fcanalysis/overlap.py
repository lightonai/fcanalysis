import hashlib
import string
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from .format import ConversationSample
from .loaders.base import FilterConfig, LoadReport

type DecisionStep = tuple[str, tuple[str, ...]]
type DecisionSequence = tuple[DecisionStep, ...]

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    return " ".join(text.split())


def _extract_seed_parts(sample: ConversationSample) -> tuple[str, list[str]]:
    first_user = ""
    for msg in sample.messages:
        if msg["role"] == "user":
            first_user = msg.get("content") or ""
            break
    names: list[str] = []
    for tool in sample.tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function")
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if isinstance(name, str):
            names.append(name)
    return first_user, names


def _hash_parts(user: str, names: list[str], *, normalize: bool) -> str:
    if normalize:
        user = normalize_text(user)
        names = [normalize_text(n) for n in names]
    payload = user + "\0" + "\0".join(sorted(names))
    return hashlib.sha256(payload.encode()).hexdigest()


def compute_seed_key(sample: ConversationSample, *, normalize: bool = True) -> str:
    """Canonical hash of (first user message, sorted tool definition names); two samples sharing it differ only in assistant trajectory."""
    user, names = _extract_seed_parts(sample)
    return _hash_parts(user, names, normalize=normalize)


def extract_decision_sequence(
    sample: ConversationSample,
    *,
    normalize: bool = True,
) -> DecisionSequence:
    """Ordered tuple of (function_name, argument_keys) per assistant tool_call; used to compare trajectories beyond the seed key."""
    steps: list[DecisionStep] = []
    for msg in sample.messages:
        if msg["role"] != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            func: dict[str, Any] = tc.get("function", {})
            name: str = func.get("name", "")
            if normalize:
                name = normalize_text(name)
            args_raw = func.get("arguments", "{}")
            if isinstance(args_raw, str | bytes):
                try:
                    parsed = orjson.loads(args_raw)
                except orjson.JSONDecodeError:
                    parsed = None
            else:
                parsed = args_raw
            keys = tuple(sorted(parsed.keys())) if isinstance(parsed, dict) else ()
            steps.append((name, keys))
    return tuple(steps)


@dataclass(slots=True)
class OverlapResult:
    dataset_a: str
    dataset_b: str
    samples_a: int
    samples_b: int
    unique_seed_keys_a: int
    unique_seed_keys_b: int
    shared_seed_keys: int
    only_in_a: int
    only_in_b: int
    shared_redundant: int
    shared_augmented: int
    shared_without_normalization: int
    normalization_delta: int
    duplicate_seed_keys_a: int
    duplicate_seed_keys_b: int
    samples_a_in_overlap: int
    samples_b_in_overlap: int
    examples_redundant: list[dict[str, str]] = field(default_factory=list)
    examples_augmented: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class DuplicateReport:
    dataset: str
    total_samples: int
    unique_seed_keys: int
    duplicate_seed_keys: int
    redundant_groups: int
    augmented_groups: int
    redundant_extra_samples: int
    augmented_extra_samples: int
    examples_redundant: list[dict[str, str]] = field(default_factory=list)
    examples_augmented: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class WithinDedupResult:
    input_samples: int
    output_samples: int
    removed_samples: int
    unique_seed_keys: int
    duplicate_seed_keys: int


@dataclass(slots=True)
class CrossDedupResult:
    primary_name: str
    secondary_name: str
    primary_seed_keys: int
    secondary_input_samples: int
    secondary_output_samples: int
    removed_samples: int
    shared_seed_keys: int


def _build_dataset_map(
    samples: list[ConversationSample],
) -> tuple[dict[str, list[DecisionSequence]], set[str], dict[str, str]]:
    key_map: dict[str, list[DecisionSequence]] = defaultdict(list)
    raw_keys: set[str] = set()
    preview: dict[str, str] = {}
    for s in samples:
        user, names = _extract_seed_parts(s)
        sk_norm = _hash_parts(user, names, normalize=True)
        sk_raw = _hash_parts(user, names, normalize=False)
        ds = extract_decision_sequence(s, normalize=True)
        key_map[sk_norm].append(ds)
        raw_keys.add(sk_raw)
        if sk_norm not in preview:
            preview[sk_norm] = user[:200]
    return key_map, raw_keys, preview


def measure_overlap(
    samples_a: list[ConversationSample],
    samples_b: list[ConversationSample],
    *,
    dataset_a_name: str = "dataset_a",
    dataset_b_name: str = "dataset_b",
    max_examples: int = 5,
) -> OverlapResult:
    map_a, raw_keys_a, preview_a = _build_dataset_map(samples_a)
    map_b, raw_keys_b, _ = _build_dataset_map(samples_b)

    keys_a = set(map_a)
    keys_b = set(map_b)
    shared = keys_a & keys_b
    shared_raw = raw_keys_a & raw_keys_b

    redundant = 0
    augmented = 0
    ex_red: list[dict[str, str]] = []
    ex_aug: list[dict[str, str]] = []

    for sk in shared:
        seqs_a = set(map_a[sk])
        seqs_b = set(map_b[sk])
        if seqs_a & seqs_b:
            redundant += 1
            if len(ex_red) < max_examples:
                ex_red.append(
                    {
                        "seed_key": sk[:16],
                        "user_msg": preview_a.get(sk, ""),
                        "decision_seq": str(list(map_a[sk][0][:3])),
                    }
                )
        else:
            augmented += 1
            if len(ex_aug) < max_examples:
                ex_aug.append(
                    {
                        "seed_key": sk[:16],
                        "user_msg": preview_a.get(sk, ""),
                        "seq_a": str(list(map_a[sk][0][:3])),
                        "seq_b": str(list(map_b[sk][0][:3])),
                    }
                )

    return OverlapResult(
        dataset_a=dataset_a_name,
        dataset_b=dataset_b_name,
        samples_a=len(samples_a),
        samples_b=len(samples_b),
        unique_seed_keys_a=len(keys_a),
        unique_seed_keys_b=len(keys_b),
        shared_seed_keys=len(shared),
        only_in_a=len(keys_a - keys_b),
        only_in_b=len(keys_b - keys_a),
        shared_redundant=redundant,
        shared_augmented=augmented,
        shared_without_normalization=len(shared_raw),
        normalization_delta=len(shared) - len(shared_raw),
        duplicate_seed_keys_a=sum(1 for v in map_a.values() if len(v) > 1),
        duplicate_seed_keys_b=sum(1 for v in map_b.values() if len(v) > 1),
        samples_a_in_overlap=sum(len(map_a[sk]) for sk in shared),
        samples_b_in_overlap=sum(len(map_b[sk]) for sk in shared),
        examples_redundant=ex_red,
        examples_augmented=ex_aug,
    )


def find_duplicates(
    samples: list[ConversationSample],
    *,
    dataset_name: str = "dataset",
    max_examples: int = 5,
) -> DuplicateReport:
    key_map: dict[str, list[DecisionSequence]] = defaultdict(list)
    preview: dict[str, str] = {}

    for s in samples:
        user, names = _extract_seed_parts(s)
        sk = _hash_parts(user, names, normalize=True)
        ds = extract_decision_sequence(s, normalize=True)
        key_map[sk].append(ds)
        if sk not in preview:
            preview[sk] = user[:200]

    redundant_groups = 0
    augmented_groups = 0
    redundant_extra = 0
    augmented_extra = 0
    ex_red: list[dict[str, str]] = []
    ex_aug: list[dict[str, str]] = []

    for sk, seqs in key_map.items():
        if len(seqs) < 2:
            continue
        extra = len(seqs) - 1
        unique_seqs = set(seqs)
        if len(unique_seqs) == 1:
            redundant_groups += 1
            redundant_extra += extra
            if len(ex_red) < max_examples:
                ex_red.append(
                    {
                        "seed_key": sk[:16],
                        "user_msg": preview.get(sk, ""),
                        "count": str(len(seqs)),
                        "decision_seq": str(list(seqs[0][:3])),
                    }
                )
        else:
            augmented_groups += 1
            augmented_extra += extra
            if len(ex_aug) < max_examples:
                ex_aug.append(
                    {
                        "seed_key": sk[:16],
                        "user_msg": preview.get(sk, ""),
                        "count": str(len(seqs)),
                        "unique_seqs": str(len(unique_seqs)),
                        "seq_first": str(list(seqs[0][:3])),
                    }
                )

    return DuplicateReport(
        dataset=dataset_name,
        total_samples=len(samples),
        unique_seed_keys=len(key_map),
        duplicate_seed_keys=redundant_groups + augmented_groups,
        redundant_groups=redundant_groups,
        augmented_groups=augmented_groups,
        redundant_extra_samples=redundant_extra,
        augmented_extra_samples=augmented_extra,
        examples_redundant=ex_red,
        examples_augmented=ex_aug,
    )


def dedup_within(
    samples: list[ConversationSample],
) -> tuple[list[ConversationSample], WithinDedupResult]:
    """Keep the first sample for each seed_key, drop later duplicates; returns (kept_samples, result)."""
    seen: Counter[str] = Counter()
    kept: list[ConversationSample] = []
    for s in samples:
        sk = compute_seed_key(s, normalize=True)
        if seen[sk] == 0:
            kept.append(s)
        seen[sk] += 1
    return kept, WithinDedupResult(
        input_samples=len(samples),
        output_samples=len(kept),
        removed_samples=len(samples) - len(kept),
        unique_seed_keys=len(seen),
        duplicate_seed_keys=sum(1 for c in seen.values() if c > 1),
    )


def dedup_cross(
    primary: list[ConversationSample],
    secondary: list[ConversationSample],
    *,
    primary_name: str = "primary",
    secondary_name: str = "secondary",
) -> tuple[list[ConversationSample], CrossDedupResult]:
    """Drop samples from `secondary` whose seed_key appears in `primary`; `primary` is returned unchanged."""
    primary_keys = {compute_seed_key(s, normalize=True) for s in primary}

    kept: list[ConversationSample] = []
    shared_keys: set[str] = set()
    for s in secondary:
        sk = compute_seed_key(s, normalize=True)
        if sk in primary_keys:
            shared_keys.add(sk)
        else:
            kept.append(s)

    return kept, CrossDedupResult(
        primary_name=primary_name,
        secondary_name=secondary_name,
        primary_seed_keys=len(primary_keys),
        secondary_input_samples=len(secondary),
        secondary_output_samples=len(kept),
        removed_samples=len(secondary) - len(kept),
        shared_seed_keys=len(shared_keys),
    )


def format_report(result: OverlapResult) -> str:
    r = result
    pct_a = (
        f"{r.shared_seed_keys / r.unique_seed_keys_a * 100:.1f}%"
        if r.unique_seed_keys_a
        else "N/A"
    )
    pct_b = (
        f"{r.shared_seed_keys / r.unique_seed_keys_b * 100:.1f}%"
        if r.unique_seed_keys_b
        else "N/A"
    )
    shared = r.shared_seed_keys or 1

    lines = [
        f"# {r.dataset_a} / {r.dataset_b} Overlap Report\n",
        "## Dataset sizes\n",
        "| Dataset | Samples | Unique seed keys | Duplicate seed keys |",
        "|---|---|---|---|",
        f"| {r.dataset_a} | {r.samples_a:,} | {r.unique_seed_keys_a:,}"
        f" | {r.duplicate_seed_keys_a:,} |",
        f"| {r.dataset_b} | {r.samples_b:,} | {r.unique_seed_keys_b:,}"
        f" | {r.duplicate_seed_keys_b:,} |",
        "",
        "## Tier 1: seed key overlap\n",
        f"| Metric | Count | % of {r.dataset_a} keys | % of {r.dataset_b} keys |",
        "|---|---|---|---|",
        f"| Shared seed keys | {r.shared_seed_keys:,} | {pct_a} | {pct_b} |",
        f"| Only in {r.dataset_a} | {r.only_in_a:,}"
        + (
            f" | {r.only_in_a / r.unique_seed_keys_a * 100:.1f}% | |"
            if r.unique_seed_keys_a
            else " | | |"
        ),
        f"| Only in {r.dataset_b} | {r.only_in_b:,}"
        + (
            f" | | {r.only_in_b / r.unique_seed_keys_b * 100:.1f}% |"
            if r.unique_seed_keys_b
            else " | | |"
        ),
        "",
        f"Samples in overlapping seed keys:"
        f" {r.samples_a_in_overlap:,} from {r.dataset_a},"
        f" {r.samples_b_in_overlap:,} from {r.dataset_b}.",
        "",
        f"## Tier 2: decision sequence overlap ({r.shared_seed_keys:,} shared seed keys)\n",
        "| Classification | Count | % of shared |",
        "|---|---|---|",
        f"| Redundant (same decisions) | {r.shared_redundant:,}"
        f" | {r.shared_redundant / shared * 100:.1f}% |",
        f"| Augmented (different decisions) | {r.shared_augmented:,}"
        f" | {r.shared_augmented / shared * 100:.1f}% |",
        "",
        "## Normalization validation\n",
        "| | Shared seed keys |",
        "|---|---|",
        f"| With normalization | {r.shared_seed_keys:,} |",
        f"| Without normalization | {r.shared_without_normalization:,} |",
        f"| Delta (gained by normalizing) | {r.normalization_delta:,} |",
    ]

    if r.examples_redundant:
        lines.extend(["", "## Examples: redundant matches\n"])
        for i, ex in enumerate(r.examples_redundant, 1):
            lines.extend(
                [
                    f"{i}. Seed key `{ex['seed_key']}...`",
                    f"   User: {ex['user_msg']!r}",
                    f"   Decisions (first 3): `{ex['decision_seq']}`",
                    "",
                ]
            )

    if r.examples_augmented:
        lines.extend(["", "## Examples: augmented (same seed, different decisions)\n"])
        for i, ex in enumerate(r.examples_augmented, 1):
            lines.extend(
                [
                    f"{i}. Seed key `{ex['seed_key']}...`",
                    f"   User: {ex['user_msg']!r}",
                    f"   {r.dataset_a}: `{ex['seq_a']}`",
                    f"   {r.dataset_b}: `{ex['seq_b']}`",
                    "",
                ]
            )

    return "\n".join(lines) + "\n"


def format_duplicate_report(result: DuplicateReport) -> str:
    r = result
    dup_total = r.duplicate_seed_keys or 1
    extra_total = r.redundant_extra_samples + r.augmented_extra_samples

    lines = [
        f"# {r.dataset} Within-Dataset Duplicate Report\n",
        "## Summary\n",
        "| Metric | Count |",
        "|---|---|",
        f"| Total samples | {r.total_samples:,} |",
        f"| Unique seed keys | {r.unique_seed_keys:,} |",
        f"| Duplicate seed keys | {r.duplicate_seed_keys:,} |",
        f"| Extra samples (removable) | {extra_total:,} |",
        f"| After dedup | {r.unique_seed_keys:,} |",
        "",
        "## Duplicate group classification\n",
        "| Classification | Groups | Extra samples | % of duplicate groups |",
        "|---|---|---|---|",
        f"| Redundant (all same decisions) | {r.redundant_groups:,}"
        f" | {r.redundant_extra_samples:,}"
        f" | {r.redundant_groups / dup_total * 100:.1f}% |",
        f"| Augmented (different decisions) | {r.augmented_groups:,}"
        f" | {r.augmented_extra_samples:,}"
        f" | {r.augmented_groups / dup_total * 100:.1f}% |",
    ]

    if r.examples_redundant:
        lines.extend(["", "## Examples: redundant duplicates\n"])
        for i, ex in enumerate(r.examples_redundant, 1):
            lines.extend(
                [
                    f"{i}. Seed key `{ex['seed_key']}...` ({ex['count']} samples)",
                    f"   User: {ex['user_msg']!r}",
                    f"   Decisions (first 3): `{ex['decision_seq']}`",
                    "",
                ]
            )

    if r.examples_augmented:
        lines.extend(["", "## Examples: augmented duplicates\n"])
        for i, ex in enumerate(r.examples_augmented, 1):
            lines.extend(
                [
                    f"{i}. Seed key `{ex['seed_key']}...`"
                    f" ({ex['count']} samples, {ex['unique_seqs']} unique sequences)",
                    f"   User: {ex['user_msg']!r}",
                    f"   First sequence (first 3): `{ex['seq_first']}`",
                    "",
                ]
            )

    return "\n".join(lines) + "\n"


_UNIVERSAL_FILTERS = FilterConfig(
    strip_thinking=True,
    require_parseable_arguments=True,
    require_balanced_cardinality=True,
    require_defined_functions=True,
    require_valid_arguments=True,
)


def _load_dataset(
    name: str, split: str = ""
) -> tuple[list[ConversationSample], LoadReport]:
    match name:
        case "nemotron_agentic_v1":
            from .loaders import nemotron_agentic_v1

            return nemotron_agentic_v1.load(
                dataset_config=nemotron_agentic_v1.NemotronAgenticV1Config(
                    splits=("tool_calling",),
                    drop_orphan_samples=True,
                    drop_conflicting_duplicate_tools=True,
                    drop_empty_system=True,
                ),
                filter_config=_UNIVERSAL_FILTERS,
            )

        case "nemotron_agentic_v2":
            from .loaders import nemotron_agentic_v2

            return nemotron_agentic_v2.load(filter_config=_UNIVERSAL_FILTERS)

        case "nemotron_terminal":
            from .loaders import nemotron_terminal

            return nemotron_terminal.load(
                dataset_config=nemotron_terminal.NemotronTerminalConfig(
                    strip_malformed=True,
                    drop_orphans=True,
                    drop_incomplete=True,
                ),
                filter_config=_UNIVERSAL_FILTERS,
            )

        case "dolci":
            from .loaders import dolci

            return dolci.load(
                dataset_config=dolci.DolciConfig(
                    drop_consecutive_text_text_assistant=True,
                    merge_text_fc_assistant=True,
                ),
                filter_config=_UNIVERSAL_FILTERS,
            )

        case "apigen_mt":
            from .loaders import apigen_mt

            return apigen_mt.load(
                dataset_config=apigen_mt.APIGenMTConfig(
                    strip_think_tool=True,
                    drop_undefined_function_calls=True,
                    drop_repeated_tool_call_streaks=True,
                    drop_error_recovery_loops=True,
                    drop_consecutive_assistant=True,
                ),
                filter_config=_UNIVERSAL_FILTERS,
            )

        case "toolmind":
            from .loaders import toolmind

            return toolmind.load(
                dataset_config=toolmind.ToolMindConfig(
                    sources=["graph_syn_datasets/graphsyn.jsonl"],
                    seed_group_filter="longest_clean",
                    drop_non_object_arguments=True,
                    drop_consecutive_text_assistant=True,
                    merge_split_assistant=True,
                    strip_think_tool=True,
                ),
                filter_config=_UNIVERSAL_FILTERS,
            )

        case "txt360":
            from .loaders import txt360

            return txt360.load(
                split=split or "high",
                dataset_config=txt360.TxT360Config(
                    seed_group_filter="latest_clean_prefix",
                    require_non_empty_user=True,
                    drop_user_tool_call_samples=True,
                ),
                filter_config=_UNIVERSAL_FILTERS,
            )

        case _:
            raise ValueError(
                f"Unknown dataset: {name!r}. Supported: nemotron_agentic_v1, "
                f"nemotron_agentic_v2, nemotron_terminal, dolci, apigen_mt, "
                f"toolmind, txt360"
            )


def _load_timed(name: str, split: str) -> list[ConversationSample]:
    print(f"Loading {name}...", flush=True)
    t0 = time.monotonic()
    samples, _ = _load_dataset(name, split)
    print(f"  {len(samples):,} samples in {time.monotonic() - t0:.1f}s")
    return samples


def _save_or_print(output: Path | None, content: str) -> None:
    if output is not None:
        output.write_text(content)
        print(f"  Written to {output}")
    else:
        print()
        print(content)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-dataset overlap and dedup tools."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ov = sub.add_parser("overlap", help="Measure cross-dataset overlap")
    p_ov.add_argument("dataset_a")
    p_ov.add_argument("dataset_b")
    p_ov.add_argument("--split-a", default="")
    p_ov.add_argument("--split-b", default="")
    p_ov.add_argument("--output", "-o", type=Path)
    p_ov.add_argument("--max-examples", type=int, default=5)

    p_dup = sub.add_parser("duplicates", help="Analyze within-dataset duplicates")
    p_dup.add_argument("dataset")
    p_dup.add_argument("--split", default="")
    p_dup.add_argument("--output", "-o", type=Path)
    p_dup.add_argument("--max-examples", type=int, default=5)

    p_dw = sub.add_parser("dedup-within", help="Dedup within a dataset")
    p_dw.add_argument("dataset")
    p_dw.add_argument("--split", default="")

    p_dc = sub.add_parser(
        "dedup-cross", help="Remove secondary samples overlapping primary"
    )
    p_dc.add_argument("primary")
    p_dc.add_argument("secondary")
    p_dc.add_argument("--split-primary", default="")
    p_dc.add_argument("--split-secondary", default="")

    args = parser.parse_args()

    match args.command:
        case "overlap":
            _cmd_overlap(args)
        case "duplicates":
            _cmd_duplicates(args)
        case "dedup-within":
            _cmd_dedup_within(args)
        case "dedup-cross":
            _cmd_dedup_cross(args)


def _cmd_overlap(args: Any) -> None:
    samples_a = _load_timed(args.dataset_a, args.split_a)
    samples_b = _load_timed(args.dataset_b, args.split_b)

    print("Measuring overlap...", flush=True)
    t0 = time.monotonic()
    result = measure_overlap(
        samples_a,
        samples_b,
        dataset_a_name=args.dataset_a,
        dataset_b_name=args.dataset_b,
        max_examples=args.max_examples,
    )
    print(f"  Done in {time.monotonic() - t0:.1f}s")
    print(f"  Shared seed keys: {result.shared_seed_keys}")
    _save_or_print(args.output, format_report(result))


def _cmd_duplicates(args: Any) -> None:
    samples = _load_timed(args.dataset, args.split)

    print("Finding duplicates...", flush=True)
    t0 = time.monotonic()
    result = find_duplicates(
        samples, dataset_name=args.dataset, max_examples=args.max_examples
    )
    print(f"  Done in {time.monotonic() - t0:.1f}s")
    print(f"  Duplicate seed keys: {result.duplicate_seed_keys}")
    print(
        f"  Redundant: {result.redundant_groups}, augmented: {result.augmented_groups}"
    )
    _save_or_print(args.output, format_duplicate_report(result))


def _cmd_dedup_within(args: Any) -> None:
    samples = _load_timed(args.dataset, args.split)

    print("Deduplicating...", flush=True)
    t0 = time.monotonic()
    _, result = dedup_within(samples)
    print(f"  Done in {time.monotonic() - t0:.1f}s")
    print(f"  Input: {result.input_samples:,}")
    print(f"  Output: {result.output_samples:,}")
    print(f"  Removed: {result.removed_samples:,}")
    print(f"  Duplicate seed keys: {result.duplicate_seed_keys:,}")


def _cmd_dedup_cross(args: Any) -> None:
    primary = _load_timed(args.primary, args.split_primary)
    secondary = _load_timed(args.secondary, args.split_secondary)

    print("Cross-deduplicating...", flush=True)
    t0 = time.monotonic()
    _, result = dedup_cross(
        primary,
        secondary,
        primary_name=args.primary,
        secondary_name=args.secondary,
    )
    print(f"  Done in {time.monotonic() - t0:.1f}s")
    print(f"  Secondary input: {result.secondary_input_samples:,}")
    print(f"  Secondary output: {result.secondary_output_samples:,}")
    print(f"  Removed: {result.removed_samples:,}")
    print(f"  Shared seed keys: {result.shared_seed_keys:,}")


if __name__ == "__main__":
    main()
