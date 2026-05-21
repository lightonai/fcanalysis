from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.nemotron_agentic_v1 import NemotronAgenticV1Config
from fcanalysis.loaders.nemotron_agentic_v1 import load as load_v1
from fcanalysis.loaders.txt360 import TxT360Config
from fcanalysis.loaders.txt360 import load as load_txt360
from fcanalysis.overlap import (
    dedup_cross,
    dedup_within,
    find_duplicates,
    format_duplicate_report,
    format_report,
    measure_overlap,
)


_PROD_FILTER = FilterConfig(
    strip_thinking=True,
    require_parseable_arguments=True,
    require_balanced_cardinality=True,
    require_defined_functions=True,
    require_valid_arguments=True,
)


def main() -> None:
    print("Loading nvidia/Nemotron-Agentic-v1 (tool_calling split)...")
    v1_samples, _ = load_v1(
        dataset_config=NemotronAgenticV1Config(
            splits=("tool_calling",),
            drop_orphan_samples=True,
            drop_empty_system=True,
            drop_conflicting_duplicate_tools=True,
        ),
        filter_config=_PROD_FILTER,
    )
    print(f"  {len(v1_samples):,} samples\n")

    print("Loading LLM360/TxT360-3efforts (high split)... [~2-3 min, 1.4M raw rows]")
    txt360_samples, _ = load_txt360(
        dataset_config=TxT360Config(
            seed_group_filter="latest_clean_prefix",
            require_non_empty_user=True,
            drop_user_tool_call_samples=True,
        ),
        filter_config=_PROD_FILTER,
    )
    print(f"  {len(txt360_samples):,} samples\n")

    # Within-dataset duplicates: same (first-user-msg, tools) seed key,
    # potentially with the same decision sequence.
    dup_v1 = find_duplicates(v1_samples, dataset_name="nemotron_agentic_v1")
    dup_txt = find_duplicates(txt360_samples, dataset_name="txt360")
    print(format_duplicate_report(dup_v1))
    print()
    print(format_duplicate_report(dup_txt))
    print()

    # Cross-dataset overlap measurement (read-only: counts shared seed keys
    # and classifies overlap as redundant vs. augmented).
    overlap = measure_overlap(
        v1_samples,
        txt360_samples,
        dataset_a_name="nemotron_agentic_v1",
        dataset_b_name="txt360",
    )
    print(format_report(overlap))
    print()

    # Mutating dedup: collapse intra-dataset duplicates, then drop from
    # TxT360 anything whose seed key already appears in the v1 set.
    v1_unique, v1_within = dedup_within(v1_samples)
    txt_unique, txt_within = dedup_within(txt360_samples)
    print(
        f"v1 dedup_within:    {v1_within.input_samples:>9,} → {v1_within.output_samples:>9,}  ({v1_within.removed_samples:,} removed)"
    )
    print(
        f"txt360 dedup_within:{txt_within.input_samples:>9,} → {txt_within.output_samples:>9,}  ({txt_within.removed_samples:,} removed)"
    )

    txt_after_cross, cross = dedup_cross(
        primary=v1_unique,
        secondary=txt_unique,
        primary_name="nemotron_agentic_v1",
        secondary_name="txt360",
    )
    print(
        f"txt360 minus v1 overlap: "
        f"{cross.secondary_input_samples:,} → {cross.secondary_output_samples:,}  "
        f"({cross.removed_samples:,} removed via {cross.shared_seed_keys:,} shared seed keys)"
    )
    print(f"Final txt360 list length: {len(txt_after_cross):,}")


if __name__ == "__main__":
    main()
