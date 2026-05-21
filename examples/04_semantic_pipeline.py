from fcanalysis.behavioral import (
    analyze_dataset_behavior,
    prepare_semantic_layer_inputs,
)
from fcanalysis.core import analyze_sample
from fcanalysis.cross_tabulation import cross_tabulate, print_cross_tab_report
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.dolci import DolciConfig, load
from fcanalysis.semantic import build_prompt
from fcanalysis.semantic_filter import (
    compute_quality_summary,
    filter_ams,
    format_quality_summary,
)


def main() -> None:
    samples, _ = load(
        dataset_config=DolciConfig(
            drop_consecutive_text_text_assistant=True,
            merge_text_fc_assistant=True,
            drop_conflicting_duplicate_tools=True,
        ),
        filter_config=FilterConfig(
            strip_thinking=True,
            require_parseable_arguments=True,
            require_balanced_cardinality=True,
            require_defined_functions=True,
            require_valid_arguments=True,
        ),
    )

    # Behavioral layer identifies no-FC turns (the structural prerequisite
    # for semantic classification: every turn the LLM did NOT call a tool).
    messages_list = [s.messages for s in samples]
    analyses = [analyze_sample(m, extract_function_names=True) for m in messages_list]
    beh = analyze_dataset_behavior(samples, analyses)
    sem_inputs = prepare_semantic_layer_inputs(samples, beh)
    print(
        f"{len(samples):,} samples → {len(sem_inputs):,} have no-FC turns to classify.\n"
    )

    # What the LLM would actually see for the first sample.
    if sem_inputs:
        first_prompt = build_prompt(sem_inputs[0])
        user_msg = first_prompt[1]["content"]
        print("=== build_prompt output (user message preview) ===")
        print(user_msg[:600] + ("..." if len(user_msg) > 600 else ""))
        print()

    # In production: `semantic.run_semantic_layer(...)` calls a vLLM endpoint
    # and writes JSONL of per-sample classifications. Then
    # `cross_tabulation.load_semantic_results(path)` reads it back as a
    # `dict[sample_id, dict[turn_index, classification]]`.
    #
    # Below we synthesize that dict on a slice so the example runs without
    # a live LLM. Every no-FC turn is labelled S4_DIRECT_ANSWER except
    # one labelled ANTI_MANUAL_SOLVE.
    synthetic_semantic: dict[str | int, dict[int, dict[str, object]]] = {}
    target = sem_inputs[:200]
    for i, inp in enumerate(target):
        synthetic_semantic[inp.sample_id] = {
            t: {
                "category": "ANTI_MANUAL_SOLVE" if i % 25 == 0 else "S4_DIRECT_ANSWER",
                "justified": i % 25 != 0,
                "reasoning": "synthetic",
            }
            for t in inp.no_fc_turn_indices
        }
    target_ids = {inp.sample_id for inp in target}
    target_samples = [s for s in samples if s.sample_id in target_ids]
    target_beh = [a for a in beh if a.sample_id in target_ids]

    # Cross-tabulate behavioral × semantic: which categories correlate
    # with longer user messages, deeper turn positions, or more tools.
    print("=== cross_tabulate (behavioral × semantic) ===")
    report = cross_tabulate(target_beh, synthetic_semantic)
    print(print_cross_tab_report(report))
    print()

    # Quality summary: aggregate AMS rate, RTC rate, justified vs unjustified.
    quality = compute_quality_summary(
        target_samples, synthetic_semantic, dataset_name="dolci (synthetic semantic)"
    )
    print(format_quality_summary(quality))
    print()

    # filter_ams drops samples with ANY ANTI_MANUAL_SOLVE turn.
    kept, filter_result = filter_ams(target_samples, synthetic_semantic)
    print(
        f"filter_ams: {filter_result.input_samples:,} → {filter_result.output_samples:,} "
        f"({filter_result.removed_samples:,} removed, {filter_result.removal_rate:.1%})"
    )
    print(f"  flagged by category: {filter_result.samples_flagged_by_category}")
    print(f"  returned list length: {len(kept):,}")


if __name__ == "__main__":
    main()
