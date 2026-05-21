from pathlib import Path

from fcanalysis.behavioral import (
    analyze_dataset_behavior,
    compute_bias_report,
    print_bias_report,
)
from fcanalysis.core import analyze_sample
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.dolci import DolciConfig, load
from fcanalysis.reporter import (
    print_full_report,
    render_metrics_markdown,
    save_text_report,
)
from fcanalysis.statistics import aggregate_enhanced_statistics, aggregate_statistics


_OUT = Path("examples/out")


def main() -> None:
    samples, report = load(
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

    messages_list = [s.messages for s in samples]
    tools_list = [s.tools for s in samples]
    analyses = [analyze_sample(m, extract_function_names=True) for m in messages_list]

    stats = aggregate_statistics(
        analyses=analyses,
        messages_list=messages_list,
        tools_list=tools_list,
    )
    enhanced = aggregate_enhanced_statistics(
        analyses=analyses,
        messages_list=messages_list,
    )
    print(f"Loaded {len(samples):,} samples.\n")
    print(
        f"Function diversity: {stats['function_diversity']['total_unique_functions']:,} unique functions."
    )
    print(
        "Single vs multi turn: "
        f"{enhanced['single_vs_multi_turn']['single_turn_samples']['count']:,} / "
        f"{enhanced['single_vs_multi_turn']['multi_turn_samples']['count']:,}"
    )

    behavioral = analyze_dataset_behavior(samples, analyses)
    bias = compute_bias_report(behavioral)

    _OUT.mkdir(exist_ok=True)
    save_text_report(stats, str(_OUT / "report.txt"))
    (_OUT / "report.md").write_text(
        render_metrics_markdown(
            samples,
            report,
            analyses=analyses,
            beh_analyses=behavioral,
            bias=bias,
            stats=stats,
        )
    )
    (_OUT / "bias.txt").write_text(print_bias_report(bias))

    print(f"\nText report → {_OUT / 'report.txt'}")
    print(f"Markdown    → {_OUT / 'report.md'}")
    print(f"Bias        → {_OUT / 'bias.txt'}")
    print()
    print_full_report(stats)


if __name__ == "__main__":
    main()
