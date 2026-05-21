from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.dolci import DolciConfig, load


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

    print(report.summary())
    print()
    print(f"Kept {len(samples):,} samples.")
    print()

    s = samples[0]
    print(f"sample_id={s.sample_id!r} dataset={s.dataset!r}")
    print(f"tools: {len(s.tools)}  messages: {len(s.messages)}")
    print(f"first message role={s.messages[0]['role']!r}")
    print(f"last  message role={s.messages[-1]['role']!r}")


if __name__ == "__main__":
    main()
