# fcanalysis

Systematic analysis of function-calling datasets for training and evaluating tool-using language models.

`fcanalysis` ships format-aware loaders for seven public function-calling datasets, each producing a unified `ConversationSample` with configurable per-dataset transforms and universal filters. On top of the loaders sit structural and statistical analyzers (turn classification, calling-pattern distributions, FC coverage, tool-call validation against schemas, bias-detection metrics), an LLM-driven semantic classifier for no-FC turns (vLLM endpoint), and a cross-dataset deduplicator keyed by canonical seed. Every loader's output is pinned byte-for-byte by a regression-test suite against committed fixtures.

## Installation

```sh
uv add fcanalysis
```

Requires Python 3.14+.

## Quickstart

### Load a dataset

```python
from fcanalysis.loaders.base import FilterConfig
from fcanalysis.loaders.dolci import DolciConfig, load

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
print(f"{len(samples)} samples after filtering")
```

Each loader returns a `list[ConversationSample]` and a `LoadReport` documenting the conversion, dataset-specific config, and universal-filter pipeline.

### Compute statistics

```python
from fcanalysis.core import analyze_sample
from fcanalysis.statistics import aggregate_statistics

analyses = [analyze_sample(s.messages, extract_function_names=True) for s in samples]
stats = aggregate_statistics(
    analyses=analyses,
    messages_list=[s.messages for s in samples],
    tools_list=[s.tools for s in samples],
)
```

### Render a report

```python
from fcanalysis.reporter import print_full_report

print_full_report(stats)
```

### Deduplicate within and across datasets

```python
from fcanalysis.overlap import dedup_within, dedup_cross

unique_samples, within_report = dedup_within(samples)
secondary_kept, cross_report = dedup_cross(
    primary=unique_samples,
    secondary=other_dataset,
)
```

## Supported loaders

| Loader | HuggingFace ID |
|---|---|
| `apigen_mt` | `Salesforce/APIGen-MT-5k` |
| `dolci` | `allenai/Dolci-Instruct-SFT-Tool-Use` |
| `nemotron_agentic_v1` | `nvidia/Nemotron-Agentic-v1` |
| `nemotron_agentic_v2` | `nvidia/Nemotron-SFT-Agentic-v2` |
| `nemotron_terminal` | `nvidia/Nemotron-Terminal-Corpus` |
| `toolmind` | `Nanbeige/ToolMind` |
| `txt360` | `LLM360/TxT360-3efforts` |

## Architecture

- `fcanalysis.format.ConversationSample`: the universal data type. Five fields (`messages`, `tools`, `dataset`, `sample_id`, `raw`); slotted dataclass with `raw` repr-hidden.
- `fcanalysis.loaders.base`: `FilterConfig`, `LoadReport`, `apply_filters`. Universal Stage-3 filter logic.
- `fcanalysis.loaders.{name}`: per-dataset Stage-1 conversion plus Stage-2 dataset-specific config.
- `fcanalysis.core`: turn classification (`identify_real_turns`, `classify_turn_pattern`).
- `fcanalysis.validation`: JSON Schema subset for argument validation.
- `fcanalysis.statistics`: per-aspect aggregators (function diversity, calling patterns, turn structure, FC coverage, abstention, termination, single-vs-multi-turn breakdowns).
- `fcanalysis.behavioral`: structural facts per turn and per sample, plus bias-detection aggregates.
- `fcanalysis.semantic`: vLLM-driven LLM classification of no-FC turns.
- `fcanalysis.cross_tabulation`: join semantic results with behavioral analyses.
- `fcanalysis.semantic_filter`: drop samples whose no-FC turns match excluded semantic categories.
- `fcanalysis.overlap`: within- and cross-dataset deduplication by canonical seed key.
- `fcanalysis.reporter`: text and markdown report rendering.

## Regression-test contract

Every loader×config pair has a committed fixture under [`tests/fixtures/loaders/{loader}/{config_id}/`](tests/fixtures/loaders/). Each fixture pins:

- The `LoadReport` (`report.json`).
- The SHA-256 hash of the full canonicalized sample list (`output.hash`).
- A deterministic 100-sample subset (`sample.jsonl`).
- The exact loader configuration (`config.json`).

The end-to-end test (`tests/e2e/test_loaders.py`) re-runs each loader and asserts byte-equivalence against its fixture. Any divergence is a behavioral regression.

## License

MIT. See [LICENSE](LICENSE).
