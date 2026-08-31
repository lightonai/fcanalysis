# fcanalysis

`fcanalysis` is a Python toolkit for loading, normalizing, validating, and
analyzing public function-calling datasets. It provides seven source-specific
loaders, a shared conversation model, structural and statistical analysis,
schema-aware tool-call validation, deterministic overlap checks, and an
optional LLM-based classifier for turns where no tool was called.

The repository contains source code and small regression metadata. It does not
contain the source datasets, generated dataset samples, semantic-judge outputs,
model checkpoints, or benchmark results. It does not train or evaluate models.

Version 0.1.0 is an alpha release and requires Python 3.14 or newer. The package
uses Python 3.14 syntax deliberately. It is not published on PyPI.

## Installation

Install the tagged source directly from GitHub:

```sh
uv add "fcanalysis @ git+https://github.com/lightonai/fcanalysis.git@v0.1.0"
```

For development:

```sh
git clone https://github.com/lightonai/fcanalysis.git
cd fcanalysis
uv sync --locked
```

## Minimal example

The zero-network smoke example constructs one normalized sample, analyzes its
tool-calling pattern, and validates its arguments:

```sh
uv run python examples/00_smoke.py
```

Equivalent library use:

```python
from fcanalysis import ConversationSample
from fcanalysis.core import analyze_sample

sample = ConversationSample(
    dataset="example",
    sample_id="1",
    tools=[],
    messages=[
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello!"},
    ],
)

analysis = analyze_sample(sample.messages)
print(analysis.turn_patterns[0].value)  # no_calls
```

## Loading a dataset

Each loader module exposes a source-specific configuration dataclass and a
`load(...) -> tuple[list[ConversationSample], LoadReport]` function. Loaders
read pinned Hugging Face revisions unless the loader explicitly supports a
local `path` override.

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

The registered loaders are:

| Loader | Source dataset |
| --- | --- |
| `apigen_mt` | `Salesforce/APIGen-MT-5k` |
| `dolci` | `allenai/Dolci-Instruct-SFT-Tool-Use` |
| `nemotron_agentic_v1` | `nvidia/Nemotron-Agentic-v1` |
| `nemotron_agentic_v2` | `nvidia/Nemotron-SFT-Agentic-v2` |
| `nemotron_terminal` | `nvidia/Nemotron-Terminal-Corpus` |
| `toolmind` | `Nanbeige/ToolMind` |
| `txt360` | `LLM360/TxT360-3efforts` (now served as `IFM/TxT360-3efforts`) |

An experimental TOUCAN loader is under development on the
[`toucan-wip`](https://github.com/lightonai/fcanalysis/tree/toucan-wip) branch.
It is not part of v0.1.0's supported loader registry or its 69 fixture cases.

Exact source revisions, source-declared licenses, attribution requirements, and
source-specific limitations are documented in [Dataset sources and
licenses](docs/datasets.md). The MIT license for `fcanalysis` does not relicense
any source dataset.

## Normalized data model

`fcanalysis.format.ConversationSample` is a slotted dataclass with six fields:

- `messages: list[dict[str, object]]`: normalized OpenAI-style conversation;
- `tools: list[dict[str, object]]`: normalized function definitions;
- `dataset: str`: source identity, sometimes including a split or teacher;
- `sample_id: str | int`: source-local row identity;
- `annotations: dict[str, object]`: curation metadata that is not serialized
  into downstream training examples; and
- `raw: object`: optional source payload retained for inspection.

In this documentation, *model-visible* means serialized into the training
examples presented to the downstream model being trained. Under the intended
conversion path, only `messages` and `tools` are serialized into those examples;
`annotations` and `raw` are curation-side records and must remain outside chat
templates and training serialization.

Normalization occurs in three ordered stages:

1. source conversion into `ConversationSample`;
2. explicitly selected source-specific transforms and drops; and
3. explicitly selected universal filters from `FilterConfig`.

Defaults are intentionally conservative: callers choose filtering policy. A
`LoadReport` records input counts, transformations, drop reasons, and the
configuration applied.

## Analysis and validation

The main library surfaces are:

- `fcanalysis.core`: real-turn extraction and tool-calling patterns;
- `fcanalysis.validation`: defined-function and JSON Schema subset checks;
- `fcanalysis.statistics`: aggregate dataset statistics;
- `fcanalysis.behavioral`: per-turn facts and bias-detection summaries;
- `fcanalysis.overlap`: deterministic within- and cross-dataset deduplication;
- `fcanalysis.reporter`: terminal, Markdown, and JSON reports;
- `fcanalysis.semantic`: optional semantic classification; and
- `fcanalysis.semantic_filter`: category-based filtering of classified turns.

Repository examples exercise the larger workflows:

```sh
uv run python examples/01_load.py
uv run python examples/02_analyze.py
uv run python examples/03_dedup.py
uv run python examples/04_semantic_pipeline.py
```

These examples load full public datasets and can require substantial network,
memory, and disk resources. Read [examples/README.md](examples/README.md) before
running them.

To inspect the semantic command without contacting a model endpoint:

```sh
uv run python -m fcanalysis.semantic --help
```

To check the correction invariant in existing semantic output:

```sh
uv run python scripts/validate_invariant.py /path/to/results
```

See [Semantic classification](docs/semantic-classification.md) for the endpoint
contract, staged workflow, and evidence limitations.

## Regression fixtures and reproducibility

The repository registers 69 loader/configuration cases. Each tracked fixture
directory contains exactly three contract files:

- `config.json`: canonical loader and filter configuration;
- `report.json`: serialized `LoadReport`; and
- `output.hash`: SHA-256 of the complete canonical uncompressed JSONL output.

The default test suite is fully runnable from a clean clone and uses synthetic
unit inputs:

```sh
uv run pytest
```

The 69 full-dataset E2E cases are intentionally separate because they download
pinned source revisions and reconstruct complete outputs:

```sh
uv run pytest -m e2e
uv run pytest -m e2e -k dolci
```

Reproducibility means deterministic processing of the same pinned source bytes,
configuration, dependency lock, and iteration order. It does not guarantee
that an upstream host will retain a revision forever, that stochastic semantic
inference can be recreated without its full serving configuration, or that a
fixture hash proves scientific validity.

## Development checks

```sh
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
uv run prek run --all-files
bash -n scripts/*.sh
git diff --check
uv build --clear
```

The project does not publish to PyPI. Inspect built artifacts locally before
distributing them.

## Known limitations

- Python 3.14 is the only declared and tested interpreter line.
- Loaders materialize large datasets in memory; they are analysis tools, not a
  streaming ingestion service.
- Loader fixtures establish implementation stability, not dataset quality,
  contamination freedom, or downstream model performance.
- Deduplication uses a canonical seed key and is not complete semantic or
  trajectory-tree deduplication.
- Universal reasoning stripping covers specific structured fields and exact
  inline patterns; it is not exhaustive and can be lossy.
- Semantic classifications are model-dependent research annotations; no
  benchmark or causal performance claim is established by this package.
- Source datasets can contain synthetic errors, tool failures, stale responses,
  bias, unsafe content, or additional upstream-license obligations.

## License and acknowledgment

The `fcanalysis` source is licensed under the [MIT License](LICENSE). Dataset
content remains under its source terms; see [docs/datasets.md](docs/datasets.md).

This research was supported by the OpenEuroLLM project, co-funded by the
Digital Europe Programme under GA no. 101195233.

We acknowledge the EuroHPC Joint Undertaking for awarding this project access
to the EuroHPC supercomputer
[LEONARDO](https://www.hpc.cineca.it/systems/hardware/leonardo/), hosted by
CINECA (Italy) and the LEONARDO consortium through a EuroHPC Access call.

The contents presented herein reflects only the author's view, and the
Commission is not responsible for any use that may be made of the information
it contains.
