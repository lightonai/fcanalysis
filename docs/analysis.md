# Analyzing conversations

[Documentation](README.md) · Prerequisite: [Getting started](getting-started.md)

Keep loader settings and counting units alongside analysis results: source
selection, reasoning settings, and curation change the measured population.
These examples analyze native `tool_calls`, not Terminal's textual actions.

## Inspect tool-use structure

This example runs offline from the repository root. Replace `samples` with
another loader's output to analyze your dataset:

```python
import json

from docs.examples.weather_loader import example_row, load
from fcanalysis.core import analyze_sample
from fcanalysis.reporter import print_full_report
from fcanalysis.statistics import aggregate_statistics

samples, report = load([json.dumps(example_row())])
analyses = [
    analyze_sample(sample.messages, extract_function_names=True)
    for sample in samples
]
stats = aggregate_statistics(
    analyses=analyses,
    messages_list=[sample.messages for sample in samples],
    tools_list=[sample.tools for sample in samples],
)
print_full_report(stats)
```

This aggregates in-memory lists. For incremental structural supervision counts,
use the accumulator below.

[`core.py`](../src/fcanalysis/core.py) starts a *real turn* at each user message
with non-whitespace content. It classifies the assistant's call batches before
the next such user message:

| Pattern | Recorded structure |
| --- | --- |
| `no_calls` | No call batch |
| `single_call` | One batch containing one call |
| `parallel` | One batch containing several calls |
| `sequential` | Several batches, each containing one call |
| `hybrid` | Several batches, at least one containing several calls |

Here, “parallel” describes batch width. It does not independently prove
concurrent execution or authorize unordered comparison in curation.
`extract_function_names=True` also fills `all_turns` with detailed turn data.
Calls before the first qualifying user are outside these turn totals;
`describe_supervision()` reports them separately as `calls_before_first_user`.

The [statistics module](../src/fcanalysis/statistics.py) adds tool diversity,
coverage, argument diagnostics, and supervision-related counts. Its diagnostic
validation does not replace an adapter's ordered source/final checks.
[The analysis example](../examples/02_analyze.py) shows behavioral measurements
and Markdown/text report generation for a loaded dataset.

## Count assistant decisions and call episodes

```python
from fcanalysis.supervision import SupervisionSummary, describe_supervision

summary = SupervisionSummary()
for sample in samples:
    features = describe_supervision(sample)
    summary.add(features)

print(summary.as_dict())
```

`describe_supervision()` distinguishes conversation, real-turn,
assistant-decision, and strict call-episode units. A strict episode continues
through tool feedback and ends at **any** user, system, or no-call assistant
message. Real-turn boundaries instead follow the nonempty-user rule above.

The accumulator keeps histograms and source breakdowns without transcripts;
memory grows with source count. It counts supplied records without deduplication
or weighting. Assistant decisions are candidates for training, not a count of
admitted targets. Sequential depth and parallel width measure different structures.

## Save and read canonical conversations

The remaining examples use the `samples` list above. Choose an output path
outside the checkout. This writes normalized data, omitting `raw`:

```python
import json
from pathlib import Path

output_path = Path("/path/to/conversations.jsonl")  # Choose your output path.
with output_path.open("x", encoding="utf-8") as stream:
    for sample in samples:
        payload = {
            name: getattr(sample, name)
            for name in ("messages", "tools", "dataset", "sample_id", "annotations")
        }
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
```

`"x"` creates a new file and fails if it already exists. Keep `raw` if your use
requires the source record; missing source data cannot be reconstructed by the
reader. For a large iterator, complete and close the load before treating a
written artifact as complete; an interrupted write can leave a partial file.

```python
from contextlib import closing

from fcanalysis.serialization import iter_samples_jsonl

with closing(iter_samples_jsonl(output_path)) as rows:
    for sample in rows:
        print(sample.dataset, sample.sample_id)
```

The reader supports UTF-8 JSONL and `.jsonl.gz`. Required root fields are
`messages`, `tools`, `dataset`, and `sample_id`; `annotations` and `raw` are
optional. Unknown root fields, blank lines, duplicate keys, nonfinite numbers,
and invalid containers fail with file/line context. It does not rerun source
interpretation, message validation, or curation.
An omitted `raw` becomes an empty dictionary.

## Curate a mixture of retained streams

[`deduplicate_mixture()`](../src/fcanalysis/mixture.py) curates an ordered union
of retained native-call or genuine text-only streams. Here the same list is
supplied twice to demonstrate a duplicate across streams; replace these lists
with your loaded populations:

```python
from fcanalysis.loaders.curation import MixtureScope
from fcanalysis.mixture import MixtureSource, deduplicate_mixture

with deduplicate_mixture(
    [
        MixtureSource("first", samples),
        MixtureSource("second", samples),
    ],
    scope=MixtureScope("combined", "train"),
) as result:
    for sample in result:
        print(sample.dataset, sample.sample_id)
    print(result.report.complete)  # True
```

Stream names must be unique; their order supplies tie priority. Dataset labels
and winner values survive. This selects only among the supplied retained inputs;
it cannot recover earlier removals or establish train/evaluation decontamination.

The mixture consumes all inputs before yielding. Unlike a loader's `final_count`,
its `report.complete` marks completed selection before output exhaustion; it does
not mean the caller consumed every winner.

The mixture utility compares native call/result batches in their current
canonical order. It does not infer permission to permute parallel batches;
explicit unordered bindings are rejected. Textual action protocols such as
Nemotron-Terminal need their own qualified comparison and are unsupported here.
The result owns temporary resources and should be used as a context manager.

## Further analysis

| Capability | Entry point | Interpretation |
| --- | --- | --- |
| Behavioral patterns and bias reports | [behavioral.py](../src/fcanalysis/behavioral.py) | Observed structure and heuristics on supplied conversations. |
| Deduplication and overlap exploration | [deduplication.py](../src/fcanalysis/deduplication.py), [overlap.py](../src/fcanalysis/overlap.py) | Different comparison rules from loader curation; coarse overlap groups are not proof of duplicate conversations. Use the mixture utility above to curate retained streams. |
| Semantic classification of no-call turns | [semantic.py](../src/fcanalysis/semantic.py), [example](../examples/04_semantic_pipeline.py) | Optional model inference through a configured endpoint; labels require review. |
| Cross-tabulation and semantic selection | [cross_tabulation.py](../src/fcanalysis/cross_tabulation.py), [semantic_filter.py](../src/fcanalysis/semantic_filter.py) | Analysis of supplied classification outputs and explicit selection policies. |
| Structural semantic-annotation contract | [supervision_semantics.py](../src/fcanalysis/supervision_semantics.py) | Draft record/quoted-evidence validation; performs no classification or accuracy certification. |
| Qwen3.5 token accounting | [tokenization.py](../src/fcanalysis/tokenization.py), [token_analysis.py](../src/fcanalysis/token_analysis.py) | Optional, model-specific measurement view. |

## Count tokens

Token analysis requires the optional dependencies:

```sh
uv sync --locked --extra tokenization
```

With the pinned tokenizer assets cached:

```python
from fcanalysis.tokenization import load_qwen35_counter

counter = load_qwen35_counter()
print(counter.count_batch(samples)[0].as_dict())
```

For the first download, call `load_qwen35_counter(local_files_only=False)`;
this fetches tokenizer assets, not model weights. A local asset directory can
be supplied with `tokenizer_path=...`.

Its pinned Qwen3.5 measurement view retains historical native reasoning and
counts assistant bodies separately from context. Those counts express that
rendering convention. Actual SFT exposure must be measured with the training
renderer, target admission, and loss masks you use. This path is not qualified
for Terminal's textual-action token categories.
