# Semantic-label release

The semantic annotations used by the frozen function-calling training-data
curation campaign are distributed as optional assets on the
[`v0.1.0` GitHub release](https://github.com/lightonai/fcanalysis/releases/tag/v0.1.0).
They are not Git objects, are not included in a clone or source archive, and are
not downloaded when `fcanalysis` is installed.

The release covers the three annotated source components used by that
campaign:

- `nvidia/Nemotron-Agentic-v1`, `tool_calling` split;
- `nvidia/Nemotron-SFT-Agentic-v2`, `interactive_agent` split; and
- `LLM360/TxT360-3efforts` (now `IFM/TxT360-3efforts`), `agent/high`.

It does not redistribute source conversations, messages, tool definitions, or
training examples. Each row contains a source-local sample ID and derived
annotations that can be joined to the corresponding pinned source data. This
makes the release useful for dataset auditing, selection, and alternative
curation experiments without folding the annotations into model-visible
training text.

## Downloading

GitHub Release assets are fetched with the GitHub CLI, `gh`, rather than core
Git. To download all three compressed JSONL files:

```sh
gh release download v0.1.0 \
  --repo lightonai/fcanalysis \
  --pattern 'semantic-labels-*.jsonl.gz' \
  --dir semantic-labels
```

To download only one component:

```sh
gh release download v0.1.0 \
  --repo lightonai/fcanalysis \
  --pattern 'semantic-labels-nemotron-agentic-v1.jsonl.gz' \
  --dir semantic-labels
```

Download the machine-readable manifest separately when verifying bytes:

```sh
gh release download v0.1.0 \
  --repo lightonai/fcanalysis \
  --pattern 'semantic-labels-manifest-v1.0.0.json' \
  --dir semantic-labels
```

## Released files

| Asset | Rows | Error rows | Turn classifications | Compressed bytes | Compressed SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| `semantic-labels-nemotron-agentic-v1.jsonl.gz` | 105,280 | 621 | 133,116 | 13,550,596 | `0044bea236e55fb3b954d8e95643d58412850096ff267bcd72420b733438da0c` |
| `semantic-labels-nemotron-agentic-v2-interactive-agent.jsonl.gz` | 199,115 | 217 | 318,359 | 20,764,375 | `7982f4b075ea5c533ace958921f80a1fde701f4477975500befe47e9ec3417a5` |
| `semantic-labels-txt360-high.jsonl.gz` | 100,294 | 231 | 143,638 | 17,890,749 | `6ed8a1574e093dc4633d9397b1c534f95b55e3a6755a622d88a7b4cee8a431a0` |
| **Total** | **404,689** | **1,069** | **595,113** | **52,205,720** | — |

Rows with an `error` are retained for complete accounting. Successful rows
have three valid first-pass ensemble generations. Consumers should skip error
rows rather than treat them as unlabeled samples.

Exact uncompressed hashes, source revisions, join policy, and per-file audit
counts are recorded in the
[semantic-label manifest](semantic-labels-manifest-v1.0.0.json).

## Joining annotations to source data

Join `sample_id` to the source-local sample identity emitted by the matching
loader and pinned configuration. The frozen campaign used this policy:

- error rows: account for them and do not apply labels;
- source rows without a semantic result: keep them;
- annotation rows without a matching source row: account for them and do not
  remap them to another row; and
- downstream curation: consume the final `category`, not the provisional
  first-pass category.

The annotation rows cover samples submitted for classification because they
contained eligible real turns where the assistant did not call a tool. They are
not a one-row-per-record mirror of the complete upstream datasets.

## Method and row schema

The classifier first requests three independent structured classifications for
all eligible no-call turns in a sample and takes the per-turn majority result.
Every turn initially flagged as an anti-pattern is then sent through targeted
verification and correction using the full conversation and agent policy. That
second operation is called “stage 2” in some historical field and CLI names; it
is not a separate semantic-analysis method.

Top-level fields are:

- `sample_id`: source-local sample identity;
- `num_valid_generations`: number of valid first-pass ensemble responses;
- `classifications`: final per-turn annotations; and
- `error`: present on rows where the required ensemble could not be completed.

Each classification includes `turn_index`, final `category`, final
`justified`, final `reasoning`, and first-pass `vote_counts`. Optional audit
fields include:

- `stage1_category` and `stage1_reasoning`: the initial ensemble result before
  targeted verification;
- `verification`: the verification verdict, including any proposed
  `correction`;
- `contested`: whether valid verification verdicts disagreed; and
- `verify_votes_valid`, `reverification`, `tied`, and `uniformity`: additional
  voting and audit metadata recorded by some runs.

See [Semantic classification](semantic-classification.md) for category
definitions, voting behavior, and the verification agreement rule. Optional
fields vary across rows, so consumers should not assume they are always
present.

## Source pins and licensing

| Component | Frozen source revision | Upstream license statement |
| --- | --- | --- |
| Nemotron Agentic v1 | [`650d590978ca35c8f1ecea2faf136e5fac421b62`](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1/tree/650d590978ca35c8f1ecea2faf136e5fac421b62) | CC BY 4.0; upstream identifies an Apache 2.0 component |
| Nemotron Agentic v2 | [`49e79a3be5ab8cf7511a12958b95cfd6408cd8db`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2/tree/49e79a3be5ab8cf7511a12958b95cfd6408cd8db) | CC BY 4.0 with Apache 2.0 and MIT components |
| TxT360 | [`bfc4a082d11967cd7810fe0b773be87bf54fb32e`](https://huggingface.co/datasets/LLM360/TxT360-3efforts/tree/bfc4a082d11967cd7810fe0b773be87bf54fb32e) | CC BY 4.0 |

This annotation release is provided under CC BY 4.0. The source datasets are
not redistributed; resolving sample IDs remains subject to their respective
licenses, component terms, attribution requirements, and dataset cards.

## Evidence boundary

These labels are model-dependent research annotations, not ground truth or a
dataset-quality ranking. Category prevalence should not be compared across
sources without accounting for their different domains, structures, and
eligible no-call populations.

The frozen rows do not record the judge model and revision, prompt hash,
serving-software version, decoding parameters, or run ID. Their schema and
behavior are consistent with the documented ensemble plus targeted
verification-and-correction pipeline, but the exact stochastic inference run
cannot be reproduced from these files and the public source alone. The files
can be consumed downstream byte-for-byte using the hashes and join policy in
the manifest.

Only the three files used by the frozen campaign are published. A separately
observed ToolMind annotation artifact is intentionally excluded because it was
not an input to that campaign.
