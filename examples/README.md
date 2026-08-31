# Examples

Each script is self-contained. Run from the repo root:

```sh
uv run python examples/00_smoke.py
uv run python examples/01_load.py
uv run python examples/02_analyze.py
uv run python examples/03_dedup.py
uv run python examples/04_semantic_pipeline.py
```

`00_smoke.py` is deterministic and does not use the network. First-time runs of
the loader examples pull complete pinned datasets into the Hugging Face cache;
review the source terms in [`docs/datasets.md`](../docs/datasets.md) first.

| Script | Loader(s) | What it covers |
|---|---|---|
| `00_smoke.py` | none | Public import, one synthetic `ConversationSample`, turn-pattern analysis, and argument validation. |
| `01_load.py` | Dolci | `DolciConfig` + `FilterConfig`, `LoadReport.summary()`, sample structure. |
| `02_analyze.py` | Dolci | `analyze_sample`, `aggregate_statistics`, `aggregate_enhanced_statistics`, `analyze_dataset_behavior`, `compute_bias_report`, `print_full_report`, `render_metrics_markdown`, `save_text_report`. Writes text + markdown + bias reports to `examples/out/`. |
| `03_dedup.py` | nemotron_agentic_v1 + txt360 | `find_duplicates`, `measure_overlap`, `dedup_within`, `dedup_cross`, `format_duplicate_report`, `format_report`. The richest overlap is between these two: TxT360's 1.4 M re-generated trajectories share seed keys with Nemotron-Agentic-v1's tool-calling split. |
| `04_semantic_pipeline.py` | Dolci | `prepare_semantic_layer_inputs`, `build_prompt` (what the LLM would see), then a synthetic classifier output to demonstrate `cross_tabulate`, `compute_quality_summary`, `filter_ams` without needing a live vLLM endpoint. |

The LLM-driven layer that *produces* the semantic classifications lives in `fcanalysis.semantic` and needs vLLM infrastructure; it is not exercised here.
