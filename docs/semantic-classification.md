# Semantic classification

`fcanalysis.semantic` is an optional research pipeline for classifying real
conversation turns where the assistant did not call an available tool. It uses
an OpenAI-compatible chat-completions endpoint. Dataset loading, structural
analysis, and validation do not require a judge endpoint.

## Current method

Version 0.1.0 exposes prompt V3 only. The staged workflow is:

1. build one prompt covering the no-call real turns in a sample;
2. request three independent classifications;
3. take a validated majority vote; and
4. optionally re-check flagged turns with the stage-2 verification prompt and
   an agreement gate.

The categories separate several justified no-call reasons from four
anti-pattern labels:

- `ANTI_MANUAL_SOLVE`;
- `ANTI_UNJUSTIFIED_REFUSAL`;
- `ANTI_PRESSURE_CAVE`; and
- `OTHER_UNJUSTIFIED`.

Prompts, response validation, vote logic, retry limits, and usage accounting are
versioned in `src/fcanalysis/semantic.py`. Model and serving behavior remain
external inputs.

## Running the command

Inspect every option first:

```sh
uv run python -m fcanalysis.semantic --help
```

A small pilot against an OpenAI-compatible endpoint looks like:

```sh
export VLLM_KEY=EMPTY
uv run python -m fcanalysis.semantic \
  --datasets dolci \
  --limit 150 \
  --base-url http://localhost:8003/v1 \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --api-key-env VLLM_KEY \
  --verify-and-correct-flags \
  --verify-votes 2 \
  --no-cache-warmup \
  --filter-mode all \
  --output-dir semantic_results_pilot
```

Use a real secret only through the environment variable named by
`--api-key-env`. Do not write credentials into commands, source, output files,
or logs.

`scripts/run_judge.sh` wraps the same V3 plus stage-2 recipe with bounded resume
passes. It accepts a caller-owned output directory and resolves the repository
from its own location:

```sh
CONCURRENCY=48 VLLM_KEY=EMPTY \
  ./scripts/run_judge.sh semantic_results_run dolci
```

`scripts/run_judge_server.sh` is an optional Docker/vLLM launcher. Its default
397B FP8 model requires an appropriately large multi-GPU host and substantial
model storage. Cache paths are configurable; no cluster path is assumed.

## Required validation

Before a full run on a new dataset or judge model:

1. measure prompt lengths and set a context limit that covers the intended
   population plus output headroom;
2. run a bounded pilot;
3. manually adjudicate a sample of disagreements and failure cases;
4. record model identity, serving version, decoding parameters, concurrency,
   prompt version, and source revision; and
5. inspect output and retry errors before using classifications for filtering.

After a staged run, validate the correction invariant:

```sh
uv run python scripts/validate_invariant.py semantic_results_run
```

The checker fails on a missing or empty input, malformed JSON, or a justified
classification carrying a correction. It reports legacy anti-pattern rows that
lack a usable correction separately.

## Evidence boundary

Semantic output is model-dependent annotation, not ground truth. The current
V3 prompts and tests establish pipeline behavior; they do not establish judge
accuracy for every dataset, model, language, or domain.

An unfinished blind relabeling and adjudication effort exists outside this
public source release. It is not a completed validation set and must not be
described or used as a gold standard. Frozen historical result files also omit
some information needed to reproduce the exact stochastic inference run, such
as complete model revision, prompt hash, temperature, and run identity.

Downstream users should report these limitations and must not infer causal
model-quality or benchmark-performance effects from classifier output alone.
