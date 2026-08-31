# Semantic classification

`fcanalysis.semantic` is an optional research pipeline for classifying real
conversation turns where the assistant did not call an available tool. It uses
an OpenAI-compatible chat-completions endpoint. Dataset loading, structural
analysis, and validation do not require a judge endpoint.

## Semantic method

The release exposes one semantic-analysis method for function-calling training
data: classify every real conversation turn where the assistant made no tool
call, given the tools available in that sample (which may be empty), then verify
and, when necessary, correct the turns initially judged to be anti-patterns.

### 1. Prepare the judge input

For each sample, the pipeline identifies its no-call real turns. The judge sees:

- the available tool definitions;
- the full conversation with positional message indices; and
- the indices of the no-call turns to classify.

Thinking-tag spans and structured `reasoning_content` are removed from the copy
shown to the judge. The source `ConversationSample` is not mutated.

### 2. Ensemble classification

The pipeline requests three independent, non-thinking JSON classifications for
the same sample. Each response must contain exactly one valid entry for every
requested turn, using a supported category, integer turn index, Boolean
`justified` value, and textual reason. Empty, malformed, schema-invalid, or
incomplete responses are retried; if all three generation slots do not produce
valid responses, the sample is emitted as an error instead of receiving a
partial vote.

For each turn, the category with the most votes becomes the first-pass result.
The output retains the vote counts and a representative reason from the winning
category.

The justified categories describe cases where not calling a tool was correct:

- `S3_CLARIFICATION`: a required parameter is missing and the assistant asks
  for it;
- `S_PREREQUISITE`: the agent policy requires unfinished authentication,
  verification, consent, or another mandatory prerequisite;
- `S4_DIRECT_ANSWER`: no available tool matches and a direct answer is
  appropriate;
- `S5_DECLINE`: no tool matches and the assistant appropriately declines;
- `S6_NEAR_MISS`: a superficially related tool cannot perform the requested
  operation;
- `S10_NO_TOOLS`: no tools are defined;
- `M3_IRRELEVANT`: this turn does not match a tool in a multi-turn sample;
- `M5_PARTIAL_PARAMS`: the user supplied only some required parameters;
- `M8_JUSTIFIED_PERSISTENCE`: the assistant maintains a correct refusal after
  user pressure;
- `E1_EXPLANATION`: the user requests explanation rather than execution and no
  relevant retrieval tool exists; and
- `OTHER_JUSTIFIED`: another justified reason not covered above.

The anti-pattern categories mean that the assistant should not have produced
the observed no-call response:

- `ANTI_MANUAL_SOLVE`: a fitting tool and its required parameters were
  available, but the assistant performed the task directly;
- `ANTI_UNJUSTIFIED_REFUSAL`: a fitting tool was callable, but the assistant
  refused or asked an unnecessary question;
- `ANTI_PRESSURE_CAVE`: the assistant abandoned a previously correct refusal
  under user pressure; and
- `OTHER_UNJUSTIFIED`: another unjustified response, including fabricated tool
  calls, results, actions, or verification claims.

### 3. Targeted verification and correction

Every first-pass anti-pattern is then audited separately with the full
conversation and the agent's own policy. This is called *stage 2* in existing
CLI option names. The verifier explicitly checks whether:

1. an agent-policy prerequisite is incomplete;
2. a genuinely required parameter is missing;
3. the nominally matching tool cannot satisfy the request;
4. the information was already provided or the request was already served; or
5. the assistant fabricated a tool, result, action, or verification claim.

The default campaign recipe requests two independent verification verdicts per
flagged turn and applies a precision-first agreement rule:

- if every valid verdict confirms an anti-pattern, the final label remains
  unjustified and the verifier must provide a corrected assistant turn;
- if every valid verdict finds the no-call justified, the first-pass flag is
  overturned;
- if valid verdicts disagree, the final label is justified and the turn is
  marked `contested`; and
- if no valid verification verdict is returned, the first-pass result is kept
  without a verification object.

When fewer verification verdicts validate than requested, the output records
the valid count. The original first-pass category and reason are retained when
verification changes or confirms the result.

### 4. Output used by curation

Each JSONL row records the sample ID, the final per-turn category, whether the
no-call was justified, the final reason, first-pass vote counts, and the number
of valid ensemble generations. Verified flags also record the first-pass result
and the complete verification verdict. Downstream semantic filtering consumes
the final category, not the provisional first-pass category.

The prompts, response validation, vote logic, retry limits, and agreement rule
are implemented in `src/fcanalysis/semantic.py`. Judge model identity, model
revision, serving software, and decoding behavior remain explicit external
inputs and must be recorded for each run.

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

`scripts/run_judge.sh` runs the complete ensemble, verification, and correction
recipe with bounded resume passes. It accepts a caller-owned output directory
and resolves the repository from its own location:

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
4. record model identity and revision, serving version, decoding parameters,
   concurrency, source commit, and dataset revision; and
5. inspect output and retry errors before using classifications for filtering.

After a complete run, validate the correction invariant:

```sh
uv run python scripts/validate_invariant.py semantic_results_run
```

The checker fails on a missing or empty input, malformed JSON, or a justified
classification carrying a correction. It reports anti-pattern rows without a
usable correction separately so they can be investigated before filtering.

## Evidence boundary

Semantic output is model-dependent annotation, not ground truth. The shipped
prompts and tests establish pipeline behavior; they do not establish judge
accuracy for every dataset, model, language, or domain. Reproducing a run also
requires the exact judge model revision, serving configuration, decoding
parameters, source commit, dataset revision, and stochastic outputs; the source
code alone is not an inference-result artifact.

Downstream users should report these limitations and must not infer causal
model-quality or benchmark-performance effects from classifier output alone.
