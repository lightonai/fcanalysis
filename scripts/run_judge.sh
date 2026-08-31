#!/usr/bin/env bash
# ==============================================================================
# THE standardized semantic-judge pipeline — one command per (new) dataset.
#
#   ./scripts/run_judge.sh <output_dir> <dataset> [dataset ...]
#
#   e.g. ./scripts/run_judge.sh semantic_results_mynewset mynewset
#        SPLIT=medium ./scripts/run_judge.sh semantic_results_txt360_med txt360
#
# What it runs (the validated recipe):
#   stage 1: v3 prompt, 3 generations, majority vote        (recall-tuned)
#   stage 2: strengthened prompt (fabricated-claims rule), VOTES=2 independent
#            samples per flagged turn, agreement gate:
#              all anti -> confirmed (+ correction)
#              all justified -> overturned
#              split -> justified + contested=true           (precision-first)
#   resume loop: up to MAX_PASSES passes; stops when error rows hit 0 or
#            stop decreasing (over-length/pathological stragglers don't loop).
#   summary: per-file rows / anti / contested / error counts.
#
# Before the FULL run on any NEW dataset (meta-rule, do not skip):
#   1. length pre-flight: max-model-len must be >= inclusion cutoff + 3072 + margin
#      (server default is 40960 => includes prompts up to ~37k tokens);
#   2. a --limit 150 pilot + hand-adjudication of ~12 disagreements
#      (prompts are not portable across models or datasets).
#
# Env overrides: CONCURRENCY (128), MAX_PASSES (3), SPLIT (high, txt360 only),
#                VOTES (2), BASE_URL, MODEL, VLLM_KEY, UV.
# Resume-safe: re-running skips classified samples. Never overwrites other dirs.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CALLER_DIR="$PWD"
OUTPUT_DIR="${1:?usage: run_judge.sh <output_dir> <dataset> [dataset ...]}"
shift
[ "$#" -ge 1 ] || { echo "usage: run_judge.sh <output_dir> <dataset> [dataset ...]" >&2; exit 1; }
DATASETS=("$@")

if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$CALLER_DIR/$OUTPUT_DIR"
fi

CONCURRENCY="${CONCURRENCY:-128}"
MAX_PASSES="${MAX_PASSES:-3}"
SPLIT="${SPLIT:-high}"
VOTES="${VOTES:-2}"
BASE_URL="${BASE_URL:-http://localhost:8003/v1}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
UV="${UV:-uv}"
export VLLM_KEY="${VLLM_KEY:-EMPTY}"

cd "$REPO"
mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# semantic judge — standardized pipeline"
echo "# start:    $(date -Is)"
echo "# datasets: ${DATASETS[*]} (split=$SPLIT)  ->  $OUTPUT_DIR"
echo "# stage2-votes=$VOTES  concurrency=$CONCURRENCY"
echo "############################################################"

echo "Checking judge server at $BASE_URL/models ..."
if ! curl -sf "$BASE_URL/models" >/dev/null 2>&1; then
    echo "ERROR: judge server not responding. Start scripts/run_judge_server.sh first." >&2
    exit 1
fi
echo "Server OK."

count_errors() {
    JUDGE_OUTPUT_DIR="$OUTPUT_DIR" "$UV" run --locked python -c '
import glob
import json
import os

count = 0
for filename in glob.glob(os.path.join(os.environ["JUDGE_OUTPUT_DIR"], "*.jsonl")):
    with open(filename, encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and "error" in json.loads(line):
                count += 1
print(count)
'
}

prev=-1
for pass in $(seq 1 "$MAX_PASSES"); do
    echo ""
    echo "===== PASS $pass / $MAX_PASSES   ($(date -Is)) ====="
    "$UV" run --locked python -m fcanalysis.semantic \
        --datasets "${DATASETS[@]}" --split "$SPLIT" \
        --base-url "$BASE_URL" --model "$MODEL" --api-key-env VLLM_KEY \
        --verify-and-correct-flags --verify-votes "$VOTES" \
        --no-cache-warmup --concurrency "$CONCURRENCY" \
        --filter-mode all --output-dir "$OUTPUT_DIR"
    errs=$(count_errors)
    echo "===== after pass $pass: $errs error rows ====="
    if [ "$errs" -eq 0 ] || [ "$errs" -eq "$prev" ]; then
        echo "(errors at 0 or stabilized — stopping the resume loop)"
        break
    fi
    prev=$errs
done

echo ""
echo "===== FINAL SUMMARY   ($(date -Is)) ====="
JUDGE_OUTPUT_DIR="$OUTPUT_DIR" "$UV" run --locked python -c '
import glob
import json
import os

anti_categories = {
    "ANTI_MANUAL_SOLVE",
    "ANTI_UNJUSTIFIED_REFUSAL",
    "ANTI_PRESSURE_CAVE",
    "OTHER_UNJUSTIFIED",
}
for filename in sorted(
    glob.glob(os.path.join(os.environ["JUDGE_OUTPUT_DIR"], "*.jsonl"))
):
    rows = ok = errors = turns = anti = contested = 0
    with open(filename, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            result = json.loads(line)
            if "error" in result:
                errors += 1
                continue
            ok += 1
            for classification in result.get("classifications", []):
                turns += 1
                if classification.get("category") in anti_categories:
                    anti += 1
                if classification.get("contested"):
                    contested += 1
    print(
        f"  {os.path.basename(filename)}: rows={rows} ok={ok} errors={errors} "
        f"turns={turns} anti={anti} contested={contested}"
    )
'
echo "ALL PASSES DONE"
