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
#                VOTES (2), BASE_URL, MODEL.
# Resume-safe: re-running skips classified samples. Never overwrites other dirs.
# ==============================================================================

set -uo pipefail

REPO="/mnt/nfs/ytahtah/fcanalysis-clean"
OUTPUT_DIR="${1:?usage: run_judge.sh <output_dir> <dataset> [dataset ...]}"
shift
[ "$#" -ge 1 ] || { echo "usage: run_judge.sh <output_dir> <dataset> [dataset ...]" >&2; exit 1; }
DATASETS=("$@")

CONCURRENCY="${CONCURRENCY:-128}"
MAX_PASSES="${MAX_PASSES:-3}"
SPLIT="${SPLIT:-high}"
VOTES="${VOTES:-2}"
BASE_URL="${BASE_URL:-http://localhost:8003/v1}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"

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
    .venv/bin/python -c "import json,glob
n=0
for f in glob.glob('$OUTPUT_DIR/*.jsonl'):
    for line in open(f):
        line=line.strip()
        if not line: continue
        try:
            if 'error' in json.loads(line): n+=1
        except Exception: pass
print(n)"
}

prev=-1
for pass in $(seq 1 "$MAX_PASSES"); do
    echo ""
    echo "===== PASS $pass / $MAX_PASSES   ($(date -Is)) ====="
    VLLM_KEY=EMPTY .venv/bin/python -m fcanalysis.semantic \
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
.venv/bin/python -c "
import json, glob
ANTI={'ANTI_MANUAL_SOLVE','ANTI_UNJUSTIFIED_REFUSAL','ANTI_PRESSURE_CAVE','OTHER_UNJUSTIFIED'}
for f in sorted(glob.glob('$OUTPUT_DIR/*.jsonl')):
    rows=ok=err=turns=anti=contested=0
    for line in open(f):
        line=line.strip()
        if not line: continue
        rows+=1
        try: d=json.loads(line)
        except Exception: continue
        if 'error' in d: err+=1; continue
        ok+=1
        for c in d.get('classifications',[]):
            turns+=1
            if c.get('category') in ANTI: anti+=1
            if c.get('contested'): contested+=1
    print(f'  {f.split(chr(47))[-1]}: rows={rows} ok={ok} errors={err} turns={turns} anti={anti} contested={contested}')
"
echo "ALL PASSES DONE"
