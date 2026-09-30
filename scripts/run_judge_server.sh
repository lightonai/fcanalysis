#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Launch a vLLM server hosting Qwen3.5-397B-A17B-FP8 as the fcanalysis
# semantic-layer JUDGE (the no-FC turn classifier).
#
# Adapted from tau2-bench/run_usersim_server.sh. Differences for the judge:
#   - 397B-A17B-FP8 on ALL 8 H100-80GB (TP=8): FP8 weights are 406 GB
#     (~47 GiB/GPU); at util=0.9 that leaves ~24 GiB/GPU for KV cache. The
#     BF16 variant (~794 GB) does NOT fit on 8×80 — use the -FP8 checkpoint.
#   - --default-chat-template-kwargs '{"enable_thinking": false}': the judge
#     runs NON-thinking. Stage-1 majority vote needs temperature diversity
#     (thinking ignores temperature), and stage-2 is a strong single-pass
#     verify on a 397B model. Disabling thinking server-side means the
#     fcanalysis pipeline needs NO code change. (No --reasoning-parser, since
#     there is no <think> block to parse.)
#   - NO --tool-call-parser: the judge emits JSON classifications via
#     response_format=json_object, not tool calls.
#   - Batch invariance OFF by default: the judge samples at temperature>0 for
#     vote diversity, so cross-run determinism is pointless, and
#     VLLM_BATCH_INVARIANT is risky on a large MoE. Pass --batch-invariant to
#     force it on.
#   - Its own port / container / caches so it coexists with the tau2 servers.
#
# Usage:
#   ./scripts/run_judge_server.sh                       # 397B on GPUs 0-7, port 8003
#   ./scripts/run_judge_server.sh --model Qwen/Qwen3.5-122B-A10B-FP8 --gpus 4,5,6,7
#   ./scripts/run_judge_server.sh --no-wait             # don't block on readiness
#
# NOTE: the default model is very large. Set HF_HOME to an existing model cache
# or pre-download it before starting the server. Smaller compatible judge models
# can be selected with --model, but their prompts require separate validation.
# ==============================================================================

GPUS="0,1,2,3,4,5,6,7"
PORT=8003
CONTAINER_NAME="vllm-judge"
VLLM_IMAGE="vllm/vllm-openai:v0.19.0"
MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
SERVE_NAME=""
SHM_SIZE="32g"   # TP=8 NCCL/IPC needs more shared memory than TP=4
TP_SIZE=""
# Measured on nemotron_agentic_v2 (199,115 samples): prompt p50 ~4.8K, p99
# ~7.6K, MAX ~22K tokens. With the 3K stage-2 output budget the longest
# sequence is ~25K, so 32768 covers 100% of samples with headroom while
# leaving more KV for concurrency than 64K would. (16384 would truncate ~10
# outlier samples.) Bump toward 40960 only if you enable thinking (8K output).
# Sizing rule: max-model-len >= (largest prompt to include) + 3072 stage-2 output
# + margin. 40960 admits prompts up to ~37k tokens (the user's "include ~32k
# samples" policy, 2026-06-05); KV cost is driven by ACTUAL lengths (p50 3-5k),
# so this is nearly free vs 32768 — just ignore the scarier "Maximum concurrency"
# boot line (it assumes full-length requests) and watch real KV% instead.
MAX_MODEL_LEN=40960
GPU_MEM_UTIL=0.9
# Throughput knobs (empty => vLLM defaults; vLLM's max_num_batched_tokens is
# 8192). The judge workload is PREFILL-heavy (long-ish prompts, short JSON
# outputs), so widening the per-step prefill budget can lift throughput. To test:
#   --max-num-batched-tokens 16384
MAX_NUM_BATCHED_TOKENS=""
MAX_NUM_SEQS=""
BATCH_INVARIANT=false
# 397B-FP8 weight load + 8-way CUDA-graph capture can take a while; first run
# may also download ~406 GB. Pre-download (see header) or use --no-wait.
MAX_WAIT=3600
NO_WAIT=false

CACHE_ROOT="${XDG_CACHE_HOME:-${HOME:?HOME must be set}/.cache}"
HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
VLLM_CACHE="${VLLM_CACHE_DIR:-$CACHE_ROOT/vllm/fcanalysis-judge}"
TMP_DIR="${JUDGE_TMP_DIR:-${TMPDIR:-/tmp}/fcanalysis-judge}"
TRITON_CACHE="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton/fcanalysis-judge}"

usage() {
    cat <<'USAGE'
Usage: ./scripts/run_judge_server.sh [options]

Options:
  --model <hf_id>          HuggingFace model ID (default: Qwen/Qwen3.5-397B-A17B-FP8)
  --serve-name <name>      vLLM API name (default: same as --model)
  --gpus <ids>             Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
  --port <port>            Server port (default: 8003)
  --tp-size <N>            Tensor parallel size (default: number of GPUs)
  --max-model-len <N>      vLLM --max-model-len (default: 40960)
  --gpu-mem-util <F>       vLLM --gpu-memory-utilization (default: 0.9)
  --max-num-batched-tokens <N>  vLLM per-step prefill budget (default: vLLM's 8192; try 16384)
  --max-num-seqs <N>       vLLM max running batch size (default: vLLM default)
  --batch-invariant        Enable VLLM_BATCH_INVARIANT (off by default; risky on MoE)
  --container-name <name>  Docker container name (default: vllm-judge)
  --vllm-image <image>     vLLM Docker image (default: vllm/vllm-openai:v0.19.0)
  --shm-size <size>        Shared memory size (default: 32g)
  --vllm-cache-dir <dir>   Mounted as /root/.cache/vllm (default: XDG cache)
  --tmp-dir <dir>          Mounted as /tmp (default: TMPDIR/fcanalysis-judge)
  --triton-cache-dir <dir> Mounted as /root/.triton (default: XDG cache)
  --max-wait <seconds>     Max seconds to wait for readiness (default: 3600)
  --no-wait                Don't wait for the server to be ready
  -h, --help               Show this help
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)             MODEL="$2"; shift 2 ;;
        --serve-name)        SERVE_NAME="$2"; shift 2 ;;
        --gpus)              GPUS="$2"; shift 2 ;;
        --port)              PORT="$2"; shift 2 ;;
        --tp-size)           TP_SIZE="$2"; shift 2 ;;
        --max-model-len)     MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-mem-util)      GPU_MEM_UTIL="$2"; shift 2 ;;
        --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
        --max-num-seqs)      MAX_NUM_SEQS="$2"; shift 2 ;;
        --batch-invariant)   BATCH_INVARIANT=true; shift ;;
        --container-name)    CONTAINER_NAME="$2"; shift 2 ;;
        --vllm-image)        VLLM_IMAGE="$2"; shift 2 ;;
        --shm-size)          SHM_SIZE="$2"; shift 2 ;;
        --vllm-cache-dir)    VLLM_CACHE="$2"; shift 2 ;;
        --tmp-dir)           TMP_DIR="$2"; shift 2 ;;
        --triton-cache-dir)  TRITON_CACHE="$2"; shift 2 ;;
        --max-wait)          MAX_WAIT="$2"; shift 2 ;;
        --no-wait)           NO_WAIT=true; shift ;;
        -h|--help)           usage ;;
        *)                   echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
TP_SIZE="${TP_SIZE:-$NUM_GPUS}"
SERVE_NAME="${SERVE_NAME:-$MODEL}"

command -v docker >/dev/null 2>&1 || {
    echo "ERROR: docker is required to launch the judge server." >&2
    exit 1
}

echo "============================================================"
echo "vLLM fcanalysis Judge Server (Qwen3.5, non-thinking)"
echo "============================================================"
echo "Model:            $MODEL"
echo "Serve name:       $SERVE_NAME"
echo "GPUs:             $GPUS ($NUM_GPUS GPUs, TP=$TP_SIZE)"
echo "Port:             $PORT"
echo "Container:        $CONTAINER_NAME"
echo "Image:            $VLLM_IMAGE"
echo "max_model_len:    $MAX_MODEL_LEN"
echo "gpu_mem_util:     $GPU_MEM_UTIL"
echo "max_num_batched:  ${MAX_NUM_BATCHED_TOKENS:-<vllm default 8192>}"
echo "max_num_seqs:     ${MAX_NUM_SEQS:-<vllm default>}"
echo "Batch invariant:  $BATCH_INVARIANT"
echo "HF_HOME:          $HF_HOME"
echo "vLLM cache:       $VLLM_CACHE"
echo "tmp dir:          $TMP_DIR"
echo "Triton cache:     $TRITON_CACHE"
echo "============================================================"

mkdir -p "$HF_HOME" "$VLLM_CACHE" "$TMP_DIR" "$TRITON_CACHE"

echo ""
echo "Stopping existing container '$CONTAINER_NAME' if any..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

DOCKER_ARGS=(
    run -d
    --name "$CONTAINER_NAME"
    --gpus "\"device=$GPUS\""
    --shm-size "$SHM_SIZE"
    -p "${PORT}:8000"
    -v "${HF_HOME}:/hf_home"
    -v "${VLLM_CACHE}:/root/.cache/vllm"
    -v "${TMP_DIR}:/tmp"
    -v "${TRITON_CACHE}:/root/.triton"
    -e HF_HOME=/hf_home
)

if [[ -n "${HF_TOKEN:-}" ]]; then
    # Pass the host variable by name so its value is not printed in the command.
    DOCKER_ARGS+=(--env HF_TOKEN)
fi

if [[ "$BATCH_INVARIANT" == "true" ]]; then
    DOCKER_ARGS+=(-e VLLM_BATCH_INVARIANT=1)
fi

DOCKER_ARGS+=(
    "$VLLM_IMAGE"
    --model "$MODEL"
    --served-model-name "$SERVE_NAME"
    --tensor-parallel-size "$TP_SIZE"
    --port 8000
    --trust-remote-code
    --language-model-only
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
)
if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
    DOCKER_ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
if [[ -n "$MAX_NUM_SEQS" ]]; then
    DOCKER_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
# FLASH_ATTN is a solid default and is required if --batch-invariant is enabled.
DOCKER_ARGS+=(--attention-backend FLASH_ATTN)
# Judge runs non-thinking: preserves stage-1 temperature/vote diversity and
# needs no pipeline code change. Remove this (and add --reasoning-parser qwen3)
# only if you later want per-request thinking on stage-2.
DOCKER_ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')

printf "Running: docker"
printf " %q" "${DOCKER_ARGS[@]}"
printf "\n"
docker "${DOCKER_ARGS[@]}"

if [[ "$NO_WAIT" == "true" ]]; then
    echo ""
    echo "Container started. Skipping readiness check (--no-wait)."
    echo "Tail logs:  docker logs -f ${CONTAINER_NAME}"
    echo "Stop:       docker rm -f ${CONTAINER_NAME}"
    exit 0
fi

echo ""
echo "Waiting for server to be ready (first run may download ~406 GB)..."
WAITED=0
while ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
    sleep 10
    WAITED=$((WAITED + 10))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "ERROR: Server did not start within ${MAX_WAIT}s"
        echo "Last logs:"
        docker logs --tail 100 "$CONTAINER_NAME"
        exit 1
    fi
    echo "Waiting... (${WAITED}s)"
done

echo ""
echo "Server is ready on port ${PORT}."
echo "Point the fcanalysis judge at it (no code change needed):"
echo "  VLLM_KEY=EMPTY uv run --locked python -m fcanalysis.semantic \\"
echo "    --datasets nemotron_agentic_v2 --limit 150 \\"
echo "    --base-url http://localhost:${PORT}/v1 \\"
echo "    --model ${SERVE_NAME} --api-key-env VLLM_KEY \\"
echo "    --verify-and-correct-flags --no-cache-warmup \\"
echo "    --concurrency 48 \\"
echo "    --output-dir pilot_local"
echo "  Notes for the local (vLLM) judge:"
echo "    --no-cache-warmup : this model has NO prefix caching (vLLM disables APC"
echo "                        for hybrid GDN/linear-attention models), so the"
echo "                        warmup-then-fan-out path only adds latency here."
echo "    --concurrency 48  : KV holds ~300k tokens (vLLM reports max ~34.6x"
echo "                        concurrency at 32k); the 256 unknown-model default"
echo "                        oversubscribes ~7x. Watch 'docker logs ${CONTAINER_NAME}'"
echo "                        for preemption and adjust."
echo "    Do NOT pass --verify-thinking: stage-2 thinking is not wired for Qwen"
echo "                        (collides with json_object) -- keep stage 2 non-thinking."
echo "Tail logs:  docker logs -f ${CONTAINER_NAME}"
echo "Stop:       docker rm -f ${CONTAINER_NAME}"
