#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-8B}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-3}"
PORT="${PORT:-8001}"
LOG_DIR="$ROOT_DIR/logs/merged_semantic_train1000_v1"
mkdir -p "$LOG_DIR"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Set MODEL_PATH to the Qwen3-8B directory; not found: $MODEL_PATH" >&2
  exit 2
fi

if ! curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name qwen3-8b \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-seqs 24 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --generation-config vllm \
    --host 127.0.0.1 \
    --port "$PORT" >"$LOG_DIR/vllm_a6000_gpu${GPU_ID}.log" 2>&1 &
fi

for _ in $(seq 1 120); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready:-0}" != 1 ]]; then
  echo "vLLM did not become ready on port $PORT" >&2
  exit 1
fi

for run in 4 5; do
  RUN_INDEX="$run" \
  BASE_URL="http://127.0.0.1:${PORT}/v1" \
  MODEL_PATH="$MODEL_PATH" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash "$ROOT_DIR/run_merged_semantic_train1000.sh"
done
