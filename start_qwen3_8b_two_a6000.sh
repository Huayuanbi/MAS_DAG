#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 /absolute/path/to/Qwen3-8B" >&2
  exit 2
fi

MODEL_PATH="$1"
VLLM_BIN="${VLLM_BIN:-vllm}"
LOG_DIR="${LOG_DIR:-logs/a6000_vllm}"
mkdir -p "$LOG_DIR"

for spec in "0:8000" "1:8001"; do
  gpu="${spec%%:*}"
  port="${spec##*:}"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name qwen3-8b \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --generation-config vllm \
    --host 127.0.0.1 \
    --port "$port" \
    >"$LOG_DIR/gpu${gpu}_port${port}.log" 2>&1 &
  echo "$!" >"$LOG_DIR/gpu${gpu}_port${port}.pid"
done

for port in 8000 8001; do
  for attempt in $(seq 1 120); do
    if curl --fail --silent "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "ready=http://127.0.0.1:${port}/v1"
      break
    fi
    if (( attempt == 120 )); then
      echo "service on port ${port} did not become ready; inspect ${LOG_DIR}" >&2
      exit 1
    fi
    sleep 5
  done
done
