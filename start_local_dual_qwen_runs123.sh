#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/data1/yz/MAS_DAG/Qwen3-8B}"
VLLM_BIN="${VLLM_BIN:-/home/gzy/.conda/envs/vllm-cu124/bin/vllm}"
LOG_DIR="$ROOT_DIR/logs/merged_semantic_train1000_v1"
mkdir -p "$LOG_DIR"

start_server() {
  local gpu="$1"
  local port="$2"
  local log="$LOG_DIR/vllm_gpu${gpu}_${port}.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$VLLM_BIN" serve "$MODEL_PATH" \
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
    --port "$port" >"$log" 2>&1 &
  echo "$!"
}

wait_server() {
  local port="$1"
  for _ in $(seq 1 120); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      return
    fi
    sleep 2
  done
  echo "vLLM on port $port did not become ready" >&2
  exit 1
}

pid0=$(start_server 0 8001)
pid1=$(start_server 1 8002)
echo "vLLM pids: gpu0=$pid0 gpu1=$pid1"
wait_server 8001
wait_server 8002

for run in 1 2 3; do
  RUN_INDEX="$run" \
  BASE_URL_0=http://127.0.0.1:8001/v1 \
  BASE_URL_1=http://127.0.0.1:8002/v1 \
  bash "$ROOT_DIR/run_merged_semantic_train1000.sh"
done
