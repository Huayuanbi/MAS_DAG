#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$ROOT_DIR/data/merged_semantic_train1000_v1"
OUTPUT_DIR="$INPUT_DIR/runs"
LOG_DIR="$ROOT_DIR/logs/merged_semantic_train1000_v1"
MODEL_PATH="${MODEL_PATH:-/data1/yz/MAS_DAG/Qwen3-8B}"
SERVED_MODEL="${SERVED_MODEL:-qwen3-8b}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8001/v1}"
BASE_URL_0="${BASE_URL_0:-$BASE_URL}"
BASE_URL_1="${BASE_URL_1:-$BASE_URL}"
PYTHON_BIN="${PYTHON_BIN:-/home/gzy/.conda/envs/vllm-cu124/bin/python}"
RUN_INDEX="${RUN_INDEX:-1}"

if (( RUN_INDEX < 1 || RUN_INDEX > 5 )); then
  echo "RUN_INDEX must be between 1 and 5" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
seed=$((2000 + RUN_INDEX))
pids=()
for shard in 0 1 2 3; do
  if (( shard < 2 )); then
    shard_base_url="$BASE_URL_0"
  else
    shard_base_url="$BASE_URL_1"
  fi
  input="$INPUT_DIR/candidates_shard${shard}.json"
  output="$OUTPUT_DIR/run${RUN_INDEX}_shard${shard}.json"
  log="$LOG_DIR/run${RUN_INDEX}_shard${shard}.log"
  "$PYTHON_BIN" "$ROOT_DIR/run_mas.py" \
    --input "$input" \
    --output "$output" \
    --backend vllm \
    --model "$SERVED_MODEL" \
    --tokenizer "$MODEL_PATH" \
    --base-url "$shard_base_url" \
    --evaluator auto \
    --max-context-tokens 8192 \
    --temperature 0.7 \
    --seed "$seed" \
    --concurrency 4 \
    --request-timeout 900 \
    --checkpoint-every 10 \
    --resume \
    --retry-errors \
    >"$log" 2>&1 &
  pids+=("$!")
  echo "started pid=$! run=$RUN_INDEX shard=$shard base_url=$shard_base_url output=$output"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
