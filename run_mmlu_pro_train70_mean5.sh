#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="$ROOT_DIR/data/mmlu_pro/train70_single_best_candidates.json"
OUTPUT_DIR="$ROOT_DIR/data/mmlu_pro/train70_mean5_shards"
LOG_DIR="$ROOT_DIR/logs/mmlu_pro_train70_mean5"
MODEL_PATH="${MODEL_PATH:-/data1/yz/MAS_DAG/Qwen3-8B}"
SERVED_MODEL="${SERVED_MODEL:-qwen3-8b}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8001/v1}"
PYTHON_BIN="${PYTHON_BIN:-/home/gzy/.conda/envs/vllm-cu124/bin/python}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

TOTAL=8422
SHARDS=4
SHARD_SIZE=$(( (TOTAL + SHARDS - 1) / SHARDS ))

for run in 1 2 3 4 5; do
  seed=$((1000 + run))
  for shard in 0 1 2 3; do
    start=$((shard * SHARD_SIZE))
    remaining=$((TOTAL - start))
    if (( remaining <= 0 )); then
      continue
    fi
    count=$SHARD_SIZE
    if (( remaining < count )); then
      count=$remaining
    fi
    output="$OUTPUT_DIR/run${run}_shard${shard}.json"
    log="$LOG_DIR/run${run}_shard${shard}.log"
    "$PYTHON_BIN" "$ROOT_DIR/run_mas.py" \
      --input "$INPUT" \
      --output "$output" \
      --backend vllm \
      --model "$SERVED_MODEL" \
      --tokenizer "$MODEL_PATH" \
      --base-url "$BASE_URL" \
      --concurrency 2 \
      --max-context-tokens 8192 \
      --temperature 0.7 \
      --seed "$seed" \
      --query-start "$start" \
      --max-queries "$count" \
      --checkpoint-every 100 \
      --resume \
      --retry-errors \
      >"$log" 2>&1 &
    echo "$! run=$run shard=$shard start=$start count=$count output=$output"
  done
done

wait
