#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gzy/.conda/envs/vllm-cu124/bin/python3.11}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8001/v1}"
MODEL_PATH="${MODEL_PATH:-/data1/yz/MAS_DAG/Qwen3-8B}"
SERVED_MODEL="${SERVED_MODEL:-qwen3-8b}"
CONCURRENCY="${CONCURRENCY:-32}"
LOG_DIR="$ROOT_DIR/logs/aime_full_mean5"

mkdir -p "$LOG_DIR"

"$PYTHON_BIN" "$ROOT_DIR/prepare_aime_full_mean5.py" --mode prepare

for run in 1 2 3 4 5; do
  output="$ROOT_DIR/data/aime/full_candidate_mean5_run${run}.json"
  "$PYTHON_BIN" "$ROOT_DIR/run_mas.py" \
    --backend vllm \
    --model "$SERVED_MODEL" \
    --tokenizer "$MODEL_PATH" \
    --base-url "$BASE_URL" \
    --input "$ROOT_DIR/data/aime/candidate_graphs.json" \
    --output "$output" \
    --evaluator math \
    --concurrency "$CONCURRENCY" \
    --max-new-tokens 3072 \
    --max-context-tokens 8192 \
    --checkpoint-every 1 \
    --store-node-outputs \
    --resume \
    --retry-errors \
    2>&1 | tee "$LOG_DIR/run${run}.log"
done

"$PYTHON_BIN" "$ROOT_DIR/prepare_aime_full_mean5.py" --mode aggregate \
  2>&1 | tee "$LOG_DIR/aggregate.log"

echo "AIME full candidate Mean-5 complete"
