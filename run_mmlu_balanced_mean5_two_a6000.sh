#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:?export MODEL_PATH=/absolute/path/to/Qwen3-8B}"
CONCURRENCY="${CONCURRENCY:-16}"
DATA_DIR="$ROOT_DIR/data/mmlu_pro_balanced_660"
LOG_DIR="$ROOT_DIR/logs/mmlu_pro_balanced_660"
mkdir -p "$DATA_DIR/runs" "$LOG_DIR"

for port in 8000 8001; do
  if ! curl --fail --silent "http://127.0.0.1:${port}/v1/models" >/dev/null; then
    echo "Qwen service is not ready on port ${port}" >&2
    exit 1
  fi
done

"$PYTHON_BIN" "$ROOT_DIR/prepare_mmlu_balanced_full_candidates.py" --mode generate

for run in 1 2 3 4 5; do
  seed=$((1000 + run))
  pids=()
  for shard in 0 1; do
    port=$((8000 + shard))
    "$PYTHON_BIN" "$ROOT_DIR/run_mas.py" \
      --input "$DATA_DIR/candidates_shard${shard}.json" \
      --output "$DATA_DIR/runs/run${run}_shard${shard}.json" \
      --backend vllm \
      --model qwen3-8b \
      --tokenizer "$MODEL_PATH" \
      --base-url "http://127.0.0.1:${port}/v1" \
      --concurrency "$CONCURRENCY" \
      --max-context-tokens 16384 \
      --max-new-tokens 2048 \
      --temperature 0.7 \
      --seed "$seed" \
      --evaluator mmlu_pro \
      --checkpoint-every 1 \
      --store-node-outputs \
      --resume --retry-errors \
      >"$LOG_DIR/run${run}_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "run=$run complete"
done

"$PYTHON_BIN" "$ROOT_DIR/prepare_mmlu_balanced_full_candidates.py" --mode aggregate \
  2>&1 | tee "$LOG_DIR/aggregate.log"
