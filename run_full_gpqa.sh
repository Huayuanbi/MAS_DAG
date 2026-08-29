#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/gzy/miniforge3/envs/vllm-cu124/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/gzy/Qwen3-8B}"
SERVED_MODEL="${SERVED_MODEL:-qwen3-8b}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"

cd "$ROOT"
"$PYTHON" run_mas.py \
  --input data/gpqa/train_candidate_graphs.json \
  --output data/gpqa/train_candidate_graphs_scored_with_outputs.json \
  --backend vllm \
  --model "$SERVED_MODEL" \
  --tokenizer "$MODEL_PATH" \
  --base-url "$BASE_URL" \
  --evaluator auto \
  --max-new-tokens 3072 \
  --concurrency 8 \
  --request-timeout 900 \
  --checkpoint-every 1 \
  --store-node-outputs \
  --resume
