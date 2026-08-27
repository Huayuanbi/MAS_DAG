#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/data/gzy/Qwen3-8B}"
MODEL_NAME="${MODEL_NAME:-qwen3-8b}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000/v1}"
CONCURRENCY="${CONCURRENCY:-12}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" run_mas.py \
  --backend vllm --model "${MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
  --base-url "${VLLM_URL}" \
  --input data/humaneval/candidate_graphs.json \
  --output data/humaneval/candidate_graphs_scored.json \
  --evaluator auto --evaluation-timeout 5 \
  --concurrency "${CONCURRENCY}" --max-new-tokens 2048 \
  --checkpoint-every 5 --token-penalty 0 --time-penalty 0 \
  --resume --retry-errors 2>&1 | tee -a data/humaneval/full_run.log
