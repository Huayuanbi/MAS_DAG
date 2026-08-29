#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "用法: $0 INPUT.json OUTPUT.json RUN.log" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
INPUT="$1"
OUTPUT="$2"
LOG="$3"

MODEL="${MODEL:-qwen3-4b}"
TOKENIZER="${TOKENIZER:-${ROOT}/Qwen3-4B}"
CONCURRENCY="${CONCURRENCY:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"

cd "${ROOT}"
nohup python "${SCRIPT_DIR}/run_mas.py" \
  --backend vllm \
  --model "${MODEL}" \
  --tokenizer "${TOKENIZER}" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --evaluator math \
  --concurrency "${CONCURRENCY}" \
  --temperature "${TEMPERATURE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --resume \
  --retry-errors \
  > "${LOG}" 2>&1 &

echo "pid=$! log=${LOG} output=${OUTPUT}"
