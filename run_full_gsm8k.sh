#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yz/Documents/MAS_DAG/AGP-minimal"
PYTHON_BIN="/home/yz/.conda/envs/agp/bin/python"
MODEL_PATH="${MODEL_PATH:-/data1/yz/MAS_DAG/Qwen3-8B}"
MODEL_NAME="${MODEL_NAME:-qwen3-8b}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000/v1}"
INPUT_PATH="${INPUT_PATH:-${PROJECT_ROOT}/data/gsm8k/candidate_graphs.json}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/data/gsm8k/candidate_graphs_scored.json}"
CONCURRENCY="${CONCURRENCY:-12}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
TOKEN_PENALTY="${TOKEN_PENALTY:-0.0}"
TIME_PENALTY="${TIME_PENALTY:-0.0}"
LOG_PATH="${LOG_PATH:-${PROJECT_ROOT}/data/gsm8k/full_run.log}"

mkdir -p "$(dirname "${OUTPUT_PATH}")" "$(dirname "${LOG_PATH}")"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "agp Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "candidate dataset not found: ${INPUT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/tokenizer_config.json" ]]; then
  echo "local tokenizer not found: ${MODEL_PATH}" >&2
  exit 1
fi
if ! curl --noproxy '*' --fail --silent "${VLLM_URL}/models" >/dev/null; then
  echo "vLLM is not reachable at ${VLLM_URL}" >&2
  exit 1
fi

echo "input=${INPUT_PATH}"
echo "output=${OUTPUT_PATH}"
echo "vllm=${VLLM_URL} model=${MODEL_NAME}"
echo "concurrency=${CONCURRENCY} max_new_tokens=${MAX_NEW_TOKENS}"
echo "token_penalty=${TOKEN_PENALTY} time_penalty=${TIME_PENALTY}"
echo "log=${LOG_PATH}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" run_mas.py \
  --backend vllm \
  --model "${MODEL_NAME}" \
  --tokenizer "${MODEL_PATH}" \
  --base-url "${VLLM_URL}" \
  --input "${INPUT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --concurrency "${CONCURRENCY}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --token-penalty "${TOKEN_PENALTY}" \
  --time-penalty "${TIME_PENALTY}" \
  --resume \
  --retry-errors \
  2>&1 | tee -a "${LOG_PATH}"
