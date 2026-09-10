#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${GPU_INDEX:?set GPU_INDEX to a verified idle GPU}"
MODEL_PATH="${MODEL_PATH:-/data1/yz/MAS_DAG/Qwen3-8B}"
PORT="${PORT:-8020}"
INPUT="${INPUT:-data/gaia/validation_balanced28_candidates_v2.json}"
OUTPUT="${OUTPUT:-data/gaia/validation_balanced28_scored_qwen3_8b.json}"
VLLM_ENV="${VLLM_ENV:-/home/gzy/.conda/envs/vllm-cu124}"
VLLM_BIN="${VLLM_BIN:-$VLLM_ENV/bin/vllm}"
VLLM_LIBRARY_PATH="${VLLM_LIBRARY_PATH:-$VLLM_ENV/lib:/usr/local/cuda/lib64}"

CUDA_VISIBLE_DEVICES="$GPU_INDEX" LD_LIBRARY_PATH="$VLLM_LIBRARY_PATH" "$VLLM_BIN" serve \
  --model "$MODEL_PATH" --served-model-name qwen3-8b \
  --host 127.0.0.1 --port "$PORT" --dtype bfloat16 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if curl --noproxy '*' --fail --silent "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 5
done
curl --noproxy '*' --fail --silent "http://127.0.0.1:${PORT}/v1/models" >/dev/null

python run_mas.py --backend vllm --model qwen3-8b --tokenizer "$MODEL_PATH" \
  --base-url "http://127.0.0.1:${PORT}/v1" --input "$INPUT" --output "$OUTPUT" \
  --concurrency 4 --max-new-tokens 1536 --max-context-tokens 16384 \
  --enable-tools --tool-max-steps 12 --store-node-outputs \
  --checkpoint-every 1 --resume --retry-errors

python analyze_gaia_results.py --input "$OUTPUT" --output "${OUTPUT%.json}_summary.json"
