#!/usr/bin/env bash
set -euo pipefail

# Qwen3-4B 可以单卡容纳。大量相互独立的候选 DAG 更适合使用四个单卡
# 数据并行副本，而不是让每次推理都通过 TP=4 跨四卡通信。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${ROOT}/Qwen3-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
# 以下两个调度限制作用于每个数据并行副本。四个副本合计最多可同时
# 调度约 4 * MAX_NUM_SEQS 个序列。
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port 8000 \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --enable-prefix-caching
