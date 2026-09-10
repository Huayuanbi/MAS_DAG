#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
DATA_DIR="$ROOT_DIR/data/mmlu_pro/router_splits_seed42"
OUTPUT_DIR="$ROOT_DIR/checkpoints/mmlu_pro_router"
LOG_DIR="$ROOT_DIR/logs/mmlu_pro_router"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

run_experiment() {
  local name="$1"
  local learning_rate="$2"
  local preferred_weight="$3"
  local seed="$4"

  "$PYTHON_BIN" "$ROOT_DIR/train.py" \
    --data "$DATA_DIR/train.json" \
    --validation-data "$DATA_DIR/validation.json" \
    --objective pairwise \
    --max-records 0 \
    --epochs 30 \
    --batch-size 32 \
    --lr "$learning_rate" \
    --ranking-temperature 1.0 \
    --preferred-fit-weight "$preferred_weight" \
    --min-reward-gap 0.2 \
    --max-pairs-per-question 1 \
    --reward-gap-scale 0.2 \
    --reward-gap-power 0.5 \
    --early-stopping-patience 6 \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --embedding-device cpu \
    --device cpu \
    --seed "$seed" \
    --output "$OUTPUT_DIR/$name.pt" \
    >"$LOG_DIR/$name.train.log" 2>&1

  "$PYTHON_BIN" "$ROOT_DIR/evaluate_topology.py" \
    --checkpoint "$OUTPUT_DIR/$name.pt" \
    --data "$DATA_DIR/validation.json" \
    --fixed-best-generator star \
    --device cpu \
    --output "$OUTPUT_DIR/$name.validation.json" \
    >"$LOG_DIR/$name.validation.log" 2>&1
}

# Validation-only model selection. The isolated test.json is intentionally
# untouched until all three validation results have been compared.
run_experiment lr3e5_seed7 3e-5 0.05 7
run_experiment lr1e4_seed7 1e-4 0.05 7
run_experiment lr3e5_seed17 3e-5 0.05 17

echo "validation experiments complete; test split has not been evaluated"
