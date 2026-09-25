#!/usr/bin/env bash
# Cross-encoder on the GPU box. Everything else (blocking, features, LightGBM, stacking) runs on the laptop.
#   bash jarvis/run_gpu.sh smoke     # ~2-5 min end-to-end test; must pass before the real run
#   bash jarvis/run_gpu.sh ce        # real run -> ce_bundle/ce_train.parquet, ce_bundle/ce_test.parquet
# Override the model / params:  CE_ARGS="--epochs 1 --max_train 800000" bash jarvis/run_gpu.sh ce
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
BUNDLE="${BUNDLE:-$HOME/ce_bundle}"
CE_MODEL="${CE_MODEL:-microsoft/mdeberta-v3-base}"
CE_ARGS="${CE_ARGS:---folds 3 --epochs 2 --bs 64}"
ts() { date +%H%M%S; }
nvidia-smi --query-gpu=name,memory.total --format=csv

for s in "${@:-smoke}"; do
  echo "===== $s ($(date)) ====="
  case $s in
    smoke) python src/cross_encoder.py --bundle "$BUNDLE" --model "$CE_MODEL" --smoke 2>&1 | tee "logs/smoke_$(ts).log" ;;
    ce)    python src/cross_encoder.py --bundle "$BUNDLE" --model "$CE_MODEL" $CE_ARGS 2>&1 | tee "logs/ce_$(ts).log" ;;
    *) echo "unknown step $s"; exit 1 ;;
  esac
done
ls -la "$BUNDLE"
