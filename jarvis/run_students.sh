#!/usr/bin/env bash
# France students v5: mDeBERTa teacher-student on run30's cleaned France pseudo-labels (decoys = hard negatives).
#   bash jarvis/run_students.sh smoke   # ~3 min check
#   bash jarvis/run_students.sh full    # two seeds -> ~/out_v5/s42, ~/out_v5/s7
set -euo pipefail
cd "$(dirname "$0")/.."
B="$HOME/da_france_v5"; O="$HOME/out_v5"; mkdir -p "$O" logs
ARGS="--bundle $B --model microsoft/mdeberta-v3-base --da none --pseudo_w 1.0 --aug 0.1 --epochs 1 --bs 32 --lr 2e-5"
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "${1:-smoke}" = smoke ]; then
  python src/da_encoder.py $ARGS --out "$O/smoke" --smoke 2>&1 | tee logs/students_smoke.log
else
  for s in 42 7; do
    python src/da_encoder.py $ARGS --seed $s --out "$O/s$s" 2>&1 | tee logs/students_s$s.log
  done
  cd "$HOME" && zip -r out_v5.zip out_v5 -x "out_v5/smoke/*" && ls -la out_v5.zip
fi
