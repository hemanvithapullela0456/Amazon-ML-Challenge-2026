#!/usr/bin/env bash
# Full pipeline on the EC2 box. Run inside tmux so it survives SSH disconnects:
#   tmux new -s er        then   bash aws/run_all.sh
#   (detach: Ctrl+b then d   |   re-attach later: tmux attach -t er)
# Steps can be picked:  bash aws/run_all.sh eda blocking train predict
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
[ -f .er_env ] && source .er_env
mkdir -p logs
STEPS="${*:-eda blocking train predict}"
ts() { date +%H%M%S; }

for s in $STEPS; do
  echo "===== $s ($(date)) ====="
  case $s in
    eda)      python src/eda.py 2>&1 | tee "logs/eda_$(ts).log" ;;
    blocking) python src/blocking.py --split train 2>&1 | tee "logs/blocking_$(ts).log" ;;
    train)    python src/train.py --no-cache --country-check 2>&1 | tee "logs/train_$(ts).log" ;;
    predict)  python src/run_pipeline.py --no-cache 2>&1 | tee "logs/predict_$(ts).log" ;;
    baseline) python src/run_pipeline.py --baseline 2>&1 | tee "logs/baseline_$(ts).log" ;;
    *) echo "unknown step $s"; exit 1 ;;
  esac
done
echo "done. outputs:"; ls -la output/ 2>/dev/null || true
