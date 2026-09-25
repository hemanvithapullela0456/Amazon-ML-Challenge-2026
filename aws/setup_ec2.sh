#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 24.04 EC2 instance. Run from the project folder:
#   bash aws/setup_ec2.sh                       # code only
#   bash aws/setup_ec2.sh ~/dataset.zip         # code + unzip the challenge dataset
set -euo pipefail
cd "$(dirname "$0")/.."
PROJ="$(pwd)"

sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip unzip tmux htop

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

if [ "${1:-}" != "" ]; then
  echo "unzipping $1 ..."
  unzip -q -o "$1" -d "$PROJ"
fi

# locate the dataset wherever the zip put it and remember it for run_all.sh
SRC1="$(find "$PROJ" -path "$PROJ/.venv" -prune -o -name 'train_source1.tsv' -print | grep -v __MACOSX | head -1 || true)"
if [ -n "$SRC1" ]; then
  DATA_DIR="$(dirname "$(dirname "$SRC1")")"
  echo "export ER_DATA_DIR=\"$DATA_DIR\"" > "$PROJ/.er_env"
  echo "dataset found at $DATA_DIR"
else
  echo "dataset not found yet - upload/unzip it, then re-run: bash aws/setup_ec2.sh <zip>"
fi

python -c "import lightgbm, rapidfuzz, sklearn; print('python env OK')"
nproc; free -h | head -2
