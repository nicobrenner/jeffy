#!/bin/bash
# Setup and run Pile scaling experiment on a Vast.ai GPU instance.
#
# Usage on the remote instance:
#   bash vast_setup.sh 20m          # train 20M model (2.1B tokens)
#   bash vast_setup.sh 100m         # train 100M model (2.1B tokens)
#   bash vast_setup.sh 20m 50e6     # quick test with 50M tokens

set -e

MODEL_SIZE="${1:?Usage: bash vast_setup.sh <model_size> [tokens]}"
TOKENS="${2:-2.1e9}"

echo "=== Pile Scaling Experiment ==="
echo "Model: ${MODEL_SIZE}"
echo "Tokens: ${TOKENS}"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q transformers datasets tokenizers numpy

# Run training with nohup so it survives SSH disconnect
echo "Starting training..."
mkdir -p pile_runs

nohup python3 train_pile_scaling.py \
    --model-size "${MODEL_SIZE}" \
    --tokens "${TOKENS}" \
    --seq-len 1024 \
    --batch-size 32 \
    --accumulate 4 \
    --lr 6e-4 \
    --eval-every 500 \
    --save-every 2000 \
    --output-dir pile_runs \
    > "pile_runs/train_${MODEL_SIZE}.log" 2>&1 &

PID=$!
echo "Training started as PID ${PID}"
echo "Log: pile_runs/train_${MODEL_SIZE}.log"
echo ""
echo "Monitor with: tail -f pile_runs/train_${MODEL_SIZE}.log"
echo "Check status: ps aux | grep train_pile"
