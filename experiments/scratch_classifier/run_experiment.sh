#!/bin/bash
# Full Pythia scaling experiment: download, tokenize, train 20M and 100M models.
# Run on the Vast.ai GPU instance.
#
# Usage: nohup bash run_experiment.sh > experiment.log 2>&1 &

set -e

echo "=== Pythia Scaling Experiment ==="
echo "Started: $(date)"
echo ""

# Install deps if needed
pip install -q transformers datasets tokenizers numpy zstandard 2>/dev/null

# Save tokenizer locally to avoid repeated HF downloads
if [ ! -d ~/neox_tokenizer ]; then
    echo "Saving tokenizer locally..."
    python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
t.save_pretrained('/root/neox_tokenizer')
print('Tokenizer saved to /root/neox_tokenizer')
"
fi

DATA_DIR="/root/pile_runs/shared_data"
mkdir -p "$DATA_DIR"

# Step 1: Fast tokenize The Pile (download raw shard + tokenize)
if [ ! -f "$DATA_DIR/train.bin" ]; then
    echo ""
    echo "=== Step 1: Tokenizing The Pile ==="
    python3 /root/fast_tokenize_pile.py \
        --output-dir "$DATA_DIR" \
        --tokens 2.2e9 \
        --seq-len 1024
else
    echo "Tokenized data already exists in $DATA_DIR"
    ls -lh "$DATA_DIR"/*.bin
fi

# Step 2: Train 20M model (faster, ~1-2h on RTX 3090)
echo ""
echo "=== Step 2: Training 20M model ==="
echo "Started: $(date)"
python3 /root/train_pile_scaling.py \
    --model-size 20m \
    --tokens 2.1e9 \
    --seq-len 1024 \
    --batch-size 32 \
    --accumulate 4 \
    --lr 1e-3 \
    --eval-every 500 \
    --save-every 2000 \
    --output-dir /root/pile_runs \
    --data-dir "$DATA_DIR" \
    --tokenizer-path /root/neox_tokenizer

echo "20M training finished: $(date)"

# Step 3: Train 100M model (~3-5h on RTX 3090)
echo ""
echo "=== Step 3: Training 100M model ==="
echo "Started: $(date)"
python3 /root/train_pile_scaling.py \
    --model-size 100m \
    --tokens 2.1e9 \
    --seq-len 1024 \
    --batch-size 16 \
    --accumulate 8 \
    --lr 6e-4 \
    --eval-every 500 \
    --save-every 2000 \
    --output-dir /root/pile_runs \
    --data-dir "$DATA_DIR" \
    --tokenizer-path /root/neox_tokenizer

echo ""
echo "=== All training complete ==="
echo "Finished: $(date)"
echo ""
echo "Results:"
for d in /root/pile_runs/pile_*/; do
    if [ -f "$d/summary.json" ]; then
        echo "  $(basename $d):"
        python3 -c "import json; s=json.load(open('${d}summary.json')); print(f'    params={s[\"n_params\"]:,}, best_ppl={s[\"best_val_ppl\"]:.2f}, tok/param={s[\"tok_param_ratio\"]:.1f}, time={s[\"training_time_hours\"]:.1f}h')"
    fi
done
