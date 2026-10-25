#!/bin/bash
set -e

echo "=== MNIST GPU Run: 2D RoPE + 1D Classification Eval ==="
echo "Started: $(date)"

# Download scripts and data via wget (huggingface_hub pip is broken on this image)
HF_BASE="https://huggingface.co/datasets/nicobrenner/jeffy-pile-tokenized/resolve/main"

mkdir -p byte_mnist pile_runs/rope_byte

echo "Downloading scripts..."
wget -q "$HF_BASE/train_rope_byte_2d.py" -O train_rope_byte_2d.py
wget -q "$HF_BASE/train_rope_byte.py" -O train_rope_byte.py
wget -q "$HF_BASE/eval_mnist_classification.py" -O eval_mnist_classification.py

echo "Downloading MNIST byte data..."
wget -q "$HF_BASE/byte_mnist/train.bytes" -O byte_mnist/train.bytes
wget -q "$HF_BASE/byte_mnist/val.bytes" -O byte_mnist/val.bytes

# Check data exists
ls -la byte_mnist/
if [ ! -f byte_mnist/train.bytes ]; then
    echo "ERROR: MNIST byte data not found. Listing HF repo..."
    pip install -q huggingface_hub 2>/dev/null || true
    python3 -c "from huggingface_hub import list_repo_tree; [print(f.rfile_path) for f in list_repo_tree('nicobrenner/jeffy-pile-tokenized', repo_type='dataset')]" 2>/dev/null || true
    exit 1
fi

pip install -q scikit-learn torchvision 2>/dev/null

echo ""
echo "=========================================="
echo "  TRACK 1: 2D RoPE d=64 bidirectional"
echo "=========================================="

# 2D RoPE at 1800 tok/param
python3 -u train_rope_byte_2d.py \
    --d-model 64 --n-layers 4 --n-heads 4 \
    --tok-param 1800 --batch-size 64 \
    --bidirectional --eval-classification \
    --data-dir byte_mnist --device cuda

# 2D RoPE at 7200 tok/param (overtraining comparison)
python3 -u train_rope_byte_2d.py \
    --d-model 64 --n-layers 4 --n-heads 4 \
    --tok-param 7200 --batch-size 64 \
    --bidirectional --eval-classification \
    --data-dir byte_mnist --device cuda

# Also run causal 2D for comparison (same 2D positions but causal mask)
python3 -u train_rope_byte_2d.py \
    --d-model 64 --n-layers 4 --n-heads 4 \
    --tok-param 1800 --batch-size 64 \
    --causal --eval-classification \
    --data-dir byte_mnist --device cuda

echo ""
echo "=========================================="
echo "  TRACK 2: 1D Classification Eval"
echo "=========================================="

# Fix data-dir for eval script (it expects pile_runs/custom_vocab/byte_mnist)
mkdir -p pile_runs/custom_vocab/byte_mnist
cp byte_mnist/train.bytes pile_runs/custom_vocab/byte_mnist/
cp byte_mnist/val.bytes pile_runs/custom_vocab/byte_mnist/

python3 -u eval_mnist_classification.py

echo ""
echo "=========================================="
echo "  UPLOADING RESULTS"
echo "=========================================="

pip install -q huggingface_hub 2>/dev/null || true
python3 -c "
from huggingface_hub import HfApi
import glob, os
api = HfApi()
for d in glob.glob('pile_runs/rope_byte/*/'):
    for f in glob.glob(d + '*.json') + glob.glob(d + '*.pt'):
        path = f
        print(f'Uploading {path}...')
        try:
            api.upload_file(path_or_fileobj=path, path_in_repo=f'mnist_gpu_results/{path}',
                          repo_id='nicobrenner/jeffy-pile-tokenized', repo_type='dataset')
        except Exception as e:
            print(f'  Upload failed: {e}')
print('Done uploading.')
" || echo "Upload failed, results are in pile_runs/rope_byte/"

echo ""
echo "=== ALL DONE ==="
echo "Finished: $(date)"
