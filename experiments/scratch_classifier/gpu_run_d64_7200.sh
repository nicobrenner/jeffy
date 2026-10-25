#!/bin/bash
# GPU setup for RoPE byte d=64 at 7200 tok/param (extreme overtraining test)
# Run on any GPU instance with PyTorch pre-installed
set -e

echo "=== Setting up d=64 @ 7200 t/p on GPU ==="

# Download data and script from HuggingFace
pip install -q huggingface_hub

python3 -c "
from huggingface_hub import hf_hub_download
import os, shutil

os.makedirs('pile_runs/custom_vocab/byte', exist_ok=True)

for f, dest in [
    ('byte_data/train.bytes', 'pile_runs/custom_vocab/byte/train.bytes'),
    ('byte_data/val.bytes', 'pile_runs/custom_vocab/byte/val.bytes'),
    ('train_rope_byte.py', 'train_rope_byte.py'),
]:
    print(f'Downloading {f}...')
    path = hf_hub_download('nicobrenner/jeffy-pile-tokenized', f, repo_type='dataset')
    if not os.path.exists(dest):
        shutil.copy2(path, dest)
    print(f'  -> {dest}')
"

echo "=== Data ready, launching training ==="
echo "Model: 215K params, 92.4% reasoning ratio"
echo "Target: 7200 tok/param = 1.55B tokens"
echo ""

# batch_size=64 for GPU utilization (4x fewer steps, same total tokens)
python3 -u train_rope_byte.py \
    --d-model 64 \
    --n-layers 4 \
    --n-heads 4 \
    --tok-param 7200 \
    --batch-size 64 \
    --seq-len 256 \
    --lr 3e-4 \
    --device cuda \
    --data-dir pile_runs/custom_vocab/byte \
    --resume

echo ""
echo "=== Training complete! Uploading results ==="

python3 -c "
from huggingface_hub import HfApi
import os

api = HfApi()
run_dir = 'pile_runs/rope_byte/run_d64_tp7200'

for fname in ['results.json', 'best_model.pt', 'checkpoint.pt']:
    fpath = os.path.join(run_dir, fname)
    if os.path.exists(fpath):
        remote = f'rope_byte_results/d64_tp7200/{fname}'
        print(f'Uploading {fpath} -> {remote}')
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=remote,
            repo_id='nicobrenner/jeffy-pile-tokenized',
            repo_type='dataset',
        )
print('Results uploaded to HF!')
"
