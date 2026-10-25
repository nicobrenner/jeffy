#!/bin/bash
# =============================================================================
# HIGH-REPETITION OVERFITTING TEST — A100 40GB
# =============================================================================
# 800K model on 2M token micro-dataset repeated to 1000 tok/param
# Three phases: MLM pretrain → Causal from MLM → Baseline causal
#
# Conservative time estimate: ~2.5-3 hours total
# Cost at ~$1/hr (A100 40GB): ~$2.50-3.00
#
# PRE-REQUISITES:
# 1. Pre-tokenized micro data (2M tokens) must be at pile_runs/micro_train.bin
#    and validation data at pile_runs/shared_data/val.bin
# 2. Both must be uploaded to HF or the GPU before renting
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"
TRAIN_BIN="${WORK_DIR}/pile_runs/micro_train.bin"
VAL_BIN="${WORK_DIR}/pile_runs/shared_data/val.bin"
OUTPUT_BASE="${WORK_DIR}/pile_runs/repetition_test"

# Model: 800K (834,400 params)
# Target: 1000 tok/param = 834,400,000 tokens = ~834M tokens
# On 2M data = 417 repetitions
MODEL="800k"
TOTAL_TOKENS="834400000"
MLM_TOKENS="50000000"  # 50M tokens for MLM pretrain (25 reps)

# Training config - batch_size=64 for better GPU utilization on A100
BATCH=64
ACCUM=2
SEQ=1024
EVAL_EVERY=250
SAVE_EVERY=1000
LR="6e-4"

echo "$(date): === HIGH-REPETITION OVERFITTING TEST ==="
echo "  Model: ${MODEL} (834,400 params)"
echo "  Total tokens per phase: ${TOTAL_TOKENS} (1000 tok/param)"
echo "  Data: 2M tokens repeated ~417x"
echo "  MLM pretrain: ${MLM_TOKENS} tokens (25 reps)"
echo "  Batch: ${BATCH} x ${ACCUM} accumulate = $(( BATCH * ACCUM )) effective"
echo "  Tokens/step: $(( BATCH * ACCUM * SEQ ))"
echo ""

# Set up data directory with expected filenames
DATA_DIR="${OUTPUT_BASE}/data"
mkdir -p "${DATA_DIR}"

if [ ! -f "${TRAIN_BIN}" ]; then
    echo "ERROR: Training data not found at ${TRAIN_BIN}"
    echo "  Expected: 2M token micro-dataset (pre-tokenized)"
    exit 1
fi
if [ ! -f "${VAL_BIN}" ]; then
    echo "ERROR: Validation data not found at ${VAL_BIN}"
    exit 1
fi

# Symlink micro data as train.bin (the training script expects this name)
ln -sf "${TRAIN_BIN}" "${DATA_DIR}/train.bin"
ln -sf "${VAL_BIN}" "${DATA_DIR}/val.bin"

TRAIN_SIZE=$(stat -f%z "${TRAIN_BIN}" 2>/dev/null || stat -c%s "${TRAIN_BIN}" 2>/dev/null)
echo "  Train data: ${TRAIN_BIN} ($(( TRAIN_SIZE / 1024 / 1024 ))MB = ~$(( TRAIN_SIZE / 2 )) tokens)"
echo "  Data dir: ${DATA_DIR} (symlinked)"
echo ""

# =============================================================================
# PHASE 1: MLM pretrain (50M tokens, ~25 reps of 2M data)
# =============================================================================
echo "$(date): === PHASE 1: MLM PRETRAIN ==="
MLM_DIR="${OUTPUT_BASE}/mlm_pretrain"
mkdir -p "${MLM_DIR}"

python train_pile_scaling.py \
    --model-size ${MODEL} \
    --tokens ${MLM_TOKENS} \
    --batch-size ${BATCH} \
    --accumulate ${ACCUM} \
    --seq-len ${SEQ} \
    --lr ${LR} \
    --tie-weights \
    --eval-every ${EVAL_EVERY} \
    --save-every ${SAVE_EVERY} \
    --output-dir "${MLM_DIR}" \
    --data-dir "${DATA_DIR}" \
    --pretrain-mode mlm \
    --no-resume \
    2>&1 | tee "${MLM_DIR}.log"

echo "$(date): === PHASE 1 DONE ==="
echo ""

# Find the best MLM checkpoint
MLM_CKPT="${MLM_DIR}/pile_${MODEL}/best_model.pt"
if [ ! -f "${MLM_CKPT}" ]; then
    echo "ERROR: MLM checkpoint not found at ${MLM_CKPT}"
    exit 1
fi

# =============================================================================
# PHASE 2: Causal from MLM (834M tokens = 1000 tok/param)
# =============================================================================
echo "$(date): === PHASE 2: CAUSAL FROM MLM ==="
CAUSAL_MLM_DIR="${OUTPUT_BASE}/causal_from_mlm"
mkdir -p "${CAUSAL_MLM_DIR}"

python train_pile_scaling.py \
    --model-size ${MODEL} \
    --tokens ${TOTAL_TOKENS} \
    --batch-size ${BATCH} \
    --accumulate ${ACCUM} \
    --seq-len ${SEQ} \
    --lr ${LR} \
    --tie-weights \
    --eval-every ${EVAL_EVERY} \
    --save-every ${SAVE_EVERY} \
    --output-dir "${CAUSAL_MLM_DIR}" \
    --data-dir "${DATA_DIR}" \
    --pretrain-mode causal \
    --init-from "${MLM_CKPT}" \
    --no-resume \
    2>&1 | tee "${CAUSAL_MLM_DIR}.log"

echo "$(date): === PHASE 2 DONE ==="
echo ""

# =============================================================================
# PHASE 3: Baseline causal (834M tokens = 1000 tok/param)
# =============================================================================
echo "$(date): === PHASE 3: BASELINE CAUSAL ==="
BASELINE_DIR="${OUTPUT_BASE}/baseline_causal"
mkdir -p "${BASELINE_DIR}"

python train_pile_scaling.py \
    --model-size ${MODEL} \
    --tokens ${TOTAL_TOKENS} \
    --batch-size ${BATCH} \
    --accumulate ${ACCUM} \
    --seq-len ${SEQ} \
    --lr ${LR} \
    --tie-weights \
    --eval-every ${EVAL_EVERY} \
    --save-every ${SAVE_EVERY} \
    --output-dir "${BASELINE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --pretrain-mode causal \
    --no-resume \
    2>&1 | tee "${BASELINE_DIR}.log"

echo "$(date): === PHASE 3 DONE ==="
echo ""

# =============================================================================
# RESULTS SUMMARY
# =============================================================================
echo ""
echo "=== REPETITION TEST RESULTS ==="
for phase in mlm_pretrain causal_from_mlm baseline_causal; do
    LOG="${OUTPUT_BASE}/${phase}/pile_${MODEL}/training_log.json"
    if [ -f "${LOG}" ]; then
        echo "--- ${phase} ---"
        python3 -c "
import json, sys
with open('${LOG}') as f:
    data = json.load(f)
best = min(data, key=lambda x: x['val_ppl'])
final = data[-1]
print(f'  Best: ppl={best[\"val_ppl\"]:.1f} at step {best[\"step\"]} ({best[\"tokens_seen\"]/834400:.0f} tok/param)')
print(f'  Final: ppl={final[\"val_ppl\"]:.1f} at step {final[\"step\"]} ({final[\"tokens_seen\"]/834400:.0f} tok/param)')
degradation = (final['val_ppl'] - best['val_ppl']) / best['val_ppl'] * 100
if degradation > 3:
    print(f'  *** OVERFITTING DETECTED: final {final[\"val_ppl\"]:.0f} > best {best[\"val_ppl\"]:.0f} ({degradation:.1f}%) ***')
else:
    print(f'  No significant overfitting ({degradation:.1f}% from best)')
"
    else
        echo "--- ${phase} --- (no log found)"
    fi
done

echo ""
echo "$(date): === ALL DONE ==="
