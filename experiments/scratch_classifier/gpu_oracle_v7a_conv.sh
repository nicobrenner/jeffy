#!/bin/bash
# Tiny Oracle v7a -> convergence: same recipe as v7a (TinyStories 8.5M init `best_lm_8m_custom_tok.pt`, d384 L4 H8 dff1280,
# 5,500-token BPE, chat_v6plain) but ONE cosine schedule at lr 3e-4 for EPOCHS epochs instead of 8 @ 1e-4 (+4 @ 3e-4).
# Motivation (RESEARCH_LOG, ATL r2): the equal-compute control (+4 epochs) improved held-out NLL by -0.18, more than any
# teacher data; every Oracle comparison so far sat on an under-trained base. Run on an A100; poll, pull best ckpt every
# minute, destroy the instance on exit (trap). Usage: INSTANCE_ID=... SSH_PORT=... SSH_HOST=... ./gpu_oracle_v7a_conv.sh
: "${INSTANCE_ID:?}" "${SSH_PORT:?}" "${SSH_HOST:?}"
TAG=${TAG:-oracle_v7a_conv}; EPOCHS=${EPOCHS:-40}; DATA=${DATA:-chat_v6plain}; LR=${LR:-3e-4}
INIT=${INIT:-best_lm_8m_custom_tok.pt}; TOK=${TOK:-tokenizer_tinystories.json}
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
SSH="ssh $SSH_OPTS -p $SSH_PORT root@$SSH_HOST"
SCP="scp $SSH_OPTS -P $SSH_PORT"
LOCAL_DIR="/Users/nico/Documents/Projects/jeffy/experiments/scratch_classifier"
CKPT_DIR="$LOCAL_DIR/checkpoints"; LOG_DIR="$LOCAL_DIR/logs"
VASTAI="/Users/nico/miniforge3/bin/vastai"
REMOTE=/workspace/jeffy

cleanup() { echo "CLEANUP: Destroying instance $INSTANCE_ID..."; yes | $VASTAI destroy instance $INSTANCE_ID 2>/dev/null || true; echo "Instance $INSTANCE_ID destroyed"; }
trap cleanup EXIT

echo "=== ORACLE $TAG ($DATA, $EPOCHS epochs, lr $LR, init $INIT) $(date) ==="
echo "STEP 1: Waiting for boot..."
for i in $(seq 1 60); do
    STATUS=$($VASTAI show instance $INSTANCE_ID --raw 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','unknown'))" 2>/dev/null || echo "err")
    if [ "$STATUS" = "running" ]; then echo "RUNNING after $((i*15))s"; break; fi
    if [ "$STATUS" = "exited" ] || [ "$STATUS" = "error" ]; then echo "FATAL: $STATUS"; exit 1; fi
    sleep 15
done
[ "$STATUS" != "running" ] && { echo "FATAL: boot timeout"; exit 1; }
sleep 10
echo "STEP 2: SSH..."
for i in 1 2 3 4 5 6; do $SSH "echo SSH_OK" 2>&1 | grep -q SSH_OK && break; [ $i -eq 6 ] && { echo "FATAL: SSH failed"; exit 1; }; sleep 10; done
echo "SSH connected"
echo "STEP 3: pip install tokenizers..."; $SSH "pip install -q tokenizers 2>&1 | grep -v WARNING | tail -1" 2>&1
echo "STEP 4: Upload (script, tokenizer, data, init checkpoint)..."
$SSH "mkdir -p $REMOTE/data $REMOTE/checkpoints" 2>&1
$SCP "$LOCAL_DIR/train_lm.py" "$LOCAL_DIR/$TOK" root@$SSH_HOST:$REMOTE/ 2>&1
$SCP "$LOCAL_DIR/data/${DATA}_train.jsonl" "$LOCAL_DIR/data/${DATA}_val.jsonl" root@$SSH_HOST:$REMOTE/data/ 2>&1
$SCP "$CKPT_DIR/$INIT" root@$SSH_HOST:$REMOTE/checkpoints/ 2>&1
echo "Files uploaded"
echo "STEP 5: Patch imports + verify CUDA..."
$SSH "cd $REMOTE && python3 -c \"
with open('train_lm.py') as f: c = f.read()
c = c.replace('from transformers import AutoTokenizer', 'try:\\n    from transformers import AutoTokenizer\\nexcept ImportError:\\n    AutoTokenizer = None')
c = c.replace('from datasets import load_dataset', 'try:\\n    from datasets import load_dataset\\nexcept ImportError:\\n    load_dataset = None')
with open('train_lm.py','w') as f: f.write(c)
import torch; assert torch.cuda.is_available(), 'NO CUDA'
from train_lm import CausalTransformer, CustomTokenizerWrapper
t = CustomTokenizerWrapper('$TOK'); print(f'OK vocab={t.vocab_size} cuda={torch.cuda.get_device_name(0)}')
\"" 2>&1 | grep -q "OK vocab" || { echo "FATAL: verify failed"; exit 1; }
echo "Verify OK"
echo "STEP 6: Launch under nohup..."
$SSH "cd $REMOTE && setsid nohup python3 -u train_lm.py --chat-data data/${DATA}_train.jsonl --tokenizer $TOK \
    --d-model 384 --n-layers 4 --n-heads 8 --d-ff 1280 --seq-len 256 --batch-size 32 --lr $LR --epochs $EPOCHS \
    --n-iterations 1 --tie-weights --sample-every 10 --resume checkpoints/$INIT --fresh-opt \
    --tag $TAG > train_$TAG.log 2>&1 < /dev/null &
    echo LAUNCHED" 2>&1 &
LAUNCH_PID=$!
for i in $(seq 1 30); do kill -0 $LAUNCH_PID 2>/dev/null || break; sleep 1; done
kill $LAUNCH_PID 2>/dev/null && echo "launch ssh released after 30s"
DEV=""
for i in $(seq 1 30); do
    DEV=$($SSH "grep -m1 Device $REMOTE/train_$TAG.log 2>/dev/null" 2>/dev/null)
    [ -n "$DEV" ] && break
    sleep 5
done
echo "Device line: $DEV"
echo "$DEV" | grep -q "Device: cuda" || { echo "FATAL: not on CUDA ($DEV)"; $SSH "pkill -f train_lm.py"; exit 1; }
echo "GPU CONFIRMED"
echo "STEP 7: Polling (every 60s)..."
mkdir -p "$CKPT_DIR" "$LOG_DIR"; LAST=""
for i in $(seq 1 120); do
    sleep 60
    $SCP "root@$SSH_HOST:$REMOTE/train_$TAG.log" "$LOG_DIR/train_$TAG.log" 2>/dev/null
    $SCP "root@$SSH_HOST:$REMOTE/checkpoints/best_lm_$TAG.pt" "$CKPT_DIR/best_lm_$TAG.pt" 2>/dev/null && P="ckpt" || P="-"
    EP=$(grep -E "^Epoch" "$LOG_DIR/train_$TAG.log" 2>/dev/null | tail -1); [ "$EP" != "$LAST" ] && { echo "$EP [$P]"; LAST="$EP"; }
    grep -qE "Traceback|Error" "$LOG_DIR/train_$TAG.log" 2>/dev/null && { echo "FATAL: training error"; grep -A4 Traceback "$LOG_DIR/train_$TAG.log" | tail -6; break; }
    ALIVE=$($SSH "pgrep -c -f '[p]ython3 -u train_lm.py' || true" 2>/dev/null | grep -E '^[0-9]+$' | tail -1); [ "$ALIVE" = "0" ] && { echo "Training process finished"; break; }
    grep -qE "^Epoch +$EPOCHS/$EPOCHS" "$LOG_DIR/train_$TAG.log" 2>/dev/null && { echo "Final epoch logged"; sleep 30; break; }
done
echo "STEP 8: Final download + verify..."
$SCP "root@$SSH_HOST:$REMOTE/checkpoints/best_lm_$TAG.pt" "$CKPT_DIR/best_lm_$TAG.pt" 2>&1 && echo "GOT best ckpt" || echo "WARN: no best ckpt"
$SCP "root@$SSH_HOST:$REMOTE/checkpoints/last_lm_$TAG.pt" "$CKPT_DIR/last_lm_$TAG.pt" 2>&1 && echo "GOT last ckpt" || echo "WARN: no last ckpt"
$SCP "root@$SSH_HOST:$REMOTE/train_$TAG.log" "$LOG_DIR/train_$TAG.log" 2>&1 && echo "GOT log"
ls -la "$CKPT_DIR/best_lm_$TAG.pt" "$CKPT_DIR/last_lm_$TAG.pt" 2>/dev/null; grep -E "^Epoch" "$LOG_DIR/train_$TAG.log" | tail -1
echo "=== ORACLE $TAG COMPLETE $(date) ==="
