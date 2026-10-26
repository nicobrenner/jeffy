#!/bin/bash
# Grow-then-pretrain on an A100: continue TinyStories pretraining of a grown checkpoint with train_lm.py (same data/tokenizer as the
# direct 8.5M base), poll, pull best ckpt, destroy on exit. Usage: INSTANCE_ID=... SSH_PORT=... SSH_HOST=... INIT=checkpoints/grown_2m_to_8m.pt TAG=grown_2m_to_8m EPOCHS=20 ./gpu_grow.sh
: "${INSTANCE_ID:?}" "${SSH_PORT:?}" "${SSH_HOST:?}" "${INIT:?}"
TAG=${TAG:-grown}; EPOCHS=${EPOCHS:-20}; LR=${LR:-3e-4}; TOK=${TOK:-tokenizer_tinystories.json}
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
SSH="ssh $SSH_OPTS -p $SSH_PORT root@$SSH_HOST"; SCP="scp $SSH_OPTS -P $SSH_PORT"
LOCAL_DIR="/Users/nico/Documents/Projects/jeffy/experiments/scratch_classifier"; VASTAI="/Users/nico/miniforge3/bin/vastai"; REMOTE=/workspace/grow
cleanup() { echo "CLEANUP: destroying $INSTANCE_ID"; yes | $VASTAI destroy instance $INSTANCE_ID 2>/dev/null || true; }
trap cleanup EXIT
echo "=== GROW $TAG (init $INIT, $EPOCHS ep, lr $LR) $(date) ==="
for i in $(seq 1 60); do
  STATUS=$($VASTAI show instance $INSTANCE_ID --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status','?'))" 2>/dev/null || echo err)
  [ "$STATUS" = "running" ] && { echo "RUNNING after $((i*15))s"; break; }; sleep 15
done
[ "$STATUS" != "running" ] && { echo "FATAL: boot"; exit 1; }
sleep 10
for i in 1 2 3 4 5 6; do $SSH "echo SSH_OK" 2>&1 | grep -q SSH_OK && break; [ $i -eq 6 ] && { echo "FATAL: ssh"; exit 1; }; sleep 10; done
echo "SSH ok; uploading..."
$SSH "mkdir -p $REMOTE/checkpoints && pip install -q tokenizers datasets 2>&1 | grep -v WARNING | tail -0" 2>&1 | grep -v "Welcome\|fun"
$SCP "$LOCAL_DIR/train_lm.py" "$LOCAL_DIR/$TOK" root@$SSH_HOST:$REMOTE/ 2>&1 | grep -v "Welcome\|fun"
$SCP "$LOCAL_DIR/$INIT" root@$SSH_HOST:$REMOTE/checkpoints/ 2>&1 | grep -v "Welcome\|fun"
$SSH "cd $REMOTE && python3 -c \"
with open('train_lm.py') as f: c = f.read()
c = c.replace('from transformers import AutoTokenizer', 'try:\\n    from transformers import AutoTokenizer\\nexcept ImportError:\\n    AutoTokenizer = None')
with open('train_lm.py','w') as f: f.write(c)
import torch; assert torch.cuda.is_available(); print('CUDA', torch.cuda.get_device_name(0))\"" 2>&1 | grep CUDA || { echo "FATAL: no cuda"; exit 1; }
CK=$(basename "$INIT")
$SSH "cd $REMOTE && setsid nohup python3 -u train_lm.py --tokenizer $TOK --max-stories 50000 --d-model 384 --n-layers 4 --n-heads 8 --d-ff 1280 --seq-len 256 --batch-size 32 --lr $LR --epochs $EPOCHS --n-iterations 1 --tie-weights --sample-every 10 --resume checkpoints/$CK --fresh-opt --tag $TAG > train_$TAG.log 2>&1 < /dev/null & echo LAUNCHED" 2>&1 | grep LAUNCHED &
LAUNCH_PID=$!; for i in $(seq 1 30); do kill -0 $LAUNCH_PID 2>/dev/null || break; sleep 1; done; kill $LAUNCH_PID 2>/dev/null; pkill -f "ssh .*-p $SSH_PORT root@$SSH_HOST.*LAUNCHED" 2>/dev/null; echo "launch step released"
LAST=""
for i in $(seq 1 90); do
  sleep 60
  $SCP "root@$SSH_HOST:$REMOTE/train_$TAG.log" "$LOCAL_DIR/logs/train_$TAG.log" 2>/dev/null
  $SCP "root@$SSH_HOST:$REMOTE/checkpoints/best_lm_$TAG.pt" "$LOCAL_DIR/checkpoints/best_lm_$TAG.pt" 2>/dev/null
  EP=$(grep -E "^Epoch" "$LOCAL_DIR/logs/train_$TAG.log" 2>/dev/null | tail -1); [ "$EP" != "$LAST" ] && { echo "$EP"; LAST="$EP"; }
  grep -qE "Traceback|Error" "$LOCAL_DIR/logs/train_$TAG.log" 2>/dev/null && { echo "FATAL: training error"; grep -A5 Traceback "$LOCAL_DIR/logs/train_$TAG.log" | tail -8; break; }
  grep -qE "^Epoch +$EPOCHS/$EPOCHS" "$LOCAL_DIR/logs/train_$TAG.log" 2>/dev/null && { echo "Final epoch logged"; sleep 20; break; }
done
$SCP "root@$SSH_HOST:$REMOTE/checkpoints/best_lm_$TAG.pt" "$LOCAL_DIR/checkpoints/best_lm_$TAG.pt" 2>&1 | grep -v "Welcome\|fun"
$SCP "root@$SSH_HOST:$REMOTE/train_$TAG.log" "$LOCAL_DIR/logs/train_$TAG.log" 2>&1 | grep -v "Welcome\|fun"
echo "=== DONE $(date) ==="; grep -E "^Epoch" "$LOCAL_DIR/logs/train_$TAG.log" | tail -2
