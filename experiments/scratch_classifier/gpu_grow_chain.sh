#!/bin/bash
# Matched second stage + chat recipe for the GROWN base, on ONE A100 (three stages, one instance, destroy on exit):
#  1. TinyStories extension exactly as the direct base got it (train_8m_extend: resume best @ep19, --epochs 65, lr 3e-4, fresh scheduler, no --fresh-opt)
#  2. chat_v6plain_fix 40 ep lr 3e-4 fresh-opt  (= oracle_v7a_conv recipe)
#  3. chat_v6persona2 4 ep lr 3e-4 fresh-opt    (= conv_persona2 recipe)
# Usage: INSTANCE_ID=... SSH_PORT=... SSH_HOST=... ./gpu_grow_chain.sh   (INIT defaults to the 20-epoch grown checkpoint)
: "${INSTANCE_ID:?}" "${SSH_PORT:?}" "${SSH_HOST:?}"
INIT=${INIT:-best_lm_grown_2m_to_8m.pt}; TOK=${TOK:-tokenizer_tinystories.json}; T1=${T1:-grown_2m_to_8m_ext}; T2=${T2:-oracle_grown_conv}; T3=${T3:-oracle_grown_persona2}
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
SSH="ssh $SSH_OPTS -p $SSH_PORT root@$SSH_HOST"; SCP="scp $SSH_OPTS -P $SSH_PORT"
LOCAL_DIR="/Users/nico/Documents/Projects/jeffy/experiments/scratch_classifier"; VASTAI="/Users/nico/miniforge3/bin/vastai"; REMOTE=/workspace/grow
ARCH="--tokenizer $TOK --d-model 384 --n-layers 4 --n-heads 8 --d-ff 1280 --seq-len 256 --batch-size 32 --n-iterations 1 --tie-weights --sample-every 10"
cleanup() { echo "CLEANUP: destroying $INSTANCE_ID"; yes | $VASTAI destroy instance $INSTANCE_ID 2>/dev/null || true; }
trap cleanup EXIT
echo "=== GROW CHAIN init=$INIT  $(date) ==="
for i in $(seq 1 60); do
  STATUS=$($VASTAI show instance $INSTANCE_ID --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status','?'))" 2>/dev/null || echo err)
  [ "$STATUS" = "running" ] && { echo "RUNNING after $((i*15))s"; break; }; sleep 15
done
[ "$STATUS" != "running" ] && { echo "FATAL: boot"; exit 1; }
sleep 10
for i in 1 2 3 4 5 6; do $SSH "echo SSH_OK" 2>&1 | grep -q SSH_OK && break; [ $i -eq 6 ] && { echo "FATAL: ssh"; exit 1; }; sleep 10; done
echo "SSH ok; uploading..."
$SSH "mkdir -p $REMOTE/checkpoints $REMOTE/data && pip install -q tokenizers datasets 2>&1 | grep -v WARNING | tail -0" 2>&1 | grep -v "Welcome\|fun"
$SCP "$LOCAL_DIR/train_lm.py" "$LOCAL_DIR/$TOK" root@$SSH_HOST:$REMOTE/ 2>&1 | grep -v "Welcome\|fun"
$SCP "$LOCAL_DIR/data/chat_v6plain_fix_train.jsonl" "$LOCAL_DIR/data/chat_v6plain_fix_val.jsonl" "$LOCAL_DIR/data/chat_v6persona2_train.jsonl" "$LOCAL_DIR/data/chat_v6persona2_val.jsonl" root@$SSH_HOST:$REMOTE/data/ 2>&1 | grep -v "Welcome\|fun"
$SCP "$LOCAL_DIR/checkpoints/$INIT" root@$SSH_HOST:$REMOTE/checkpoints/ 2>&1 | grep -v "Welcome\|fun"
$SSH "cd $REMOTE && python3 -c \"
with open('train_lm.py') as f: c = f.read()
c = c.replace('from transformers import AutoTokenizer', 'try:\\n    from transformers import AutoTokenizer\\nexcept ImportError:\\n    AutoTokenizer = None')
with open('train_lm.py','w') as f: f.write(c)
import torch; assert torch.cuda.is_available(); print('CUDA', torch.cuda.get_device_name(0))\"" 2>&1 | grep CUDA || { echo "FATAL: no cuda"; exit 1; }

# run_stage TAG FINAL_EPOCH "train args"   — launch under nohup, poll (pull log + best ckpt each minute), stop at the final epoch line
run_stage() {
  local TAG=$1 FINAL=$2 ARGS=$3
  echo "--- STAGE $TAG ($FINAL epochs) $(date)"
  $SSH "cd $REMOTE && setsid nohup python3 -u train_lm.py $ARGS --tag $TAG > train_$TAG.log 2>&1 < /dev/null & echo LAUNCHED" 2>&1 | grep LAUNCHED &
  local LP=$!; for i in $(seq 1 30); do kill -0 $LP 2>/dev/null || break; sleep 1; done; kill $LP 2>/dev/null; pkill -f "ssh .*-p $SSH_PORT root@$SSH_HOST.*LAUNCHED" 2>/dev/null
  local LAST=""
  for i in $(seq 1 120); do
    sleep 60
    $SCP "root@$SSH_HOST:$REMOTE/train_$TAG.log" "$LOCAL_DIR/logs/train_$TAG.log" 2>/dev/null
    $SCP "root@$SSH_HOST:$REMOTE/checkpoints/best_lm_$TAG.pt" "$LOCAL_DIR/checkpoints/best_lm_$TAG.pt" 2>/dev/null
    local EP=$(grep -E "^Epoch" "$LOCAL_DIR/logs/train_$TAG.log" 2>/dev/null | tail -1); [ "$EP" != "$LAST" ] && { echo "$EP"; LAST="$EP"; }
    grep -qE "Traceback|Error" "$LOCAL_DIR/logs/train_$TAG.log" 2>/dev/null && { echo "FATAL: training error in $TAG"; grep -A5 Traceback "$LOCAL_DIR/logs/train_$TAG.log" | tail -8; return 1; }
    grep -qE "^Epoch +$FINAL/$FINAL" "$LOCAL_DIR/logs/train_$TAG.log" 2>/dev/null && { echo "final epoch logged"; sleep 20; break; }
  done
  $SCP "root@$SSH_HOST:$REMOTE/checkpoints/best_lm_$TAG.pt" "$LOCAL_DIR/checkpoints/best_lm_$TAG.pt" 2>&1 | grep -v "Welcome\|fun"
  $SCP "root@$SSH_HOST:$REMOTE/train_$TAG.log" "$LOCAL_DIR/logs/train_$TAG.log" 2>&1 | grep -v "Welcome\|fun"
  grep -qE "^Epoch +$FINAL/$FINAL" "$LOCAL_DIR/logs/train_$TAG.log" || { echo "FATAL: $TAG did not finish"; return 1; }
  ls -la "$LOCAL_DIR/checkpoints/best_lm_$TAG.pt" | awk '{print $5, $9}'
}
run_stage $T1 65 "$ARCH --max-stories 50000 --lr 3e-4 --epochs 65 --resume checkpoints/$INIT" || exit 1
run_stage $T2 40 "$ARCH --chat-data data/chat_v6plain_fix_train.jsonl --lr 3e-4 --epochs 40 --resume checkpoints/best_lm_$T1.pt --fresh-opt" || exit 1
run_stage $T3 4  "$ARCH --chat-data data/chat_v6persona2_train.jsonl --lr 3e-4 --epochs 4 --resume checkpoints/best_lm_$T2.pt --fresh-opt" || exit 1
echo "=== DONE $(date) ==="
