#!/bin/bash
# Two-scale prior (phase (1,1) | phase (0,0)) on an A100: upload script+deps+data, train EPOCHS epochs, poll, pull ckpts, destroy on exit.
# Usage: INSTANCE_ID=... SSH_PORT=... SSH_HOST=... [EPOCHS=6] ./gpu_two_scale.sh
: "${INSTANCE_ID:?}" "${SSH_PORT:?}" "${SSH_HOST:?}"
TAG=${TAG:-ts_d64}; EPOCHS=${EPOCHS:-6}; WPE=${WPE:-60000}; BS=${BS:-32}; LR=${LR:-3e-4}
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
SSH="ssh $SSH_OPTS -p $SSH_PORT root@$SSH_HOST"; SCP="scp $SSH_OPTS -P $SSH_PORT"
LOCAL_DIR="/Users/nico/Documents/Projects/jeffy/experiments/scratch_classifier"; VASTAI="/Users/nico/miniforge3/bin/vastai"; REMOTE=/workspace/ts
cleanup() { echo "CLEANUP: destroying $INSTANCE_ID"; yes | $VASTAI destroy instance $INSTANCE_ID 2>/dev/null || true; }
trap cleanup EXIT
echo "=== TWO-SCALE $TAG ($EPOCHS ep x $WPE, bs $BS, lr $LR) $(date) ==="
for i in $(seq 1 60); do
  STATUS=$($VASTAI show instance $INSTANCE_ID --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status','?'))" 2>/dev/null || echo err)
  [ "$STATUS" = "running" ] && { echo "RUNNING after $((i*15))s"; break; }; sleep 15
done
[ "$STATUS" != "running" ] && { echo "FATAL: boot"; exit 1; }
sleep 10
for i in 1 2 3 4 5 6; do $SSH "echo SSH_OK" 2>&1 | grep -q SSH_OK && break; [ $i -eq 6 ] && { echo "FATAL: ssh"; exit 1; }; sleep 10; done
echo "SSH ok; uploading (data 289 MB)..."
$SSH "mkdir -p $REMOTE/data $REMOTE/checkpoints" 2>&1 | grep -v "Welcome\|fun"
$SCP "$LOCAL_DIR/two_scale_prior.py" "$LOCAL_DIR/shared_prior_model.py" "$LOCAL_DIR/train_rope_byte_2d.py" root@$SSH_HOST:$REMOTE/ 2>&1 | grep -v "Welcome\|fun"
$SCP "$LOCAL_DIR/data/two_scale_train.npy" "$LOCAL_DIR/data/two_scale_kodak.npz" root@$SSH_HOST:$REMOTE/data/ 2>&1 | grep -v "Welcome\|fun"
$SCP "$LOCAL_DIR/checkpoints/prior_full_d64_20ep_best.pt" root@$SSH_HOST:$REMOTE/checkpoints/ 2>&1 | grep -v "Welcome\|fun"
$SSH "cd $REMOTE && python3 -c 'import torch; assert torch.cuda.is_available(); print(\"CUDA\", torch.cuda.get_device_name(0))'" 2>&1 | grep CUDA || { echo "FATAL: no cuda"; exit 1; }
$SSH "cd $REMOTE && pip install -q pillow 2>&1 | tail -0; setsid nohup python3 -u two_scale_prior.py train --epochs $EPOCHS --windows-per-epoch $WPE --batch-size $BS --lr $LR --tag $TAG > train_$TAG.log 2>&1 < /dev/null & echo LAUNCHED" 2>&1 | grep LAUNCHED
LAST=""
for i in $(seq 1 90); do
  sleep 60
  $SCP "root@$SSH_HOST:$REMOTE/train_$TAG.log" "$LOCAL_DIR/logs/train_$TAG.log" 2>/dev/null
  $SCP "root@$SSH_HOST:$REMOTE/checkpoints/prior_${TAG}_best.pt" "$LOCAL_DIR/checkpoints/prior_${TAG}_best.pt" 2>/dev/null
  EP=$(grep -E "^Epoch" "$LOCAL_DIR/logs/train_$TAG.log" 2>/dev/null | tail -1); [ "$EP" != "$LAST" ] && { echo "$EP"; LAST="$EP"; }
  grep -qE "Traceback|Error" "$LOCAL_DIR/logs/train_$TAG.log" 2>/dev/null && { echo "FATAL: training error"; grep -A5 Traceback "$LOCAL_DIR/logs/train_$TAG.log" | tail -8; break; }
  grep -qE "^Epoch $EPOCHS/$EPOCHS" "$LOCAL_DIR/logs/train_$TAG.log" 2>/dev/null && { echo "Final epoch logged"; sleep 20; break; }
done
$SCP "root@$SSH_HOST:$REMOTE/checkpoints/prior_${TAG}_best.pt" "$LOCAL_DIR/checkpoints/prior_${TAG}_best.pt" 2>&1 | grep -v "Welcome\|fun"
$SCP "root@$SSH_HOST:$REMOTE/checkpoints/prior_${TAG}_last.pt" "$LOCAL_DIR/checkpoints/prior_${TAG}_last.pt" 2>&1 | grep -v "Welcome\|fun"
$SCP "root@$SSH_HOST:$REMOTE/train_$TAG.log" "$LOCAL_DIR/logs/train_$TAG.log" 2>&1 | grep -v "Welcome\|fun"
echo "=== DONE $(date) ==="; grep -E "^Epoch" "$LOCAL_DIR/logs/train_$TAG.log" | tail -2
