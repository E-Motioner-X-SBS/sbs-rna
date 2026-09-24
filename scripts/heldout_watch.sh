#!/usr/bin/env bash
# Score every new training checkpoint on the fixed held-out sample.
#
# The running trainer predates the in-loop held-out evaluation, so its only
# self-report is a rolling mean over whatever shard is streaming -- a number
# that moved 1.488 -> 1.863 bits and back while the model improved
# monotonically. Until that run restarts, this supplies the curve from outside:
# it waits for the checkpoint's step counter to change, copies it (the trainer
# overwrites the same path every 250 steps, so a copy is the only way a
# checkpoint survives to be compared), scores it, and appends to
# runs/heldout_mlm.csv.
#
# Deliberately polite about the card: one short eval per checkpoint, roughly
# once every 40 minutes, against a trainer holding 68 GiB.
#
#   nohup bash scripts/heldout_watch.sh &   # start
#   touch data/samples/analysis/cron/HELDOUT_STOP   # stop after the next tick
set -u
REPO=/store/shuvam/E-motioner-X-SBS/sbs-rna
cd "$REPO" || exit 1
PY=/store/shuvam/.venv/bin/python
CKPT=data/derived/checkpoints/pretrain_shared400.pt
KEEP=data/derived/checkpoints/heldout
STOP=data/samples/analysis/cron/HELDOUT_STOP
LOG=data/samples/analysis/runs/heldout_watch.log
mkdir -p "$KEEP" "$(dirname "$LOG")"
last=""
lastmt=""
while [ ! -f "$STOP" ]; do
    if [ -f "$CKPT" ]; then
        # mtime first. Reading the step means loading a 4.3 GB checkpoint, and
        # doing that every five minutes to learn nothing competes with the
        # trainer for memory bandwidth for no reason. The trainer writes
        # atomically (temp file + rename), so a changed mtime means a complete
        # new checkpoint, never a half-written one.
        mt=$(stat -c %Y "$CKPT" 2>/dev/null)
        if [ -n "$mt" ] && [ "$mt" != "$lastmt" ]; then
            lastmt="$mt"
            step=$($PY - <<'PYEOF' 2>/dev/null
import torch
try:
    print(torch.load("data/derived/checkpoints/pretrain_shared400.pt",
                     map_location="cpu", weights_only=False)["step"])
except Exception:
    print("")
PYEOF
)
        else
            step=""
        fi
        if [ -n "$step" ] && [ "$step" != "$last" ]; then
            cp -f "$CKPT" "$KEEP/step${step}.pt" 2>/dev/null
            echo "$(date -Is) scoring step $step" >> "$LOG"
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
              timeout 900 $PY -u scripts/eval_mlm_checkpoint.py \
                --ckpt "$KEEP/step${step}.pt" --device cuda \
                --n-seq 1024 --token-budget 8192 >> "$LOG" 2>&1
            # keep only the newest few: each is 4.3 GB and the disk is at 92%
            ls -1t "$KEEP"/step*.pt 2>/dev/null | tail -n +4 | xargs -r rm -f
            last="$step"
        fi
    fi
    sleep 300
done
echo "$(date -Is) stopped (HELDOUT_STOP present)" >> "$LOG"
rm -f "$STOP"
