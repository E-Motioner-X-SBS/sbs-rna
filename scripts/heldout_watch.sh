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
tries=0
pending=""

# Score one saved checkpoint. Returns the evaluator's exit status, which the
# caller MUST look at -- see the note at the call site.
score_ckpt() {
    local f="$1" st="$2"
    echo "$(date -Is) scoring step $st" >> "$LOG"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      timeout 900 $PY -u scripts/eval_mlm_checkpoint.py \
        --ckpt "$f" --device cuda --n-seq 1024 --token-budget 8192 >> "$LOG" 2>&1
}
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
        # A step whose eval failed is retried from the SAVED COPY before
        # anything else, while that copy still exists. Without this, not
        # advancing `last` achieves nothing: the mtime guard means `step` is
        # only recomputed when the trainer writes a NEW checkpoint, so the
        # failed step would still never be scored again.
        if [ -n "$pending" ] && [ -f "$KEEP/step${pending}.pt" ]; then
            score_ckpt "$KEEP/step${pending}.pt" "$pending"
            if [ $? -eq 0 ]; then
                echo "$(date -Is) recovered step $pending" >> "$LOG"
                pending=""
                tries=0
            else
                tries=$((tries + 1))
                if [ "$tries" -ge 3 ]; then
                    echo "$(date -Is) giving up on step $pending after $tries attempts; the curve has a GAP here" >> "$LOG"
                    pending=""
                    tries=0
                fi
            fi
        fi
        if [ -n "$step" ] && [ "$step" != "$last" ]; then
            cp -f "$CKPT" "$KEEP/step${step}.pt" 2>/dev/null
            score_ckpt "$KEEP/step${step}.pt" "$step"
            rc=$?
            # keep only the newest few: each is 4.3 GB and the disk is at 92%
            ls -1t "$KEEP"/step*.pt 2>/dev/null | tail -n +4 | xargs -r rm -f
            # Advance ONLY on success.
            #
            # `last="$step"` used to run unconditionally, so a failed eval
            # retired that step for good and the curve simply lost the point.
            # That is what happened at step 8,250: the evaluator's load guard
            # refused a checkpoint carrying two new buffers, the loop recorded
            # the step as done, and the next reading would have been 8,500 --
            # an hour of coverage gone with nothing in the CSV to show a gap
            # had occurred. Same shape as everything else this audit has found:
            # a command ran, it failed, and its exit status was discarded.
            #
            # Retries are capped so a permanently broken evaluator polls every
            # five minutes rather than spinning, and gives up loudly.
            last="$step"
            if [ "$rc" -ne 0 ]; then
                echo "$(date -Is) eval FAILED for step $step (rc=$rc); queued for retry" >> "$LOG"
                pending="$step"
                tries=0
            fi
        fi
    fi
    sleep 300
done
echo "$(date -Is) stopped (HELDOUT_STOP present)" >> "$LOG"
rm -f "$STOP"
