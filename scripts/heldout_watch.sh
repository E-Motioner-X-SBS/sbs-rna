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
EVERY_N=1000

# Score one saved checkpoint. Returns the evaluator's exit status, which the
# caller MUST look at -- see the note at the call site.
score_ckpt() {
    local f="$1" st="$2" b
    # Halving backoff on the token budget.
    #
    # The stratified sample has a mean length of 324 nt against the legacy
    # sample's 185, and attention is B*L^2 at a fixed B*L, so at the same
    # budget it costs about 1.75x the memory and its worst batch -- 1,024-nt
    # sequences -- costs five times more. Against a trainer that has grown to
    # 73.4 GiB plus an unrelated 1.7 GiB service on the same card, 8,192
    # OOMed three times and the curve lost step 10,000. The budget only sets
    # how the fixed sample is packed, so a smaller one scores the SAME
    # sequences; it is slower and otherwise identical.
    # CPU, not CUDA. The backoff below is kept, but it was treating the
    # wrong problem: the trainer has grown to 71.6 GiB, an unrelated service
    # holds 1.7 more, and the card has about 3 GiB free -- not enough for
    # 394M parameters plus activations at ANY token budget. 8,192 OOMed three
    # times and lost step 10,000; 3,072 and 1,536 then OOMed too, and one of
    # those retries collided with a manual eval I was running at the same
    # time. There is no GPU-side budget that fits.
    #
    # On CPU it takes about 35 minutes at 512 sequences against a checkpoint
    # every 40, so it keeps up, and it cannot compete with the trainer. The
    # numeric path differs -- fp32 here, bf16 autocast on cuda -- so the csv
    # now carries a `device` column and a series must not mix them.
    echo "$(date -Is) scoring step $st on cpu" >> "$LOG"
    # 8 threads of 24, and a 90-minute ceiling.
    #
    # Unthrottled this took 11 cores and 26+ minutes for 512 sequences, and the
    # trainer's throughput fell from 15.9k to 14.7k tok/s -- it needs CPU of its
    # own for parquet decode and masking, so an evaluation that takes half the
    # machine is charged to the run it is measuring. Capped at 8 threads and
    # 384 sequences, and gated to every 1,000 steps by the caller, the cost is
    # roughly one reading every 2.7 hours for a few percent of throughput.
    #
    # 192, not 384. Measured rather than estimated: 512 sequences on 11 cores
    # did NOT finish in 50 minutes -- about 2.0e14 FLOPs at an effective
    # 67 GFLOP/s -- so 384 on 8 cores would have been marginal against even a
    # 90-minute ceiling. 192 sequences is ~15,000 masked positions, which pins
    # the bits figure to about +/-0.01, well inside the 0.05 the curve already
    # scatters by.
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    timeout 3600 $PY -u scripts/eval_mlm_checkpoint.py \
        --ckpt "$f" --device cpu --n-seq 192 --token-budget 4096 \
        --split stratified \
        --append-csv data/samples/analysis/runs/heldout_stratified.csv \
        >> "$LOG" 2>&1
}
# `--split stratified`, and a NEW csv.
#
# The legacy sample is `eldors_c020_shard0004` alone: 52.7% of it falls in the
# 20-79 nt band against the corpus's 6.5%, an eight-fold over-representation of
# the shortest sequences, and it measured the model 0.19 bits better than a
# corpus-weighted sample does. Its readings also swung +0.303 bits over three
# checkpoints where the corpus-weighted figure moved +0.074.
#
# A new file rather than more rows in the old one: the two are different
# samples and appending would silently splice two series. heldout_mlm.csv is
# frozen at step 9,750 and stays as the record of what was tracked overnight.
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
        # Only every EVERY_N steps. A CPU evaluation costs about 40 minutes
        # and some of the trainer's own cores; at one per checkpoint that is a
        # permanent tax for a curve whose points are 250 steps apart and
        # 0.05 bits noisy. Every 1,000 steps is four times cheaper and loses
        # nothing that the noise was not already hiding.
        if [ -n "$step" ] && [ "$step" != "$last" ] \
           && [ $((step % EVERY_N)) -ne 0 ]; then
            echo "$(date -Is) skipping step $step (not a multiple of $EVERY_N)" >> "$LOG"
            last="$step"
            step=""
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
