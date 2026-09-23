#!/usr/bin/env bash
# Fired by cron every 30 minutes: if the shared GPU is genuinely free, run the
# architecture's validation and training pipeline; otherwise exit quietly.
#
# Why a system cron and not an in-session watcher: the A100 in this box is
# shared and has stayed occupied for many hours at a time. A watcher that lives
# inside a Claude session dies with the session. This does not.
#
# The guards, each for a failure this job can actually hit:
#
#   flock          cron fires every 30 min and a training run takes hours, so
#                  without a lock the second fire starts a second run on the
#                  same card and both OOM.
#   sustained      a brief dip between two phases of somebody else's job looks
#     free-check   exactly like a free card. The floor must hold across a
#                  re-check separated by SETTLE seconds before anything starts.
#   done markers   once a stage has finished successfully it is not repeated;
#                  cron keeps firing forever and this job should not.
#   stage ordering validation first. If verify_claims.py fails, the architecture
#                  is inconsistent with its own measurements and training it
#                  would produce a number about the wrong model.
#
# Everything lands in data/samples/analysis/cron/ -- one log per run, plus a
# status.json that says what state the pipeline is in.
set -uo pipefail

REPO=/store/shuvam/E-motioner-X-SBS/sbs-rna
PY=/store/shuvam/.venv/bin/python
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# The batch width is quantised to 16 distinct values, so the caching allocator
# cycles through 16 differently-shaped activation sets and fragments: an OOM at
# 40,960 tokens reported 417 MiB free with 1.14 GiB "reserved but unallocated".
# Expandable segments grow a single mapping instead of allocating a new block
# per shape, which is the case this setting exists for. PyTorch's own OOM
# message names it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Free memory required, GiB. 40 was a guess and it was too low: stage 1 peaks
# at 51.8 GiB measured, so a 40 GiB floor lets a run start on a card it will
# then OOM on -- which is the exact failure the floor exists to prevent.
NEED_GIB=${NEED_GIB:-60}
SETTLE=${SETTLE:-90}           # seconds between the two free-checks
LOGDIR=$REPO/data/samples/analysis/cron
LOCK=$LOGDIR/run.lock
STATUS=$LOGDIR/status.json
STAMP=$(date +%Y%m%d-%H%M%S)
LOG=$LOGDIR/run-$STAMP.log

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

# ---- one run at a time -----------------------------------------------------
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "$(date -Is) another run holds the lock; skipping" >> "$LOGDIR/skipped.log"
    exit 0
fi

note() { echo "$(date -Is) $*" | tee -a "$LOG"; }

# A killed run used to leave status.json saying "running" forever, because
# nothing wrote a status on the way out. Checking on the pipeline then reported
# a stage that had not been alive for half an hour. The trap fires on SIGTERM,
# SIGINT and SIGHUP -- which is what a kill, a Ctrl-C and a closed terminal
# send -- so the file says what is true.
on_signal() {
    note "received a signal; standing down"
    write_status interrupted "stopped by a signal; resumes next fire" "${F2:-0}"
    exit 143
}
trap on_signal TERM INT HUP

free_gib() {
    nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | awk '{printf "%d", $1/1024}'
}

write_status() {
    cat > "$STATUS" <<JSON
{
 "updated": "$(date -Is)",
 "state": "$1",
 "detail": "$2",
 "free_gib_at_check": ${3:-null},
 "log": "$(basename "$LOG")",
 "stages": {
  "verify":  $([ -f "$LOGDIR/.done-verify" ]  && echo '"done"' || echo 'null'),
  "stage1":  $([ -f "$LOGDIR/.done-stage1" ]  && echo '"done"' || echo 'null'),
  "scorer":  $([ -f "$LOGDIR/.done-scorer" ]  && echo '"done"' || echo 'null'),
  "seqstages": $([ -f "$LOGDIR/.done-seqstages" ] && echo '"done"' || echo 'null'),
  "stage5":  $([ -f "$LOGDIR/.done-stage5" ]  && echo '"done"' || echo 'null')
 }
}
JSON
}

# ---- is the card actually free, and does it stay free? ---------------------
F1=$(free_gib)
if [ -z "$F1" ]; then
    note "no GPU visible to nvidia-smi"
    write_status waiting "nvidia-smi returned nothing" null
    exit 0
fi
if [ "$F1" -lt "$NEED_GIB" ]; then
    write_status waiting "GPU busy: ${F1} GiB free, need ${NEED_GIB}" "$F1"
    exit 0
fi
note "GPU reports ${F1} GiB free; re-checking in ${SETTLE}s before committing"
sleep "$SETTLE"
F2=$(free_gib)
if [ -z "$F2" ] || [ "$F2" -lt "$NEED_GIB" ]; then
    note "dropped to ${F2:-?} GiB during settle; standing down"
    write_status waiting "free memory did not hold: ${F1} -> ${F2:-?} GiB" "${F2:-0}"
    exit 0
fi
note "GPU free and holding (${F1} -> ${F2} GiB). Starting."
write_status running "pipeline started" "$F2"

# ---- 1. the architecture must agree with its own measurements first --------
if [ ! -f "$LOGDIR/.done-verify" ]; then
    note "=== verify_claims.py (259 checks, 9 test suites) ==="
    if $PY scripts/sampling/verify_claims.py >> "$LOG" 2>&1; then
        touch "$LOGDIR/.done-verify"
        note "verification passed"
    else
        note "VERIFICATION FAILED -- not training against an inconsistent spec"
        write_status failed "verify_claims.py failed; see the log" "$F2"
        exit 1
    fi
fi

# ---- 2. stage 1: MLM pretraining, the base of the curriculum --------------
# First, because everything after it is supposed to FINE-TUNE what it builds.
# §12.1 is a curriculum; stages that do not pass weights are four unrelated
# runs, and stage 5 run from random weights is what produced r = 0.049 on
# unseen folds. It resumes from its own checkpoint, so interruption costs at
# most `--ckpt-every` steps rather than the whole run.
#
# `--token-budget 0` sizes the batch from free VRAM rather than hardcoding it.
# Measured, compiled, expandable segments: 24,576 tokens peaks at 50.9 GiB and
# runs at 36.5k tok/s; 36,864 peaks at 74.1 and runs at 41.1k. The larger is
# 13% faster and leaves five gigabytes on an 80 GiB card somebody else also
# uses, so the budget is computed from what is free at start instead of chosen
# once. Peak is linear at 2.01 GiB per 1,000 tokens to within 3%.
#
# 2e9 tokens, not the 5e8 this first ran with. 5e8 is 8 tokens per active
# parameter -- 0.4x Chinchilla-optimal, an undertrained base for everything
# downstream. 2e9 is 31.5 tok/param, and at the measured 36.1k tok/s it is
# about 15 hours, which a resumable job on a shared card can actually finish.
# The spec's 25B (410 tok/param) is 8 days at this throughput and is a
# multi-GPU figure, not a single-card one.
if [ ! -f "$LOGDIR/.done-stage1" ]; then
    note "=== pretrain_mlm.py (curriculum stage 1) ==="
    if $PY -u scripts/pretrain_mlm.py \
            --device cuda --min-free-gib "$NEED_GIB" --size "${MLM_SIZE:-shared400}" \
            --tokens "${MLM_TOKENS:-4e9}" --token-budget 0 \
            --corpus data/derived/parquet_starter data/derived/parquet_mars \
            --n-loops 2 >> "$LOG" 2>&1; then
        touch "$LOGDIR/.done-stage1"
        note "stage 1 finished"
    else
        note "stage 1 did not finish; will retry next fire (it resumes)"
        write_status interrupted "stage 1 stopped; resumes next fire" "$F2"
        exit 1
    fi
fi

PRETRAIN=$REPO/data/derived/checkpoints/pretrain_small.pt
INIT=""
[ -f "$PRETRAIN" ] && INIT="--init-from $PRETRAIN"

# ---- 3. stages 2 and 3: secondary structure and probing, co-trained -------
if [ ! -f "$LOGDIR/.done-seqstages" ]; then
    note "=== train_sequence_stages.py (curriculum stages 2-3) ==="
    if $PY -u scripts/train_sequence_stages.py \
            --device cuda --min-free-gib "$NEED_GIB" --size "${MLM_SIZE:-shared400}" \
            --epochs 2 $INIT >> "$LOG" 2>&1; then
        touch "$LOGDIR/.done-seqstages"
        note "stages 2-3 finished"
    else
        note "stages 2-3 did not finish; will retry next fire"
        write_status interrupted "stages 2-3 stopped" "$F2"
        exit 1
    fi
fi

SEQCK=$REPO/data/derived/checkpoints/seqstages_small.pt
[ -f "$SEQCK" ] && INIT="--init-from $SEQCK"

# ---- 4. R1: can a model find the occupied blocks? -------------------------
# The block scorer is a DIFFERENT architecture -- a convolutional residue
# encoder, not the trunk -- so it does not chain from the curriculum and is
# trained standalone. That is a property of the experiment, not an oversight.
#
# Sized to the card and to the STEP COUNT, which matters more: a 131k-token
# budget gives 102 batches/epoch, and a 153M model at 2,448 steps came out
# worse at every epoch than 4.4M at ~11,700. 61.9M at a 32k budget for 30
# epochs is 10,710 steps, the count the small run had.
if [ ! -f "$LOGDIR/.done-scorer" ]; then
    note "=== train_block_scorer.py (R1) ==="
    if $PY -u scripts/train_block_scorer.py \
            --device cuda --min-free-gib "$NEED_GIB" \
            --d-model 768 --d-block 512 --n-conv 10 --n-attn 6 \
            --token-budget 32768 --lr 5e-4 --epochs 30 >> "$LOG" 2>&1; then
        touch "$LOGDIR/.done-scorer"
        note "block scorer finished"
    else
        note "block scorer did not finish; will retry next fire"
        write_status interrupted "block scorer stopped" "$F2"
        exit 1
    fi
fi

# ---- 5. stage 5: the multi-task structural head, chained off stages 1-3 ----
# PHAROS-Small at a 32k budget: 43.1 GiB peak, sampled recycling. $INIT carries
# whatever representation the curriculum has built so far, which is the whole
# point of running the stages in order.
if [ ! -f "$LOGDIR/.done-stage5" ]; then
    note "=== train_pharos.py (curriculum stage 5) ==="
    if $PY -u scripts/train_pharos.py \
            --device cuda --min-free-gib "$NEED_GIB" \
            --size small --epochs 8 --token-budget 32768 \
            $INIT >> "$LOG" 2>&1; then
        touch "$LOGDIR/.done-stage5"
        note "stage 5 finished"
    else
        note "stage 5 did not finish; will retry next fire"
        write_status interrupted "stage 5 stopped" "$F2"
        exit 1
    fi
fi

# ---- 6. re-verify: training must not have moved a pinned claim -------------
note "=== re-verifying after training ==="
$PY scripts/sampling/verify_claims.py >> "$LOG" 2>&1 \
    && note "claims still reproduce" \
    || note "WARNING: a pinned claim moved after training -- inspect the log"

write_status complete "block scorer, stage 5 and stage 1 all finished" "$F2"
note "=== pipeline complete ==="
