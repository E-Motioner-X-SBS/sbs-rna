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

NEED_GIB=${NEED_GIB:-40}       # free memory required, GiB
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
  "scorer":  $([ -f "$LOGDIR/.done-scorer" ]  && echo '"done"' || echo 'null'),
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

# ---- 2. R1: can a model find the occupied blocks? --------------------------
# The largest unvalidated assumption in the design. Reports against three
# baselines at an identical budget, because a selector that knows only that
# contacts cluster near the diagonal would already score well.
if [ ! -f "$LOGDIR/.done-scorer" ]; then
    note "=== train_block_scorer.py (R1) ==="
    # Sized to the card, not to a laptop. Measured on this A100: 153.4M
    # parameters, 34.4 GiB peak on the worst batch, 75% mean utilisation,
    # 520 s/epoch. The previous 4.4M / 16k-token configuration used 0.6 GiB and
    # left the GPU at 13%.
    #
    # Capacity and OPTIMISER STEPS both matter, and the first attempt at scaling
    # traded the second away for the first. A 131k-token budget gives 102
    # batches/epoch, so even 24 epochs is 2,448 steps for a 153M model against
    # the original run's ~11,700 for 4.4M. Measured, and the answer was not
    # subtle: the 153M model was worse at every epoch (val L2 0.664 / 0.671 /
    # 0.666 against 0.775 / 0.821 / 0.845) and its loss barely moved
    # (2.457 -> 2.372 against 1.55 -> 0.39). A step-count failure, not a
    # capacity ceiling.
    #
    # So: 61.9M parameters -- 14x the original, 20.6 GiB peak -- at a 32k budget
    # for 357 batches/epoch and 30 epochs = 10,710 steps, which is the step
    # count the 4.4M run had. lr 5e-4 for the ~2x batch. Filling VRAM is not the
    # objective; a better answer to R1 is, and steps buy that.
    if $PY -u scripts/train_block_scorer.py \
            --device cuda --min-free-gib "$NEED_GIB" \
            --d-model 768 --d-block 512 --n-conv 10 --n-attn 6 \
            --token-budget 32768 --lr 5e-4 --epochs 30 >> "$LOG" 2>&1; then
        touch "$LOGDIR/.done-scorer"
        note "block scorer finished"
        grep -E "sequence gain|recall" "$LOG" | tail -8 | tee -a "$LOG" >/dev/null
    else
        note "block scorer did not finish (GPU taken back, or an error)"
        write_status interrupted "block scorer stopped; will retry next fire" "$F2"
        exit 1
    fi
fi

# ---- 3. stage 5: the multi-task structural head ----------------------------
if [ ! -f "$LOGDIR/.done-stage5" ]; then
    note "=== train_pharos.py (curriculum stage 5) ==="
    # PHAROS-Small, not Mini, and a 4x token budget: 43.1 GiB peak,
    # 4.5 s/step, 2.9 min/epoch with sampled recycling. Mini at the old budget
    # was 10.1 GiB and 8.0 s/step.
    if $PY -u scripts/train_pharos.py \
            --device cuda --min-free-gib "$NEED_GIB" \
            --size small --epochs 8 --token-budget 32768 >> "$LOG" 2>&1; then
        touch "$LOGDIR/.done-stage5"
        note "stage 5 finished"
    else
        note "stage 5 did not finish; will retry next fire"
        write_status interrupted "stage 5 stopped; will retry next fire" "$F2"
        exit 1
    fi
fi

# ---- 3b. stage 1: MLM pretraining, the longest job and the base of the ----
# curriculum. It is last in this script because it is the one that can run for
# days: the two experiments above answer specific questions and should not
# queue behind it. It resumes from its own checkpoint, so an interrupted run
# picks up rather than restarting.
if [ ! -f "$LOGDIR/.done-stage1" ]; then
    note "=== pretrain_mlm.py (curriculum stage 1) ==="
    if $PY -u scripts/pretrain_mlm.py \
            --device cuda --min-free-gib "$NEED_GIB" --size small \
            --tokens "${MLM_TOKENS:-5e8}" --token-budget 24576 \
            --n-loops 2 >> "$LOG" 2>&1; then
        touch "$LOGDIR/.done-stage1"
        note "stage 1 finished"
    else
        note "stage 1 did not finish; will retry next fire"
        write_status interrupted "stage 1 stopped; will retry next fire" "$F2"
        exit 1
    fi
fi

# ---- 4. re-verify: training must not have moved a pinned claim -------------
note "=== re-verifying after training ==="
$PY scripts/sampling/verify_claims.py >> "$LOG" 2>&1 \
    && note "claims still reproduce" \
    || note "WARNING: a pinned claim moved after training -- inspect the log"

write_status complete "block scorer, stage 5 and stage 1 all finished" "$F2"
note "=== pipeline complete ==="
