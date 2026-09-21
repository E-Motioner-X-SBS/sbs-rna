#!/usr/bin/env bash
# Wait for the GPU to actually free, then train the block scorer.
#
# The A100 in this box is shared. `train_block_scorer.py` refuses to start on a
# busy card rather than OOM halfway through, so something has to watch for the
# card coming free -- polling by hand wastes the window, and starting on a card
# with 20 GB free only to have the other job grow into it wastes the run.
#
# Two guards against grabbing the card at the wrong moment:
#   * a memory floor, and
#   * a SUSTAINED check -- the floor must hold across N consecutive polls, so a
#     brief dip between two phases of somebody else's job does not look like
#     the card being free.
set -uo pipefail
NEED_GIB=${NEED_GIB:-40}
SUSTAIN=${SUSTAIN:-4}          # consecutive polls that must pass
INTERVAL=${INTERVAL:-120}      # seconds
MAX_WAIT=${MAX_WAIT:-86400}
LOG=${LOG:-data/samples/analysis/block_scorer_train.log}

cd "$(dirname "$0")/.."
PY=/store/shuvam/.venv/bin/python
waited=0; streak=0

echo "[wait] need ${NEED_GIB} GiB free on ${SUSTAIN} consecutive polls, every ${INTERVAL}s"
while [ "$waited" -lt "$MAX_WAIT" ]; do
  free_mib=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$free_mib" ] && [ "$free_mib" -ge $((NEED_GIB * 1024)) ]; then
    streak=$((streak + 1))
    echo "[wait] $(date +%H:%M:%S) ${free_mib} MiB free  (streak ${streak}/${SUSTAIN})"
  else
    [ "$streak" -gt 0 ] && echo "[wait] $(date +%H:%M:%S) dropped to ${free_mib:-?} MiB; streak reset"
    streak=0
  fi
  if [ "$streak" -ge "$SUSTAIN" ]; then
    echo "[wait] GPU free and holding -- starting training at $(date -Is)"
    exec "$PY" -u scripts/train_block_scorer.py \
        --device cuda --min-free-gib "$NEED_GIB" --epochs "${EPOCHS:-8}" \
        >> "$LOG" 2>&1
  fi
  sleep "$INTERVAL"
  waited=$((waited + INTERVAL))
done
echo "[wait] gave up after ${MAX_WAIT}s without a free GPU"
exit 1
