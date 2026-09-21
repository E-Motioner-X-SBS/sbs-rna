#!/usr/bin/env bash
# Fired every 30 minutes: check that the architecture is still consistent with
# itself, and record anything that changed.
#
# It checks, then it works, then it says what it is waiting for. It does not
# stop: when the only work left needs the GPU and the GPU has not been offered,
# it prints "I am waiting for GPU permission" and exits 0. That is a normal
# tick, not a failure.
#
# Separate from gpu_cron_runner.sh on purpose. That one trains; this one only
# reads and runs CPU-only work by default, so the two must not block each other
# and do not share a lock. It is safe while training is running --
# verify_claims.py forces CUDA_VISIBLE_DEVICES="" for exactly this reason, after
# a neighbour's memory use once made a passing suite report drift.
#
# It takes the card ONLY when data/samples/analysis/cron/GPU_PERMITTED exists
# AND enough memory is free. Remove the file to take the card back.
set -uo pipefail

REPO=/store/shuvam/E-motioner-X-SBS/sbs-rna
PY=/store/shuvam/.venv/bin/python
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
LOGDIR=$REPO/data/samples/analysis/cron
LOCK=$LOGDIR/architecture_watch.lock
LOG=$LOGDIR/architecture_watch.log

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

# One at a time. A full check runs 13 test suites and can outlast the interval.
exec 9>"$LOCK"
flock -n 9 || exit 0

{
    echo "----- $(date -Is) -----"
    timeout 3000 "$PY" scripts/architecture_watch.py 2>&1
    echo "exit $?"
} >> "$LOG" 2>&1

# Keep the log from growing without bound: 4,000 lines is several weeks of
# half-hourly checks, and the durable record is CHANGELOG.md, not this.
if [ "$(wc -l < "$LOG")" -gt 4000 ]; then
    tail -2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi
