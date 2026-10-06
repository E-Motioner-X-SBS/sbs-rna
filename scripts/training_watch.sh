#!/usr/bin/env bash
# The DURABLE half of the training watch. Fired by the system crontab every
# half hour; it does not know or care whether an assistant session is alive.
#
# Why two halves. The in-session scheduler (CronCreate) holds its jobs in
# memory and loses them when the session ends -- which is exactly what a
# usage limit does. So the mechanical checking lives here, in the system
# crontab, where it survives a dead session, a closed terminal and a reboot;
# and it writes its findings to files, so a session that was asleep for four
# hours reads the whole backlog instead of only seeing now.
#
# Guards, each for a failure this job can hit:
#   flock      the previous fire may still be reading 4 GiB checkpoints
#   HOLD       `touch data/samples/analysis/watch/HOLD` stops it, and the
#              stop survives the scheduler, same convention as the GPU runner
#   nice       it must never compete with the trainer it is watching
#   no exit 1  "found problems" is not "the watcher failed"; cron mail on a
#              non-zero exit would fire every half hour forever
set -uo pipefail
REPO=/store/shuvam/E-motioner-X-SBS/sbs-rna
PY=/store/shuvam/.venv/bin/python
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
export PYTHONPATH=$REPO/src:$REPO/scripts
export PYTHONUNBUFFERED=1
W=$REPO/data/samples/analysis/watch
mkdir -p "$W"
cd "$REPO" || exit 0

[ -f "$W/HOLD" ] && exit 0

exec 9>"$W/watch.lock"
flock -n 9 || exit 0

# A watcher that silently stops is worse than none, so every fire leaves a
# heartbeat whether or not the python half succeeds.
date -Is > "$W/heartbeat"

if ! timeout 900 nice -n 15 "$PY" scripts/training_watch.py >> "$W/watch.log" 2>&1; then
    {
        echo "# training watch — $(date -Is)"
        echo
        echo "**THE WATCHER ITSELF FAILED.** Exit non-zero or timed out."
        echo "Tail of watch.log:"
        echo '```'
        tail -30 "$W/watch.log"
        echo '```'
    } > "$W/ALERT"
fi
exit 0
