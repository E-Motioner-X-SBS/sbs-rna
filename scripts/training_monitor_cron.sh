#!/usr/bin/env bash
# Report on stage 1 every 45 minutes. Fired by cron every 5; gated here.
#
# `*/45 * * * *` does NOT mean "every 45 minutes". Cron's step syntax walks
# the 0-59 minute field, so it fires at :00 and :45 and then waits 15 -- a
# 45/15 alternation that looks right in the crontab and is wrong in the log.
# A 45-minute period does not divide the hour, so it cannot be written as a
# minute pattern at all. Either four lines on a 3-hour cycle, or a frequent
# tick with the interval enforced in the script. This repository's sibling
# (legal_memorization/scripts/checkup_cron.sh) uses the second, so this does
# too, at 5-minute granularity rather than 1: the report is at most 5 minutes
# late and the other 8 ticks cost a stat and an exit.
#
#   crontab:  */5 * * * * .../scripts/training_monitor_cron.sh
#   stop it:  touch data/samples/analysis/cron/MONITOR_HOLD
set -uo pipefail

REPO=/store/shuvam/E-motioner-X-SBS/sbs-rna
PY=/store/shuvam/.venv/bin/python
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
LOGDIR=$REPO/data/samples/analysis/cron
LOG=$LOGDIR/training_monitor.log
STAMP=$LOGDIR/.monitor_last
GAP_S=2700          # 45 minutes

cd "$REPO" || exit 1
mkdir -p "$LOGDIR"

[ -f "$LOGDIR/MONITOR_HOLD" ] && exit 0

# Has 45 minutes passed? `date -r` on a missing file is an error, not 0, so
# the first run has to be handled explicitly rather than by arithmetic on an
# empty string -- which would silently evaluate to "yes, run" every tick.
now=$(date +%s)
if [ -f "$STAMP" ]; then
    last=$(stat -c %Y "$STAMP" 2>/dev/null || echo 0)
    [ $((now - last)) -lt "$GAP_S" ] && exit 0
fi
touch "$STAMP"

# One at a time. A report that overlaps itself interleaves its output and
# corrupts the state file both halves write.
exec 9>"$LOGDIR/monitor.lock"
flock -n 9 || exit 0

{
    "$PY" scripts/training_monitor.py
    echo "    exit $?"
    echo
} >> "$LOG" 2>&1

# Keep the log bounded: ~32 reports a day at ~16 lines is 4 weeks in 15k.
if [ "$(wc -l < "$LOG")" -gt 15000 ]; then
    tail -7500 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi
