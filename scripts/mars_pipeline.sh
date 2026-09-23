#!/usr/bin/env bash
# Download MARS and convert it as it arrives, so the archives never accumulate.
#
# 427 GB of .tgz will not fit beside its own output on a disk with 599 GB free,
# and the output is worth far less than the input suggests: measured on 279,232
# records, the structured-ncRNA filter keeps 8.0%. So each archive is converted
# to parquet and then deleted, and peak archive footprint is bounded by however
# many downloads are in flight (8 x ~14 GB).
#
# Downloading and converting run as two loops against the same directory. The
# downloader writes `<name>.tgz.part` and renames on completion, so a file
# named `.tgz` is complete by construction and the converter can take it
# without coordination.
set -uo pipefail

REPO=/store/shuvam/E-motioner-X-SBS/sbs-rna
PY=/store/shuvam/.venv/bin/python
cd "$REPO" || exit 1
LOG=$REPO/data/samples/analysis/mars_pipeline.log
LOCK=$REPO/data/samples/analysis/cron/mars_pipeline.lock
WORKERS=${WORKERS:-8}
MODE=${MODE:-structured}

exec 9>"$LOCK"
flock -n 9 || { echo "$(date -Is) already running" >> "$LOG"; exit 0; }

say() { echo "$(date -Is) $*" >> "$LOG"; }

say "=== pipeline start (workers=$WORKERS mode=$MODE) ==="

# 1. downloader, in the background: all 30 parts, resumable
"$PY" scripts/acquire_all.py --only mars --workers "$WORKERS" >> "$LOG" 2>&1 &
DL=$!
say "downloader pid $DL"

# 2. converter loop: take any completed archive, convert, delete
while kill -0 "$DL" 2>/dev/null || ls data/sequences/mars/*.tgz >/dev/null 2>&1; do
    for f in data/sequences/mars/*.tgz; do
        [ -e "$f" ] || continue
        say "converting $(basename "$f")"
        timeout 14400 "$PY" scripts/mars_to_parquet.py \
            --archive "$f" --mode "$MODE" --delete-archive \
            --dedup-against data/derived/parquet_starter data/derived/parquet_mars \
            >> "$LOG" 2>&1
    done
    sleep 300
done

say "downloader finished; final sweep"
timeout 28800 "$PY" scripts/mars_to_parquet.py --all --mode "$MODE" \
    --delete-archive \
    --dedup-against data/derived/parquet_starter data/derived/parquet_mars \
    >> "$LOG" 2>&1
say "=== pipeline done: $(ls data/derived/parquet_mars/*.parquet 2>/dev/null | wc -l) shards ==="
