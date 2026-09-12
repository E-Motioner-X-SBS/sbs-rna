#!/bin/bash
# Resilient downloader: keeps resuming until the expected size is reached.
# Usage: resilient_dl.sh <url> <dest> <expected_bytes>
URL="$1"; DEST="$2"; EXPECTED="$3"
LOG="${DEST}.dl.log"
echo "[$(date)] resilient downloader started: $DEST (expect $EXPECTED bytes)" >> "$LOG"
for attempt in $(seq 1 200); do
  CURRENT=0
  [ -f "$DEST" ] && CURRENT=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
  if [ "$CURRENT" -ge "$EXPECTED" ]; then
    echo "[$(date)] COMPLETE after $attempt checks ($CURRENT bytes)" >> "$LOG"
    touch "${DEST}.done"
    exit 0
  fi
  echo "[$(date)] attempt $attempt: have $CURRENT / $EXPECTED" >> "$LOG"
  curl -sSL --fail --retry 3 --retry-delay 10 --connect-timeout 30 --speed-time 120 --speed-limit 10240 \
       -C - -o "$DEST" "$URL" >> "$LOG" 2>&1
  sleep 20
done
echo "[$(date)] GAVE UP" >> "$LOG"
