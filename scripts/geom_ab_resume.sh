#!/usr/bin/env bash
# Auto-resume the finding-100 A/B arm the moment the card frees up.
#
# The arm died at epoch 25/40 (finding 104) and a vLLM server then took
# 66.5 GiB, leaving less than stage 5's 60 GiB guard. The run resumes from
# its own checkpoint, so waiting costs nothing but wall-clock. This script
# is NOT the cron runner and does not touch its HOLD.
set -uo pipefail
REPO=/store/shuvam/E-motioner-X-SBS/sbs-rna
LOG="$REPO/data/samples/analysis/geom_ab/train.log"
CK="$REPO/data/derived/checkpoints/pharos_shared400_geom.pt"
NEED=62000                     # MiB free before we start; the guard wants 60 GiB
HB="$REPO/data/samples/analysis/geom_ab/heartbeat"
cd "$REPO" || exit 1

while true; do
  # already finished?
  if /store/shuvam/.venv/bin/python - <<'PY' 2>/dev/null
import sys, torch
ck = torch.load("data/derived/checkpoints/pharos_shared400_geom.pt",
                map_location="cpu", mmap=True, weights_only=False)
sys.exit(0 if (ck.get("epoch", 0) + 1) >= 40 and ck.get("epoch_done") else 1)
PY
  then echo "[resume] arm already at epoch 40; done." ; exit 0 ; fi

  # Still running? `pgrep -f` on a pattern that also appears in whatever
  # shell is asking matches that shell, so exclude self and the parent.
  if pgrep -f "train_pharos\.py" | grep -qv -e "^$$\$" -e "^$PPID\$"; then
    sleep 120; continue
  fi

  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  free=$(( total - used ))
  echo "[resume] $(date -Is) ${free} MiB free, need ${NEED}" >> "$HB"
  if [ "$free" -ge "$NEED" ]; then
    echo "[resume] $(date -Is) ${free} MiB free; resuming the arm" | tee -a "$LOG"
    PYTHONPATH=src nice -n 5 /store/shuvam/.venv/bin/python -u scripts/train_pharos.py \
      --device cuda --min-free-gib 60 --size shared400 --epochs 40 \
      --token-budget 16384 --ckpt "$CK" >> "$LOG" 2>&1
    echo "[resume] $(date -Is) trainer exited rc=$?" | tee -a "$LOG"
  fi
  sleep 120
done
