#!/usr/bin/env bash
# Run the queued architecture A/Bs back to back, whenever the card is free.
#
# Three changes are built and all are OFF by default, because landing them
# together would confound all three (findings 100, 109, 103). This runs them
# one at a time, each from the SAME stage-4 init, each at the baseline's
# settings -- 40 epochs, 16,384-token budget, lr 2e-4 -- and each followed
# by prediction and scoring, so a free card produces a RESULT and not just a
# checkpoint.
#
# NEVER edit this file while it is running (finding 99). It is launched from
# a copy outside the repo for exactly that reason.
set -uo pipefail
REPO=/store/shuvam/E-motioner-X-SBS/sbs-rna
PY=/store/shuvam/.venv/bin/python
AN="$REPO/data/samples/analysis"
CKDIR="$REPO/data/derived/checkpoints"
INIT="$CKDIR/seqstages_shared400.pt"
LOG="$AN/geom_ab/queue.log"
HB="$AN/geom_ab/heartbeat"
NEED=62000
cd "$REPO" || exit 1

say() { echo "[queue] $(date -Is) $*" | tee -a "$LOG"; }

# name | checkpoint | extra train flags | predictions dir | blind json
ARMS=(
  "geom|$CKDIR/pharos_shared400_geom.pt||$AN/pred_geom|$AN/blind_geom.json"
  "tri|$CKDIR/pharos_shared400_tri.pt|--triangle-layers 2|$AN/pred_tri|$AN/blind_tri.json"
  "bigdn|$CKDIR/pharos_shared400_bigdn.pt|--bidirectional-gdn|$AN/pred_bigdn|$AN/blind_bigdn.json"
  # Finding 112: the union coevolution cache. Not a train-time flag but an
  # environment selection, so the arm carries ENV= in its flags field and
  # `run_arm` exports it. Deeper MI for 58.5% of chains against 31.9%.
  "coevfull|$CKDIR/pharos_shared400_coevfull.pt|ENV:PHAROS_COEV_CACHE=data/derived/coevolution_full|$AN/pred_coevfull|$AN/blind_coevfull.json"
)

done_at_40() {   # $1 = checkpoint path
  "$PY" - "$1" <<'PY' 2>/dev/null
import sys, torch
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    sys.exit(1)
ck = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
sys.exit(0 if (ck.get("epoch", 0) + 1) >= 40 and ck.get("epoch_done") else 1)
PY
}

free_mib() {
  local u t
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  echo $(( t - u ))
}

for spec in "${ARMS[@]}"; do
  IFS='|' read -r NAME CK FLAGS PRED BLIND <<< "$spec"

  if [ -f "$BLIND" ]; then say "$NAME already scored; skipping"; continue; fi

  while ! done_at_40 "$CK"; do
    if pgrep -f "[t]rain_pharos\.py" >/dev/null; then sleep 120; continue; fi
    f=$(free_mib)
    echo "[queue] $(date -Is) $NAME waiting: ${f} MiB free, need ${NEED}" >> "$HB"
    if [ "$f" -lt "$NEED" ]; then sleep 120; continue; fi

    # Resume if a checkpoint exists, otherwise start from the stage-4 init.
    if [ -f "$CK" ]; then INITARG=""; else INITARG="--init-from $INIT"; fi
    say "$NAME training (${f} MiB free) $FLAGS"
    # An arm may select its data with an environment variable rather than
    # a flag; `ENV:` marks that and the rest of FLAGS stays CLI.
    ENVARG=""; CLIFLAGS="$FLAGS"
    case "$FLAGS" in
      ENV:*) ENVARG="${FLAGS#ENV:}"; CLIFLAGS="" ;;
    esac
    # shellcheck disable=SC2086
    env ${ENVARG:+$ENVARG} PYTHONPATH=src nice -n 5 "$PY" -u scripts/train_pharos.py \
        --device cuda --min-free-gib 60 --size shared400 --epochs 40 \
        --token-budget 16384 --ckpt "$CK" $INITARG $CLIFLAGS >> "$LOG" 2>&1
    say "$NAME trainer rc=$?"
  done

  say "$NAME at epoch 40; predicting"
  ENVARG=""; case "$FLAGS" in ENV:*) ENVARG="${FLAGS#ENV:}" ;; esac
  # The SAME data selection at inference. Scoring an arm against a
  # different coevolution cache than it trained on measures neither.
  env ${ENVARG:+$ENVARG} PYTHONPATH=src nice -n 5 "$PY" -u scripts/predict_structure.py \
      --ckpt "$CK" --size shared400 --targets rna_puzzles \
      --out "$PRED" >> "$LOG" 2>&1
  say "$NAME predict rc=$?"

  say "$NAME scoring"
  PYTHONPATH=src nice -n 5 "$PY" -u scripts/eval_blind_tests.py \
      --pred "$PRED" --ckpt "$CK" --out "$BLIND" >> "$LOG" 2>&1
  say "$NAME eval rc=$?"
done

say "all arms done; writing the comparison"
PYTHONPATH=src "$PY" scripts/ab_report.py >> "$LOG" 2>&1
say "QUEUE COMPLETE -- see geom_ab/RESULT.md"
