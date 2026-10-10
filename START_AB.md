# The A/B queue — armed, not running

Disarmed on request: it will **not** start by itself. Nothing runs on the
GPU until you say so.

## To run it

    cd /store/shuvam/E-motioner-X-SBS/sbs-rna
    cp scripts/ab_queue.sh /tmp/ab_queue.run.sh      # never edit a script a shell is running (finding 99)
    nohup bash /tmp/ab_queue.run.sh > data/samples/analysis/geom_ab/queue_stdout.log 2>&1 &

It waits for ≥62 GB free, then runs five arms back to back, each from the
same stage-4 init at the baseline's settings (40 epochs, 16,384-token
budget, lr 2e-4), each followed by prediction on 17 RNA-Puzzles targets
and scoring against the field.

## To run one arm instead

    PYTHONPATH=src python scripts/train_pharos.py --device cuda \
        --size shared400 --epochs 40 --token-budget 16384 \
        --ckpt data/derived/checkpoints/pharos_shared400_tri.pt \
        --init-from data/derived/checkpoints/seqstages_shared400.pt \
        --triangle-layers 2

| arm | finding | how it is switched on | state |
|---|---|---|---|
| geom | 100 — decoder could not see current geometry | (already in its checkpoint) | **resumes at epoch 24/40** |
| tri | 109 — pair track not 3D-embeddable | `--triangle-layers 2` | from scratch |
| bigdn | 103 — two thirds of the trunk is causal | `--bidirectional-gdn` | from scratch |
| coevfull | 112 — coevolution below its noise floor | `PHAROS_COEV_CACHE=data/derived/coevolution_full` | from scratch |
| cbias | 92 — pair feature is additive, pairing is not | `--contact-bias` | from scratch |

Each is zero-gated, so switching one on does not change a loaded
checkpoint's forward pass; it only gives the model something it may learn
to use.

## The verdict

    PYTHONPATH=src python scripts/ab_report.py     # -> data/samples/analysis/geom_ab/RESULT.md

Paired, per target, against all three baseline runs, with an exact
two-sided sign test. The baselines span 0.0046 at identical settings and
the init-RNG noise floor is 0.0023, so the bar is stated against that
spread and not against zero: **13 of 17 targets** for two-sided p < 0.05.

Baseline to beat: mean TM **0.0936**, field median **0.3013**.
