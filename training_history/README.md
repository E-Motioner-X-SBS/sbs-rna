# Training history

One folder per run, with the raw log beside a reading of it. The reading is the
point: a log says `ce 1.2505` and a reading says whether that meant anything.

| run | dates | stage | tokens | what happened |
|---|---|---|---|---|
| [run 1](2026-09-24_stage1_run1.md) | 24–25 Sep 2026 | 1 (MLM) | 0.395B → 1.574B | Trained. Then the audit found the curve measuring it was 52.7% one length band, and the pipeline feeding it was serving one length-sorted shard at a time. Stopped at step 10,800 and restarted. |
| [run 2](2026-09-25_stage1_run2.md) | 25 Sep 2026 → | 1 (MLM) | 1.569B → 8B | Same weights, corrected pipeline: 8B budget, loops 1–3 sampled, shard interleaving, `max_batch` 2048, stratified evaluation. |

`logs/` holds the raw files unedited. Numbers in the readings are reproducible
from `scripts/sampling/verify_claims.py` (`R1`–`R13`) unless marked otherwise.
