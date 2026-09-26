# Architecture changelog

Written by `scripts/architecture_watch.py`, which checks every 30 minutes and records only what changed. The architecture itself is in [`ARCHITECTURE.md`](ARCHITECTURE.md); what did not work is in [`history_of_failed_attempts/`](../../history_of_failed_attempts/).

## 2026-09-22 00:34 IST — baseline

- specification split from its history; watcher installed (`816f6d7`)
- 330 pinned checks, components 30/30, stages 6/6, scheduled 4/4,
  resumable 4/4, telemetry 4/4

## 2026-09-22 01:13 IST

- **The watcher's first two entries were both the same non-change** (`94dcaa6`)

## 2026-09-22 01:19 IST

- **The 30-minute loop now works rather than only watching, and never stops** (`18315d0`)

## 2026-09-22 02:04 IST

- **The watcher was writing a changelog entry about committing the changelog** (`d29f573`)

## 2026-09-23 15:34 IST

- **Training telemetry: stage 1 from the 188.6M resume onward** (`8f91948`)
- tokens: 188601237 → **244695353**

## 2026-09-23 16:04 IST

- tokens: 244695353 → **295733186**

## 2026-09-23 16:34 IST

- tokens: 295733186 → **348598312**

## 2026-09-23 17:04 IST

- tokens: 348598312 → **402503125**

## 2026-09-23 17:34 IST

- **Stage 1 telemetry: 238M -> 447M tokens, 1.813 -> 1.773 bits** (`4023262`)
- tokens: 402503125 → **457899328**

## 2026-09-23 18:04 IST

- **Stage 1 telemetry to 505M tokens; the curve has flattened** (`46c503d`)
- tokens: 457899328 → **510639413**

## 2026-09-23 18:34 IST

- **Stage 1 telemetry to 557M tokens** (`567da13`)
- tokens: 510639413 → **563089954**

## 2026-09-23 19:04 IST

- **Stage 1 telemetry to 603M tokens** (`673c4f4`)
- claims_ok: 330 → **333**
- tokens: 563089954 → **611235903**

## 2026-09-23 19:34 IST

- **Stage 1 telemetry to 648M tokens** (`7d367c3`)
- tokens: 611235903 → **650302874**

## 2026-09-23 20:04 IST

- **Stage 1 telemetry to 688M tokens** (`45e6133`)
- tokens: 650302874 → **692448758**

## 2026-09-23 20:34 IST

- **Stage 1 telemetry to 731M tokens** (`3565f19`)
- tokens: 692448758 → **738483249**

## 2026-09-23 21:04 IST

- claims_ok: 333 → **334**
- tokens: 738483249 → **782435360**

## 2026-09-23 21:34 IST

- **First MARS archive converted; the length window does most of the filtering** (`054c950`)
- tokens: 782435360 → **837784557**

## 2026-09-23 22:04 IST

- tokens: 837784557 → **889602990**

## 2026-09-23 22:34 IST

- **Stage 1 to 934M tokens; MARS 5 archives converted** (`7005331`)
- tokens: 889602990 → **945887893**

## 2026-09-23 23:04 IST

- **Halfway: 984.8M tokens, 1.746 bits, accuracy 0.4421 -- the plateau broke** (`2a81af8`)
- tokens: 945887893 → **992472983**

## 2026-09-23 23:34 IST

- tokens: 992472983 → **1025441062**

## 2026-09-24 00:04 IST

- **Coevolution, for real: APC-MI that rebuilds the tRNA cloverleaf untrained** (`5ccb32a`)
- claims_ok: 334 → **336**
- tokens: 1025441062 → **1064647612**

## 2026-09-24 00:34 IST

- **Head 3 becomes a diffusion decoder; the coordinate MLP could not have worked** (`1664057`)
- claims_ok: 336 → **321**
- tokens: 1064647612 → **1103566302**

## 2026-09-24 01:34 IST

- claims_ok: 321 → **336**

## 2026-09-24 02:34 IST

- **512 experts sharing one network: 61M active parameters becomes 302M** (`bdc8bcc`)
- claims_ok: 336 → **321**

## 2026-09-24 03:34 IST

- claims_ok: 321 → **336**

## 2026-09-24 04:34 IST

- **The blind-test field, and three fixes the trainer needed to run shared400** (`2b5ef1c`)

## 2026-09-24 05:04 IST

- **The spec describes what is actually running: shared400, diffusion, coevolution** (`81af0ed`)
- claims_ok: 336 → **337**

## 2026-09-24 06:04 IST

- **A verification timeout was an uncaught exception that would block training** (`0a71379`)

## 2026-09-24 08:34 IST

- **Stage 1 telemetry: shared400 through step 1700 (1.488 bits, acc 0.540)** (`de51c94`)
- claims_ok: 337 → **333**

## 2026-09-24 09:04 IST

- **Close the loop: sequence -> sampled backbone -> PDB -> scored against the field** (`251fd08`)

## 2026-09-24 09:34 IST

- **A fixed-sample evaluator, because the training loss was measuring the data** (`0e47225`)

## 2026-09-24 10:04 IST

- **Score every checkpoint on the held-out sample without a human in the loop** (`1b3750e`)

## 2026-09-24 10:34 IST

- **The curriculum could not hand off: stage 5 built a different model than stage 1** (`5a1cc3b`)

## 2026-09-24 11:04 IST

- **Stage 5 refuses to silently discard pretraining, and it now actually runs** (`990b878`)

## 2026-09-24 11:34 IST

- **Is the accuracy real? Partly: 6 of the 38 points are an artefact** (`a6ec973`)

## 2026-09-24 12:04 IST

- **Switch stage 1 to Muon, and let resume survive an architecture that grew** (`b31cb6e`)

## 2026-09-24 12:34 IST

- **The spec records the optimiser decision and the honest accuracy** (`d5342f8`)

## 2026-09-24 13:04 IST

- **The model is under-CONFIDENT, not just under-trained; stop decaying the tied embedding** (`b2788cb`)

## 2026-09-24 13:34 IST

- **Muon recovers from the restart dip and is at parity by step 3500** (`f8d815d`)

## 2026-09-24 14:04 IST

- **Muon verdict: no measurable difference at scale. Not reverting, not claiming a win.** (`bf3eb57`)

## 2026-09-24 14:34 IST

- **Catch structures that are not molecules, and two heads the spec claims but lacks** (`c59a110`)
- claims_ok: 333 → **332**

## 2026-09-24 16:04 IST

- **Heads 9 and 10 now exist, with targets that were already in the archive** (`b5f4f4b`)

## 2026-09-24 17:04 IST

- **The corpus build died at the last step because meta() held an ndarray** (`bab2429`)

## 2026-09-24 17:34 IST

- **The held-out set was being trained on. It is now excluded at the source.** (`00f18cf`)

## 2026-09-24 18:04 IST

- **The 3D corpus carries base-pair geometry: heads 9 and 10 are live** (`2a1c27e`)

## 2026-09-24 19:04 IST

- **No leak: the training loss is reading easy shards, and the clean curve is good** (`8bab7ca`)
- claims_ok: 332 → **0**

## 2026-09-24 20:04 IST

- claims_ok: 0 → **332**

## 2026-09-24 21:34 IST

- **The block scorer's headline recall is a chain-weighted mean of mostly tiny chains** (`8192cb6`)

## 2026-09-24 22:04 IST

- claims_ok: 332 → **0**

## 2026-09-24 23:04 IST

- claims_ok: 0 → **333**

## 2026-09-25 00:04 IST

- **Four measurements that could not fail, and so were not measurements** (`e73c299`)

## 2026-09-25 01:04 IST

- claims_ok: 333 → **334**

## 2026-09-25 02:04 IST

- **The routing-width claim now has a measurement under it** (`8fbbaed`)
- claims_ok: 334 → **353**

## 2026-09-25 02:34 IST

- **The router's top length bin is 4.5 octaves wide, and takes 40% of stage 5** (`6d3e708`)
- claims_ok: 353 → **363**

## 2026-09-25 03:04 IST

- **Every head reported a loss with no floor; now they report the floor too** (`42e2cf9`)

## 2026-09-25 03:34 IST

- **The held-out curve's "noise" is largely the batch size, and it is measurable** (`b9fd406`)
- claims_ok: 363 → **367**

## 2026-09-25 04:04 IST

- **Withdrawn: the batch size does not explain step 8,500** (`3026197`)

## 2026-09-25 04:34 IST

- **Dim 13 of the chemistry vector is never set by anything that trains the model** (`85ee502`)
- claims_ok: 367 → **133**

## 2026-09-25 05:34 IST

- **What is the rest of stage 1 worth? Not identifiable from this curve** (`307476e`)
- claims_ok: 133 → **0**

## 2026-09-25 06:34 IST

- **The one number that says whether this competes was pinned nowhere** (`5936475`)
- claims_ok: 0 → **393**

## 2026-09-25 07:04 IST

- **The csv declared 16 columns and wrote 12, dropping the pairing measurement** (`c646e7e`)
- claims_ok: 393 → **404**

## 2026-09-25 07:34 IST

- **Coevolution is attached to the right chain: 28x enrichment, measured** (`c0130c4`)
- claims_ok: 404 → **410**

## 2026-09-25 08:04 IST

- **The CPU evaluation was being charged to the run it measures** (`c8e57e4`)

## 2026-09-25 08:34 IST

- **An audit record, because sixteen findings in commit messages are not actionable** (`a20e212`)

## 2026-09-25 09:04 IST

- **Restart: 8B tokens, loops 1-3 sampled, corrected data pipeline, dim 13 wired** (`ea60c91`)

## 2026-09-25 10:04 IST

- **Two correct-looking rules that, composed, starved the held-out curve** (`697fc70`)
- claims_ok: 410 → **413**

## 2026-09-25 10:34 IST

- **The results file had no idea whether it already held a row** (`6f3284f`)

## 2026-09-25 12:04 IST

- **--help has been broken all day, and the fix I nearly shipped was wrong** (`c624415`)

## 2026-09-25 13:04 IST

- **The token budget was a one-way ratchet, and it just cost half the batch** (`962e559`)
- claims_ok: 413 → **417**

## 2026-09-25 13:34 IST

- **A full disk would have killed the run, and /store is at 96%** (`d764a1f`)

## 2026-09-25 14:04 IST

- **The "fixed" held-out sample was not fixed, and the log said so all along** (`63ede58`)

## 2026-09-25 14:34 IST

- **The telemetry I un-dropped this morning was printed and then dropped again** (`6c52cbb`)

## 2026-09-25 15:04 IST

- **The 2.8x speedup was one point; with a fourth it is a wash** (`70d9012`)

## 2026-09-25 15:34 IST

- **Interleaving calmed the training loss 6.6x, and run 1 had calm windows too** (`95fc8c1`)

## 2026-09-25 17:34 IST

- **Stage 5's validation reported two heads of ten, and not the one that matters** (`a4b251c`)

## 2026-09-25 18:04 IST

- **Stage 1 stopped at step 13,750 / 2.0686B tokens, on a checkpoint** (`66b8d55`)

## 2026-09-25 18:34 IST

- **The stop would have lasted 30 minutes: the scheduler had no way to be told** (`25183fe`)

## 2026-09-25 19:04 IST

- **Run 2 closed: 1.80266 bits at 2.069B tokens, 10.6% better than base frequencies** (`74669fe`)

## 2026-09-25 22:04 IST

- claims_ok: 417 → **0**

## 2026-09-26 00:04 IST

- claims_ok: 0 → **417**

## 2026-09-26 03:04 IST

- claims_ok: 417 → **0**

## 2026-09-26 05:04 IST

- claims_ok: 0 → **417**

## 2026-09-26 06:04 IST

- claims_ok: 417 → **0**

## 2026-09-26 07:04 IST

- claims_ok: 0 → **417**

## 2026-09-26 08:04 IST

- claims_ok: 417 → **0**

## 2026-09-26 12:04 IST

- claims_ok: 0 → **417**

## 2026-09-26 19:04 IST

- claims_ok: 417 → **416**
