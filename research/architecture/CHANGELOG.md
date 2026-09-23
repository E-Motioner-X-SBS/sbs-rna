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
