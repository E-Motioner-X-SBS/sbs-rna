# Rigor Recheck — PHAROS empirical claims

**Note on location**: the repo's `plans/` already holds RNA data-acquisition
documents (12-*, 13-*, CONTINUATION_STATE). Rigor-mode state lives in this
subfolder to avoid collision.

| Field | Value |
|---|---|
| Task | Recheck every empirical claim in the PHAROS architecture work |
| Cycle | **11** — the oldest outstanding items (C13, C14) |
| Phase | 6 AUDIT (cycle 9) |
| Started | 2026-09-13 (cycle 1) |
| Open TODO | 0 open, 4 deferred (3 need training/data, 1 is OQ-7) |
| Defects to date | **28** |

## The arc of cycles 7-10

Cycles 0-6 audited the work: measurements, derivations, specifications,
definitions, and the propagation of corrections. Cycles 7-10 audited the
**instruments** — and found a defect in every one. Four for four.

| Cycle | Instrument | Defect | Consequence |
|---|---|---|---|
| 7 | the RNA resolver | **#25** — parsed only one of mmCIF's two serialisations | **published a false finding** ("G2 strengthens") |
| 8 | the shared mmCIF `loops()` parser | **#26** — the same bug, never checked for | 9 rows, all headlines unchanged |
| 9 | the guard's tolerances | **#27** — 45 of 54 wider than their own precision | the #25 retraction came within 1.5x of passing undetected |
| 10 | the guard's token check | **#28** — `tok in text` proves the wrong thing | 3 of 5 guards vacuous; suite exited 0 with the claims deleted |

**The tooling had never been audited, and it had a 100% defect rate when it
finally was.** That is the central finding of this stretch: nine cycles of
re-deriving numbers through a shared instrument does not test the instrument, and
agreement between results that share a tool is not independent evidence.

## What survives

All 54 numeric claims pass at **exact quoted precision** after cycle 9's
tightening — the values were accurate to their stated digits throughout; only the
proof was weak. G2 and G3 land back on their originally published figures. The
only finding that genuinely moved across three cycles of definitional work is
**G7**, which was 8.5x too high and whose argument is now stronger than the one it
replaced.

## Defect base rate

| Cycle | Defects | Character |
|---|---|---|
| 0 (build) | 5 | implementation bugs producing plausible numbers |
| 1 (audit) | 4 | overclaims + a physics constant |
| 2 | 3 | wrong population, unfair comparison, unimplemented loss |
| 3 | 2 | misattributed citation, impossible hardware |
| 4 | 4 | mixed units, an omitted term in every cost, unspecified components |
| 5 | 2 | circularity, copied template without its active capacity |
| 6 | 3 | no definition of the unit of measurement; a self-damaging repair |
| 7 | 1 | **the resolver — published a false finding** |
| 8 | 1 | the shared parser — same bug, negligible impact, real class |
| 9 | 1 | the guard's tolerances — 45 of 54 claims under-checked |
| **10** | **1** | **the guard's token check — 3 of 5 tested guards vacuous** |

| 11 | 0 defects | **C13/C14 closed** — definition sensitivity measured, shared geometry loader added |

**Twenty-eight defects across eleven cycles.** The location moved decisively over
cycles 7-10 — four consecutive defects in the verification apparatus rather than
in the architecture. **Cycle 11 found no defect**, the first such cycle, and
closed the two oldest outstanding items instead.

## Where it stands

Every to-do is closed, deferred with a reason, or waiting on data that does not
exist yet. The architecture's one decision that these measurements feed — the
`c = 20` sparse-track budget — **survives**, but with **4.8% headroom at the worst
chain rather than the 16% a mean-over-long-chains figure implied** (OQ-7).

What cannot be closed by auditing: **nothing has been trained**, so every claim
about model behaviour remains an argument rather than a result.
