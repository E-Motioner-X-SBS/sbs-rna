# Master To-Do — Cycle 1

Legend: [ ] open · [x] done+tested · [BLOCKED: r] · [DEFERRED: r]

## Priority 0 — known discrepancy
- [x] T1  X-RAY COUNT CONFLICT -> RESOLVED, not a data bug. 63 is correct;
          rigidity uses 31 after a <30-RNA-residue guard. See doubts D1.
- [x] T20 -> FIXED in all three deliverables
- [x] T22 -> RETRACTED in all three; PK omission disclosed
- [x] T23 -> FIXED in all three (b=1.40 A, xi=5.11, theta=0.804)
- [x] T21 -> ADDED to all three deliverables

## Priority 1 — re-derive measurements independently
- [x] T2  Ion inventory -> PASS (independent parser: MG 17,428, K 1,868, OP share 0.828)
- [x] T3  Rigidity gradient -> PASS, reproduces 1.760 sigma exactly
- [x] T4a Contact prefilter safety -> PASS (bound 22.20 A, 0 missed)
- [x] T4b Contact sparsity -> PASS; bins n=3/6/9 flagged as thin
- [x] T5  Proposal recall index arithmetic -> PASS
- [x] T6  Contact separation -> PASS
- [x] T7  Block sparsity effective_c -> PASS (2.9% conservative overstatement)
- [x] T8  Coevolution -> Neff PASS; ground truth omits 9.87% PK pairs
          (conservative); depth split OVERCLAIMED -> see T22

## Priority 2 — statistical validity
- [x] T9  Sample sizes + bootstrap CIs computed; thin bins flagged
- [x] T10 Depth split -> NOT adequately supported. p=0.046 post-hoc,
          Spearman +0.224, deep arm n=2, shallow family reaches 1.000
- [x] T11 Mg/rigidity confound -> PASS, partial corr +0.372 vs +0.420 raw
- [x] T12 -> NOT load-bearing. O(L) rests on n=38 and n=61 ends. Flagged.

## Priority 3 — derived arithmetic
- [x] T13 Parameter budget -> PASS exact (910.5M / 382.0M)
- [x] T14 8x attention -> PASS (8.0x quadratic-only; 7.53x incl. SWA)
- [x] T15 309 GB -> PASS exact
- [x] T16 Physics -> **FAIL**: b~5.9-7.0 A is wrong (contour vs axial).
          Correct A-RNA: b=1.40 A, xi=5.11, theta=0.804 -> T23

## Priority 4 — literature + consistency
- [x] T17 -> PASS; benchmark is n=52 monomers, now stated in the report
- [x] T18 -> found a silent edit failure (6 vs 7 novelty rows); fixed. 15/15 now consistent
- [x] T19 -> ONE OVERREACH FOUND (coevolution depth gate); retracted

- [x] T24 [ADDED-CYCLE-1] Regression guard scripts/sampling/verify_claims.py
