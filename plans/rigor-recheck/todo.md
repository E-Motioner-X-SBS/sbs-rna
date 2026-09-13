# Master To-Do — Cycle 1

Legend: [ ] open · [x] done+tested · [BLOCKED: r] · [DEFERRED: r]

## Priority 0 — known discrepancy
- [x] T1  X-RAY COUNT CONFLICT -> RESOLVED, not a data bug. 63 is correct;
          rigidity uses 31 after a <30-RNA-residue guard. See doubts D1.
- [ ] T20 [ADDED-CYCLE-1] DOC FIX (2 defects): (a) all three deliverables omit
          the >=30-residue filter; (b) they say the Mg gradient covers 31 X-ray
          structures when it covers 15 (the Mg-containing subset).
- [ ] T22 [ADDED-CYCLE-1] **REVERSAL**: walk back the coevolution depth-gate
          claim in ARCHITECTURE.md / main.tex / blueprint.html; disclose the
          9.87% pseudoknot omission.
- [ ] T23 [ADDED-CYCLE-1] **PHYSICS FIX**: correct b, xi and theta in all three
          deliverables; note theta~0.76 is B-DNA, A-RNA is ~0.80.
- [ ] T21 [ADDED-CYCLE-1] STRENGTHEN: add the partial-correlation and
          within-structure results to the deliverables - they materially
          improve the claim and were not previously reported.

## Priority 1 — re-derive measurements independently
- [ ] T2  Ion inventory + coordination (A1-A6) via independent reimplementation
- [x] T3  Rigidity gradient -> PASS, reproduces 1.760 sigma exactly
- [x] T4a Contact prefilter safety -> PASS (bound 22.20 A, 0 missed)
- [ ] T4b Contact sparsity numbers (C1-C2) themselves
- [x] T5  Proposal recall index arithmetic -> PASS
- [ ] T6  Contact separation (E1)
- [x] T7  Block sparsity effective_c -> PASS (2.9% conservative overstatement)
- [x] T8  Coevolution -> Neff PASS; ground truth omits 9.87% PK pairs
          (conservative); depth split OVERCLAIMED -> see T22

## Priority 2 — statistical validity
- [ ] T9  Sample sizes + CIs for every headline number
- [x] T10 Depth split -> NOT adequately supported. p=0.046 post-hoc,
          Spearman +0.224, deep arm n=2, shallow family reaches 1.000
- [x] T11 Mg/rigidity confound -> PASS, partial corr +0.372 vs +0.420 raw
- [ ] T12 Contact-sparsity length bins have n=3 and n=6 — are they load-bearing?

## Priority 3 — derived arithmetic
- [x] T13 Parameter budget -> PASS exact (910.5M / 382.0M)
- [x] T14 8x attention -> PASS (8.0x quadratic-only; 7.53x incl. SWA)
- [x] T15 309 GB -> PASS exact
- [x] T16 Physics -> **FAIL**: b~5.9-7.0 A is wrong (contour vs axial).
          Correct A-RNA: b=1.40 A, xi=5.11, theta=0.804 -> T23

## Priority 4 — literature + consistency
- [ ] T17 Verify cited numbers (ERNIE-RNA, trRosettaRNA, Motif Atlas, NucleicBERT)
- [ ] T18 Cross-document consistency ARCHITECTURE.md / main.tex / blueprint.html
- [ ] T19 Confirm conclusions follow from data (no overreach)
