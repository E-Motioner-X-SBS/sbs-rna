# Final Verification Report — PHAROS empirical claims

**Date**: 2026-09-13 · **Cycles**: 1 · **Verdict**: claims stand after 4 corrections

## Scope
Every empirical claim in `research/` re-derived independently (separately-written
parsers, not re-runs of the original scripts), with statistical validity checks,
derived-arithmetic recomputation, literature verification, and a cross-document
consistency scan.

## Results

### PASSED unchanged (re-derived independently)
| Claim | Status |
|---|---|
| Mg²⁺ 17,428 / K⁺ 1,868 / 96 Mg-bearing structures | reproduces exactly |
| Inner-sphere OP1+OP2 share 0.83 | 0.8284 independent |
| Mg/rigidity gradient 1.760 sigma, monotonic | reproduces exactly |
| contacts/nt median 4.397; long-chain density 0.353% | reproduces |
| flat-proposal recall 0.200; random baseline 0.746 | reproduces |
| block occupancy 1.34%, effective c 17.24 | reproduces (17.12 strict) |
| coevolution mean precision@L/5 0.670 | reproduces |
| HPT benchmark: L=4096 at 0.96% of dense, c=19.6 | reproduces |
| parameter budget 910.5M / 382.0M | exact |
| 309 GB dense activations at L=2048 | exact |
| SOTA TM-scores (trRosettaRNA 0.548 etc.) | VERIFIED vs primary source (n=52 monomers) |

### DEFECTS FOUND AND CORRECTED

**1. PHYSICS ERROR (substantive).** Documents stated axial charge spacing
`b ~ 5.9-7.0 A` together with `theta ~ 0.76`. These are mutually inconsistent
(b=5.9 A gives theta=0.175). Manning's `b` is the **axial** charge spacing, not
the P-P contour distance. Corrected to A-form RNA values **b = 1.40 A,
xi = 5.11, theta = 0.804** (`q_eff = -0.196`); noted that 0.76 is the **B-DNA**
figure. Verified against literature (B-DNA xi=4.2/theta=0.76; A-RNA atmosphere
neutralises ~0.8 of phosphate charge).

**2. OVERCLAIM (substantive).** The coevolution depth split was described as
"the decisive pattern" and "the strongest argument for MoE in the whole design".
Testing shows: exact permutation p=0.0455 on a **post-hoc** threshold,
**Spearman +0.224** over n=12, deep arm **n=2**, and a *shallow* family reaching
precision 1.000. Retracted in all three deliverables; the architectural case now
rests on the literature and on measured between-family variance (0.333-1.000).

**3. SAMPLE MIS-STATEMENT.** The Mg/rigidity gradient was reported over
"31 X-ray structures". It is measured over **15** (the Mg-bearing subset); 31 is
the set used for density/sequence-context analyses. The `>=30-RNA-residue` guard
was also unstated. Both corrected.

**4. SILENT EDIT FAILURE.** An unguarded string replacement had silently failed,
leaving the report with **6 novelty rows while the other documents had 7**, and
leaving two retracted phrases in the report after a failed retry. Found by the
cross-document scan; fixed. All subsequent edits use asserted replacements.

### STRENGTHENED (new analysis, not previously reported)
- **Confound control on the headline 1.76 sigma claim**: partial correlation of
  z_B with Mg-distance controlling for packing density is **+0.372** vs **+0.420**
  raw, so the signal is largely independent of packing. Within-structure estimate
  **+1.514 sigma**, positive in 4/4 structures, bootstrap CI [+1.256, +1.661].
  This materially improves the claim and is now in all three deliverables.
- **30 A centroid prefilter proven safe**: theoretical bound 22.20 A, observed
  max 18.59 A, zero contacts missed in an exact no-prefilter test.
- **Ground-truth incompleteness disclosed**: the coevolution parser omits WUSS
  pseudoknot brackets, 53 of 537 pairs (9.87%). Bias is conservative (deflates
  precision), so 0.670 is a lower bound.

### KNOWN LIMITATIONS (flagged, not resolved)
- Contact-sparsity bins at L=100-200 (n=6), 200-500 (n=3) and 500-1500 (n=9) are
  thin. The O(L) conclusion rests on the well-populated ends (n=38 and n=61),
  with bootstrap CIs [1.335,2.063] and [4.757,4.957] respectively. The
  interpolating bins are illustrative, not load-bearing. [FLAGGED]
- Within-structure rigidity estimate rests on 4 structures. [FLAGGED]
- Minor ion-count drift between parsers (Zn 362 vs 359, ~0.8%) attributable to
  mmCIF loop-termination handling. Below tolerance; headline numbers unaffected. [FLAGGED]
- `effective_c` naive accounting overstates by 1.029x (17.61 vs 17.12 strict) —
  conservative, left as-is with a note. [ACCEPTED]
- "8x cheaper attention" counts only the quadratic term; including sliding-window
  cost the honest ratio is 7.53x. Footnoted in the report. [CORRECTED]
- **R1 remains open**: the reference implementation proves the hierarchical
  track's cost and structure, not that a trained scorer finds occupied blocks.

## Regression guard
`scripts/sampling/verify_claims.py` re-derives every headline number, checks the
derived arithmetic and closed-form physics, verifies all 14 shared tokens appear
in all three documents, and asserts the four retracted phrases are absent.
Exit 0 = reproduces. Currently: **ALL CLAIMS REPRODUCE**.

## Confidence assessment
**MEDIUM-HIGH.** The measurements are reproducible and now confound-controlled;
the two substantive defects (physics, overclaim) were in the *interpretation*,
not the data. Confidence is not HIGH because: the sample is 180 structures and
12 alignments; several length bins are thin; and the central architectural
claim (learned block detection) is still unproven by construction.

Seven defects have now been found across two sessions (five implementation, two
in the write-up). That base rate argues for keeping the regression guard in CI
and re-running it before any future publication of these numbers.
