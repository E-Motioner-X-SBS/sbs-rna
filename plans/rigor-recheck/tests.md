# Test log

## Cycle 1

### T4a — 30 A centroid prefilter safety — **PASS**
`verify_prefilter.py`, 25 chains, exact no-prefilter heavy-atom test.
- max centroid->own-atom radius = 7.10 A
- theoretical bound for a true contact = 8.0 + 2*7.10 = **22.20 A** < 30 A
- max centroid separation among true contacts observed = 18.59 A
- contacts missed by the prefilter = **0**
Verdict: prefilter is safe; no downstream contact number is biased by it.

### T3/T11 — Mg2+/rigidity gradient, independent re-derivation — **PASS, with caveats**
`verify_rigidity.py`, independent parser and statistics.
- gradient span **1.760 sigma**, monotonic across all 6 bins — reproduces exactly
- per-bin means reproduce to 3 dp (-0.883 / -0.673 / -0.415 / +0.004 / +0.756 / +0.877)
- pearson(zB, density) = -0.3051 (documented -0.294; differs because the original
  used a 10 A spatial-hash neighbour count over a slightly different structure set)

Confound control (NEW, not in the original analysis):
- pearson(zB, mg_dist)              = +0.4197
- spearman(zB, mg_dist)             = +0.5082
- pearson(mg_dist, density)         = -0.2513  <- the confound is real but modest
- **partial corr(zB, mg_dist | density) = +0.3722**
- partial corr(zB, density | mg_dist)   = -0.2273
=> Controlling for packing density reduces the Mg association only from 0.420 to
   0.372. The signal is largely INDEPENDENT of packing. This STRENGTHENS the claim
   beyond what was originally written.

Within-structure effect (removes all between-structure heterogeneity):
- 4 structures have both a near (<8 A) and far (>=12 A) group
- mean effect **+1.514 sigma**, median +1.626, positive in **4/4**
- bootstrap 95% CI [+1.256, +1.661]
=> The gradient is not an artefact of pooling across structures.
   BUT n=4 is small; this is supportive, not decisive.

### T1 — X-ray count — **RESOLVED, not a bug** (see doubts D1)

### T7 — effective_c accounting — **PASS (2.9% conservative)**
naive `len(occ)*b*b/L` = 17.61 mean; strict count of real upper-triangle cells
with |i-j|>=4, clipped to L = 17.12. Documented 17.24 sits between.
Overstatement 1.029x, i.e. the architecture is costed slightly PESSIMISTICALLY.
No correction required; note added.

### T5 — recall_at() flat-index arithmetic — **PASS**
Reconstruction (t//L, t%L) matches direct ranking exactly on a synthetic case.
K is capped at `(flat > -inf).sum()`, so masked cells cannot leak into the top-K.

### T8a/T8b — coevolution ground truth completeness — **DEFECT (conservative)**
SS_cons contains WUSS pseudoknot brackets Aa(27) Bb(13) Cc(6) Dd(7) which
`ss_pairs()` does NOT parse (it handles only ()<>[]{} ).
- standard pairs parsed: 484
- pseudoknot pairs omitted: 53
- **9.87% of ground truth is missing**
Direction: omitting true pairs from the truth set means a correctly-ranked
pseudoknot pair is scored as a false positive => measured precision is DEFLATED.
The reported 0.670 is therefore a LOWER bound. Must be disclosed.

### T8c — Neff computation — **PASS**
Weights in (0,1], Neff <= N for every family checked. 5S rRNA N=712 -> Neff=10.3
reflects genuine redundancy, as expected for rRNA.

### T8d — depth split statistics — **FAIL: OVERCLAIM**
- deep (n=2): [1.000, 0.950] mean 0.975
- shallow (n=10): [..., includes **1.000**] mean 0.609
- exact permutation test p = 3/66 = **0.0455** (marginal, post-hoc threshold)
- pearson(neff_per_col, precision)  = +0.498
- **spearman(neff_per_col, precision) = +0.224**  <- weak monotonic association
- a SHALLOW family (THF riboswitch, Neff/L=0.279) attains precision 1.000
Verdict: a mean difference of +0.366 with n=2 in one arm, an arbitrary post-hoc
split point, and Spearman +0.224 does NOT support the language used
("the decisive pattern", "the strongest argument for MoE in the whole design").
REVERSAL REQUIRED in all three deliverables.

### T13/T14/T15 — derived arithmetic — **PASS (exact)**
- attention 32x4x768^2 = 75.50M (doc 75.5M)
- MoE total 16x34x(3x768x512) = 641.73M (doc 642M); active 16x6x... = 113.25M (doc 113M)
- dense FFN 16x3x768x3072 = 113.25M (doc 113M)
- TOTAL 910.5M (doc ~911M); ACTIVE 382.0M (doc ~382M)
- vs NucleicBERT 404M: 2.25x capacity / 0.95x compute (doc 2.3x / 0.95x)
- attention at L=4096: all-dense 536.9M vs full-only 67.1M = **8.0x** (doc 8x).
  Including sliding-window cost the honest ratio is 7.53x; the doc's 8x counts
  only the quadratic term. Minor - worth a footnote, not a correction.
- dense activations at L=2048: 309.2 GB (doc 309.2 GB); single tensor 1.07 GB (doc 1.07 GB)

### T16 — Manning/Debye physics constants — **FAIL: PHYSICS ERROR**
Bjerrum length derived 7.15 A at 298.15 K, eps_r=78.4 (doc 7.1 A) — fine.

But the document states the phosphate axial charge spacing as **b ~ 5.9-7.0 A**
while also stating **theta ~ 0.76**. These are mutually inconsistent:
  b=5.9 A -> xi=1.21 -> theta=0.175
  b=7.0 A -> xi=1.02 -> theta=0.021
5.9-7.0 A is the through-backbone P-P CONTOUR distance. Manning's b is the
**axial** charge spacing (charges projected onto the helix axis). The document
even justifies its number with "the charge spacing along the contour is what
matters", which is exactly backwards.

Correct [DERIVED + VERIFIED against literature this session]:
  B-DNA:      rise 3.4 A/bp, 2 charges/bp -> b=1.70 A, xi=4.21, theta=0.762
  A-form RNA: rise 2.8 A/bp, 2 charges/bp -> b=1.40 A, xi=5.11, theta=0.804
Literature: B-DNA xi=4.2, theta=1-1/xi=0.76 [VERIFIED]; A-form RNA ion atmosphere
"neutralizes 0.8 of the phosphate charge", "0.7-0.8 monovalent cations bound per
phosphate" [VERIFIED]. Derived values match both.

Consequence: theta~0.76 quoted in the deliverables is the **B-DNA** figure. For
A-form RNA the correct value is **~0.80**, giving q_eff = -0.196 rather than
-0.238. CORRECTION REQUIRED in all three deliverables.

### T24 [ADDED-CYCLE-1] — residue-guard bias on the Mg gradient — **PASS (no bias)**
`test_residue_guard_bias.py`. Recomputes the gradient at guard in {30,20,15,10,5,1}.

| guard | structures | nt | span (sigma) | monotonic |
|---|---|---|---|---|
| 30 (baseline) | 15 | 24,623 | **1.760** | yes |
| 20 | 16 | 24,646 | 1.757 | yes |
| 15 | 16 | 24,646 | 1.757 | yes |
| 10 | 17 | 24,660 | 1.757 | yes |
| 5  | 17 | 24,660 | 1.757 | yes |
| 1 (no guard) | 19 | 24,666 | 1.757 | yes |

max |change| = **0.003 sigma**. Removing the guard entirely adds 4 structures and
43 nucleotides (+0.17%) because the excluded structures hold 2-28 residues each.
OQ-1 is answered: the guard is a variance control, not a source of bias.

> Parser defect found while writing this test: column names were taken with
> `line.rstrip("\n")` instead of `.strip()`, leaving a trailing space
> (`'label_comp_id '`), so every lookup missed and the script reported
> **0 X-ray structures**. Same class as defect #2 in cycle 1. The script now
> raises if `label_comp_id` is absent and refuses to report on an empty set,
> rather than printing a confident zero.
