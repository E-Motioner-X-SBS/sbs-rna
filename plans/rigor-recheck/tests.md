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

### T25 [ADDED-CYCLE-1] Guard-sensitivity of the Mg gradient — **PASS**
A concurrent session published a claim that the >=30-RNA-residue guard is a
variance control rather than a bias. Verified independently rather than accepted:

| guard | structures | nucleotides | span (sigma) | monotonic |
|---|---|---|---|---|
| >=30 | 15 | 24,623 | 1.760 | yes |
| >=20 | 16 | 24,646 | 1.757 | yes |
| >=15 | 16 | 24,646 | 1.757 | yes |
| >=10 | 17 | 24,660 | 1.757 | yes |
| >=5  | 17 | 24,660 | 1.757 | yes |
| none | 19 | 24,666 | 1.757 | yes |

Span moves 0.003 sigma across the whole sweep, monotonic throughout.
The cross-session claim is CORRECT. **OQ-1 is now closed.**

### T26 Base-pair geometry extraction — **PASS after 2 defects fixed**
- mmCIF WRAPS long rows across physical lines; requiring all fields on one line
  yielded ZERO rows for every wide category (43-column steps arrive as 24 + 19).
- Metal-vs-ligand assignment used a length/case heuristic that classified the
  nucleotides G, A, U, C as metals.
After fixing: 103,964 steps, 76 stiffness contexts, means reproduce canonical
A-form RNA (GG/CC rise 3.14 twist 29.98; AU/AU rise 2.81 twist 33.94).
Curated metalc records independently reproduce the distance-based coordination
ranking (OP2 > OP1 > O6 > O4 > O2' > N7).

### T27 Stiffness headroom — **PASS, justifies the encoder**
M0 19.724 / M1 17.579 (sequence table) / M2 14.551 (sequence x structure).
Structural context contributes MORE than sequence (+3.028 vs +2.144 nats).

### T25 [CROSS-CHECK] — gradient fix verified from outside — **PASS**

The second audit found the L1/L2 block scorers dead (scores fed only `topk`,
which is non-differentiable) and applied two fixes. Verified independently here
with a fresh script that runs a real backward pass and inspects every parameter,
rather than re-running the audit's own test:

| component | params | with grad | non-zero grad | max abs grad |
|---|---|---|---|---|
| l1 (coarse b=16 scorer) | 3 | 3 | **3** | 2.838e-02 |
| l2 (b=4 scorer) | 3 | 3 | **3** | 8.699e-02 |
| rest of module | 16 | — | 16 | — |

Every block-scorer parameter now receives a non-zero gradient. The selectors are
trainable.

Cost of the fix, re-measured (the audit claimed none):

| L | time | pairs | % of dense | effective c |
|---|---|---|---|---|
| 1024 | 0.12 s | 20,108 | 3.839% | 19.64 |
| 2048 | 0.22 s | 40,198 | 1.918% | 19.63 |
| 4096 | **0.45 s** | 80,396 | **0.959%** | 19.63 |

Matches the pre-fix figures (0.45 s, 0.959%, c=19.63); pair counts differ by 6 of
80,390 at L=4096, from sigmoid gating altering a tie-break. The claim that the
fix is free stands.

> Worth noting *why* the original benchmark could not have caught this: it ran
> under `torch.no_grad()`, so it measured only forward cost. A speed benchmark
> cannot detect an untrainable parameter. Nine silent defects now, every one
> producing a plausible result rather than an error.

### T26 [CROSS-CHECK] — stiffness ratio was inflated by rounding — **DEFECT (#10)**

The "134x stiffness span" is quoted to three significant figures, but
`extract_basepair_geometry.py` stored force constants with `round(F[i,i], 4)`.
Twist/roll/tilt constants run to ~1e-4, so the softest values were left with
**one significant figure** and the ratio inherited that.

Recomputed at 6 significant figures:

| quantity | value |
|---|---|
| twist force constant, stiffest (UG/UG) | 1.34269e-2 |
| twist force constant, floppiest (AA/UA) | 1.16663e-4 |
| **true ratio** | **115.1x** (not 134x) |
| twist sd span, same endpoints | 11.48 deg -> 102.44 deg = 8.9x |

The named endpoints were correct; only the ratio was wrong. (At 4 dp, AU/AA and
AA/UA both read 0.0001, so which one is the minimum was decided by dict order,
not by the data.)

Also added: the covariance **condition number** per context. 3 of 76 exceed 1e4
(AU/AA, GA/AA, GC/AC), i.e. `F = kT C^-1` is least trustworthy exactly where the
steps are floppiest, so the softest constants carry the largest uncertainty.
This was not previously disclosed.

Corrected in all six places the figure appeared: ARCHITECTURE.md, main.tex,
blueprint.html, prior-art 07, diagram 07 (source + re-rendered SVG/PNG), and the
audit's own final-report.md. Script now rounds to significant figures rather
than fixed decimals.

> Defect #10, and the second caused by *formatting* rather than logic (the first
> being trailing-space column names). Numbers that survive a correctness review
> can still be wrong at the precision they are quoted to.

## Cycle 2

### S1 — disorder-label count RNA-only? — **DEFECT #11**
Independent parser reproduces the documented total exactly (143,871), then splits:
protein/other 97,131 (67.5%) · **RNA 46,447 (32.3%)** · DNA 293.
`extract_basepair_geometry.py:163` had `n_unobs += len(rows)` with no residue
filter. A novelty claim was overstated **3.1x**. Corrected everywhere; the
extractor now reports RNA/DNA/protein separately and the guard re-derives both.

### S2 — A100-hours 299 vs 79 — **PASS, not a discrepancy**
The lever table is cumulative, not parallel. Chained: 1883 /1.42 -> 1326
/4.43 -> 299 /2.00 -> 150 /1.60 -> 94 /1.19 -> 79. Total 24.0x. Both figures
are correct at different points in the chain. **My suspicion was wrong.**

### S3 — right-sizing arithmetic — **PASS exact**
attention 4d^2*16 = 16.78M (doc 16.8M); MoE total 34*3*512*128*16 = 106.95M
(doc 107.0M); active 6/34 -> 18.87M (doc 18.9M); TOTAL 148.73M (doc 149M);
ACTIVE 60.65M (doc 61M); 16x8 = 128 vs 32x3 = 96 layers; 1401/149 = 9.40x.
Size ladder consistent with cost ~ 6*N_active: Small 299 h, Mini predicted
147.0 vs doc 148, Micro predicted 58.8 vs doc 61.

### S4 — stiffness headroom NLLs — **DEFECT #12, and it REVERSES the claim**
Each model was fitted **in-sample** and scored on its **own covered subset**
(M0 103,964 / M1 90,098 / M2 78,076 steps). Finer partitioning lowers in-sample
NLL mechanically, and M2's subset is the better-populated, more regular steps.

Rescored on the common 78,076 steps, 2-fold held-out
(`measure_stiffness_headroom_heldout.py`):

| Model | in-sample (published) | held-out, common subset |
|---|---|---|
| M0 global | 19.7235 | 18.1607 |
| M1 sequence | 17.5792 | 16.3760 (gain **1.7847**) |
| M_struct structure only | 17.8470 | 16.4198 (gain 1.7409) |
| M2 sequence x structure | 14.5510 | **15.3424** (gain over sequence **1.0336**) |

structure-over-sequence: **3.0282 -> 1.0336, a 2.9x shrink, and the order
INVERTS** — sequence adds more than structure.

**Survives**: structure still adds a real +1.03 nats beyond sequence;
sequence-alone (+1.78) and structure-alone (+1.74) are near-equal and
complementary (3.53 if independent vs 2.82 actual). The learned-encoder
decision stands; only the superlative is retracted (REV-2). The §7c acceptance
gate moved from 14.551 to **15.3424** (fair held-out M2).

> Also refactored `collect()` out of the original `main()` so both scripts share
> one extractor rather than a duplicated parser that could drift. Verified the
> refactor reproduces all four published NLLs exactly before relying on it.

### S5 — aux block-occupancy loss: implemented? — **DEFECT #13, now fixed**
The documents claimed two fixes "both implemented". Only one was. Grep across
the repo found **no** BCE / occupancy loss anywhere — the reference code merely
*exposed* the scores in an `aux` dict so a loss could be attached later. The R1
downgrade rested on a hook.

Implemented `block_occupancy_loss()` (recall-weighted BCE, `pos_weight=neg/pos`)
and added a permanent test:

| Check | Result |
|---|---|
| gradient reaches l1 selector from the aux loss ALONE | 3/3 params, PASS |
| gradient reaches l2 selector from the aux loss ALONE | 3/3 params, PASS |
| optimising it decreases it | 3.0278 -> 0.0000, PASS |
| block recall improves | l1 0.000 -> 1.000, l2 0.000 -> 1.000, PASS |

> Honest scope: a single-example overfit on a synthetic stem-plus-cluster
> pattern. It shows the mechanism (gradient path + optimisability), **not**
> generalization. Recall 1.000 on one memorised pattern is expected.
