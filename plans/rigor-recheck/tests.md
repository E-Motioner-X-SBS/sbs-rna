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

### S6 — literature claims, verified by search this session — **PASS with 2 corrections**

| Claim | Verdict |
|---|---|
| gRNAde multi-state 3-5%, best at 3 | **VERIFIED** this session |
| RNAnneal: 16 experimentally-resolved riboswitch conformations, 10-state | **VERIFIED** this session |
| Pentameric scale = minimum range of elastic couplings | **VERIFIED** this session |
| Muon ~2x compute efficiency; Moonlight 3B/16B MoE, 5.7T tokens, ~52% FLOPs | **VERIFIED exactly** |
| HRM 27M, ~1000 examples, 40.3% ARC-AGI; loop drives it, hierarchy ~5pp | **VERIFIED** (ARC Prize independent analysis) |
| FP8 <0.25% loss error vs BF16 | **VERIFIED**, but **misattributed** |

**Correction 1 (defect #14, misattribution).** The documents said "<0.25% loss
error at 671B scale". The DeepSeek-V3 FP8 framework was validated at
**V2-Lite / V2 scale over ~1T tokens**. V3 is 671B, but the ablation
establishing the figure was smaller. Corrected in ARCHITECTURE.md (2 sites),
main.tex and the blueprint.

**Correction 2 (missing caveat).** The ARC Prize analysis that credits HRM's
outer loop *also* reports limited cross-task transfer and that **most of HRM's
ARC-AGI performance comes from memorising solutions to the evaluation tasks**.
That caveat existed in prior-art 07 but **did not propagate** to ARCHITECTURE.md,
main.tex or the blueprint, where HRM is used to justify right-sizing. Since §10e
independently established that the evidence base is narrow, citing a
substantially-memorisation result without the qualification is doubly risky.
Caveat propagated to all three; the loop stands on its measured merits, but the
parameter-count argument no longer leans on HRM.

### S7 — cross-document consistency — **PASS**
19/19 shared tokens present in ARCHITECTURE.md, main.tex and blueprint.html.
All 7 retracted phrasings confirmed absent ("decisive pattern", "strongest
argument for MoE", "0.76 for monovalent RNA", "contributes/adds more than
sequence", "at 671B scale", "143,871 unobserved-residue records").
The 4 apparent gaps were formatting, verified not drift: main.tex uses LaTeX
thousand separators (`11{,}478`, `17{,}428`, `103{,}964`) and the blueprint
writes `8.9%` where the others write `8.90%`.

## Cycle 3

### P1 — precision formats, verified by search — **PASS, 1 decisive finding**
NF4: W4A16 only, base **frozen**, gradients to LoRA adapters only -> **cannot
train from scratch**. NVFP4: E2M1, 16-elem blocks, FP8 scale; pretraining recipe
reports no measurable loss vs FP8, validated at **8B/1T = 125 tok/param**.
MXFP4 needs ~36% more tokens. FP8 <0.25% at V2/V2-Lite scale.

### P2 — is precision the bottleneck? — **NO, measured**
elDORS entropy **2.0167 bits/nt**; 512-dim bf16 token = 8,192 bits = **4,062x**.
Fully attributed nucleotide 58.9 bits -> still **139x**. Coordinates: median
resolution 3.10 A, so fp16 is **3,100x** finer than the noise floor.

### P3 — defect #15, hardware incoherence — **CONFIRMED**
A100 (SM80) has no FP8 tensor cores. Cost quoted in A100-hours with a 1.60x FP8
lever. Independent recomputation: A100 bf16 **299 h** (matches the doc's 299,
confirming that figure was right and the later levers broke the unit);
H100 fp8 **47 h**; ratio 6.3x. The published 79 sits between them.

### P4 — token attributes — **16 inventoried, coverage measured**
58.9 bits total; **26.0 available at inference**; 6 structure-only (supervision,
not input); 3 MSA-gated; 1 not acquired (SHAPE/DMS).

### P5 — repo layout — **created and tested**
`src/pharos/{model,data,train,eval,physics}` + `configs/`. The physics module is
real and locked: independently reproduces A-RNA theta **0.8044** and B-DNA
**0.7625**, and demonstrates ion-conditioning (Debye length 9.61 -> 6.88 A as
salt rises, |B_elec| falling with it). `test_manning.py` ALL TESTS PASS.

### P6 — the joint decision — **3 reversals (REV-3/4/5)**
QiD scaling law scored across the ladder; PHAROS-Small at 265x Chinchilla is the
worst case. Token budget cut 323B -> 25B (12.9x saving, no accuracy cost)
dominates any precision lever (FP8 1.6x).

## Cycle 4 — internal coherence after ~20 corrections

### C1 — recycle count: pipeline says x3, config says 8 loops — **DEFECT #16**
Two different quantities in one column. Section 10b.3 says "train 8-16, serve 3",
so Small's 8 is a TRAIN count while Base-v2's 3 reads as a SERVE count. At a
matched serve-3 setting Small is 16x3 = **48** effective layers against
Base-v2's 32x3 = **96**. The headline "more effective depth than Base-v2
(128 vs 96)" holds only at training and **reverses at inference**.
Fixed: the ladder now carries separate `train depth` and `serve@3 depth` columns.

### C2 — were the loops ever costed? — **DEFECT #17, the largest yet**
`train_flops()` computed `6 * n_active * tokens` with **no loop factor**.
Deep supervision at every segment with a one-step (detached) gradient means each
segment is its own forward+backward, so compute scales linearly with loops.
The design even states loops "cost compute but no additional activation memory"
and then omits the compute.

| Model | loops | published h | REAL h | understated |
|---|---|---|---|---|
| Micro | 16 | 61 | 976 | 16x |
| Mini | 12 | 148 | 1,776 | 12x |
| **Small** | **8** | **299** | **2,392** | **8x** |
| Base-v2 | 3 | 1,325 | 3,975 | 3x |

Consequences:
- **"4.4x smaller than Base-v2" collapses to 1.65x.** By active params 61M vs
  269M is 4.4x; by effective compute (active x loops) 488M vs 807M is 1.65x.
- **Lever chain 24x -> 5.6x**, because right-sizing is 1.65x not 4.43x.
- Corrected cost at the decided 25B budget: **78 A100-h bf16**, **12 H100-h fp8**.

Verified by **two independently written cost paths** (`precision_and_hardware.py`
patched to include loops, and a fresh `recompute_cost_with_loops.py`) which agree
at 78 h. Guard now re-derives effective compute, the 1.65x ratio and the 78 h.

> Activation *memory* genuinely does not scale with loops -- the detach buys
> that -- so that half of the claim survives. Only the compute half was wrong.

### C3/C4 — two spec gaps — **DEFECTS #18, #19**
- **#18** the attention head count was never specified anywhere (`d=512`, heads=?).
  Fixed at 8 x 64. No parameter change, but genuinely absent.
- **#19** "Motif bank + heads + decoder ~10M" was a placeholder, and the decoder
  was never specified at all, so the line was unverifiable. Itemised:

| Component | params |
|---|---|
| 9 prediction heads | 1.65M (cycle-3 additions only 0.80M) |
| motif bank, frozen | 0.17M |
| frame-diffusion decoder, 4 blocks | **12.59M** |
| total | **14.41M** vs a 10M line |

Totals 149M/61M -> **~153M/~65M** (+3% / +7%). Heads are cheap; the decoder is
not. An unspecified component is not a small one.

### C5 — self-inflicted edit corruption — **DEFECT #18b**
My own cycle-2 global `replace("143,871", "46,447 RNA")` rewrote the number
inside parentheticals that were *describing the all-polymer total*, producing
"46,447 RNA all-polymer" in 5 places across 4 files. Repaired; both diagrams
re-rendered. Exactly the unguarded-replace failure mode cycle 1 flagged -- and I
reintroduced it while fixing a different defect.

## Cycle 5 — buildability audit

### B1 — could this be built from the spec? — **10 of 37 components not specified**
27 SPECIFIED. Seven genuinely absent: learning rate, batch size, warmup, the five
loss weights, decoder diffusion steps, MoE bias-update rate, GDN state size. Two
existed only outside the configs (`d_pair` in the reference impl; the md5 split
rule in the catalogue). All closed or marked **PROVISIONAL with the procedure
that fixes them** rather than invented.

### B2 — circularity sweep — **DEFECT #20**
G6 found the router reads features produced downstream of itself. The same sweep
had never been run on anything else. **`B_motif` has the identical shape**: it
enters the *trunk's* attention bias but comes from a bank keyed on the
*interaction graph*, which is the pair track's output — produced after the trunk.
`B_elec` was given a first-pass rule; `B_motif` never was, and G6's router fix was
described but never written into the bias equation. All three now share one
explicit `attention_bias_recycle_schedule`.

### B3 — MoE active capacity — **DEFECT #21**
Checked against the *verified* DeepSeek-V3 config (d=7168, d_ff=2048, 1+256
experts, 8 activated):

| | d_ff/d | active | active FFN / d |
|---|---|---|---|
| DeepSeek-V3 | 0.286 | 9 | **2.571x** |
| PHAROS-Small | 0.250 | 6 | **1.500x** |
| dense | — | — | 4.000x |

Per-expert width matches the template; **active capacity does not** — 1.7x below
DeepSeek, 2.7x below dense. The design copied the segmentation and not the
activation. Fix is cheap: top-8 gives 2.50x for +12.6M active and ~0 total.
Design gap, not an error; decide at the 5B checkpoint.

## Cycle 6 — one canonical definition of "an RNA residue"

### C1 — do the parsers agree? — **DEFECT #22**
Two scripts reading the same 180 files reported different totals, and the gap
had been visible since cycle 2 without being chased:

| script | RNA residues | structures | definition used |
|---|---|---|---|
| `analyze_ions_motifs.py` | 307,965 | 179 | hardcoded list of 16 residue names |
| `audit_generalization.py` | 308,370 | 180 | `len(comp) <= 3 and not water and group=ATOM` |

Per-structure diff: they disagree on **72 of 180 structures, in both
directions**. 7PKT: the hardcoded list is **767 residues short** (it misses every
modification not on the list). 8JDJ: the permissive rule is **156 short the
other way** (it drops RNA coded as HETATM). Neither is authoritative because
**neither is a definition** — each script invented one.

### C2-C4 — the canonical definition — **DEFECT #23**
mmCIF *declares* polymer type. First implementation (`_entity_poly.type ==
polyribonucleotide`, chains from `pdbx_strand_id`):

```
RNA residues: 404,205     outside A/C/G/U: 24.87%
```

24.87% is not a modification rate, it is contamination. Composition dump:
**HOH 37,857 and MG 9,997 in the first 60 files alone** — water and ions sit in
the *same auth chain* as the RNA they solvate. Fix: mmCIF assigns `label_seq_id`
only to polymer positions, so a non-polymer entry carries `.`:

```python
if r.get("label_seq_id", ".") in (".", "?"):
    continue
```

| | before | after |
|---|---|---|
| RNA residues | 404,205 | **306,857** |
| outside A/C/G/U | 24.87% | **1.04%** |
| solvent leakage | HOH 37,857, MG 9,997 | **0 — CLEAN** |

### C5 — canonical re-derivation of the G-findings
`scripts/sampling/audit_generalization_canonical.py`:

| Finding | published | **canonical** | verdict |
|---|---|---|---|
| structures with a declared RNA entity | 180 | **162** | 18 had none |
| RNA residues | 308,370 | **306,857** | −0.5% |
| G1 longest chain median / max | 67 / 3,679 | **86.5 / 3,764** | non-RNA chains dragged the median down |
| G2 structures with protein | 159/180 = 88.3% | **157/162 = 96.9%** | **stronger** |
| G2 structures with >1 RNA chain | 150/180 = 83.3% | **118/162 = 72.8%** | −10.5 pts |
| G2 RNA residues in complexes | 98.96% | **99.70%** | **stronger** |
| G3 ribosome-like entries | 61/180 = 33.9% | **60/162 = 37.0%** | +3.1 pts |
| G3 residues ribosomal | 93.07% | **93.35%** | unchanged |
| G7 outside A/C/G/U | 8.90% | **1.04%** | **8.6x too high** |
| G7 distinct modification types | 8 | **76** | undercounted 9.5x |

G7's published 8.90% was DNA and UNK: DT 4,608 + DG 4,556 + DA 4,440 + DC 4,135
+ UNK 7,036 = **24,775 of 27,437** "modified" instances — 90% contamination. The
*argument* survives on better ground: 3,189 genuinely-RNA modified residues
across 76 distinct types, led by PSU (999), OMG (373), A2M (365), OMC/OMU (268
each), inosine (80). "Seventy-six chemically distinct modifications in 180
structures" is a stronger case for widening the vocabulary than a percentage
inflated by DNA ever was.

### C7 — audit of the propagation — **DEFECT #24**
The correction was propagated and the propagation was then checked, on the
principle from cycle 4 that a repair can carry its own defect. It did, twice:

1. **Partial**: only **4 of the 11** affected numbers were moved to the 162
   basis. G2's protein row read `157/162` while the multi-chain row directly
   beneath it still read `150/180`. G3 carried the canonical *ratio* 93.35% over
   the stale numerator and denominator `286,990 / 308,370`. Three tables were
   left mixing two denominators inside one row-set — undetectable to a reader,
   because every individual number was defensible in isolation.
2. **Corrupted the correction table**: an unguarded global replace of the
   canonical values rewrote the **published** column of the cycle-6 comparison
   table in ARCHITECTURE.md, so it read `99.70 -> 99.70` and
   `1.04 -> 1.04 (8.6x too high)`. **Third occurrence of the unguarded-replace
   failure mode** (after #18b in cycle 4), and the first to destroy a table whose
   only purpose was to record a correction.

All 11 numbers are now on the 162 basis; the denominator is stated in each
section preamble rather than implied; every repair substitution was made with an
exact-match, count-asserted replace.

### C8 — the resolver is now tested (`src/pharos/data/test_mmcif_entities.py`)
A shared definition that is itself untested only relocates the problem.
15 properties over all 180 files — **ALL TESTS PASS**:

| Property | Result |
|---|---|
| no water/ion/ligand counted as a residue (defect #23) | clean |
| no deoxyribonucleotide counted as a residue (the 8.6x) | clean |
| every residue carries a resolved `auth_seq_id` | clean |
| structures with a declared RNA entity | 162 |
| canonical RNA residues | 306,857 |
| residues outside A/C/G/U | 3,189 |
| distinct modification types | 76 |
| non-ACGU fraction below 2% | 1.04% |
| pseudouridine is the top modification | PSU = 999 |
| repeated calls agree (determinism) | pass |
| largest entry (7QVP) parses to a chain->type map | 162 chains, 2 types |
| chain ids plausible — no stray `;`-block sequence text | max id len 2 |

### C11 — hybrid chains: measured, not assumed
`polydeoxyribonucleotide/polyribonucleotide hybrid` chains are excluded because
the declaration does not say which of their residues are RNA. That is a choice,
so it was measured: **3 entries** carry hybrid chains (7PU7 P/T, 7S3B B, 8DFA N),
holding 59 residues of which **6 are A/C/G/U — 0.002%** of 306,857. Immaterial at
this sample size, and now documented in the resolver rather than silent.

### C9-C10 — regression guard extended
`scripts/sampling/verify_claims.py` — **ALL CLAIMS REPRODUCE**:
- 12 new canonical measurements pinned (162, 306,857, 86.5, 3,764, 0.9970,
  0.7284, 60, 0.9335, 3,189, 0.01039, 76) plus the published contamination
  (24,775 of 27,437) so the 8.6x stays explainable rather than asserted.
- 9 new cross-document tokens required in all three deliverables.
- **Defect #24 guard A**: each row of the correction table must contain *both*
  columns (`308,370`&`306,857`, `98.96`&`99.70`, `93.07`&`93.35`, `8.90`&`1.04`).
  A global replace that collapses one into the other now fails the build.
- **Defect #24 guard B**: the stale denominators `150/180`, `61/180`, `159/180`
  may appear **at most once per document** — legitimate in the published column,
  a failure anywhere else.
- 5 superseded strings banned outright (`median 67 nt`, `max 3,679`,
  `286,990 / 308,370`, `44 of 180`, `56 of 180`).
- All three test suites now run inside the guard:
  `test_hierarchical_pair_track.py`, `test_manning.py`, `test_mmcif_entities.py`.

### C12 — deliverables rebuilt
| Artifact | State |
|---|---|
| `research/report/main.pdf` | **39 pages, 0 overfull, 0 underfull** |
| `research/architecture/blueprint.html` | **v23**; tags balanced (section 17/17, div 232/232, table 31/31, p 156/156, tr 163/163, td 496/496) |
| `research/architecture/ARCHITECTURE.md` | canonical basis stated in §10e preamble |
| `scripts/sampling/verify_claims.py` | ALL CLAIMS REPRODUCE |

## Cycle 7 — auditing the audit's own instrument

### C15-C17 — the 18 structures were not empty, they were unparsed — **DEFECT #25**
Cycle 6 deferred OQ-3 on the reasoning that the 18 structures "contribute zero
residues under either definition, so no published number depends on it". That
assumes the zero is real. Raw inspection:

```
7LNE   entity_poly rows=0   types=[]          <- 16 of these
7PU7   entity_poly rows=3   hybrid + polypeptide
7S3B   entity_poly rows=2   hybrid + polypeptide
```

**16 of 18 returned zero `_entity_poly` rows.** The raw file (7LNE):

```
_entity_poly.entity_id                      1
_entity_poly.type                           polyribonucleotide
_entity_poly.pdbx_strand_id                 A,B
```

mmCIF serialises a category in **two** forms: the `loop_` form (headers then one
line per row) and a **key-value form** used when the category has exactly one
row. `entity_poly_types()` handled only the loop form.

**The failure class is not random.** A single polymer entity means one RNA and no
protein partner — a *small isolated RNA*. That is exactly the population G2
quantifies. Recovering the 16:

| | cycle 6 | **cycle 7** |
|---|---|---|
| structures with countable RNA | 162 | **178** |
| RNA residues | 306,857 | **309,191** |
| **G2 RNA residues in complexes** | **99.70%** | **98.95%** |

The published figure was 98.96%. **Cycle 6's headline finding — "G2 strengthens"
— was entirely the artifact of this bug**, and it was published one commit before
this cycle caught it.

### C19 — hybrid chains resolved rather than excluded
Cycle 6 excluded `polydeoxyribonucleotide/polyribonucleotide hybrid` chains and
measured the loss at 6 residues. Cycle 7 resolves them instead, using a
*structural* test that does not reintroduce the curated-list problem of defect
#22: **ribose carries an `O2'` atom, deoxyribose does not.**

| entry | hybrid chain | has O2' | no O2' |
|---|---|---|---|
| 7S3B | B | **U 3, C 2, G 1** | DU 1, BRU 1 |
| 7PU7 | P, T | — | DA 11, DG 8, DC 7, DT 7 |
| 8DFA | N | — | DC 6, DG 6, DT 3, DA 3 |

BRU (5-bromo-deoxyuridine) is correctly classed as DNA by the atom test though
its name resembles a uridine. Coverage closes at **179 of 180**; the exception is
7PU7, whose only nucleic entity is declared hybrid and modelled entirely as
deoxyribonucleotide — there is no RNA in its coordinates to count.

### C20 — the G-findings, finally
| Finding | published | cycle 6 | **canonical (cycle 7)** |
|---|---|---|---|
| structures with countable RNA | 180 | 162 | **179** |
| RNA residues | 308,370 | 306,857 | **309,197** |
| G1 longest chain, median / max | 67 / 3,679 | 86.5 / 3,764 | **70 / 3,764** |
| G2 structures with protein | 88.3% | 96.9% | **88.3%** |
| G2 structures with >1 RNA chain | 83.3% | 72.8% | **73.7%** |
| G2 RNA residues in complexes | 98.96% | 99.70% | **98.95%** |
| G3 ribosome-like entries | 33.9% | 37.0% | **33.5%** |
| G3 residues ribosomal | 93.07% | 93.35% | **92.65%** |
| G7 outside A/C/G/U | 8.90% | 1.04% | **1.05%** |
| G7 distinct modification types | 8 | 76 | **82** |

**G2 and G3 land back on the published figures.** The only finding that genuinely
moves is G7, which was **8.5x too high** — 24,775 of its 27,437 "modified"
instances were DNA and UNK. Its argument is stronger than before: 3,237 modified
RNA residues across **82 distinct types**, led by pseudouridine (1,003).

New measurement, not present before: **21 of 179 structures contain no protein at
all, holding 3,257 residues = 1.05% of the corpus.** Isolated RNA is 11.7% of
structures but 1% of residues, because isolated RNAs are small. Any "train on
autonomous folds only" mitigation works with ~1% of the supervision.

### C22 — resolver test suite extended to 20 properties — **ALL TESTS PASS**
Added over the cycle-6 twelve:

| Property | Result |
|---|---|
| key-value form parses (7LNE) | 2 RNA chains |
| key-value form parses (8FEQ) | 1 RNA chain |
| no structure silently resolves to an empty declaration | 179/180 |
| 7S3B hybrid contributes its 6 ribonucleotides | C 2, U 3, G 1 |
| 7S3B hybrid excludes DU and BRU (no O2') | deoxy excluded |
| 7PU7 hybrid is all-deoxy, contributes nothing | 0 residues |
| structures with countable RNA | 179 |
| canonical RNA residues | 309,197 |
| residues outside A/C/G/U | 3,237 |
| distinct modification types | 82 |
| pseudouridine top | PSU = 1,003 |

### C23 — the guard now protects the retraction itself
`verify_claims.py` — **ALL CLAIMS REPRODUCE**. The correction-table guard gained
two pairs so the cycle-6 error cannot be quietly erased from the record:

```
both columns present: cycle-6 vs cycle-7 totals     306,857 -> 309,197
both columns present: the retracted G2 strengthening  99.70 -> 98.95
```

A future edit that deletes the wrong cycle-6 numbers — tidying away the mistake —
now fails the build. The record of a correction is itself a claim under guard.

### C24 — deliverables
| Artifact | State |
|---|---|
| `research/report/main.pdf` | **39 pages, 0 overfull, 0 underfull** |
| `research/architecture/blueprint.html` | **v24**; tags balanced (div 233/233, table 31/31, p 157/157, tr 164/164, td 510/510, th 111/111) |
| `research/architecture/ARCHITECTURE.md` | three-column published/cycle-6/canonical table |
| `scripts/sampling/verify_claims.py` | ALL CLAIMS REPRODUCE |
