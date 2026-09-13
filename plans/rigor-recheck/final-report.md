# Final Verification Report — Cycle 5 (buildability)

| Field | Value |
|---|---|
| Cycles | 5 |
| Defects this cycle | **2** (#20, #21) + 10 unspecified components |
| **Total defects** | **21** |
| Macro-audit | **8/8 YES** |
| Open to-dos | 0 (2 deferred to the 5B checkpoint) |

## The question this cycle asked

After #18 (attention heads) and #19 (decoder) — both components *never specified
at all*, then silently assumed small — cycle 5 asked of every component:
**could an engineer implement this from the spec, without inventing a number?**

**37 enumerated, 27 specified, 10 not.** Seven genuinely absent (learning rate,
batch size, warmup, the five loss weights, diffusion steps, MoE bias-update rate,
GDN state size); two present only in reference code or the catalogue, never in a
config; one exposed a circularity.

All closed or marked **PROVISIONAL with the sweep that fixes them**. A guessed
hyperparameter is worse than an acknowledged gap, and the 5B checkpoint costs
~1.5% of the run.

## Defect #20 — the circularity sweep had only ever been run on the router

G6 (cycle 2) found the router consumes features produced downstream of itself.
**That sweep was never run on anything else.** `B_motif` has the identical shape:
it enters the *trunk's* attention bias but is keyed on the *interaction graph*,
which is the pair track's output — produced after the trunk. `B_elec` had a
first-pass rule; `B_motif` never did; and G6's own fix was described in prose but
never written into the bias equation. All three now share one explicit
`attention_bias_recycle_schedule`.

## Defect #21 — the MoE copied one DeepSeek ratio and not the other

| | d_ff/d | active | **active FFN / d** |
|---|---|---|---|
| DeepSeek-V3 (verified) | 0.286 | 9 | **2.571x** |
| **PHAROS-Small** | **0.250** | **6** | **1.500x** |
| dense | — | — | 4.000x |

Per-expert width matches the cited template. **Active capacity is 1.7x below it
and 2.7x below a dense FFN.** The design copied the segmentation and not the
activation, and the ratio had never been examined. Fix is cheap — top-8 gives
2.50x for +12.6M active and ~0 total, since the experts already exist.

## Confidence

**MEDIUM-HIGH measurements · MEDIUM-LOW derived claims · LOW outcomes.**

Unchanged from cycle 4, and for the same reason: measurements keep surviving
independent re-derivation; things built on top of them keep not.

## Defect base rate, five cycles

| Cycle | Defects | Character |
|---|---|---|
| 0 (build) | 5 | silent parser/logic bugs |
| 1 (audit) | 4 | wrong physics, overclaim, disclosure |
| pre-2 | 1 | precision/rounding |
| 2 | 4 | inflated count, unfair comparison, unimplemented code |
| 3 | 1 | hardware incoherence |
| 4 | 5 | **derived quantities**: cost, depth, budget |
| **5** | **2** | **components never specified at all** |

**21 defects. The rate has not fallen across five cycles.** The *character* has
shifted though: cycles 0-2 found things that were wrong, cycles 4-5 found things
that were **missing or never checked against their own stated template**. That
is a meaningfully different failure mode and argues the audit should continue to
target *absences* rather than errors.

---

# Final Verification Report — Cycle 4

## Summary

| Field | Value |
|---|---|
| Cycles used | 4 |
| **Defects found (cycle 4)** | **4** (#16-#19) |
| Total defects, all cycles | **19** |
| Macro-audit | **6/6 YES** |
| Open to-dos | 0 (3 deferred: need hardware or training) |

Cycle 4 asked a question no earlier cycle had: after ~20 individually-correct
corrections, **does the specification still cohere as a whole?** It did not.

## Defects

| # | Defect | Impact |
|---|---|---|
| **16** | The ladder's `loops` column mixed **train-time** (Small: 8) with **serve-time** (Base-v2: 3) | At matched serve-3, Small is **48** effective layers vs Base-v2's **96** — the depth claim **reverses at inference** |
| **17** | **Refinement loops were never in the FLOP budget.** `6*N_active*T` costs one pass; Small runs 8 loops with deep supervision at each | Every cost understated by its loop count. **"4.4x smaller" is really 1.65x**; lever chain **24x -> 5.6x**; cost **78 A100-h** at 25B |
| **18** | Attention **head count never specified anywhere** | Spec gap; fixed at 8 x 64 |
| **18b** | **My own cycle-2 global replace** corrupted 5 parentheticals into "46,447 RNA all-polymer" | Repaired; the exact failure mode cycle 1 warned about, reintroduced while fixing a different defect |
| **19** | "Motif bank + heads + decoder ~10M" was a placeholder; the **decoder was never specified at all** | Itemised: heads 1.65M, bank 0.17M, decoder **12.59M** — the decoder alone exceeds the line. Totals **149M/61M -> ~153M/~65M** |

## The most consequential finding

Defect #17. The design stated that loops "cost compute but no additional
activation memory" — and then **omitted the compute**. The memory half was
right; the compute half was simply missing from every calculation.

| Model | loops | published h | real h |
|---|---|---|---|
| Micro | 16 | 61 | 976 |
| Mini | 12 | 148 | 1,776 |
| **Small** | **8** | **299** | **2,392** |
| Base-v2 | 3 | 1,325 | 3,975 |

Verified by **two independently written cost paths** agreeing at 78 A100-h for
the decided 25B budget.

Note what survives: the cycle-3 token cut (12.9x) was large enough that the
corrected absolute cost is still modest — **78 A100-h bf16, 12 H100-h fp8**.
The *claims* were wrong; the *plan* remains affordable.

## Confidence

**MEDIUM-HIGH on measurements; MEDIUM-LOW on derived claims; LOW on outcomes.**

Derived claims drop a notch: four of the last five defects were in *derived*
quantities (cost, depth, budget) rather than measurements. The measurements have
held up under independent re-derivation; the arithmetic built on top of them has
repeatedly not.

## Defect base rate

Cycle 0: 5 · cycle 1: 4 · pre-2: 1 · cycle 2: 4 · cycle 3: 1 · **cycle 4: 5**.
**Nineteen defects. The rate has not fallen.** Two were caused by formatting,
two by unfair statistical comparison, one by code that did not exist, and now
two by quantities that were never specified at all. The newest category is the
most concerning: **unspecified components silently assumed to be small.**

---

# Final Verification Report — Cycle 2

> Cycle 1's report follows below, unchanged. This covers the 6,259 lines added
> across 55 files *after* cycle 1 closed, plus the architecture-level audit the
> user requested mid-cycle ("check if all of our architecture is efficient and
> will be able to generalize").

## Summary

| Field | Value |
|---|---|
| Cycles used | 2 |
| Searches run (cycle 2) | 3 (Muon, HRM ablation, DeepSeek FP8) |
| Tests logged (cycle 2) | 8 (S1–S7 + G1/G2/G3/G7) |
| To-do completion | 14 done, 0 open, **1 DEFERRED** (G5, needs training) |
| Doubts | 3 logged, 3 resolved |
| **Defects found** | **4** (#11–#14) |
| **Major reversals** | **1** (REV-2) |
| Macro-audit | **9/9 YES** |

## Defects found in cycle 2

| # | Defect | Impact |
|---|---|---|
| **11** | `_pdbx_unobs_or_zero_occ_residues` counted **all polymers**. 143,871 rows are 67.5% protein; RNA-only is **46,447**. | A novelty claim overstated **3.1x**. Claim survives at one third the size. |
| **12** | Stiffness headroom fitted every group **in-sample** and scored each model on its **own covered subset** (103,964 / 90,098 / 78,076 steps). | **REV-2**: structure-over-sequence falls 3.0282 -> **1.0336** (2.9x) and **the order inverts**. |
| **13** | The auxiliary block-occupancy loss was documented as "implemented". Only a **hook** existed — no BCE anywhere in the repo. | The R1 downgrade rested on unwritten code. Now implemented and tested. |
| **14** | FP8 "<0.25% loss error **at 671B scale**" — the DeepSeek ablation was at **V2/V2-Lite scale (~1T tokens)**. Plus the HRM memorisation caveat existed in prior-art 07 but **never propagated** to where the argument is made. | Misattribution + a material caveat missing from the load-bearing citation. |

## The reversal (REV-2)

**Retracted**: "structural context contributes more than sequence context"
(+3.028 vs +2.144 nats), used to justify the learned stiffness encoder.

**Corrected**, held-out on the common 78,076-step subset:

| Model | in-sample | held-out |
|---|---|---|
| M0 global | 19.7235 | 18.1607 |
| M1 sequence | 17.5792 | 16.3760 (gain **1.7847**) |
| M_struct structure only | 17.8470 | 16.4198 (gain 1.7409) |
| M2 sequence x structure | 14.5510 | **15.3424** (gain over sequence **1.0336**) |

**Survives**: structure still adds a real **+1.03 nats beyond sequence**;
sequence-alone (+1.78) and structure-alone (+1.74) are near-equal and
complementary (3.53 if independent vs 2.82 actual). The learned-encoder decision
stands — a sequence-only lookup still leaves ~1.03 nats unused. The §7c
acceptance gate moved 14.551 -> **15.3424**.

## Architecture audit — the user's question

**"Is it efficient?" — YES, and it is the best-verified part.** The hierarchical
pair track runs at 0.96% of dense at L=4096 while the dense baseline cannot run
at all on a 14 GB machine; parameter and FLOP arithmetic reproduces exactly
(148.73M/60.65M against documented 149M/61M); the lever chain reproduces exactly
(1,883 -> 79 A100-h, 24.0x).

**"Will it generalize to any RNA?" — NOT AS SCOPED.** Four measured limits:

| ID | Finding | Status |
|---|---|---|
| **G2** | **98.96% of RNA residues sit in protein-containing entries** (159/180 structures, median 10 protein chains). The model predicts single chains but learns partner-stabilised folds. | **Largest hazard. Not fixable by tuning.** Split reporting now mandatory. |
| **G3** | **93.07% of RNA residues come from 61 ribosome-like entries.** Every residue-weighted statistic is primarily ribosomal. | Length-binned tables happen to stratify it; the c=20 budget is safe at both ends. Headline figures relabelled. |
| **G7** | **8.90% of polymer residues fall outside {A,C,G,U}** — 17,767 DNA, 7,036 UNK, 2,634 inosine/other. | Vocab-5 cannot represent any. "Any RNA" unsupportable without widening it. |
| **G1** | Every single chain fits 4096 (max 3,679), but total RNA per entry reaches **11,478**; 44/180 entries exceed the context. | Scope statement added. |
| **G6** | The router reads structural features produced by the pair track, which runs *after* the trunk — **routing at recycle 0 is undefined**. `B_elec` got an explicit first-pass rule; the router never did. | Design gap; fix specified. |

**Honest headline**: *a single-chain RNA structure model, up to 4,096 nt, trained
predominantly on ribosomal and complex-embedded RNA, which must report
performance split by isolated vs complexed context.*

## Confidence

**MEDIUM-HIGH on the measurements; MEDIUM on the design; LOW on outcomes.**

- *Measurements* — every headline number is now re-derived by an independently
  written check, and the regression guard re-derives them on demand. High.
- *Design* — internally consistent and the efficiency case is strong, but four
  generalization limits are now measured rather than hypothetical. Medium.
- *Outcomes* — **no model has been trained.** Nothing here predicts that PHAROS
  beats TM 0.55. Low, and should stay low until a checkpoint exists.

## Known limitations carried forward

1. **R1**: block detection is now a supervised target with a tested loss, but
   the test is a single-example synthetic overfit. Generalization untested.
2. **G5 [DEFERRED]**: 61M active vs ERNIE-RNA's 86M dense while doing strictly
   more tasks. An experiment, not an argument; first checkpoint ~1 GPU-day.
3. **G2** is unresolvable with current data — it bounds the claim permanently.
4. Muon 2x and FP8 1.6x are *reported* figures, verified as citations but
   **not benchmarked on our shapes**. The 24x cost reduction depends on them.
5. Thin bins persist: contact-sparsity n=3 and n=6; coevolution deep arm n=2;
   within-structure Mg estimate n=4.
6. Ionic titration data (RMDB) still not acquired — prerequisite for the
   headline ion-conditioning claim.

## Defect base rate

Cycle 0 (build) 5 · cycle 1 (audit) 4 · pre-cycle-2 1 · **cycle 2: 4**.
**Fourteen defects, every one producing a plausible number rather than an
error.** Two were caused by formatting rather than logic; two were unfair
statistical comparisons; one was code that did not exist. The base rate has not
fallen across cycles, which is the strongest argument for keeping
`verify_claims.py` in the loop.

---

# Final Verification Report — PHAROS empirical claims

**Date**: 2026-09-13 · **Cycles**: 1 (extended) · **Verdict**: claims stand after 6 corrections

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

## Cycle 1 extension — mmCIF mining and the training design

After the audit closed, the scope extended to mining the data more fully and
designing the training. Two further defects surfaced, both in new code:

**5. mmCIF ROW WRAPPING.** Long mmCIF rows wrap across physical lines (a
43-column base-pair-step row arrives as 24 + 19 tokens). A parser requiring all
fields on one line yields **zero rows** for every wide category — silently. Fixed
with quote-aware token accumulation. This unlocked 103,964 annotated steps.

**6. METAL/LIGAND MISCLASSIFICATION.** A length-and-case heuristic classified the
nucleotides G, A, U and C as metals. Replaced with an explicit metal set.

### New measurements (all verified)
- **103,964 base-pair steps** across 155/180 structures; stiffness matrices
  `F = kT C^-1` for **76 contexts**. Validated: Watson-Crick means reproduce
  canonical A-form RNA (GG/CC rise 3.14 twist 29.98; AU/AU rise 2.81 twist 33.94).
- Stiffness spans **115x** (twist force constants; the earlier 134x came from
  4-decimal rounding that left the softest constants with one significant
  figure). GC content predicts rigidity (Pearson -0.314).
- **44,708 curated Mg²⁺ coordination records** whose coordinating-atom ranking
  (OP2 > OP1 > O6 > O4 > O2' > N7) **independently reproduces** the earlier
  distance-based result. Two methods, same conclusion.
- **Stiffness headroom**: M0 19.724 / M1 17.579 (sequence table) / M2 14.551
  (sequence x structure). Structural context contributes **more** than sequence
  (+3.028 vs +2.144 nats) => a learned encoder beats a lookup table. This
  changed the design from a tabulated force field to a learned stiffness encoder
  with a Gaussian-NLL objective and an explicit acceptance threshold.
- **Guard sensitivity (OQ-1 CLOSED)**: sweeping the >=30-residue guard down to
  none moves the Mg-gradient span only 1.760 -> 1.757 sigma, monotonic
  throughout. A concurrent session's claim to this effect was verified
  independently rather than accepted.

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

**Nine defects have now been found across this work** — seven in implementation
(BGSU HTML-as-CSV, `_exptl.method_details` overwrite, MI outer-product
broadcast, HPT L3 diagonal-only expansion, HPT L2 budget clamp, mmCIF row
wrapping, metal/ligand misclassification) and two in the write-up (the Manning
physics error and the coevolution overclaim), plus one silent unguarded string
replacement. Every one of them would have produced plausible-looking but wrong
output rather than an error.

That base rate is the central lesson: **in this kind of work the failure mode is
silence, not crashes.** Keep `verify_claims.py` in CI, re-run it before any
publication of these numbers, and treat any weak or surprising result as a
suspected bug until independently reproduced.
