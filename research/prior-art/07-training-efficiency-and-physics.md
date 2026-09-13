# Prior Art 7 — Efficient Training, and Physics That Can Guide It

The architecture is only useful if it can actually be trained. NucleicBERT used
**192 A100s** densely; that is not reproducible for this project. This surveys
what the efficiency literature offers, and what physics can be folded into the
training objective.

## 7.1 Muon — the single biggest optimiser win available

**Muon** (MomentUm Orthogonalized by Newton-Schulz) maintains SGD-style momentum
on 2D weight matrices, then applies Newton-Schulz iterations to approximate the
polar decomposition — i.e. it *orthogonalises the gradient momentum* before
applying it.

| Property | Value |
|---|---|
| Compute efficiency vs AdamW | **~2x** at compute-optimal training |
| FLOPs to match AdamW quality | **~52%** |
| Memory saving | ~33% |
| Demonstrated at scale | **Moonlight**, a 3B/16B **MoE** model, 5.7T tokens |
| Production use | Kimi K2 |

**Why it matters here specifically**: Moonlight is a *Mixture-of-Experts* model,
so Muon is already validated on the exact architecture class PHAROS uses. A ~2x
FLOP reduction is the difference between "needs a national facility" and "needs a
modest cluster".

**The catch, stated honestly**: Muon needs *fewer optimizer steps* to reach a
given loss, but **each step is more expensive** — Newton-Schulz is a cubic matrix
operation. The net win depends on the ratio of optimiser overhead to forward/
backward cost, which is favourable for large hidden dimensions and less so for
small ones. The quintic Newton-Schulz iteration runs in bfloat16 on GPU.

**Scope**: Muon applies to 2D matrix parameters. Embeddings, biases, norms and
scalar parameters stay on AdamW. For PHAROS the expert FFN matrices and attention
projections — the overwhelming majority of parameters — are Muon-eligible.

## 7.2 The DeepSeek-V3 efficiency stack

DeepSeek-V3 trained a 671B-parameter MoE on **14.8T tokens for 2.788M H800
GPU-hours** (~$5-6M). The techniques are individually adoptable:

| Technique | What it does | Applicability to PHAROS |
|---|---|---|
| **FP8 mixed precision** | all major GEMMs in FP8, with block/tile quantisation and selective high-precision accumulation. **<0.25% loss error vs BF16** | directly applicable; the single largest memory/throughput win |
| **DualPipe** | pipeline parallelism with near-total computation-communication overlap, few bubbles | applicable once the model spans nodes |
| **Auxiliary-loss-free load balancing** | per-expert bias adjusted each step instead of a competing loss term | **already in the PHAROS design** (§4.3) |
| **Multi-head Latent Attention (MLA)** | compresses the KV cache | partially redundant — PHAROS already uses linear attention for 20/32 blocks |
| **Multi-token prediction (MTP)** | predicts several tokens ahead; improves quality *and* enables speculative decoding | adaptable: predict masked *spans*, matching the span-masking objective |

**The honest read**: FP8 and aux-loss-free balancing transfer cleanly. MLA is
largely redundant against our hybrid attention. DualPipe only matters at
multi-node scale. MTP is interesting but needs reinterpretation for a masked
rather than causal objective.

## 7.3 Looped / recycled refinement

AlphaFold2's **recycling** feeds the pair representation, single representation
and predicted structure back into the network, by default **3 times**; studies
have swept 0-10 cycles and some pipelines iterate "until no further improvement
is detectable". It is a genuine accuracy mechanism, not just an inference trick.

**For PHAROS recycling is structural, not optional.** The electrostatic attention
bias `B_elec` requires a distance estimate `d_ij` to evaluate. It cannot be
computed on the first pass, so it is disabled at recycle 0 and enabled from
recycle 1 using the current predicted distance map. Each cycle sharpens the
geometry, which sharpens the physics term, which sharpens the geometry.

Two consequences:
1. **Weight sharing across cycles** means recycling buys accuracy at *zero*
   parameter cost — attractive given our data scarcity.
2. Gradients need only flow through the final cycle (AF2's approach), so the
   memory cost of recycling is one cycle, not N.

## 7.4 Physics that can guide training

### (a) Energy as a differentiable loss — the established route
**OpenMM-Loss** implements the potential energy of a predicted structure as a
PyTorch loss, feeding molecular-dynamics forces in as gradients for AlphaFold2.
Result: **comparable accuracy with lower potential energy and better MolProbity
scores**. This is the key precedent — physics terms improve *structural quality*
without costing accuracy.

### (b) RNA force fields
- **AMBER χOL3** (ff99bsc0χOL3) is the standard RNA parameter set; the revised
  dihedral parameters improved agreement with PDB conformer populations.
- **HB-CUFIX** (2025) compares against NMR and SAXS on single-stranded
  oligonucleotides and **outperforms χOL3 and ROC**, giving near-experimental
  accuracy for sequence-dependent structural preferences.
- **Turner nearest-neighbour** free energies remain the reference for secondary
  structure thermodynamics.

### (c) **Measured base-pair-step stiffness — our own contribution**
This is the term the literature does *not* already provide in usable form.
From `_ndb_struct_na_base_pair_step` across 156 structures we extracted
**103,965 annotated steps** and derived covariance-based stiffness matrices
`F = kT C^-1` for **76 dinucleotide contexts** with n >= 200.

Validation — the Watson-Crick means reproduce canonical A-form RNA:

| Context | n | rise (A) | twist (deg) | slide (A) |
|---|---|---|---|---|
| GG/CC | 7,558 | 3.14 | 29.98 | -1.69 |
| GC/GC | 5,902 | 3.11 | 33.41 | -1.51 |
| AA/UU | 1,811 | 3.01 | 30.59 | -1.48 |
| AU/AU | 1,506 | 2.81 | 33.94 | -1.58 |

(reference A-form RNA: rise ~2.8-3.1 A, twist ~32 deg, slide ~-1.5 A)

Findings:
- **Stiffness spans 115x** across contexts (twist force constants, UG/UG 1.343e-2
  to AA/UA 1.167e-4). Stiffest UG/UG (twist sd 11.5 deg);
  floppiest AA/UA (102 deg) — the floppy end is dominated by non-canonical
  contexts, exactly as expected.
- **GC content predicts rigidity**: Pearson(GC fraction, twist sd) = **-0.314**,
  roll sd -0.298, over 76 contexts. More hydrogen bonds and better stacking give
  a stiffer step.

### (c2) Do NOT hard-code this as a lookup table — learn it [measured]

The obvious implementation is a 76-entry table keyed on dinucleotide context.
**That is the same mistake the GNRA result already disproved**: sequence context
alone is a weak predictor of rigidity. We measured the headroom directly by
fitting Gaussians over the 6 step coordinates and comparing mean negative
log-likelihood per step (`measure_stiffness_headroom.py`, 103,965 steps):

| Model | conditioning | NLL (nats/step) | gain |
|---|---|---|---|
| M0 | none (one global Gaussian) | 19.724 | — |
| M1 | **sequence context** (what a lookup table achieves) | 17.579 | 2.144 |
| M2 | **sequence x structural context** | **14.551** | **+3.028 further** |
| — | structural context *alone*, ignoring sequence | 17.847 | 1.877 |

**Sequence and structural context are near-equal, complementary contributors.**
Held-out: sequence +1.7847, structure-only +1.7409, and combining them adds a
further +1.0336 beyond sequence. (An earlier version claimed structure buys
*more* than sequence, 3.028 vs 2.144; that was an in-sample artefact — see the
correction table below.)

| Model | in-sample (published) | **held-out, common subset** |
|---|---|---|
| M0 global | 19.7235 | **18.1607** |
| M1 sequence context | 17.5792 | **16.3760** (gain **1.7847**) |
| M_struct structure only | 17.8470 | **16.4198** (gain **1.7409**) |
| M2 sequence x structure | 14.5510 | **15.3424** (gain over sequence **1.0336**) |

> **Corrected in cycle 2 (REV-2).** The published figures fit every group
> in-sample and scored each model on its *own* covered subset (103,965 / 90,098
> / 78,076 steps). Finer partitioning lowers in-sample NLL mechanically, and
> M2's subset is the better-populated, more regular steps. Rescored on the
> common 78,076 steps with 2-fold held-out evaluation, structure-over-sequence
> falls from 3.0282 to **1.0336 nats (2.9x)** and **the order inverts**:
> sequence adds more than structure, not less.
>
> **What survives, and it is enough**: structural context still adds a real
> **+1.03 nats beyond sequence**. Sequence-alone (+1.78) and structure-alone
> (+1.74) are near-identical and complementary — 3.53 nats if they were
> independent against 2.82 actual. So an encoder reading **both** is correct,
> and a sequence-keyed dinucleotide lookup table still leaves ~1.03 nats
> unused. Only the comparative superlative is retracted.

The table also has a coverage hole: only 76 contexts reach n >= 200, covering
90,098 of 103,965 steps (**86.7%**). The remaining 13,866 steps — the unusual,
most structurally interesting ones — get no entry at all. An encoder generalises
to them; a table cannot.

### (c3) The resulting objective

Replace the table with a **stiffness encoder** that emits, per step, a mean
`mu_i` and a positive-definite precision `F_i` (parameterised through a Cholesky
factor so positive-definiteness is structural). Train it by the Gaussian negative
log-likelihood of the observed deformation:

```
L_stiff = sum_i [ 0.5 (x_i - mu_i)^T F_i (x_i - mu_i) - 0.5 log det F_i ]
```

The `log det F` term is essential: without it the trivial solution is `F -> 0`
(declare everything floppy, pay no penalty). With it, the model is rewarded for
*confident* predictions only where the data actually supports them.

At inference `F_i` is exactly the local harmonic force-constant field that
`E_rigid` needs, now conditioned on structure rather than looked up by sequence.

**Benchmarks the encoder must beat**: 17.579 nats/step (sequence table) and
14.551 nats/step (sequence x crude structure). Anything above the first is worse
than a lookup table and the design should be abandoned.

### (d) Other free supervision found in the same files
- **`_struct_conn` `metalc`**: **44,708 curated Mg²⁺ coordination records**
  (plus 7,009 K⁺, 1,367 Zn²⁺), with coordinating atoms ranked
  OP2 > OP1 > O6 > O4 > O2' > N7 — independently reproducing our
  distance-based finding that phosphate oxygens dominate.
- **Saenger 28-class / Leontis-Westhof 12-class** base-pair annotations:
  **29 Saenger classes present**, giving a ready-made non-canonical interaction
  vocabulary for the motif machinery.
- **`_pdbx_unobs_or_zero_occ_residues`**: **46,448 RNA records** of unmodelled
  residues — regions too disordered to resolve, i.e. a free maximal-flexibility
  label complementary to B-factors.

## Sources
- Muon is Scalable for LLM Training (Moonlight) https://arxiv.org/pdf/2502.16982
- Muon repo https://github.com/MoonshotAI/Moonlight
- Muon original note (Keller Jordan) https://kellerjordan.github.io/posts/muon/
- DeepSeek-V3 Technical Report https://arxiv.org/pdf/2412.19437
- DeepSeek-V2 (MLA) https://arxiv.org/pdf/2405.04434
- AlphaFold2 recycling / ReFOLD https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10290552/
- OpenMM-Loss: MD forces as deep learning gradients https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11393680/
- HB-CUFIX RNA force field (J Chem Phys 2025) https://pubs.aip.org/aip/jcp/article/162/20/200901/3347562/HB-CUFIX-Force-field-for-accurate-RNA-simulations
- Revised RNA dihedral parameters (χOL3) https://pubs.acs.org/doi/10.1021/acs.jctc.6b00870
- **Own measurement**: `data/samples/analysis/basepair_geometry.json`

---

## 7.6 Hierarchical Reasoning Models — adopt the loop, not the hierarchy

**HRM** (Sapient, 2025) is directly relevant because its headline is *extreme data
efficiency*: 27M parameters, ~1,000 training examples, no pretraining and no
chain-of-thought supervision, reaching 40.3% on ARC-AGI, 55.0% on Sudoku-Extreme
and 74.5% on Maze-Hard — where CoT methods score 0%. **That is our regime**: we
have 6,661 unique sequences with 3D structure, not millions.

Its stated mechanisms:
- Two interdependent recurrent modules on different timescales — a high-level
  module for slow abstract planning, a low-level one for rapid detailed computation.
- **Deep supervision**: multiple forward segments, loss and parameter update at
  the end of each, hidden states detached between segments.
- **One-step gradient approximation**: detaching hidden states before the next
  segment avoids BPTT's memory cost while stabilising training.
- **Adaptive Computation Time (ACT)**: a Q-head predicts halt/continue, so easy
  instances stop early and hard ones get more compute.

### The independent ablation matters more than the paper

The ARC Prize team ablated HRM, and the result is not what the paper emphasises:

| Component | Measured contribution |
|---|---|
| **Hierarchical H/L architecture** | **~5pp at most** — "a regular transformer comes within ~5pp without any hyperparameter optimization", and at one outer loop the two are equivalent |
| **Outer refinement loop** | **+13pp** from none to one cycle; gains double again from 1 to 8 loops. "The refinement outer loop is an essential driver" |
| Training *with* refinement | Models trained with 16 loops but run with **1** at inference still gained **>15pp** — the loop matters during *training*, not just inference |
| Cross-task transfer | small — 31% vs 41% without the extra training sets; HRM is "fundamentally a zero-pretraining test-time training approach" |
| Augmentation | 300 augmentations reach near-max, not the 1,000 claimed |

**Conclusion: the architectural novelty was oversold; the refinement loop and
task-specific training drove the gains.**

### What PHAROS should take, and what it should not

| HRM element | Verdict for PHAROS |
|---|---|
| Two-timescale H/L module split | **Do not adopt as a claimed win.** Independently measured at ~5pp, possibly noise. We already have a genuine hierarchy (coarse blocks -> fine pairs) justified by *measured* block occupancy, not by analogy to brain oscillations. |
| **Outer refinement loop** | **Adopt, and train with more cycles than inference uses.** This is the one component with a large measured effect, and PHAROS already requires recycling for `B_elec` — so the mechanism is free. |
| **Deep supervision at every segment** | **Adopt.** Our current spec computes loss only after the final recycle (AF2-style). HRM's evidence says supervising *each* segment is what pays. |
| **One-step gradient (detach between segments)** | **Adopt.** Gives effective depth at **constant** memory — it decouples the number of refinement cycles from activation memory, which is exactly the constraint that makes deep recycling expensive. |
| **ACT / Q-learning halting** | **Adopt (later).** RNA difficulty varies enormously — a clean helix needs one pass, a four-way junction with a pseudoknot needs many. Adaptive compute is better motivated here than on fixed-size puzzle grids. |

The data-efficiency evidence also supports the overall bet: HRM shows a small,
heavily-recurrent model can beat far larger ones when data is scarce and the task
is structured. RNA 3D prediction is precisely that.

## Sources (7.6)
- Hierarchical Reasoning Model https://arxiv.org/pdf/2506.21734
- HRM official release https://github.com/sapientinc/HRM
- **The Hidden Drivers of HRM's Performance on ARC-AGI** (independent ablation) https://arcprize.org/blog/hrm-analysis
