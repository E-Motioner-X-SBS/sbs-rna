# PHAROS — Architecture Specification v0.1

**P**hysics-**H**ybrid **A**ttention-**R**outed mixture-**O**f-experts for RNA **S**tructure

> Status: design draft, session 1. Not yet implemented or trained.
> Every quantitative claim is either measured (`data/samples/analysis/`) or cited
> in `research/prior-art/`. Numbers marked **[measured]** come from our own
> analysis of 180 BGSU non-redundant structures.

---

## 1. Design thesis

Three empirical facts drive every decision below.

**Fact 1 — Inductive bias beats scale in RNA, decisively.**
ERNIE-RNA (86M) beats RiNALMo (650M) and UNI-RNA (400M) on 2D structure, and
beats RiNALMo on cross-family generalisation **0.646 vs 0.355 F1**. Its only
advantage is a crude 3-valued base-pairing bias added to attention logits.
A 7.5x parameter deficit overturned by one hand-set prior.
=> *The return on better physics priors is far higher than the return on scale.*

**Fact 2 — RNA contact maps are O(L)-sparse, not O(L²).** **[measured]**
Across 117 chains, contacts per nucleotide *saturates* while density collapses:

| Chain length | n | contacts/nt | map density |
|---|---|---|---|
| 32-100 | 38 | 1.78 | 6.52% |
| 100-200 | 6 | 2.85 | 4.04% |
| 200-500 | 3 | 4.22 | 2.15% |
| 500-1500 | 9 | 4.53 | 0.97% |
| 1500+ | 61 | 4.85 | **0.353%** |

Median 4.40 contacts/nt, p95 5.42, max 5.50 — bounded by RNA's coordination
geometry, independent of length. A dense L×L pair track spends >99% of its
compute on empty space, and at L=2048 would need **309 GB** of activations.
=> *Sparsity is the ground truth, not an approximation.* But **how** you exploit
it matters: a flat top-K proposal was measured and fails on long chains (§5.1);
contacts must be selected as **blocks**, not pairs (§5.2).

**Fact 3 — Ion coordination and local rigidity are the same phenomenon.** **[measured]**
Normalised B-factor vs distance to nearest Mg²⁺, 24,623 nt in the **15** X-ray
structures that contain Mg²⁺ (of 63 X-ray in the sample; 31 survive a >=30-RNA-
residue guard, and 15 of those contain Mg²⁺):
-0.883 (0-4 A) -> -0.673 -> -0.415 -> +0.004 -> +0.756 -> +0.877 (>20 A).
A monotonic **1.76 sigma** gradient. Meanwhile 83% of inner-sphere Mg²⁺
coordination is to phosphate OP1/OP2, and Mg²⁺ outnumbers all other cations 9:1.

**The obvious confound was tested.** Mg²⁺ proximity and dense packing both mark
folded cores, so the gradient might merely restate that cores are ordered.
Controlling for local density reduces the association only from **+0.420** to
**+0.372** (partial correlation), so the signal is largely *independent* of
packing. A within-structure estimate — comparing nucleotides within 4 A against
those beyond 12 A *inside the same structure*, removing all between-structure
heterogeneity — gives **+1.514 sigma**, positive in **4/4** structures holding
both groups, bootstrap 95% CI [+1.256, +1.661]. The selection objection is also
closed: recomputing the gradient with the >=30-RNA-residue guard lowered to
20/15/10/5 and removed entirely moves the span only from **1.760 to 1.757 sigma**
(change 0.003), monotonic at every threshold — the guard is a variance control,
not a source of bias (`test_residue_guard_bias.py`). Caveats remain: B-factor
absorbs resolution and refinement choices, the closest bin holds 319 nucleotides,
and the within-structure estimate rests on 4 structures.
=> *Ions and rigidity must share one expert, and ionic condition must be a
model **input**, which no current predictor accepts.*

And one negative result that constrains the design: **GNRA k-mer context alone
predicts rigidity at 0.073 sigma — negligible.** **[measured]** Motif identity
requires the interaction graph, not sequence n-grams. Motif routing therefore
happens in the *pair* track after pairing is estimated, never from raw k-mers.

**Fact 4 — Coevolution is a strong signal, but only when the alignment is deep.** **[measured]**
APC-corrected, sequence-reweighted mutual information on 12 Rfam seed alignments
recovers curated consensus base pairs at mean precision@L/5 = **0.670** and
recall@L = **0.768**. tRNA (Neff/L = 3.5) reaches **1.000 / 1.000**. But the
signal is gated by depth:

| Alignment depth | n families | mean precision@L/5 |
|---|---|---|
| Neff/L >= 1 | **2** | 0.975 |
| Neff/L < 1 | 10 | 0.609 |

The ground truth is also **incomplete**: the parser handles only the nested WUSS
brackets and ignores the alphabetic pseudoknot brackets, which account for
**53 of 537 pairs (9.87%)**, so a correctly-ranked pseudoknot pair is scored as a
false positive. That bias pushes precision *down*, making 0.670 a lower bound.

**This split is weak evidence and must not be leaned on.** Exact permutation
p = 0.0455 on a *post-hoc* threshold; **Spearman(Neff/L, precision) = +0.224**
over n=12; the deep arm has n=2; and a *shallow* family (THF riboswitch,
Neff/L = 0.279) reaches precision 1.000. What the data does show robustly is that
**precision varies enormously between families (0.333 to 1.000)**.
=> *Coevolution strength is highly input-dependent, so a fixed-weight blend is
wrong. `Neff/L` is a defensible router feature on theoretical and literature
grounds (shallow RNA MSAs are known to degrade coupling analysis), but our own
measurement supports it only weakly.*

---

## 2. What PHAROS accepts as input

| Input | Shape | Required | Note |
|---|---|---|---|
| Sequence | L, vocab 5 (A,C,G,U,N) | yes | elDORS is DNA-alphabet; T->U at load |
| **Ionic condition** | ([Mg²⁺],[K⁺],[Na⁺],T,pH) | **yes, defaulted** | **novel — nothing else takes this.** Physics term is closed-form; the *learned* response needs titration data the PDB lacks (§7b) |
| MSA / Rfam alignment | M x L | optional | graceful degradation |
| Chemical probing | L (SHAPE/DMS) | optional | acquisition target, see §9 |

Ionic condition enters as a FiLM-style conditioning vector, like a diffusion
timestep embedding. **The same sequence at 0 mM and 10 mM Mg²⁺ must produce
different predictions.** This is the single clearest novelty claim.

---

## 3. Global layout

```
sequence + ionic condition
        |
 [A] Token Trunk  - 32 hybrid blocks, MoE FFN, physics-biased attention   O(L)
        |
 [B] Coarse block map - DENSE at b=16, only 0.4% of dense pair cost
        |
 [C] Hierarchical Pair Track - refine occupied blocks b=16 -> b=4 -> pairs
        |     effective c ~ 17 on long chains, 2.2% of dense total
        |         ^
        |         +-- [D] Motif KV Bank (frozen retrieval, 667 classes)
        |
 [E] Physics/Hamiltonian Module  - Manning implicit + Mg explicit
        |
 [F] Heads: contacts | distances | Mg sites | rigidity | reactivity | 2D
        |
 [G] Structure Decoder - frame diffusion -> 3D coordinates
        |
     recycle x3 (distance estimate feeds back into the electrostatic bias)
```

---

## 4. [A] Token trunk — hybrid attention + MoE

### 4.1 Block interleaving

32 blocks, repeating period-8 pattern (4 cycles):

```
[ GDN, GDN, SWA, GDN, GDN, SWA, GDN, FULL ]
```

- **GDN** = Gated DeltaNet (linear attention), O(L). 20/32 blocks.
- **SWA** = sliding-window softmax attention, window 128, O(L·w). 8/32 blocks.
- **FULL** = full attention with physics bias, O(L²). **4/32 blocks only.**

Justification from prior art: Jamba interleaves at 7:1 and reaches 256K context
with a 4 GB KV cache; Gated DeltaNet beats Mamba2/DeltaNet on in-context
retrieval and length extrapolation; Oryx varies the mixer *within* a sequence.

Justification from RNA: A-form helix is locally periodic and low-information —
linear attention is sufficient. Junctions, pseudoknots and kissing loops are
where global mixing is needed, and they are rare. Full attention is reserved for
the 4 blocks that feed the pair track.

**Cost at L=4096**: dense-32-block attention would be 32 x 16.8M = 537M pair
evaluations; PHAROS pays 4 x 16.8M = 67M, a **8x reduction** on the attention
term, before MoE savings.

### 4.2 Physics-biased attention

In SWA and FULL blocks:

```
A_ij = (Q_i K_j^T)/sqrt(d)  +  B_wc(i,j)  +  B_elec(i,j; c_ion)  +  B_motif(i,j)  +  A_ij^(l-1)
```

The `A^(l-1)` residual-attention term is ERNIE-RNA's accumulation, retained
because it demonstrably works.

**B_wc — learned pairing prior.** ERNIE-RNA fixes {CG:3, AU:2, GU:0.8}. We
replace it with a learned 5x5 pair-type embedding *modulated by stacking
context*, because real pairing energy is nearest-neighbour-dependent (Turner
model), not per-pair. Initialised at the ERNIE-RNA values so we start from a
known-good prior and improve from there.

**B_elec — screened electrostatics (the novel term).**
Bjerrum length `l_B = e²/(4 pi eps_0 eps_r k_B T)` ~= 7.1 A in water at 298 K.
Manning parameter `xi = l_B/b` with `b` the axial charge spacing. Condensed
fraction `theta = 1 - 1/(z xi)`, giving renormalised phosphate charge
`q_eff = -(1-theta)`. **`b` is the axial charge spacing** (charges projected onto
the helix axis), *not* the P-P contour distance: for A-form RNA, 2.8 A per base
pair carrying 2 charges gives **b = 1.40 A**, hence **xi = 5.11** and
**theta = 0.804** for monovalent counterions (`q_eff = -0.196`). The commonly
quoted 0.76 is the **B-DNA** figure (b = 1.70 A, xi = 4.21); A-form RNA condenses
more strongly. For Mg²⁺ (z=2), theta = 0.902. Debye screening
`kappa = sqrt(2 N_A e² I / (eps eps_0 k_B T))` from ionic strength `I`. Then

```
B_elec(i,j) = -lambda * q_eff² * l_B * exp(-kappa * d_ij) / d_ij
```

**Resolution of the chicken-and-egg problem**: `d_ij` is unknown on the first
pass. B_elec is therefore **disabled in recycle 0** and switched on from recycle
1 using the current predicted distance map. This is why recycling is structural
to the design rather than an optimisation — the physics term needs a geometry
estimate to evaluate, and each recycle sharpens it.

`lambda` is learned; `q_eff`, `kappa`, `l_B` are computed from `c_ion` and
temperature by closed form. **The ionic input reaches the attention logits
through this term** — that is the mechanism by which salt changes the prediction.

### 4.3 MoE feed-forward

Following DeepSeekMoE:

| Property | Value |
|---|---|
| Routed experts | 32, fine-grained, d_ff = 512 each |
| Shared (always-on) experts | 2 |
| Routing | top-4 of 32, + 2 shared = 6 active |
| Load balancing | **auxiliary-loss-free**, per-expert bias updated each step |
| MoE placement | every other block (16 of 32) |

**Why shared experts matter here specifically**: Watson-Crick helix formation is
the single most frequent pattern in all RNA. Putting it in an always-on shared
expert means the 32 routed experts never waste capacity relearning it and can
specialise on what is rare and hard — junctions, pseudoknots, ion pockets.

**Why auxiliary-loss-free balancing matters here specifically**: our 3D data is
pathologically imbalanced (5S rRNA alone = 5,976 of the BGSU clusters). An
auxiliary balance loss would actively fight the biology. A bias-based scheme
equalises utilisation without adding a competing gradient.

**Router input is structural, not lexical.** The router sees the token hidden
state *concatenated with* the current local structural estimate (pairing
probability, predicted rigidity, local density) **and the alignment depth
`Neff/L`**. The depth feature is what lets the router gate the coevolution
expert. Fact 4 shows coevolution precision varies from 0.333 to 1.000 across
families, so the model should trust it conditionally rather than blending it in
at fixed weight. **The specific `Neff/L` gate is motivated by the literature
rather than established by our n=12 measurement** (Spearman +0.224). This is also
why RhoFold+ pairs a language model *with* MSA features — the two have
complementary failure modes; PHAROS makes that complementarity routed and
explicit instead of concatenated. This is a direct consequence of
the GNRA negative result: **routing on sequence k-mers does not work.** It also
prevents router collapse onto rRNA, because structural regimes (helix /
junction / ion pocket / single-strand) are far more evenly distributed across the
corpus than families are.

### 4.4 Parameter budget (PHAROS-Base)

| Component | Total params | Active params |
|---|---|---|
| Embeddings (vocab 5) | ~0.01M | 0.01M |
| Attention (32 blocks, d=768) | 75.5M | 75.5M |
| MoE FFN (16 blocks x 34 experts x 1.18M) | 642M | 113M |
| Dense FFN (16 non-MoE blocks) | 113M | 113M |
| Sparse pair track + triangle | ~45M | 45M |
| Motif bank + heads + decoder | ~35M | 35M |
| **Total** | **~911M** | **~382M** |

Versus **NucleicBERT: 404M, all active.** PHAROS-Base carries 2.3x the capacity
at comparable active compute, and its attention term is ~8x cheaper at long
context.

---

## 5. [B]+[C] Hierarchical Pair Track (HPT)

> **Revised in session 1 after the flat design was measured and failed.**
> The original specification used a flat top-K proposal keeping K = 32L edges.
> That design is **retained only as a fallback for short chains**; see §5.1.

### 5.1 The flat top-K design was tested and does not scale [measured]

`scripts/sampling/validate_proposal_recall.py` scored every pair with a
sequence-only proposer (ERNIE-RNA pair prior + stacking context + exponential
sequence-distance decay) and measured recall of true 3D contacts in the top
K = 32L:

| Chain length | n | mean recall @ c=32 | K as % of dense |
|---|---|---|---|
| 32-100 | 29 | 0.980 | 59.4% |
| 100-200 | 6 | 0.642 | 25.5% |
| 200-500 | 3 | 0.307 | 9.0% |
| 500-1200 | 5 | **0.200** | 4.7% |

A **random** scorer reaches mean recall 0.746 at c=32 across the same set,
because on short chains 32L already covers most of the map. The flat proposer's
apparent 0.795 mean is therefore almost entirely an artefact of short chains.
**On long chains, where sparsity actually matters, it recovers one contact in
five.** That is a fatal, silent accuracy ceiling.

A banded fallback does not rescue it. Contacts are not local enough
(`analyze_contact_separation.py`, fraction of contacts within |i-j| <= band):

| Band | L 500-1500 | L 1500-3000 |
|---|---|---|
| 32 | 0.413 | 0.350 |
| 64 | 0.562 | 0.479 |
| 128 | 0.694 | 0.597 |
| 256 | 0.832 | 0.706 |
| 512 | 0.956 | **0.812** |

Median contact separation on long chains is 78 nt with a long tail; even a
+/-512 band misses 19%.

### 5.2 The fix: contacts are strongly *clustered*, so select blocks not pairs

Helical stems are contiguous anti-diagonal runs, so contacts occupy very few
*blocks* even though they are spread along the sequence. Measured block
occupancy (`analyze_block_sparsity.py`, 69 chains):

| Chain length | median L | b=4 occupancy | effective c at b=4 |
|---|---|---|---|
| 64-200 | 88 | 16.34% | 7.9 |
| 200-500 | 393 | 6.87% | 12.6 |
| 500-1500 | 1038 | 3.17% | 14.8 |
| 1500-3000 | 2861 | **1.34%** | **17.2** |

**Occupancy falls as chains get longer** — the mechanism gets *more* effective
at exactly the lengths where the flat design failed. Keeping every occupied
4x4 block costs an effective c of only ~17, *below* the flat budget of 32,
while capturing 100% of contacts by construction.

### 5.3 Three-level coarse-to-fine track

| Level | Operation | Cost (L=2861) |
|---|---|---|
| L1 | **dense** pair map at block size b=16 | 0.396% of dense |
| L2 | refine occupied b=16 blocks to b=4 | 0.460% of dense |
| L3 | refine occupied b=4 blocks to pairs | 1.346% of dense |
| | **total** | **2.202% of dense** |

Compare the flat design: 1.118% of dense at c=32, but ~20% recall. **For roughly
2x the cost we move from a hard 20% ceiling to a recall limited only by
block-level detection.**

Why this is easier to learn: identifying whether a 16x16 block contains *any*
contact aggregates 256 pair-decisions into one, so the signal-to-noise ratio at
the coarse level is far higher than for individual pair ranking. The model never
has to rank 4.1M individual pairs; it ranks 32k blocks, then 16 sub-blocks
inside each survivor.

The L1 map is genuinely dense and therefore has **no recall loss at all** at the
coarse level — at L=4096 and b=16 it is only 65k entries. Recall loss can only
enter at the *selection thresholds*, which are tunable and can be set
recall-first (keep any block above a low probability), trading a little compute
for coverage.

### 5.4 Reference implementation and measured cost [measured]

`research/architecture/reference/hierarchical_pair_track.py` implements the
track; `benchmark_pair_track.py` measures it against an AlphaFold3-style dense
baseline. Run on this machine (CPU only, 14 GB RAM, no CUDA), batch 1,
d_model 768, d_pair 128, b1=16, b2=4, budget `target_c = 20`:

| L | HPT time | HPT pairs | % of dense | effective c | Dense time | Dense peak RSS |
|---|---|---|---|---|---|---|
| 128 | 0.01 s | 750 | 9.23% | 5.9 | 0.07 s | 0.10 GB |
| 256 | 0.01 s | 3,506 | 10.74% | 13.7 | 0.28 s | 0.33 GB |
| 512 | 0.05 s | 10,054 | 7.69% | 19.6 | 1.22 s | 1.30 GB |
| 1024 | **0.10 s** | 20,102 | 3.84% | 19.6 | **8.26 s** | **5.15 GB** |
| 2048 | 0.33 s | 40,198 | 1.92% | 19.6 | *not runnable* | predicted 12.9 GB |
| 4096 | **0.45 s** | 80,390 | **0.96%** | 19.6 | *not runnable* | predicted 51.5 GB |

Two things are established here that arithmetic alone could not:

1. **The dense baseline genuinely does not run.** At L=2048 it needs a predicted
   12.9 GB of activations and at L=4096 some 51.5 GB, on a machine with 14 GB.
   The infeasibility claim in §1 (Fact 2) is not a rhetorical flourish.
2. **At the largest length both can run (L=1024), HPT is ~83x faster**
   (0.10 s vs 8.26 s) and its peak RSS is below measurement noise against the
   dense baseline's 5.15 GB.

The cost fraction falls monotonically with length (9.2% -> 0.96%) because the
pair budget is `K = c*L` while the dense map grows as `L²` — the mechanism gets
*cheaper relative to dense* exactly where it is needed, mirroring the falling
block occupancy measured in §5.2.

`target_c = 20` is set slightly above the measured requirement of 17.2 (which
retains every occupied 4x4 block on long chains) and far above the observed
4.4-4.9 contacts per nucleotide.

> **Two implementation defects found while building this**, both of a kind that
> would have silently degraded a trained model rather than crashing:
> (i) the L3 expansion originally paired block rows with block columns
> element-wise, materialising only each block's *diagonal* — 4 pairs per 4x4
> block instead of 16, silently delivering a quarter of the intended budget;
> (ii) the L2 selection budget was expressed as a fraction-of-everything and
> was clamped by masking to a quarter of its target. Both capped the effective
> `c` at 5.0 while appearing to work. Budgets are now stated in the same units
> as the measurement (`K = c*L` pairs) so a mismatch is visible immediately.

### 5.5 What is still unproven

The measurements establish that the *information* is there: contacts are
clustered enough that block selection is viable, and the cost is affordable.
They do **not** establish that a trained model will identify occupied blocks
accurately — that requires training. The relevant external evidence is
ERNIE-RNA's zero-shot Top-L/1 contact precision of 0.68 from attention maps
alone, which suggests learned representations rank contacts far better than the
sequence-only heuristic tested in §5.1.

**Block-detection recall must be reported per length bin and per level.** This
remains risk R1, but it is now a *measurable, bounded* risk attached to a
mechanism with a 100% ceiling, rather than an unbounded one attached to a
mechanism with a measured 20% ceiling.

## 6. [D] Motif KV bank — retrieval instead of memorisation

The user's "RNA is just a bunch of data / KV caching for fast retrieval"
intuition, made concrete.

- The RNA 3D Motif Atlas 4.12 compresses **4,992 loop instances into 667 motif
  classes** (413 internal + 254 hairpin) — a 7.5:1 ratio. RNA tertiary structure
  is far more repetitive than its sequence diversity suggests.
- Build a **frozen KV bank** of 667 entries:
  - **Key** = motif interaction-graph signature (non-canonical pair pattern).
  - **Value** = motif internal geometry (pairwise distance-matrix eigenfeatures
    + backbone torsion profile), computed once from Atlas instances.
- The pair track **cross-attends** into this bank. Output contributes `B_motif`
  back into the token trunk.

**Why this is the right answer to data scarcity.** With only 6,661 unique
sequences with 3D, a model cannot reliably *learn* motif geometry implicitly.
The bank hands it the geometry explicitly, so the model only has to learn
**where motifs occur**, which is a far easier problem than what they look like.

**Cost**: the bank is ~10³ entries, fixed, and shared across every sequence in a
batch. Cross-attention against it is negligible next to the pair track.

**Consistency with the negative result**: retrieval is keyed on the *interaction
graph* from the pair track, never on sequence k-mers.

---

## 7. [E] Physics module — the explicit Hamiltonian

A differentiable energy evaluated on the predicted geometry, following STEM's
hybrid implicit/explicit split:

```
E_total = E_stack + E_pair + E_elec^implicit + E_Mg^explicit + E_excl + E_rigid
```

| Term | Treatment | Source |
|---|---|---|
| `E_stack` | Turner nearest-neighbour stacking | thermodynamic tables |
| `E_pair` | canonical + non-canonical pairing | learned, initialised from Turner |
| `E_elec^implicit` | **Manning-renormalised Debye-Hückel over phosphates** | closed form from `c_ion` |
| `E_Mg^explicit` | predicted Mg²⁺ sites with **inner/outer-sphere classification** | learned head |
| `E_excl` | steric exclusion | hard-sphere |
| `E_rigid` | harmonic restraint weighted by predicted rigidity | elastic-network-style |

**Why implicit K⁺ / explicit Mg²⁺ and not both explicit**: STEM establishes that
mean-field theory captures monovalent KCl well but fails on Mg²⁺, where ion-ion
correlation and site-specific chelation dominate. Our own data agrees on which
matters: **Mg²⁺ 17,428 vs K⁺ 1,868 — a 9:1 ratio** **[measured]**.

**Why inner/outer classification is a first-class output**: STEM's central claim
is that folding dynamics are driven by *exchange between* inner-sphere
(chelated) and outer-sphere (water-mediated) coordination — the mode, not the
mere presence, is decisive. We measure both channels directly: mean 0.77
inner-sphere vs 7.00 outer-sphere RNA contacts per Mg²⁺ **[measured]**, and the
inner fraction is itself compactness-dependent (0.511 overall but 0.296 for
chains under 500 nt). So **the inner/outer ratio is a predictable readout of
tertiary compactness**, not just an input.

`E_rigid` closes the loop with Fact 3: predicted Mg²⁺ sites raise local
rigidity, which stiffens the harmonic restraints, which constrains the decoder.
That pathway is the architectural encoding of the measured 1.76 sigma gradient.

---

## 7b. Ionic-condition supervision: audited, and the claim rescoped [measured]

Risk R2 asked whether enough recorded ionic metadata exists to train the
ion-conditioning claim. It was audited on all 180 sampled structures
(`audit_ionic_metadata.py`, `audit_em_buffers.py`). The answer changed the claim.

### Coverage is about one third, and the two methods store it differently

| Method | n | where conditions live | usable ionic vector |
|---|---|---|---|
| Cryo-EM | 117 | `_em_buffer_component` — **structured** (concentration + units + formula) | **23.9%** |
| X-ray | 63 | `_exptl_crystal_grow.pdbx_details` — **free text** | 58.7% mention an ion |

Cryo-EM entries carry pH for **94%** of structures, but only 24.8% populate
`_em_buffer_component` at all. X-ray entries always carry crystallisation
details, but as unparsed prose ("0.1-0.2 M Arginine-HCl, 0.1M Tris-HCl pH 7.6,
2.6-3.0% PEG-20K..."), so extraction is noisy.

Combined, roughly **36% of structures yield any recoverable ionic condition** —
enough to train on, but a 3x reduction in usable 3D data for that stage.

### The deeper problem: the PDB is survivorship-biased on ionic conditions

Where Mg²⁺ concentration *is* recorded (n=21), the distribution is narrow:

| Ion | n | p10 | median | p90 |
|---|---|---|---|---|
| Mg²⁺ | 21 | 5.0 mM | **7.5 mM** | 15.0 mM |
| K⁺ | 13 | 30 mM | 100 mM | 150 mM |
| Na⁺ | 12 | 100 mM | 150 mM | 150 mM |

Mg²⁺ spans only 5-15 mM. This is not an accident of sampling: **nobody deposits
a structure of unfolded RNA.** Every entry in the PDB was solved under conditions
chosen *because* the RNA folds there. The dataset therefore contains almost no
contrast between folding and non-folding ionic regimes.

**Consequence: the [Mg²⁺] -> structure response cannot be learned from the PDB.**
A model trained only on deposited structures would see ionic condition as a
near-constant and correctly learn to ignore it.

### How the claim is rescoped

The ion-conditioning novelty splits into three parts with very different support:

| Component | Support | Status |
|---|---|---|
| `B_elec` Manning/Debye screening term | **closed-form physics**, no training data needed | **unaffected** — holds regardless of label coverage |
| Mg²⁺ site head + inner/outer classifier | 17,428 labelled ions from 180 structures alone | **well supported** |
| Learned *response* to varying ionic conditions | needs titration series; PDB has none | **NOT supported by current data** |

The first two stand. The third must be either dropped or fed by data the PDB
cannot provide:

- **RMDB Mg²⁺ titration series** — SHAPE/DMS probing across an Mg²⁺ ladder on
  the same sequence. This is exactly the missing contrast, and it is public.
- **SAXS Mg²⁺ titrations** — radius of gyration versus [Mg²⁺], the classic
  compaction measurement.

This raises the priority of chemical-probing acquisition (§9) from "most
valuable missing asset" to **prerequisite for the headline novelty claim**.
Ribonanza alone does not supply it; the titration series in RMDB do.

**Training stage 5 as originally specified (ion-conditioned refinement on
structures stratified by ionic condition) is not viable on PDB data alone** and
is rewritten accordingly: it becomes conditioning on *probing titrations*, with
deposited structures supplying only site-level supervision.

## 7c. Stiffness Encoder — learned, not tabulated [measured]

The physics module's `E_rigid` term needs local force constants. Two ways to
supply them, and the measurement settles it.

### The data
`_ndb_struct_na_base_pair_step` gives the 6 deformation coordinates (shift,
slide, rise, tilt, roll, twist) for every annotated step. We extracted
**103,964 steps across 155 structures** and derived covariance-based stiffness
matrices `F = kT C^-1` for 76 dinucleotide contexts (n >= 200).

Validation: Watson-Crick means reproduce canonical A-form RNA (GG/CC rise 3.14 A
twist 29.98; AU/AU rise 2.81 twist 33.94; reference ~2.8-3.1 A, ~32 deg).
**Stiffness spans 134x** between the stiffest (UG/UG, twist sd 11.5 deg) and
floppiest (AA/UA, 102 deg) contexts, and GC content predicts rigidity
(Pearson -0.314 with twist sd).

### Why a lookup table is the wrong design

| Model | conditioning | NLL (nats/step) |
|---|---|---|
| M0 | none | 19.724 |
| M1 | sequence context = **a lookup table** | 17.579 |
| M2 | sequence x structural context | **14.551** |
| — | structural context alone | 17.847 |

**Structural context contributes more than sequence context** (+3.028 vs +2.144
nats), using only a 2-bit structural descriptor. A dinucleotide table captures
under half the available signal, and covers only 86.7% of steps — the 13,866
uncovered steps being the structurally unusual ones that matter most.

This is the same lesson as the GNRA negative result (§1, Fact 3-adjacent):
**sequence context alone does not determine local structural behaviour.**

### The design

A **stiffness encoder** head consumes the trunk representation and the pair
representation at each step and emits:
- `mu_i` in R^6 — the expected deformation
- `F_i`, a 6x6 **positive-definite precision matrix**, parameterised via a
  Cholesky factor `L_i` with `F_i = L_i L_i^T` so positive-definiteness is
  structural rather than penalised.

Trained by Gaussian negative log-likelihood of the observed deformation:

```
L_stiff = sum_i [ 0.5 (x_i - mu_i)^T F_i (x_i - mu_i) - 0.5 log det F_i ]
```

The `log det F` term is load-bearing: without it the degenerate optimum is
`F -> 0` — declare everything floppy and pay nothing. With it, confidence must
be earned.

**Acceptance criterion**: the encoder must reach below **14.551 nats/step** on
held-out structures. Above **17.579** it is worse than a lookup table and the
component should be cut.

`F_i` then *is* the local force-constant field for `E_rigid`, conditioned on
structure rather than retrieved by sequence — and it composes with the measured
Mg²⁺/rigidity coupling (§1, Fact 3), since ion proximity is part of the
structural context the encoder sees.

## 8. [F]+[G] Heads and decoder

| Head | Output | Supervision | Labels available |
|---|---|---|---|
| Secondary structure | L x L pairing | bpRNA, ArchiveII, RNAStrAlign | ~160k |
| Contact map | K sparse contacts | RNA3DB, gRNAde | 27,452 chains |
| Distance map | binned distances | same | same |
| **Mg²⁺ sites** | density + inner/outer | **extracted from mmCIF ourselves** | **17,428 from 180 structures alone** |
| **Rigidity** | per-nt z_B / RMSF | B-factors from mmCIF | every X-ray structure |
| Reactivity | per-nt SHAPE/DMS | **not yet acquired** (§9) | — |
| 3D coordinates | frames -> all-atom | RNA3DB | 6,661 unique seqs |

**Decoder**: frame-based diffusion. AF3 moved to raw-atom diffusion and still
only reaches TM 0.51 on RNA; trRosettaRNA's restraint-then-fold approach leads at
0.548. We therefore keep an explicit geometric/energetic stage rather than going
fully generative — consistent with the CASP16 signal that physics+ML hybrids are
currently ahead on RNA.

**Two supervision channels cost us nothing and are currently unused by anyone**:
Mg²⁺ sites and B-factor rigidity are already sitting inside the mmCIF files
every RNA structure model already downloads. We demonstrated the extraction end
to end in this session.

---

## 9. Data mapping and the one gap

| Stage | Data | In catalog? |
|---|---|---|
| A. MLM pretraining | elDORS 1.32B + RNAcentral 46M | yes |
| B. 2D structure | bpRNA/ArchiveII/RNAStrAlign ~160k | yes |
| C. Coevolution | Rfam 15.1, 4,227 families | yes |
| D. 3D + ions + rigidity | RNA3DB + gRNAde, 27,452 chains | yes |
| E. Motif bank | BGSU Motif Atlas 4.12, 6,326 records | yes |
| **F. Chemical probing** | **Ribonanza 2.1M / RMDB** | **NO — must acquire** |
| **F2. Mg²⁺ titration series** | **RMDB SHAPE/DMS across an Mg²⁺ ladder** | **NO — prerequisite for the ion-conditioning claim (§7b)** |
| G. Blind eval | CASP15/16, RNA-Puzzles | yes |

**The gap is chemical probing, and it is the most valuable missing asset.**
Ribonanza provides **2.1M sequences with DMS+SHAPE** — ~315x more supervised
examples than the 6,661 unique 3D sequences. It is the only label source that is
simultaneously (a) abundant, (b) structure-informative, (c) already public.
`plans/13-sequence-structure-gap-strategy.md` already flagged RMDB/SHAPE as the
next acquisition target; this design makes that concrete and high-priority.

**Corpus weighting caveat**: elDORS chunks are source-partitioned, not
homogeneous — 9 of 20 chunks have median length *exactly* 151 nt (unassembled
reads). Naive uniform sampling would let read fragments dominate pretraining.
Stage A must use per-chunk weighting; chunks 6 and 9 (median 2,193-2,389 nt) are
the only real source of long-context training signal.

---

## 10. Training curriculum

| Stage | Objective | Data | Frozen |
|---|---|---|---|
| 1 | MLM, span masking | elDORS weighted + RNAcentral | — |
| 2 | + 2D structure | bpRNA/ArchiveII/RNAStrAlign | — |
| 3 | + motif bank init | Motif Atlas (bank computed, then frozen) | bank |
| 4 | + contacts, Mg sites, rigidity (multi-task) | RNA3DB + gRNAde | bank |
| 5 | + ion-conditioned refinement | structures stratified by ionic condition | bank |
| 6 | 3D decoder | RNA3DB documented split | trunk partly |

**Evaluation must be on the documented RNA3DB split plus CASP15/16 and
RNA-Puzzles held out entirely.** Given 5S rRNA is 5,976 of the clusters,
sequence-identity-based dedup is mandatory or the numbers will be meaningless.

---

## 10b. Training: making it affordable without compromising the model

NucleicBERT used **192 A100s** densely. That is the bar to get under, and the
efficiency literature offers enough to do it.

### 10b.1 Optimiser — Muon

**Muon** orthogonalises gradient momentum via Newton-Schulz iterations before
applying it. Reported: **~2x compute efficiency vs AdamW**, matching quality at
**~52% of training FLOPs**, with ~33% memory saving. Critically for us it is
already validated on an **MoE** model at scale (Moonlight, 3B/16B, 5.7T tokens)
and is in production use (Kimi K2).

- Applies to 2D matrix parameters — for PHAROS that is the expert FFNs and
  attention projections, i.e. the overwhelming majority of parameters.
- Embeddings, norms, biases and scalars stay on AdamW.
- **Honest caveat**: Muon needs *fewer steps* but each step costs more
  (Newton-Schulz is cubic). The net win favours large hidden dimensions; it must
  be benchmarked on our actual shapes before being assumed.

### 10b.2 The DeepSeek-V3 stack, filtered for what actually transfers

| Technique | Verdict for PHAROS |
|---|---|
| **FP8 mixed precision** | **Adopt.** <0.25% loss error vs BF16 at 671B scale; block/tile quantisation with selective high-precision accumulation. Largest single memory/throughput win. |
| **Auxiliary-loss-free load balancing** | **Already in the design** (§4.3). Essential here because 5S rRNA is 5,976 of the 3D clusters, so a balance *loss* would fight the biology. |
| **DualPipe** | **Defer.** Only pays once the model spans nodes. |
| **Multi-head Latent Attention** | **Skip.** Largely redundant — 20 of our 32 blocks are already linear attention, so the KV cache is small by construction. |
| **Multi-token prediction** | **Adapt, don't copy.** MTP is defined for causal LMs; our objective is span-masked. The analogue is predicting a whole masked *span* jointly rather than independently per position. Worth an ablation, not a commitment. |

### 10b.3 Looped refinement is structural, not an add-on

AlphaFold2 recycles 3x by default (swept 0-10 in the literature; some pipelines
iterate to convergence). For PHAROS recycling is **required by the physics**:
`B_elec` needs a distance estimate `d_ij`, which does not exist on the first
pass. It is disabled at recycle 0 and enabled from recycle 1 onward.

Two consequences worth stating:
1. **Weight sharing across cycles buys accuracy at zero parameter cost** —
   valuable given 6,661 unique 3D sequences.
2. Gradients need flow only through the final cycle (AF2's approach), so
   recycling costs one cycle of activation memory, not N.

### 10b.4 The full physics-guided objective

```
L = L_MLM + L_2D + L_contact + L_dist
  + lambda_1 * L_stiff        (Gaussian NLL, learned precision F_i -- §7c)
  + lambda_2 * L_Mg           (site + inner/outer class; 44,708 curated labels)
  + lambda_3 * L_elec         (Manning-screened Coulomb consistency)
  + lambda_4 * L_violation    (steric clash, bond geometry)
  + lambda_5 * L_flex         (B-factor / RMSF, and unmodelled-residue labels)
```

Precedent that this helps rather than hurts: **OpenMM-Loss** implements MD
potential energy as a differentiable loss for AlphaFold2 and reports *comparable
accuracy with lower potential energy and better MolProbity scores*. Physics terms
improve structural quality without costing accuracy.

**Where each label comes from — all of it already in the mmCIF files:**

| Term | Source | Volume (180 structures) |
|---|---|---|
| `L_stiff` | `_ndb_struct_na_base_pair_step` | **103,964 annotated steps** |
| `L_Mg` | `_struct_conn` `metalc` | **44,708 Mg²⁺ coordination records** |
| `L_flex` | B-factors + `_pdbx_unobs_or_zero_occ_residues` | **143,871 unobserved-residue records** |
| motif vocabulary | Saenger `hbond_type_28` / LW `hbond_type_12` | **29 Saenger classes** |
| `L_elec` | closed form from `c_ion` | no labels needed |

None of this required new data acquisition. It was sitting unused in files the
field already downloads.

### 10b.5 Force-field grounding

For `L_violation` and any MD-based refinement, the current best RNA parameter
sets are **AMBER χOL3** (standard; revised dihedrals improved agreement with PDB
conformer populations) and **HB-CUFIX** (2025), which outperforms χOL3 and ROC
against NMR and SAXS on single-stranded oligonucleotides. Turner
nearest-neighbour free energies remain the reference for 2D thermodynamics and
already initialise `B_wc` (§4.2).

## 11. Novelty claims, stated precisely

| # | Claim | Prior art status |
|---|---|---|
| 1 | **Ionic condition as a model input**, reaching attention logits via a Manning-screened Coulomb bias | **No existing RNA structure predictor accepts ionic conditions.** Physics term needs no labels; the learned response requires titration data (§7b) |
| 2 | First **sparse-MoE** architecture for RNA structure | AIDO.Protein is MoE for protein; no RNA equivalent |
| 3 | **Hierarchical coarse-to-fine pair track** selecting blocks not pairs, justified by measured 1.34% block occupancy | AF3/Rhoformer are dense O(L²); flat top-K sparsification measured here to fail at ~20% recall on long chains |
| 4 | **Frozen motif KV bank** as retrieval-based geometry prior | Motif Atlas is published but wired into no large model |
| 5 | **Mg²⁺ sites + B-factor rigidity as free auxiliary supervision** | Extracted from mmCIF; currently unused by structure predictors |
| 6 | **Coupled ion-rigidity expert**, justified by a measured 1.76 sigma gradient | Treated separately or not at all elsewhere |
| 8 | **Learned stiffness encoder** emitting a per-step 6x6 precision matrix trained by Gaussian NLL, replacing a tabulated force field | Nucleic-acid elasticity uses fixed per-context stiffness tables; measured here to capture <half the signal (14.551 vs 17.579 nats/step) |
| 9 | **Physics labels mined from unused mmCIF categories** — 103,964 step geometries, 44,708 curated Mg²⁺ coordinations, 143,871 disorder records | These categories ship with every RNA structure and are used by no structure predictor |
| 7 | **Depth-gated coevolution routing** (`Neff/L` as a router feature) | RhoFold+ concatenates LM and MSA features at fixed weight; none route on measured depth. *Motivated by the literature and by measured between-family variance (0.333-1.000); our own depth split is weak evidence (Spearman +0.224, n=12)* |

## 12. Risks and open questions

1. **Block-detection recall.** Superseded but not eliminated. The flat top-K
   design was measured at ~20% recall on long chains and replaced (§5.1-5.3).
   The hierarchical track has a 100% recall *ceiling* by construction, but
   whether a trained model hits it is unproven. Block-detection recall must be
   reported per length bin and per level.
2. **Ionic metadata sparsity — AUDITED, partially confirmed (§7b).** ~36% of
   structures yield a recoverable ionic condition (cryo-EM 23.9% structured,
   X-ray 58.7% free-text). Worse, recorded Mg²⁺ spans only 5-15 mM because the
   PDB is survivorship-biased: nobody deposits unfolded RNA. **The learned
   ionic response cannot come from the PDB**; it requires RMDB Mg²⁺ titration
   series. The closed-form physics term and the Mg²⁺ site head are unaffected.
3. **Crystallographic ions are not the solution ensemble.** Resolved Mg²⁺ are
   site-bound only; the diffuse Manning atmosphere is invisible to
   crystallography. The explicit head learns site binding, *not* the atmosphere —
   these must not be conflated.
4. **B-factor is a noisy rigidity proxy**, absorbing resolution and refinement
   choices. Cryo-EM ADPs were excluded here (31 of 137 usable), which shrinks the
   rigidity label set considerably.
5. **Mg²⁺/rigidity correlation is association, not isolated causation** — both
   track folded cores. Justifies use as a prior, not a mechanistic claim.
6. **MoE at this scale needs real infrastructure.** NucleicBERT used 192 A100s
   dense; MoE improves FLOP efficiency but adds routing/communication complexity.
7. **Evaluation honesty.** TM-score ~0.5 is the field ceiling. Any claim of
   improvement must be on blind sets (CASP16, RNA-Puzzles), never on
   rRNA-saturated random splits.
