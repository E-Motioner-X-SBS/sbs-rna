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
Normalised B-factor vs distance to nearest Mg²⁺, 24,623 nt in 31 X-ray structures:
-0.883 (0-4 A) -> -0.673 -> -0.415 -> +0.004 -> +0.756 -> +0.877 (>20 A).
A monotonic **1.76 sigma** gradient. Meanwhile 83% of inner-sphere Mg²⁺
coordination is to phosphate OP1/OP2, and Mg²⁺ outnumbers all other cations 9:1.
=> *Ions and rigidity must share one expert, and ionic condition must be a
model **input**, which no current predictor accepts.*

And one negative result that constrains the design: **GNRA k-mer context alone
predicts rigidity at 0.073 sigma — negligible.** **[measured]** Motif identity
requires the interaction graph, not sequence n-grams. Motif routing therefore
happens in the *pair* track after pairing is estimated, never from raw k-mers.

---

## 2. What PHAROS accepts as input

| Input | Shape | Required | Note |
|---|---|---|---|
| Sequence | L, vocab 5 (A,C,G,U,N) | yes | elDORS is DNA-alphabet; T->U at load |
| **Ionic condition** | ([Mg²⁺],[K⁺],[Na⁺],T,pH) | **yes, defaulted** | **novel — nothing else takes this** |
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
fraction `theta = 1 - 1/(z xi)` (~0.76 for monovalent RNA), giving renormalised
phosphate charge `q_eff = -(1-theta)`. Debye screening
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
probability, predicted rigidity, local density). This is a direct consequence of
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

### 5.4 What is still unproven

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

## 11. Novelty claims, stated precisely

| # | Claim | Prior art status |
|---|---|---|
| 1 | **Ionic condition as a model input**, reaching attention logits via a Manning-screened Coulomb bias | **No existing RNA structure predictor accepts ionic conditions.** |
| 2 | First **sparse-MoE** architecture for RNA structure | AIDO.Protein is MoE for protein; no RNA equivalent |
| 3 | **Hierarchical coarse-to-fine pair track** selecting blocks not pairs, justified by measured 1.34% block occupancy | AF3/Rhoformer are dense O(L²); flat top-K sparsification measured here to fail at ~20% recall on long chains |
| 4 | **Frozen motif KV bank** as retrieval-based geometry prior | Motif Atlas is published but wired into no large model |
| 5 | **Mg²⁺ sites + B-factor rigidity as free auxiliary supervision** | Extracted from mmCIF; currently unused by structure predictors |
| 6 | **Coupled ion-rigidity expert**, justified by a measured 1.76 sigma gradient | Treated separately or not at all elsewhere |

## 12. Risks and open questions

1. **Block-detection recall.** Superseded but not eliminated. The flat top-K
   design was measured at ~20% recall on long chains and replaced (§5.1-5.3).
   The hierarchical track has a 100% recall *ceiling* by construction, but
   whether a trained model hits it is unproven. Block-detection recall must be
   reported per length bin and per level.
2. **Ionic metadata sparsity.** mmCIF records crystallisation conditions
   inconsistently. Stage 5 may have far fewer usable ionic labels than hoped —
   needs an audit before committing to that stage. **Open.**
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
