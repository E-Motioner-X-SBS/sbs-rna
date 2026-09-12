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
=> *Sparsity is the ground truth, not an approximation.*

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
 [B] Pair Proposal - cheap O(L w) pairing -> select K = 32L candidates
        |
 [C] Sparse Pair Track - sparse triangle updates on K edges               O(K)
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

## 5. [B]+[C] Adaptive Sparse Pair Track (ASPT)

The main efficiency contribution, justified by Fact 2.

**Step 1 — cheap pairing proposal.** From the trunk output, compute a low-rank
outer product plus a Turner nearest-neighbour prior to score candidate pairs in
O(L·w) for local pairs plus a global top-k scan. No L×L tensor is ever
materialised.

**Step 2 — select K = c·L candidates**, c = 32 by default.
Observed requirement is 4.40 contacts/nt (p95 5.42, max 5.50) **[measured]**, so
c=32 gives **~6x headroom** over the worst chain we measured. At L=2048 this is
65,536 edges = **1.56% of dense**, 16.8 MB per tensor instead of 1.07 GB.

**Step 3 — sparse triangle updates.** AF3's triangle attention enforces the
triangle inequality on the pair graph. We run the same updates but restricted to
the induced subgraph on K edges. Triangles are formed only among retained
candidates, which is where the structural signal lives anyway.

**Step 4 — re-selection each recycle.** Candidates are re-scored after every
recycle, so a contact missed initially can be recovered. This makes the top-K a
soft, revisable commitment rather than a hard early prune.

**Failure mode and mitigation**: if the true contact is never proposed, it can
never be predicted. Mitigation is (a) the 6x headroom, (b) re-selection each
recycle, (c) a recall-oriented auxiliary loss on the proposal stage specifically
penalising missed true contacts. **Proposal recall is an explicit metric to
report** — a sparse track that quietly drops 5% of true contacts would be a
silent accuracy ceiling, and must be measured, not assumed.

---

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
| 3 | **O(L) sparse pair track** justified by measured contact scaling | AF3/Rhoformer are dense O(L²) |
| 4 | **Frozen motif KV bank** as retrieval-based geometry prior | Motif Atlas is published but wired into no large model |
| 5 | **Mg²⁺ sites + B-factor rigidity as free auxiliary supervision** | Extracted from mmCIF; currently unused by structure predictors |
| 6 | **Coupled ion-rigidity expert**, justified by a measured 1.76 sigma gradient | Treated separately or not at all elsewhere |

## 12. Risks and open questions

1. **Proposal recall ceiling.** If ASPT's proposal stage misses true contacts,
   nothing downstream recovers them. Must be measured explicitly per length bin.
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
