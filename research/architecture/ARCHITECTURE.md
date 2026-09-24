# PHAROS — architecture

**This document describes the architecture as it stands.** It states what each
part is, why it is that way, and what it is measured to do. It is not a
changelog: wrong turns, superseded numbers and rejected designs live in
[`history_of_failed_attempts/`](../../history_of_failed_attempts/), which is
kept deliberately and in full.

PHAROS predicts RNA structure from sequence. It is a hybrid-attention
mixture-of-experts trunk with an explicit physics term, a hierarchical pair
track for contacts, and ten output heads whose third is a denoising diffusion
decoder. The trained configuration, **shared400**, is **18 blocks at d=768**
— **394M** total parameters, **302M active** per token, **144 effective**
layers through eight refinement loops, and **512 experts that share one
network**.

---

## 1. What the model consumes and produces

**In:** a nucleotide sequence; a 24-dimensional chemistry vector per residue,
all of it computable from sequence; and an ionic condition (Mg²⁺ and monovalent
concentration), which is a real input the physics responds to.

**Out:** ten heads (§9) — contact map, distance distribution, 3D coordinates,
secondary structure, per-residue reactivity, Mg²⁺ site probability, local
rigidity, disorder, base-pair geometry class, and a motif-class posterior.

The design constraint that shapes everything: **the 3D corpus is saturated.**
The whole PDB holds 29,038 RNA chains across 10,520 entries, and no amount of
engineering adds another. Every decision below is made under the assumption
that structural supervision is fixed and small, and that the sequence corpus —
1.37 billion sequences — is the only thing that scales.

## 2. Data foundation

| tier | scale | what it supervises |
|---|---|---|
| sequence | 1.37B sequences (elDORS 1.32B + RNAcentral 46M) | the representation, via MLM |
| families | Rfam 15.1, 4,227 families | the split, and coevolution depth |
| secondary structure | ~160k annotated structures | head 4 |
| chemical probing | Ribonanza + RMDB | head 5, and the ionic response |
| 3D | **16,604 chains, 13.2M residues, 63.3M contacts** | heads 1-3, 6-10 |

The pyramid is steep on purpose. Stage 1 sees 10⁵ times more sequence than the
structural stages see structure, and the curriculum (§12) exists so that the
representation is built where the data is and only then specialised where it is
not.

**The pretraining corpus must span all twenty elDORS chunks.** elDORS is sorted
by length: chunk 001 averages 1,252 nt at GC 0.462, chunk 020 averages 162 nt at
GC 0.538. A subset of chunks is a length band, not a sample. The built corpus is
25M sequences drawn 1.25M from each of the twenty, and shard order is shuffled
at read time so a pool draws from across the corpus rather than walking it
longest-to-shortest.

## 3. Tokenisation

**Two vocabularies, because the sources differ.** elDORS is pre-normalised to
ACGUN, so pretraining uses **5 symbols plus specials**; a wider input vocabulary
is dead embedding rows and dead softmax mass. Structures are not normalised —
1.005% of raw PDB residues fall outside ACGU — so the structural vocabulary is
13 symbols plus a separate modification id over 370 species, resolved through
the PDB Chemical Component Dictionary rather than a hand-written list.

`N` is two tokens, not one: *unknown base, ribose modelled* is a different
statement from *nothing modelled*, and conflating them loses the distinction the
disorder head needs.

## 4. The chemistry layer

A 24-dimensional vector per residue, and the substrate the physics reasons
about. Feeding a physics module nucleotide identity alone gives it nothing to
work with.

```
 0-4   one-hot A / C / G / U / MOD
 5     purine flag
 6-11  H-bond donors and acceptors, per edge (WC, Hoogsteen/CH, sugar)
12     pKa of the protonating ring nitrogen
13     shifted-pKa flag          <- STRUCTURAL: target, not input
14-15  C3'-endo / C2'-endo pucker propensity
16-17  relative polarisability, stacking-energy prior
18     phosphate charge state
19     2'-OH present
20-22  modification class: methylation / pseudouridylation / other
23     local GC fraction, +/-16 window
```

**Every dimension except 13 is computable from sequence**, which is what lets
stage 1 train the same chemistry projection the structural stages use. Dim 13 is
set from context, and the contexts that shift a pKa are structural — so it is a
supervision target and is never fed in. Structural observables generally
(Leontis-Westhof class, B-factor, Mg distance, reactivity) live in a physically
separate array, so feeding one in as a feature is not something a caller can do
by accident.

The whole vector is a table lookup indexed by symbol, plus one cumulative sum
for the GC window — `data/chemistry_torch.py` computes it on device, asserted
element-wise identical to the reference implementation.

## 4A. Coevolution

Two residues that base-pair are under joint selection: a mutation on one side
is compensated on the other, so their alignment columns covary. That covariance
is the single strongest structural signal available from sequence alone, and it
is what MSA-based predictors are built on.

**APC-corrected mutual information** over Rfam seed alignments, with Henikoff
sequence weighting so a redundant clade does not count 50 times. The average
product correction removes the column-wise background — highly variable columns
covary with everything — which is what separates a coupling from a conservation
artefact.

Alignments come from `Rfam.seed`, not `Rfam.full`. The full-region set is
deeper but is missing RF00005 (tRNA), RF00177, RF02541 and RF02543 (rRNA) —
between them **84.4% of the structural residues in this corpus**.

**Assigned per chain, not per entry.** The corpus originally keyed its Rfam
metadata by PDB id alone, so every chain in a deposition inherited one family
— a 76-nucleotide tRNA bound to a ribosome was labelled `SSU_rRNA_bacteria`
along with the 1,500-nucleotide subunit. Chain length settles it: chains we
called `SSU_rRNA_bacteria` had a median length of 122. Families now come from
RNA3DB's Infernal `cmscan`, per chain, which disagreed with the entry-level
assignment on **59.1%** of the 13,496 chains both cover.

**Cached per family, not per chain.** The corpus's chains draw on only a few
hundred distinct families; coevolution is
a property of the alignment, so computing it per chain would do the same work
forty times over. The cache stores the strongly-coupled pairs rather than the
dense matrix — for a 1,980-column rRNA that is 8k pairs against 3.9M
mostly-noise entries. Per chain, the alignment columns are mapped back through
the seed row that best matches its sequence, and a pair whose either end falls
in a gap is dropped.

### 4A.1 What it is worth, measured

Scored against the **deposited contact sets** of real chains, per family, with
a random-pair baseline at the same sequence-separation cutoff. The baseline is
part of every row because a bare precision figure means nothing on its own:
contact density falls with length, so 15% on a 120-mer and 15% on a 2,900-mer
are not comparable claims.

| family | Neff/L | coupling precision | random | enrichment |
|---|---|---|---|---|
| tRNA | 2.084 | **44.9%** | 8.9% | 5.0× |
| 5_8S_rRNA | 0.080 | 9.2% | 2.6% | 3.5× |
| 5S_rRNA | 0.053 | 15.6% | 6.3% | 2.5× |
| SSU_rRNA_bacteria | 0.015 | 18.6% | 0.6% | 30.9× |
| LSU_rRNA_bacteria | 0.002 | 52.6% | 0.5% | 104.9× |

Read the precision column, not the enrichment column: the rRNA ratios are large
mostly because the random baseline on a long chain is near zero.

The tRNA couplings reconstruct the cloverleaf outright. Of the top 24, seven sit
at i+j=115, five at 87, three at 42 and three at 136 — four antiparallel
helices, which is the acceptor stem, the anticodon stem, the D-arm and the
T-arm. The entire secondary structure, from covariation alone.

**Two limits that travel with these numbers.**

*Alignment depth.* Rfam **seed** alignments are shallow — SSU_rRNA_bacteria is
99 rows, LSU_rRNA_bacteria 102 — and after Henikoff weighting the effective
counts are 29.5 and 10.9. Only tRNA clears Neff/L ≥ 1. Eleven effective
sequences cannot support pairwise statistics over 5,241 columns, so LSU scoring
52.6% is a **surprise to be explained, not a result to lean on**. The likely
explanation is that Rfam seeds are themselves structure-curated, so the
covariation that survives at low depth is the curated pairing — which means it
should not be assumed to generalise to a family whose alignment was built
without structure. Deeper alignments, searched out of the 1.3-billion-sequence
corpus with the Rfam covariance models, are the way to remove this caveat
rather than argue around it.

*What the earlier version of this section claimed.* It reported 45.5% on tRNA
and nothing else, because `map_to_query` selected a seed row by walking two
ungapped strings position by position. One indel decorrelates everything after
it, so real 1,500-nucleotide SSU chains scored 0.36–0.46 against **their own
family** and were rejected — SSU and LSU, 3,450 chains and the two largest
families in the corpus, silently received no coevolution while the coverage
figure counted them. The function fails closed, so this looked like success.
Selection is now by k-mer overlap and the query is genuinely aligned to the
chosen row before its positions are carried into alignment columns.

### 4A.2 How it enters the model

Through the pair track, as one scalar per pair, via its **own zero-initialised
projection** rather than extra width on `pair_proj`. Widening `pair_proj` would
change its input shape and invalidate every checkpoint trained without
coevolution; zero-init means the feature contributes exactly nothing at first,
so enabling it cannot regress a model that already works — it has to earn its
way in.

A dense `(B, L, L)` coupling tensor is not an option: the longest chain in the
corpus is 4,450 residues, so one batch element alone would be 79 MB. The batch
carries a sorted flat key array and the model binary-searches it for the pairs
actually sampled.

**Absent reads as zero**, which is sound only because the projection is
zero-initialised and the score is non-negative: "no coupling measured" and "a
coupling of zero" enter the model identically, and neither can look like
evidence *against* a contact. The 12% of the corpus with no Rfam family takes
that path on every pair.

**And "absent reads as zero" is exactly why the reach has to be measured.**
Three separate layers on this path degrade to *adds nothing* without raising:
`_coevolution_for` wraps the whole lookup in a bare `except Exception: return
None`, the `searchsorted` fills its misses with zeros, and `coev_proj` starts
at zero. A missing cache, a renamed Rfam family or a typo inside that try block
therefore produce precisely the arithmetic of a working feature that happens to
contribute nothing, and the contact loss falls either way. Nothing printed a
number, so §4A's claim that the pair track reads couplings rested on code that
could not have said otherwise.

Measured on the built corpus (`data/samples/analysis/coevolution_reach.json`,
16 batches, 927,401 sampled pairs, train split): **3.461% of the pairs the
contact head scores carry a non-zero coupling**, with 88% of sampled chains
carrying an Rfam family and no batch at zero. The trainer now refuses to start
if that rate is zero, and logs `coev_frac` and `coev_norm` — the latter being
the projection's weight norm, so *whether the feature ever left zero* is
observable rather than assumed.

## 5. Token trunk

### 5.1 Hybrid attention, period 8

```
gdn  gdn  swa  gdn  gdn  swa  gdn  full
```

Tiled to 16 blocks: **10 Gated DeltaNet, 4 sliding-window, 2 full.** GDN is a
chunked delta rule carrying O(1) state per token, so most of the stack is linear
in length; sliding-window covers the local helical regime at w=128; full
attention, twice, is where the physics bias reaches the token track.

### 5.2 Depth from refinement, not stacking

Eight loops over 16 blocks give **128 effective layers** at 16 blocks' worth of
parameters. Loops 1..N-1 run under `no_grad` and only the last is
differentiated — the one-step gradient — so activation memory is that of a
single pass while the computation is that of eight. The recycled signal is the
previous iteration's distance estimate, injected additively through its own
norm, so it enters the trunk rather than merely re-scoring a finished
representation.

### 5.3 Mixture of experts

**512 experts that share one network, and a router that may fire 1 or all of
them.** In a conventional MoE each expert owns a full feed-forward network, so
an expert costs `3 · d · d_ff` and the expert count stays small — 48 experts at
d=640 is already 17.7M parameters per block, and most of it is idle on any
given token. Here every expert reads and writes through **one shared SwiGLU**
and differs only by a gain and a bias over its hidden units: `4·d_ff + d`, or
9.9k parameters against 5.3M. Experts stop being where the parameters live.

That changes what "active" means. The previous configuration activated 61M of
240M parameters; the rest sat in experts a given token never selected, paid for
in memory and never used in a forward pass. shared400 activates **302M of
394M**, because the parameters moved into the network every token runs through.
The expert count rose at the same time — 48 to 512 — so specialisation gets
*finer* while capacity gets *denser*.

**Routing is nucleus, not top-k.** Every expert whose probability clears
`1/E` fires, so the width is 1..E per token: a poly-U tract fires one expert, a
four-way junction fires many. The argmax is always included, so the width is
never zero. The threshold is relative to uniform rather than absolute, because
an absolute threshold is not scale-free — 0.02 fires all 32 experts and none of
256, making the initial width a property of E rather than of the router.
Measured stable at 0.39–0.41 of E across E ∈ {32, 64, 128}.

**Measured on the trained model, which it had never been.** The block reports
`mean_width`, `max_width` and two router entropies at every one of 18 blocks at
every step; the trunk discarded all four before the trainer could read them, so
the paragraph above was an argument about initialisation and a claim about
training with no measurement behind it, and "no router collapse" rested on the
balance term alone with no threshold written down anywhere. At **step 7,000
(992.8M tokens)**, on the reserved shards (`scripts/audit_router.py`):

| | measured | reference |
|---|---|---|
| mean routing width | **15.28** | 512 = fixed top-k, 1 = argmax |
| widest single token | **330** | `max_k` is 512 |
| router entropy | **7.255 bits** | 9.000 = uniform over 512 |
| balance, per block | **1.517** | 1.0 = uniform, 512 = collapsed |
| dead experts (<0.1 of a uniform share) | **0.0%** | |
| effective experts, `1/Σu²` | **288.5** | of 512 |

The width is genuinely variable — the claim holds. Note that the trainer logs
the balance term SUMMED over 18 blocks, so the logged 0.197 at
`balance_weight` 0.01 is **1.09 per block, 9% above uniform**, not the 20×
concentration the raw figure invites; that misreading is why the per-block
figure is the one quoted here.

**The experts are merged before the network runs, not after.** The obvious
implementation builds the (token, expert) pair list and pushes every pair
through the shared network, which costs `width ×` the activations and produces
a pair count that changes every step. Because an expert here is a *vector*, the
router's convex combination of the selected gains and biases can be taken
first, and the SwiGLU run once. A token routed to 512 experts then costs
exactly what a token routed to one costs, which is what makes "1 or all" a real
option rather than a nominal one. This is soft merging (SMEAR, Muqeeth et al.),
and it also gives the router an *exact* gradient where discrete top-k needs a
straight-through estimator and leaves unselected experts with no signal.

Measured on an A100 80GB at 32,768 tokens:

| layout | tok/s | peak memory | MFU |
|---|---|---|---|
| per-pair dispatch | 4,378 | 31.5 GiB | 3.4% |
| merged | 17,458 | 12.0 GiB | 13.5% |
| merged + `torch.compile` | **31,642** | **12.1 GiB** | **24.5%** |

The per-pair version was not merely slower: a profile put **70% of GPU time in
gathers, scatters and elementwise multiplies against 19% in the matmuls**, and
`torch.compile` could not help because the pair count is data-dependent. Merging
makes every shape a function of the token count alone.

A Switch load-balance term keeps the load even, which is what stops a
type-conditioned router collapsing onto the 85.94% of residues that are
ribosomal. The router is conditioned on **what kind of RNA this is** — length
bin, coevolution depth (Neff/L), in-complex flag, pooled chemistry.

Load balance is measured and holds to three decimals. Whether the router
*specialises* is a separate question, and balance cannot answer it: the mean
routing distribution is uniform by construction at balance, whether every token
picks a few experts or every token spreads across all of them. Per-token
routing entropy separates the two, and `probe_router_specialisation.py` tracks
it against the live checkpoint.

**It specialises, and it starts late.** Measured on the 32-expert predecessor,
where uniform entropy is log 32 = 3.466:

| tokens | per-token entropy | % of uniform | mean top-1 | sharpest block |
|---|---|---|---|---|
| 12.0M | 3.4094 | 98.4% | 0.0490 | 3.172 |
| 55.4M | 3.4326 | 99.0% | 0.0466 | 3.178 |
| 82.1M | 3.4331 | 99.1% | 0.0486 | 3.290 |
| 188.6M | 3.4064 | 98.3% | 0.0587 | 3.260 |
| **573.3M** | **3.3546** | **96.8%** | **0.0742** | **2.986** |

Through the first 82M tokens the router is indistinguishable from uniform and
the MoE is a dense feed-forward with 32× the parameters. From roughly 100M it
begins to sharpen, monotonically across the last three points: top-1 rises from
0.049 to **0.074** against 0.031 for a coin flip, and the sharpest block falls
from 3.29 to 2.99 nats. It is still only 3.2% below uniform, so this is early
rather than strong specialisation. What can be said is that the flat reading at
82M was a measurement taken too early, not a property of the design — and it is
the direct motivation for the finer 512-expert granularity above, which gives
the router more to distinguish between.

### 5.4 Sizing

| model | d | blocks | loops | effective layers | experts | total | active |
|---|---|---|---|---|---|---|---|
| **shared400** (trained) | **768** | **18** | 8 | **144** | **512 shared** | **394M** | **302M** |
| base400 (independent experts) | 640 | 18 | 8 | 144 | 48 | 405M | 126M |
| PHAROS-Small (predecessor) | 512 | 16 | 8 | 128 | 32 | 149M | 61M |
| PHAROS-Mini | 384 | 12 | 12 | 144 | 32 | 67M | 30M |

Counted from the built model, not derived on paper. The two 400M rows are the
same budget spent differently and the difference is the active column: sharing
converts dormant expert parameters into active ones, 126M → 302M.

**Wide beats deep at equal parameters on this hardware.** d_expert 2304 at 18
blocks counts the same as 2048 at 20, but runs larger matmuls, and at d=512 the
matmuls were too small to saturate an A100 at all — the predecessor measured
6.6% MFU with only ~18% of GPU time in tensor-core GEMMs.

The predecessor was small on the argument that the structure task is
**data-limited, not capacity-limited** — evidenced in §7.4, where a 14× larger
block scorer is no better on held-out data. That argument bounds the *structure*
stage, which trains on 16,604 chains. It does not bound stage 1, which has
billions of tokens of sequence, and the capacity added here is capacity for
sequence.

## 6. Physics — an explicit Hamiltonian

### 6.1 Manning counterion condensation

RNA is a line charge dense enough to condense counterions. The Manning
parameter is ξ = l_B/b; with b = 1.40 Å for A-form RNA, **ξ = 5.11** and the
condensed fraction is **θ = 0.804**. Not the 0.76 quoted for B-DNA — A-RNA is
more densely charged, and using the DNA value understates screening.

### 6.2 Debye screening

Effective charge and screening length follow from the ionic condition, giving a
screened-Coulomb term that enters full attention as an additive bias. The ionic
condition is therefore an input that changes *what attends to what*, not a label
appended to the output.

### 6.3 Mg²⁺ and rigidity

Mg²⁺ proximity and local rigidity are coupled at **1.523 σ**, measured on 1,535
X-ray structures. Mg outnumbers K in the archive **54:1**, and inner-sphere
coordination goes to the phosphate oxygens **77.9%** of the time — which is why
the electrostatic bias is built on phosphate geometry.

## 7. Hierarchical Pair Track

### 7.1 Why not flat top-K

Contacts are not spread uniformly. A flat top-K proposer at K = 32L recovers
**0.200** of true contacts on 500-1,200 nt chains where a random scorer gets
0.746 on the same set, because rank-by-score across all L² pairs is dominated by
the diagonal. Contacts must be selected as **blocks**, not pairs.

### 7.2 Three levels, coarse to fine

L1 scores 16-residue blocks, L2 refines 4-residue blocks inside the survivors,
L3 emits pairs. Cost is O(L) rather than O(L²): at L=4096 the track runs in
0.45 s at 0.96% of dense, and the dense baseline does not run at all past
L=1024 on a 14 GB machine.

### 7.3 The pair budget

`target_c = 24` pairs per nucleotide, closed on the raw archive: across 14,106
chains the maximum effective c is 23.30 and there are **zero** breaches, with an
overflow path for the tail.

### 7.4 What the selector achieves

Trained on the family-disjoint split and evaluated against three baselines at
identical budget, on 1,450 held-out chains:

| scorer | L1 recall | L2 recall | L2 precision |
|---|---|---|---|
| random | 0.1353 | 0.5435 | 0.2020 |
| separation prior | 0.2313 | 0.6333 | 0.2415 |
| learned, prior ablated | 0.2713 | 0.8326 | 0.3370 |
| **learned** | **0.2724** | **0.8345** | 0.3377 |

**Sequence carries real information about which blocks are occupied: +0.2012 at
L2 over the separation prior.** The ablation is what establishes it — the
prior-ablated model never receives the separation prior and scores within 0.0019
of the full model, so the gain is the sequence and not the prior being read back
out.

The gain is concentrated where it is needed. On chains of 1,500 nt and up, where
flat ranking collapses, L2 recall is **0.9972** against the prior's 0.2779.

End to end, composed as deployed, the cascade recovers **0.264** of true L2
blocks against an L1 ceiling of 0.270 — L2 loses 0.006 and L1 bounds everything
else. On chains ≥1,500 nt the cascade delivers **0.902**.

## 8. Motif bank

667 frozen BGSU motif classes, queried from the pair track. RNA reuses a small
vocabulary of tertiary motifs across unrelated families; a bank makes that
reuse available without relearning it per family.

## 9. Heads

1. contact map · **2. distance distribution (distogram)** · **3. backbone
coordinates, by denoising diffusion** · 4. secondary structure · 5. per-residue reactivity ·
6. Mg²⁺ site probability · 7. local rigidity · 8. disorder · 9. base-pair
geometry class · 10. motif-class posterior.

Head 2 is supervised over **40 bins, 2–40 Å plus overflow**, on the same
sampled pairs as head 1. A binary contact says two residues are within a
cutoff; a binned distance says how far apart, which is strictly more
information from the same coordinates — it is what AlphaFold trains its pair
track on. It was defined but had no target until the corpus carried
coordinates.

### 9.1 Head 3 is a diffusion decoder, not a coordinate regressor

A structure is defined only **up to a rigid motion**, so there is no single
correct coordinate to regress towards. A squared error against one arbitrary
deposited frame trains the model towards the mean of the orbit under rotation,
which is the centroid — not a structure. Regressing coordinates directly cannot
work, and this is the reason.

Head 3 is therefore an EDM-preconditioned denoiser
(`D(x;σ) = c_skip·x + c_out·F(c_in·x, c_noise)`) over three backbone atoms per
residue — phosphate, C4′, and the glycosidic nitrogen, which is the minimum that
fixes a frame. Training samples a noise scale, corrupts a randomly rotated copy
of the deposited backbone and asks the network to recover it; inference runs
Heun sampling down the Karras schedule. The σ-weighting is calibrated so every
noise scale contributes equally, which makes the weighted loss ≈1.0 at
initialisation by construction — an invariant the tests assert, because a
constant offset there is invisible in training and silently reweights head 3
against the other nine.

Augmentation is **SO(3), never O(3)**. A reflection preserves every pairwise
distance, so it passes any distance-based check, and it mirrors chirality —
a mirrored RNA has a left-handed helix and the wrong sugar pucker. Training on
enantiomers that cannot exist is a failure mode with nothing in the loss to
object to.

Missing atoms are masked, never imputed: every 5′ terminus lacks a phosphate,
and disordered regions lose atoms anywhere. Filling them with a plausible guess
would train the model to reproduce the guess and would flatter every metric
computed against it.

Four of the remaining heads are supervised **for free** from the structure files themselves —
Mg sites, X-ray B-factor z-scores, the number of deposited models, and
unobserved residues — extracted during dataset construction rather than
annotated separately.

## 10. Dynamics

RNA in solution is an ensemble. The model emits a per-step 6×6 stiffness field,
diagonalised into a harmonic ensemble by block-tridiagonal selected inversion —
O(L), not O(L³) — with the stiffness matrix parameterised through a Cholesky
factor so it is symmetric positive-definite by construction and coupling bounded
by a Frobenius-norm criterion so the ensemble cannot run away.

## 11. Generalisation limits, stated

Two properties of the archive bound what any model trained on it can claim:

- **97.15% of structural RNA residues are in complex** with protein. Isolated
  RNA is 2.83% of residues. Performance on isolated RNA and on RNA-in-complex
  must be reported separately and never averaged.
- **85.94% of residues are ribosomal.** A random split leaks homologues; the
  split must be family-disjoint.

The four largest rRNA families are 84.4% of all structural residues, and each
exceeds any sensible held-out quota, so they are forced into training and a
separate **entry-disjoint** `test_ribosomal` set measures them. It is never
averaged with the family-disjoint test set.

## 12. Training

### 12.1 Curriculum

| stage | task | data | script |
|---|---|---|---|
| 1 | MLM + span masking | elDORS, 25M sequences | `pretrain_mlm.py` |
| 2 | secondary structure | bpRNA-SPOT | `train_sequence_stages.py` |
| 3 | chemical probing | Ribonanza | same, co-trained |
| 4 | physics | closed form, no stage | — |
| 5 | 3D multi-task | 16,604 chains | `train_pharos.py` |
| R1 | block selector | same | `train_block_scorer.py` |

**The stages chain.** Each passes its checkpoint forward with `--init-from`;
stages that do not pass weights are unrelated runs, not a curriculum. The block
selector is deliberately standalone — it is a convolutional residue encoder, not
the trunk, so there is nothing to chain from.

Span masking is geometric with mean 3, not single tokens: a helix is locally
periodic, so a single masked base is fillable from its neighbours without
learning anything about structure.

### 12.2 Budget

**Tokens per active parameter, not tokens.** At the predecessor's 61M active,
323B tokens is 5,295 per parameter — 265× Chinchilla, the worst corner of the
quantisation-degradation curve, and for no information gain, because RNA
sequence entropy is 2.0165 bits/nt and the corpus is redundant rather than
rich. At shared400's **302M active** the same reasoning gives a very different
budget: 4B tokens is 13 per active parameter, under 1× Chinchilla, so the model
is now under-trained rather than over-trained and the budget is bounded by wall
clock rather than by diminishing returns.

Perplexity is reported alongside cross-entropy, and cross-entropy is reported
**separately from the MoE balance term** — the balance loss has a non-zero
floor (0.01 × n_blocks × 1.0) and folding it into the reported loss puts a
constant under every curve. The reference point is the corpus's own 2.0165
bits/nt: a model at that number has learned base frequencies and nothing else.

Cost is `6N + 2N(loops-1)` per token, not `6N × loops`: intermediate loops run
under `no_grad` and cost a forward, not a forward and a backward.

### 12.3 Quality weighting

3D examples cannot be added, so they are weighted. MolProbity clashscore bands
map to training weights 1.00 / 0.80 / 0.55 / 0.30; unscored entries take 0.80.

### 12.4 Splits

Family-disjoint on families small enough to hold out, with the four dominant
rRNA families forced to train and measured separately as described in §11.

### 12.5 Optimiser and schedule

**Muon on the 2-D weights, AdamW on everything else.** AdamW rescales each
gradient coordinate independently, which discards the fact that a weight
*matrix* has a spectrum: a gradient concentrated in a few directions moves
those far and the rest barely. Muon replaces the momentum buffer with the
nearest semi-orthogonal matrix — every singular value set to 1 — through a
quintic Newton–Schulz iteration, five matmuls rather than an SVD, so the update
moves every direction equally.

It applies to matrices only. Embeddings, LayerNorm gains, biases and the
**per-expert modulation vectors** stay on AdamW, because orthogonalising a
vector is meaningless. On shared400 that is Muon over 222 matrices (343M
parameters) at lr 0.02 and AdamW over the rest at 6e-4. The two rates are not
comparable: an orthogonal update has unit spectral norm by construction, so
Muon's natural scale is ~50× AdamW's, and comparing them at one rate compares
nothing.

Chosen by measurement, not preference. Matched A/B — same model, same 240
batches in the same order, same seed, held-out scored on genuinely masked
positions:

| optimiser | held-out bits | accuracy |
|---|---|---|
| AdamW, lr 6e-4 | 1.9681 | 0.3067 |
| AdamW, lr 1.5e-3 | 1.9783 | 0.3018 |
| Muon, lr 0.01 | 1.9609 | 0.3163 |
| **Muon, lr 0.02** | **1.9603** | 0.3152 |
| Muon, lr 0.04 | 1.9638 | 0.3114 |

Muon wins at all three of its rates against the better of AdamW's two, and the
result reproduces on a second model. The margin is small and 240 steps is
short, so this is evidence rather than proof that it holds at 4B tokens.

**Weight decay is split.** 0.01 on the matrices, **zero** on norms, biases and
the per-expert modulation. Decaying a LayerNorm gain has no scale-invariance
argument behind it, and decaying the zero-initialised expert modulation pulls
the MoE's specialisation back toward the shared network it exists to differ
from — the `mod` tensor whose measured std is 0.008 is precisely what that
decay was fighting.

bf16 autocast, TF32 enabled. Every stage warms up then decays — stage 1 and
stages 2-3 on a cosine keyed to progress, stage 5 and the selector on
OneCycleLR. Gradient clipping at 1.0.

## 13. Engineering properties

These are part of the architecture, because on a shared GPU a design that cannot
survive interruption is a design that does not finish.

**Every stage stops and resumes.** Model, optimiser, and for OneCycleLR stages
the scheduler state, checkpointed every 100-250 steps. A resume takes precedence
over `--init-from`, because the chain already happened. `--restart` renames
rather than overwrites. Checkpoints are format-tagged so a file predating
resumability is never mistaken for training state.

**Every stage survives an OOM.** The batch is dropped, the scheduler kept
aligned with the batch count, the budget reduced, and the run continues.

**The batch is sized from free VRAM**, from a measured cost per token, so the
same command fills an empty card and backs off on a shared one.

**Telemetry is streamed, not summarised at the end.** A CSV per stage, appended
and flushed per row, beside a JSON manifest carrying argv, git commit, full
config, parameter counts and the resume point. `kind` marks each row `step`,
`epoch`, `eval` or `event`, so an OOM or a resume appears alongside the metrics
around it.

**Throughput is a property of the data path.** Batches are length-bucketed
within a shuffled pool and quantised to a fixed set of widths, which both
removes padding waste and makes `torch.compile` viable — the delta-rule chunk
loop is a Python loop, so inductor specialises on the chunk count and an
unquantised length recompiles on nearly every batch.

## 14. What is measured, and what is not

Measured and reproduced by `scripts/sampling/verify_claims.py` (331 checks, 13
test suites): every number in this document.

Not yet established:

- whether the MoE router specialises, as opposed to merely balancing;
- the pair track's L1 ceiling, which bounds the cascade at 0.270 and is set by
  L1's minimum separation being expressed in 16-residue blocks where L2's is in
  4-residue ones — a retrain with the block diagonal admitted is owed;
- stage 5 and stages 2-3 end-to-end on a chained curriculum;
- Ribonanza beyond the 335,616 rows acquired.

## 14A. Does it understand RNA, or is the accuracy an artefact?

Masked-token accuracy inflates easily, and a pooled figure hides how. Four
mechanisms produce a respectable number with no understanding behind it, and
all four are measured rather than dismissed.

**BERT's corruption scores positions the model can see.** Of the selected
positions, 10% are left unchanged and 10% replaced with a random base; both are
scored by default. Split by what was actually visible:

| what the model saw | accuracy | share |
|---|---|---|
| `[MASK]` — must infer | **0.3229** | 80.5% |
| kept — answer visible | 0.9984 | 12.3% |
| random — wrong base shown | 0.0068 | 7.2% |
| pooled | 0.3829 | |

The honest number is **0.3229, not 0.3829**. Every held-out metric in this
project now scores hidden positions only.

**Against trivial strategies**, on hidden positions: always the commonest base
0.2774, copy the previous residue 0.2794, model **0.3229**. So +4.4 points over
the best trivial strategy, not +10.

**It leans on the input.** Where a *wrong* base is shown, accuracy is 0.0068 —
far below the 0.25 of guessing. It echoes the corrupted input rather than
overruling it.

**It is skewed.** Predicted share A 42.2%, U 31.4%, C 14.7%, G 11.7%; recall
A 0.477, U 0.385, C 0.218, G 0.173. G recall is *below chance*. Not a collapse,
but not four-way competence, and a pooled figure cannot show it.

### 14A.1 The one metric that cannot be faked

Watson–Crick complementarity. One side of a real base pair — taken from a
deposited structure, not proposed by a folding program — is masked under two
conditions that differ **only** in whether the partner is readable. Same bases,
same contexts, same model, same prior; the difference is pairing being used.

| checkpoint | tokens | bits | accuracy | **WC gap** |
|---|---|---|---|---|
| step 1750 | 226M | 1.9877 | 0.2910 | **+0.0229** |
| step 3000 | 395M | 1.9628 | 0.3158 | **+0.0857** |

The gap has almost quadrupled. Loss and accuracy improved too, by amounts a
composition shift could account for; the WC gap cannot be explained that way,
and it is the evidence that the model is learning that G pairs with C. It is
measured at every checkpoint.

## 14B. Evaluation on blind tests

The only honest test sets this project has are **RNA-Puzzles, CASP15 and
CASP16** — 42 targets. Everything else is drawn from the PDB, and a structure
held out by date is not a blind test when homologues leak; these targets were
predicted by the field *before* their structures were public, which is what
makes the published numbers a real baseline rather than a self-reported one.

RNA-Puzzles also ships every group's submissions, so the benchmark is a
**leaderboard rather than a score**: 885 competitor models across 17 targets.
"PHAROS scores 0.52 TM" means nothing until you know the best submission on
that target scored 0.61 and the median scored 0.34.

### 14B.1 The metrics, and why four of them

- **RMSD** after optimal superposition is the common currency and the one that
  lies most: dominated by the worst-placed residue and growing with length, so
  8 Å on a 30-mer is bad and 8 Å on a 700-mer is a result.
- **TM-score** normalises that away with a length-dependent `d0`, using the RNA
  formula `0.6·√(L−0.5) − 2.5` — the protein formula is calibrated on protein
  compactness and inflates RNA scores by roughly 0.1. The superposition is found
  by the Zhang–Skolnick fragment search, because the TM-optimal superposition is
  not the RMSD-optimal one.
- **lDDT** needs no superposition at all, so a model that gets every local
  contact right but hinges one domain scores badly on RMSD and well here. For
  RNA — modular, with hinging junctions — that distinction is the whole game.
- **INF**, the RNA-Puzzles metric, is the Matthews correlation over the
  base-pair set. A prediction can have a respectable TM-score with the wrong
  secondary structure; this is what catches it.

Plus a clash score, because a diffusion sampler with an undertrained denoiser
produces self-intersecting chains and neither TM-score nor lDDT objects.

The base-pair detector's geometric windows were **measured** from full-atom
references rather than assumed (WC pairs sit at glycosidic N–N 8.63 ± 0.72 Å,
C4′–C4′ 14.67 ± 0.98 Å), and its agreement with the full-atom criterion —
precision 0.768, recall 0.812 — is re-derived by the tests, so an INF gap under
about 0.1 is inside the detector's own error and must not be read as a
difference between models.

**The metrics are validated by reproducing published results**, not by internal
consistency: scored against the real RNA-Puzzles round-1 submissions they
recover the published ranking, Das first at 3.30 Å and Dokholyan last at
7.26 Å.

### 14B.2 The bar

Over 17 RNA-Puzzles targets and all 885 submissions:

| | mean TM | mean lDDT |
|---|---|---|
| best submission per target | **0.459** | **0.710** |
| median submission per target | 0.301 | 0.575 |

TM 0.45 is roughly where the field calls a fold correct. **The best submission
clears it on 7 of 17 targets** — which is the honest measure of how hard this
problem still is, and the number to beat.

## 15. Provenance

Data sources, links and acquisition programs: [`data/catalog/SOURCES.md`](../../data/catalog/SOURCES.md).
Decisions and their rationale: [`DECISIONS.md`](DECISIONS.md).
What was tried and did not work: [`history_of_failed_attempts/`](../../history_of_failed_attempts/).
