# PHAROS — architecture

**This document describes the architecture as it stands.** It states what each
part is, why it is that way, and what it is measured to do. It is not a
changelog: wrong turns, superseded numbers and rejected designs live in
[`history_of_failed_attempts/`](../../history_of_failed_attempts/), which is
kept deliberately and in full.

PHAROS predicts RNA structure from sequence. It is a hybrid-attention
mixture-of-experts trunk with an explicit physics term, a hierarchical pair
track for contacts, and ten output heads. The default configuration,
**PHAROS-Small**, is **16 blocks at d=512** — **149M** total parameters,
**61M** active per token, **128 effective** layers through eight refinement
loops.

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

Fine-grained: 32 routed experts at d_expert 256, 2 always-on shared experts,
top-4 routing. The router is conditioned on **what kind of RNA this is** —
length bin, coevolution depth (Neff/L), in-complex flag, pooled chemistry — and
a Switch load-balance term keeps the load even, which is what stops a
type-conditioned router collapsing onto the 85.94% of residues that are
ribosomal.

Load balance is measured and holds to three decimals. Whether the router
*specialises* is a separate question, and balance cannot answer it: the mean
routing distribution is uniform by construction at balance, whether every token
picks a few experts or every token spreads across all of them. Per-token
routing entropy separates the two, and `probe_router_specialisation.py` tracks
it against the live checkpoint.

**It specialises, and it starts late.** Measured on one run's checkpoints, with
32 experts so uniform entropy is log 32 = 3.466:

| tokens | per-token entropy | % of uniform | mean top-1 | sharpest block |
|---|---|---|---|---|
| 12.0M | 3.4094 | 98.4% | 0.0490 | 3.172 |
| 55.4M | 3.4326 | 99.0% | 0.0466 | 3.178 |
| 82.1M | 3.4331 | 99.1% | 0.0486 | 3.290 |
| 188.6M | 3.4064 | 98.3% | 0.0587 | 3.260 |
| **573.3M** | **3.3546** | **96.8%** | **0.0742** | **2.986** |

Through the first 82M tokens the router is indistinguishable from uniform and
the MoE is a dense feed-forward with 32× the parameters. From roughly 100M it
begins to sharpen, and the movement is monotone across the last three points:
top-1 probability rises from 0.049 to **0.074**, against 0.031 for a coin flip
across 32 experts, and the sharpest of the 16 blocks falls from 3.29 to 2.99
nats.

It is still only 3.2% below uniform, so this is early specialisation rather
than strong specialisation, and the question the architecture has to answer is
whether it continues. What can be said is that the flat reading at 82M was a
measurement taken too early, not a property of the design.

### 5.4 Sizing

| model | d | blocks | loops | effective layers | total | active |
|---|---|---|---|---|---|---|
| **PHAROS-Small** (default) | **512** | **16** | 8 | **128 effective** | **149M** | **61M** |
| PHAROS-Mini | 384 | 12 | 12 | 144 | 67M | 30M |
| Base-v2 (scale-up path) | 768 | 32 | 3 | 96 | 1,401M | 269M |

Small is the default because the structure task is **data-limited, not
capacity-limited** — evidenced directly in §7.4, where a 14× larger block
scorer is no better on held-out data.

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

1. contact map · 2. distance distribution · 3. 3D coordinates · 4. secondary
structure · 5. per-residue reactivity · 6. Mg²⁺ site probability · 7. local
rigidity · 8. disorder · 9. base-pair geometry class · 10. motif-class
posterior.

Four of these are supervised **for free** from the structure files themselves —
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

**25B tokens, not 323B.** At 61M active, 323B is 5,295 tokens per parameter —
265× Chinchilla, the worst corner of the quantisation-degradation curve, and for
no information gain, because RNA sequence entropy is 2.0165 bits/nt and the
corpus is redundant rather than rich.

Cost is `6N + 2N(loops-1)` per token, not `6N × loops`: intermediate loops run
under `no_grad` and cost a forward, not a forward and a backward.

### 12.3 Quality weighting

3D examples cannot be added, so they are weighted. MolProbity clashscore bands
map to training weights 1.00 / 0.80 / 0.55 / 0.30; unscored entries take 0.80.

### 12.4 Splits

Family-disjoint on families small enough to hold out, with the four dominant
rRNA families forced to train and measured separately as described in §11.

### 12.5 Optimiser and schedule

AdamW, bf16 autocast, TF32 enabled. Every stage warms up then decays —
stage 1 and stages 2-3 on a cosine keyed to progress, stage 5 and the selector
on OneCycleLR. Gradient clipping at 1.0.

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

## 15. Provenance

Data sources, links and acquisition programs: [`data/catalog/SOURCES.md`](../../data/catalog/SOURCES.md).
Decisions and their rationale: [`DECISIONS.md`](DECISIONS.md).
What was tried and did not work: [`history_of_failed_attempts/`](../../history_of_failed_attempts/).
