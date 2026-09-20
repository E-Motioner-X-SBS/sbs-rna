# PHAROS v0.2 — Complete Architecture Specification

**P**hysics-**H**ybrid **A**ttention-**R**outed mixture-**O**f-experts for RNA **S**tructure

> **Status.** Design complete, not trained. Supersedes `ARCHITECTURE.md` (v0.1)
> wherever the two differ. v0.1 is retained: its audit trail — 28 numbered
> defects across 11 cycles — is the reason this version can state what is
> measured and what is not.
>
> **What makes v0.2 different.** v0.1 was designed against **180** structures
> fetched onto a laptop. v0.2 is validated against the corpus: **29,807 chains**,
> **7,943 raw PDB entries** (13.7 GB, acquired for this revision), **10,424
> pdb_hunter entries** with secondary structure and Rfam mapping, and **238M nt**
> of sequence. Every number below is tagged **[v0.2]** if it was re-derived at
> that scale, **[v0.1]** if it still rests on the 180-structure sample, or
> **[lit]** if it comes from the literature.

---

## 0. The validation that produced this revision

Ten v0.1 claims were re-derived on the full corpus. The result has a shape, and
the shape is the most useful thing this document can tell a reader.

| | v0.1 (n=180) | v0.2 (full corpus) | |
|---|---|---|---|
| **Central estimates** | | | |
| mean effective c, long chains | 17.2 | **17.14** | holds |
| sequence entropy | 2.0167 bits/nt | **2.0165** | holds (238M nt) |
| b=4 block occupancy | 1.34% | 1.67% | holds |
| contact separation, band 512 | 0.812 | 0.8545 | holds |
| modified-residue fraction | 1.05% | **1.005%** | holds (11.6M residues) |
| Mg–rigidity gradient | 1.76 σ | **1.523 σ**, monotonic | holds, −13% |
| **Extreme quantiles** | | | |
| max effective c | 19.04 | **21.14** | **breach** |
| longest chain | 3,764 nt | **4,450 nt** | **breach** |
| chains over 4,096 nt | 0 | **8** | **breach** |
| max contacts/nt | 5.50 | **7.66** | **breach** |
| Mg : K ratio | 9 : 1 | **54 : 1** | wrong 6× |

**Every mean held. Every maximum failed.** This is not eleven separate errors;
it is one property of an n=180 sample, which estimates a mean well and cannot
estimate a p99.9 at all. Two load-bearing parameters (`target_c`, the context
window) had been set from maxima.

**Design rule adopted for v0.2: no parameter may be set from an extreme
quantile measured on fewer than 10³ structures.** Where the corpus cannot
supply that, the parameter is set from a mean plus an explicit safety factor,
and the factor is stated.

---

## 1. Data foundation — and the fact that reframes everything

### 1.1 The 3D corpus is saturated **[v0.2]**

An RCSB search for every entry containing an RNA polymer entity returns
**10,399** entries. `pdb_hunter/RNA_Database` holds **10,424**. The overlap is
essentially total; **98** entries are missing and were acquired for this
revision.

**There is no more experimental RNA 3D structure to obtain.** Not from better
tooling, not from more bandwidth. 10.4k entries, collapsing to **6,661 unique
sequences** and **673 unique sequences at ≤ 2.5 Å**, is the entire world supply.

This single fact determines the architecture. A model whose structural knowledge
must come from 3D supervision is capped at 673 clean examples. Therefore:

> **PHAROS is designed so that 3D coordinates are the *smallest* of its
> supervision channels, not the primary one.** Everything else in this document
> — the physics module, the chemistry layer, the 2D and probing heads, the
> coevolution track — exists to move learning off a channel that cannot grow.

### 1.2 The supervision pyramid **[v0.2]**

| Channel | Volume | Growable? | Information |
|---|---|---|---|
| Sequence (elDORS + RNAcentral) | **1,369,926,204** | yes | ~80 GB |
| Chemical probing (Ribonanza) | **335,616 acquired** of 2.1M | yes | large |
| Fitness (NABench + RNAGym) | 620,372 records | yes | moderate |
| Splicing (147 species) | 8.6 GB | yes | moderate |
| 2D structure | ~117k annotations + **9,347** dot-bracket from pdb_hunter | slowly | ~9 MB |
| Coevolution (Rfam 15.1) | 4,227 families, 100 with 3D | slowly | moderate |
| **3D coordinates** | **10,399 entries → 673 clean sequences** | **NO — saturated** | **~2.8 MB** |

The last row is 2.8 MB against a model capacity of 37 MB at 2 bits/parameter.
**The 3D task is data-limited by a factor of ~13, and cannot be un-limited.**

### 1.3 What each corpus actually contains **[v0.2]**

A trap that cost this project real time: the two derivative corpora are
**stripped**, and their statistics are not RNA statistics.

| Corpus | Entries | HETATM | B-factors | Modified residues |
|---|---|---|---|---|
| RNA3DB | 15,441 chains | **none** | **zeroed** | 0.025% |
| RNASolo/gRNAde | 14,366 chains | **none** | real | 0.025% |
| **Raw PDB** (acquired for v0.2) | **8,041 entries, 13.9 GB** | **yes** | real | **1.005%** |
| pdb_hunter RNA_Database | 10,424 entries | yes | real | — |

Modified residues are **40× more frequent** in raw PDB than in the derivatives.
Any statistic about ions, rigidity, modified residues or whole-entry context
**must** be computed on raw entries. The derivatives are fine for backbone
geometry and nothing else.

---

## 2. Inputs

| Input | Shape | Required | Note |
|---|---|---|---|
| Sequence | L, vocab §3.1 | yes | elDORS is DNA-alphabet; T→U at load |
| **Chemistry vector** | L × 24 | yes, computed | **new in v0.2** — §4 |
| **Ionic condition** | ([Mg²⁺],[K⁺],[Na⁺],T,pH) | yes, defaulted | the distinguishing input |
| MSA / Rfam | M × L | optional | graceful degradation, gated on Neff/L |
| Chemical probing | L (SHAPE/DMS) | optional | Ribonanza; §9.4 |
| In-complex flag | 1 | yes, defaulted | mandated by G2 (§11.2) |

Ionic condition enters as FiLM conditioning, like a diffusion timestep. **The
same sequence at 0 mM and 10 mM Mg²⁺ must produce different predictions.** This
remains the clearest novelty claim, and §6.4 states honestly what supervises it.

---

## 3. Tokenisation

### 3.1 Vocabulary — decided jointly with the data source **[v0.2]**

v0.1 specified two vocabularies on the basis that 8.90% of structural residues
fall outside {A,C,G,U}. That figure is a **raw-PDB** figure. Measured on the
data that would actually be consumed:

| Source | non-ACGU |
|---|---|
| elDORS (pretraining) | 0.203% (all `N`) |
| RNA3DB / RNASolo | **0.025%** |
| Raw PDB | **1.005%** |

**Decision.** Pretraining vocabulary is **5 symbols + specials**; elDORS is
pre-normalised and widening it would add dead tokens (NucleicBERT's vocab-25
mistake). The structural vocabulary is **5 + DNA{A,C,G,T,U} + I + UNK + a
learned `MOD` embedding keyed by mmCIF `comp_id`, falling back to the parent
base** — and it is required **only because v0.2 trains on raw PDB entries**.
Had we trained on the derivatives, 5 symbols would have been within 0.025%.

### 3.2 `N` is two tokens, not one **[v0.2]**

The same symbol denotes two unrelated things, and v0.1 conflated them.

| | `N_seq` | `N_struct` |
|---|---|---|
| Meaning | ambiguous base call | identity unassigned by depositor |
| Rate | **0.203%** of nt | 0.025% of residues |
| Range across chunks | 0.51% → 0.0077% (67×) | — |
| Geometry | **none** | **complete** |
| Sequences affected | 1.45% carry ≥ 1 | — |

Verified directly: `8vvt_ZA` residue 1248 carries a full ribose — O5′, C5′, C4′,
O4′, C3′, O3′, C2′, **O2′**, C1′ — plus a partial base ring. Its backbone is
known; only the base is not.

**Consequences.** `N_struct` residues train every geometry head normally, and
recovering their identity is a free auxiliary task (§9.7). `N_seq` residues are
masked from identity losses but retain positional context. One shared token
throws both away.

---

## 4. The chemistry layer — new in v0.2

v0.1 carried nucleotide identity and little else. The goal is a model that
reasons about RNA *chemically*, so chemistry enters explicitly, as a per-residue
feature vector through one projection.

**Why a projection and not a wider `d_model`:** a fully attributed nucleotide
carries **58.9 bits**; a 512-dim bf16 token holds **8,192** — 139× over-provisioned
**[v0.1, arithmetic verified]**. Attributes are nearly free; width is quadratic
in the FFN. This was already the right call and is retained.

### 4.1 The 24-dimensional chemistry vector

| Group | Features | Source |
|---|---|---|
| **Identity** | one-hot A/C/G/U/mod (5) | sequence |
| **Purine/pyrimidine** | ring class (1) | derived |
| **H-bond capacity** | donors, acceptors on WC / Hoogsteen / sugar edge (6) | chemical table |
| **pKa** | N1/N3 pKa, shifted-pKa flag (2) | literature table |
| **Sugar pucker propensity** | C3′-endo / C2′-endo prior (2) | literature |
| **Stacking** | polarisability, stacking-energy prior (2) | literature |
| **Backbone** | phosphate charge state, 2′-OH present (2) | derived |
| **Modification** | methylation / pseudouridylation / other (3) | mmCIF `comp_id` |
| **Local context** | GC fraction ±16 (1) | computed |

All 24 are available at inference for any sequence. None is a structural
observable, so none leaks the answer — the separation v0.1 established between
**inputs** and **supervision targets** is preserved and is load-bearing.

### 4.2 What stays a target, never an input

Saenger class, Leontis–Westhof class, normalised B-factor, Mg²⁺ distance class,
disorder label, SHAPE/DMS reactivity. Feeding any of these in would leak.

---

## 5. Token trunk — MoE + hybrid attention + loops

### 5.1 Block interleaving **[v0.1, retained]**

16 blocks at d=512, period-8 pattern, two cycles:

```
[ GDN, GDN, SWA, GDN, GDN, SWA, GDN, FULL ]
```

- **GDN** Gated DeltaNet, linear, O(L) — 10/16
- **SWA** sliding-window softmax, w=128, O(L·w) — 4/16
- **FULL** full attention with physics bias, O(L²) — **2/16**

Justified by RNA structure: A-form helix is locally periodic and low-information,
so linear attention suffices; junctions, pseudoknots and kissing loops need
global mixing and are rare. Full attention is reserved for the two blocks that
feed the pair track.

**Honest scoping, retained from v0.1:** at L=2048 the attention terms are only
**3.7%** of training FLOPs. The hybrid stack earns its place on **memory and
inference**, not training throughput.

### 5.2 Loop engineering — depth from refinement, not stacking

**16 blocks × 8 loops = 128 effective layers**, more than a 32-block
configuration's 96, at 9.4× fewer parameters. With the one-step gradient the
loops cost compute but **no additional activation memory**.

Deep supervision is applied at every loop segment. The recycled signal is the
previous iteration's distance estimate, which feeds back into the electrostatic
bias — so physics and refinement are coupled rather than sequential.

**Caveat that must travel with the HRM citation [lit]:** the independent ARC
Prize analysis credits the outer loop, but also finds most of HRM's benchmark
performance comes from memorising evaluation-time tasks. The loop is adopted on
its measured merits; the *parameter-count* argument must not lean on HRM.

### 5.3 MoE routing

Fine-grained all-MoE feed-forward. Router inputs include length bin, `Neff/L`
(coevolution depth), in-complex flag, and the chemistry summary — so routing is
conditioned on *what kind of RNA this is*, which is the point.

**Router circularity at recycle 0 remains an open design gap [v0.1, unmeasured]:**
features derived from the pair track are unavailable on the first pass. Current
plan is a sequence-only router at recycle 0, switching at recycle ≥ 1.

### 5.4 Sizing **[v0.1 arithmetic, verified exact]**

| Model | d | blocks | loops | eff. layers | total | active | A100-h @25B |
|---|---|---|---|---|---|---|---|
| **PHAROS-Small** (default) | 512 | 16 | 8 | 128 | **149M** | **61M** | **~78** |
| PHAROS-Mini | 384 | 12 | 12 | 144 | 67M | 30M | — |
| Base-v2 (scale-up path) | 768 | 32 | 3 | 96 | 1,401M | 269M | — |

Small first: the structure task carries ~2.8 MB of information and is
data-limited, not capacity-limited. The first informative checkpoint is about
**one GPU-day**, so sizing is an experiment rather than an argument.

---

## 6. Physics module — the explicit Hamiltonian

This is where PHAROS differs most from AlphaFold-style architectures, and it is
the part best supported by measurement.

### 6.1 Manning counterion condensation **[implemented, 4/4 tests pass]**

Closed form, no learned parameters, no training data required:

```
ξ = l_B / b          b = 1.40 Å (A-RNA axial rise per phosphate)
θ = 1 − 1/ξ          ξ = 5.11,  θ = 0.804,  q_eff = −0.196
```

**`b` is the axial projection, not the P–P contour distance** — v0.1 had this
wrong, and the θ ≈ 0.76 in the earlier draft is the **B-DNA** figure. Corrected
in cycle 2, implemented, and regression-tested.

### 6.2 Debye screening

Screened Coulomb pair bias with κ from the ionic condition. Verified behaviour:
screening length shortens 9.61 → 6.88 Å and `B_elec` weakens −0.0097 → −0.0064
as salt rises. This is the mechanism by which ionic condition changes a
prediction.

### 6.3 Mg²⁺ and rigidity — one expert, now properly evidenced **[v0.2]**

The claim that ion coordination and local rigidity are the same phenomenon was
v0.1's justification for a shared expert, and it rested on **15** structures.
Re-derived on raw PDB:

| stratum | span | monotonic | nucleotides | structures |
|---|---|---|---|---|
| v0.1 published | 1.76 σ | yes | 24,623 | **15** |
| **X-ray** | **1.523 σ** | **yes** | 3,858,271 | **1,535** |
| cryo-EM | 0.934 σ | **no** | 4,615,984 | 1,885 |

**The gradient holds at 102× the structure count**, monotonic across all six
distance bins, 13% weaker than published. Quote **1.52 σ**.

**New and consequential:** cryo-EM does **not** reproduce it — 0.934 σ,
non-monotonic, on 1,885 structures. v0.1 excluded cryo-EM on the *assumption*
that ADPs are not comparable to crystallographic B-factors; that is now
**measured**. Therefore:

> **The rigidity head trains on X-ray B-factors only.** Pooling would dilute the
> signal by a third. This was not stated in v0.1 and is a real training-pipeline
> requirement.

Ion inventory, corrected in the design's favour: Mg **816,270** vs K 15,123 —
**Mg:K = 54:1**, not 9:1, and Mg outnumbers all other cations **19.7:1**.
Inner-sphere coordination to phosphate OP1/OP2 is **77.9%** (324,708 / 416,731).
The Mg-centric design is better supported than its own evidence claimed.

### 6.4 What supervises the ionic-condition input — stated honestly

| Component | Supervision | Status |
|---|---|---|
| Manning θ, κ, q_eff | closed form | **no data needed** |
| Debye pair bias | closed form | **no data needed** |
| Mg²⁺ site head | ion coordinates | **available** (raw PDB, 816k Mg sites) |
| Rigidity head | X-ray B-factors | **available** (3.86M nt) |
| **[Mg²⁺] → structure *response*** | titration series | **NOT available** |

The PDB is survivorship-biased on ionic conditions: recorded Mg²⁺ spans only
5–15 mM because nobody deposits unfolded RNA. So the closed-form terms and the
site/rigidity heads are fully supervised, but the **learned response to changing
ionic strength is not**, and needs RMDB titration ladders. RMDB exposes no bulk
endpoint; acquisition is an open action, not a solved one.

**Scope statement required:** PHAROS accepts ionic condition as an input and its
*physics* terms respond to it correctly by construction. Claims that the
*learned* prediction tracks a Mg²⁺ titration are unsupported until RMDB lands.

---

## 7. Hierarchical Pair Track

### 7.1 Why not flat top-K **[v0.1, measured]**

A flat top-K proposer keeping K = 32L recovers **0.200** of true contacts on
500–1200 nt chains, where a *random* scorer gets 0.746 across the same set. A
banded fallback does not rescue it: at L 1500–3000 even ±512 misses ~15%
**[v0.2: 0.8545 captured]**. Contacts must be selected as **blocks**, not pairs.

### 7.2 Three-level coarse-to-fine

| Level | Operation | Cost at L=2861 |
|---|---|---|
| L1 | **dense** pair map at b=16 | 0.396% of dense |
| L2 | refine occupied b=16 → b=4 | 0.460% |
| L3 | refine occupied b=4 → pairs | 1.346% |
| | **total** | **2.202% of dense** |

The L1 map is genuinely dense, so there is **no recall loss at the coarse
level**; loss can enter only at the tunable selection thresholds. Identifying
whether a 16×16 block contains any contact aggregates 256 pair-decisions into
one, which is why the coarse level is learnable where flat pair ranking is not.

### 7.3 `target_c` — raised to 24 **[v0.2]**

v0.1 set `target_c = 20` against a maximum of 19.04 observed on 180 structures.
On **20,266 chains** the maximum is **21.14** — the budget is **breached**.

| | v0.1 | v0.2 |
|---|---|---|
| chains measured | 180 | **20,266** |
| mean effective c (long) | 17.2 | 17.14 |
| p99 / p99.9 | — | 19.13 / 19.52 |
| **max** | 19.04 | **21.14** |
| `target_c` | 20 | **24** |

24 clears the observed maximum by 12% and costs proportionally more only in L3.
**And the true tail is worse than 21.14**: this was measured on derivatives
carrying 0.025% modified residues against raw PDB's 1.005%, a 40× under-
representation of the quantity that pushes a chain up. Re-deriving on raw
entries is an open action; 24 is an interim value with a stated margin, not a
measured maximum.

The track must additionally **handle overflow gracefully** — truncate to budget
and record the miss — rather than assume the budget always suffices.

### 7.4 Reference implementation **[v0.1, measured]**

L=4096 in **0.45 s** at 0.96% of dense; the AF3-style dense baseline is **not
runnable** past L=1024 on a 14 GB machine (12.9 GB predicted at 2048, 51.5 GB at
4096). Two silent indexing/budget bugs were found and fixed by building it.

**Still unproven:** a *learned* block-detection scorer. The reference proves cost
and structure, not that a model can find the occupied blocks. This is the single
largest unvalidated assumption in the design.

---

## 8. Motif bank and what does not change in RNA

RNA has a small, reused vocabulary of local structure — GNRA and UNCG
tetraloops, kink-turns, A-minor motifs, sarcin-ricin loops. These are the
"things that do not change."

**Retrieval, not memorisation:** a frozen KV bank of 667 motif classes, indexed
from the BGSU motif atlas, queried by the pair track after pairing is estimated.

**Critical negative result [v0.1, measured]:** GNRA k-mer context alone predicts
rigidity at **0.073 σ** — negligible. Motif identity requires the *interaction
graph*, not sequence n-grams. Motif routing therefore happens in the **pair**
track, never from raw k-mers. This is why the bank is queried late.

---

## 9. Heads — what the model can do

v0.1 specified six. v0.2 specifies ten, four of them on data already in hand and
previously ignored.

| # | Head | Output | Supervision | Volume |
|---|---|---|---|---|
| 1 | Contact map | L×L binary | raw PDB | 10,399 entries |
| 2 | Distance | L×L binned | raw PDB | same |
| 3 | 3D structure | coordinates, K=3 states | raw PDB | 673 clean seqs |
| 4 | Secondary structure | dot-bracket | bpRNA + **pdb_hunter 9,347** | ~126k |
| 5 | Mg²⁺ sites | per-residue | raw PDB | **816,270 sites** |
| 6 | Rigidity | normalised B | **X-ray only** | 3.86M nt |
| 7 | Reactivity | SHAPE/DMS | Ribonanza | **335,616 profiles** |
| 8 | **Fitness** | mutation effect | NABench + RNAGym | **620,372** |
| 9 | **Splicing** | site / outcome | 147 species | **8.6 GB** |
| 10 | **Base identity** | recover `N_struct` | free | 0.025% of residues |

Plus the ensemble outputs of §10.

**Heads 8 and 9 are new.** Both sit on substantial supervised data already on
disk and entirely unused by v0.1. Fitness in particular is the second-largest
labelled channel after probing, and it is the one that speaks to *function* —
the goal's "what would be the function" requirement.

---

## 10. Dynamics — ensemble, not a single structure

Much of RNA is not one structure. Once the per-step stiffness field `F_i` exists,
a harmonic ensemble is nearly free: assemble `F` → normal modes → fluctuation
amplitudes, and emit **K=3** states with weights.

**What this does NOT claim:** it is **equilibrium breathing, not a folding
pathway**. The couplings are dinucleotide-level and the model does not traverse
a barrier. Calling this "folding dynamics" would overstate it.

---

## 11. Coevolution, conservation, and generalisation limits

### 11.1 What changes and what does not

Coevolution is how the model learns which positions are free to vary and which
are not. APC-corrected, sequence-reweighted MI on Rfam seeds recovers curated
base pairs at mean precision@L/5 **0.670**, recall@L **0.768**; tRNA reaches
1.000/1.000 **[v0.1]**.

**The depth split must not be leaned on.** v0.1 called `Neff/L` "the decisive
pattern"; the rigor recheck retracted that — exact permutation p = 0.0455 on a
post-hoc threshold, Spearman only +0.224 over n=12, deep arm n=2, and a shallow
family reaching precision 1.000. What is robust is that **precision varies
enormously between families (0.333–1.000)**, so a fixed-weight blend is wrong
and `Neff/L` is a defensible *router feature* on literature grounds.

Ground truth is also incomplete: the WUSS parser ignores pseudoknot brackets,
**53 of 537 pairs (9.87%)**, which deflates precision — 0.670 is a lower bound.

### 11.2 G2 — the largest generalisation hazard **[v0.1, and still unmeasured at scale]**

**98.95%** of RNA residues in the sample come from entries containing protein.
The model takes one RNA sequence and predicts a fold that, in reality, only
exists inside a ribonucleoprotein. Isolated RNA is 11.7% of *structures* but
**1.05% of residues**.

Not fixable by tuning. Mandatory mitigation: **evaluation must be reported split
by isolated vs in-complex**, and the in-complex flag is a model input.

> Raw entries were acquired partly to re-measure this properly — G2 and G3 are
> whole-entry properties and are not computable on per-chain extracts. That
> re-derivation is an open action.

### 11.3 Scope statement

PHAROS predicts **single RNA chains**. Coverage of the corpus at a 4,096 context
is **99.97%** — v0.1 claimed 100%, but 8 chains (5 unique structures: 6HRM
4,450 nt, 7UPH, 4V6X, 8TOC, 7LHD, all large ribosomal rRNA) exceed it **[v0.2]**.
Either state 99.97% or set the context to 4,608.

It is **not** a whole-transcript or whole-ribosome model: total RNA per entry
reaches 11,478 residues, and lncRNAs (XIST ~19 knt) and viral genomes
(SARS-CoV-2 ~30 knt) are an order of magnitude beyond any context considered.

---

## 12. Training

### 12.1 Curriculum

1. **Pretrain** on sequence (MLM + span masking), 5B → 25B tokens staged.
2. **2D** on bpRNA + pdb_hunter dot-bracket.
3. **Probing** on Ribonanza — **335,616 profiles acquired** (2A3_MaP and
   DMS_MaP, 206 reactivity positions, ~177 nt sequences): **499× the 673 clean
   3D sequences**, and the largest labelled channel the model will see.
4. **Physics** terms active throughout; they are closed-form and need no stage.
5. **3D** last and briefest, on 673 clean sequences, with the ensemble head.
6. **Function** heads (fitness, splicing) as auxiliary tasks, co-trained.

### 12.2 Budget **[v0.1 arithmetic, verified exact]**

**25B tokens**, not 323B. At 61M active, 323B is 5,295 tok/param = 265×
Chinchilla, the worst corner of the quantisation-degradation law, for no
information gain — RNA entropy is **2.0165 bits/nt [v0.2]**, so the corpus is
redundant rather than rich. 25B gives 410 tok/param ≈ 20× Chinchilla.

**BF16 first, FP8 on Hopper, not 4-bit.** A100 has no FP8 tensor cores, so
"~79 A100-hours" mixed an Ampere baseline with a Hopper-only lever. Honest
figures: **~78 h on A100 bf16, ~12 h on H100 fp8**.

### 12.3 Optimiser

Muon (2× lever). Keep embeddings, norms, router and attention softmax in higher
precision regardless — this is what DeepSeek's own FP8 recipe does.

---

## 13. Open actions

| # | Action | Blocks |
|---|---|---|
| 1 | Re-derive `target_c` on raw PDB entries | final pair-track budget |
| 2 | Acquire RMDB titration ladders | the learned ionic *response* |
| 3 | Complete Ribonanza — 335,616 of 2.1M acquired; the rest needs Kaggle credentials | head 7 coverage |
| 4 | Train the block-detection scorer | §7.4, the largest unvalidated assumption |
| 5 | Re-measure G2/G3 on raw whole entries | scope claims |
| 6 | Re-derive every remaining max/min at scale | four of four failed so far |
| 7 | Restore sample structures to the server | `test_mmcif_entities.py` cannot run |

---

## 14. Provenance

Every **[v0.2]** number traces to a script and a JSON under
`data/samples/analysis/`:

```
recheck_block_sparsity_fullcorpus.py  -> block_sparsity_fullcorpus.json
recheck_contact_tails_fullcorpus.py   -> contact_tails_fullcorpus.json
recheck_ions_rigidity_rawpdb.py       -> ions_rigidity_rawpdb.json
acquire_raw_pdb_entries.py            -> raw_pdb_manifest.json
```

Full comparison against v0.1 in `plans/rigor-recheck/fullcorpus-validation.md`.
