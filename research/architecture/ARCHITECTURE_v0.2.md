# PHAROS v0.2 — Complete Architecture Specification

**P**hysics-**H**ybrid **A**ttention-**R**outed mixture-**O**f-experts for RNA **S**tructure

> **Status.** Design complete, not trained. Supersedes `ARCHITECTURE.md` (v0.1)
> wherever the two differ. v0.1 is retained: its audit trail — 28 numbered
> defects across 11 cycles — is the reason this version can state what is
> measured and what is not.
>
> **What makes v0.2 different.** v0.1 was designed against **180** structures
> fetched onto a laptop. v0.2 is validated against the corpus: **29,807 chains**,
> **10,527 raw PDB entries** (15 GB, acquired for this revision), **10,424
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
| max effective c | 19.04 | **21.14** | **breach**, closed at **23.30** on raw (§7.3) |
| longest chain | 3,764 nt | **4,450 nt** | **breach** |
| chains over 4,096 nt | 0 | **8** | **breach**, closed at 5 entries (§0.1) |
| max contacts/nt | 5.50 | **7.66** | **breach**, 7.79 on raw |
| Mg : K ratio | 9 : 1 | **54 : 1** | wrong 6× |

**Every mean held. Every maximum failed.** This is not eleven separate errors;
it is one property of an n=180 sample, which estimates a mean well and cannot
estimate a p99.9 at all. Two load-bearing parameters (`target_c`, the context
window) had been set from maxima.

> **A twelfth claim failed differently, and it was the most load-bearing of
> all.** §7.1 — the justification for selecting blocks rather than ranking
> pairs — did not fail by being a small-sample maximum. It failed by comparing
> **two different populations**: a recall measured on five long chains against a
> random baseline measured on all 43, most of them short. Re-derived on 2,994
> chains at matched budgets the conclusion is not merely restored but
> strengthened from ~2× to an order of magnitude (§7.1). The lesson generalises
> past sample size: *what* was measured matters as much as *how much*.

**Design rule adopted for v0.2: no parameter may be set from an extreme
quantile measured on fewer than 10³ structures.** Where the corpus cannot
supply that, the parameter is set from a mean plus an explicit safety factor,
and the factor is stated.

### 0.1 A third pass, on raw whole entries **[v0.2]**

Three claims — G1, G2, G3 — describe *entries*, not chains, so none of them is
computable on RNA3DB or RNASolo, which distribute per-chain extracts with the
protein stripped out. All three still rested on the 180-structure sample after
the corpus pass. The raw archive was acquired to close them: **10,520 of 10,527
entries hold an RNA polymer chain, carrying 13,348,166 RNA residues**, 43x the
309,197 the three were originally measured on.

| | v0.1 (n=180) | raw PDB (10,520 entries) | |
|---|---|---|---|
| **G2** residues in entries with protein | 98.95% | **97.15%** | holds, −1.8 pts |
| **G3** residues from ribosome-like entries | 92.65% | **85.94%** | holds, −6.7 pts |
| **G1** longest single RNA chain | 3,764 nt | **4,450 nt** | **reproduces exactly** |
| **G1** entries over 4,096 RNA residues | 24.6% | **15.06%** | over-stated |
| **G1** total RNA residues per entry, max | 11,478 | **22,345** | **1.95x breach** |
| **`target_c`** max effective c | 19.04 | **23.30** (14,106 chains) | **within 24** |

The pattern from the corpus pass repeats and then breaks once, informatively.
The two *aggregate* claims held and both moved **downward**: the corpus is less
protein-bound and less ribosome-dominated than BGSU representative sets — which
deliberately over-weight large assemblies — made it look. The entry maximum
failed at 1.95x, the fifth extreme quantile from n=180 to do so.

**The exception is the one that matters most.** The longest single RNA chain
reproduces *exactly* — 4,450 nt, 6HRM, with the same five structures above 4,096
that the derivative corpora found. A maximum that does not move when the sample
goes from 29,807 derived chains to the entire archive is not an estimate any
more; it is the value. That is what lets the context window be settled at 4,608
(D20) rather than left provisional, and it is the first extreme quantile in this
project to be closed rather than merely raised.

---

## 1. Data foundation — and the fact that reframes everything

### 1.1 The 3D corpus is saturated **[v0.2]**

An RCSB search for every entry containing an RNA polymer entity returns
**10,399** entries. `pdb_hunter/RNA_Database` holds **10,424**. The overlap is
essentially total; **98** entries are missing and were acquired for this
revision.

**There is no more experimental RNA 3D structure to obtain.** Not from better
tooling, not from more bandwidth. The raw store now holds **10,527 of 10,528**
entries in the union of RCSB's RNA query, RNA3DB, RNASolo and pdb_hunter.

The single exclusion is instructive: **9A0D** is an integrative/hybrid model —
an in-cell expressome from *M. pneumoniae* solved from cross-links plus 3DEM —
whose coordinates are `_ihm_sphere_obj_site` coarse-grained spheres with no
`_atom_site` at all. It is correctly excluded, because an atomic-coordinate
model cannot train on spheres, and the acquirer now reports such entries as
`excluded_ihm` rather than as download failures. 10.4k entries, collapsing to **6,661 unique
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
| Sequence (elDORS + RNAcentral) | **1,369,926,204** | yes, but not usefully | ~80 GB |
| Chemical probing (Ribonanza) | **335,616 acquired** of 2.1M | yes | large |
| Fitness (NABench + RNAGym) | 620,372 records | yes | moderate |
| Splicing (147 species) | 8.6 GB | yes | moderate |
| 2D structure | ~117k annotations + **9,347** dot-bracket from pdb_hunter | slowly | ~9 MB |
| Coevolution (Rfam 15.1) | 4,227 families, 100 with 3D | slowly | moderate |
| **3D coordinates** | **10,399 entries → 673 clean sequences** | **NO — saturated** | **~2.8 MB** |

The last row is 2.8 MB against a model capacity of 37 MB at 2 bits/parameter.
**The 3D task is data-limited by a factor of ~13, and cannot be un-limited.**

**Nor is the sequence channel worth growing.** MARS, the 1.7-billion-sequence
corpus NucleicBERT pretrains on, was assessed and rejected: a 40 MB range-probe
of its first tarball shows 41.4% coding mRNA and only 10.9% ncRNA-like, of which
**94.6% is rRNA or tRNA** gene and amplicon records. Projected yield of
genuinely diverse structured ncRNA is ~3M sequences, for 413 GB of download
against 290 GB of free disk — and it would deepen the ribosome skew that G3
already measures at 92.65% of structural residues. We hold 46.2M curated
RNAcentral ncRNA, more than the 30M NucleicBERT extracted from MARS. See
`plans/14-mars-acquisition-assessment.md`.

### 1.3 What each corpus actually contains **[v0.2]**

A trap that cost this project real time: the two derivative corpora are
**stripped**, and their statistics are not RNA statistics.

| Corpus | Entries | HETATM | B-factors | Modified residues |
|---|---|---|---|---|
| RNA3DB | 15,441 chains | **none** | **zeroed** | 0.025% |
| RNASolo/gRNAde | 14,366 chains | **none** | real | 0.025% |
| **Raw PDB** (acquired for v0.2) | **10,527 entries, 15 GB** | **yes** | real | **1.005%** |
| pdb_hunter RNA_Database | 10,424 entries, 154 GB | yes | real | — |

`pdb_hunter` additionally carries a **derived annotation layer** that nothing in
v0.1 accounted for, indexed by `scripts/integrate_pdb_hunter.py` into
`data/catalog/pdb_hunter_index.json` (files referenced in place, not copied):

| Annotation | Entries |
|---|---|
| dot-bracket 2D (`.dbn`) + base pairs (`.bpseq`) | **9,347** |
| backbone coordinates (`.xyz`) | 10,280 |
| **MolProbity clashscore** | **10,073** |
| **Rfam family** | **6,316** across **585 families** |
| resolution | 5,169 |

Three of these change the design rather than adding rows — see §12.3 and §12.4.

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
| **Modification** | methylation / pseudouridylation / other (3) | mmCIF `comp_id` → PDB Chemical Component Dictionary |
| **Local context** | GC fraction ±16 (1) | computed |

All 24 are available at inference for any sequence. None is a structural
observable, so none leaks the answer — the separation v0.1 established between
**inputs** and **supervision targets** is preserved and is load-bearing.

#### Implemented, and the modification classes are now measured **[v0.2]**

`src/pharos/data/chemistry.py`, with `test_chemistry.py` pinning 34 properties.
Two things changed on contact with the archive.

**The three modification dims were specified against five hand-picked examples;
the archive holds 370 species.** Over all 10,520 RNA-bearing entries, 75,434 of
13,348,166 residues are modified (0.565%):

| dim | class | residues | share |
|---|---|---|---|
| 20 | methylation | 42,555 | **56.4%** |
| 21 | pseudouridylation | 18,458 | **24.5%** |
| 22 | other | 14,421 | 19.1% |

Pseudouridine is a quarter of all modifications by itself, which retroactively
justifies giving it a dim rather than a share of "other" — and it earns that dim
mechanically, not just by frequency: it adds an N1-H donor on the Hoogsteen
face, which `chemistry.py` applies at dim 8.

**Parent resolution is by dictionary, because it cannot be anything else.**
`VOCAB.md` specifies a `MOD` embedding "falling back to the parent base", and
the obvious source — `_chem_comp.mon_nstd_parent_comp_id` — is **not in the
entry files**. They carry only id/type/mon_nstd_flag/name/synonyms/formula/
formula_weight; a first implementation parsed for the parent and got one for
zero of 55 species. The field lives in the PDB Chemical Component Dictionary
(119 MB, acquired). All 370 species are in it, and **88.6% of modified residues
resolve to a standard A/C/G/U parent**. The 11.4% that do not — inosine, UNK,
L-nucleotides, locked and fluorinated analogues — are genuinely parentless and
are reported as `N` rather than assigned one.

Classification reads the CCD's systematic *name*, which is chemistry, rather
than a list of component codes, which is the curated list defect #22 was about.
A2M is the regression case: nothing in the string says adenosine.

**The leak boundary is enforced in code.** Dim 13 (shifted pKa) is the one
feature set from structural context, so `chain_chemistry` has no parameter that
can set it — a caller must go residue-by-residue and ask for it explicitly, on
recycles ≥ 1. Setting it from a known structure in training and from nothing at
inference is a train/test mismatch that would otherwise be one keyword away.

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

### 5.4 Sizing **[v0.2 — built and counted; v0.1's table was not reproducible]**

| Model | d | blocks | loops | eff. layers | total | active | active % | A100-h @25B |
|---|---|---|---|---|---|---|---|---|
| **PHAROS-Small** (default) | 512 | 16 | 8 | 128 | **238.8M** | **62.7M** | 26.2% | **~78** |
| PHAROS-Mini | 384 | 12 | 12 | 144 | 102.1M | 27.8M | 27.2% | — |
| Base-v2 (scale-up path) | 768 | 32 | 3 | 96 | 1,060.6M | 267.9M | 25.3% | — |

MoE recipe, shared across the family: **32 experts, top-4, 2 shared,
`d_expert = d_model/2`**. Counted from `pharos.model.Pharos`, pinned by
`test_pharos.py`.

> **v0.1 quoted 149M/61M, 67M/30M and 1,401M/269M and called the arithmetic
> "verified exact". It could not have been.** The table gives `d_model`, block
> count and loop count; the two numbers it quotes are determined by
> `n_experts`, `d_expert`, `top_k` and `n_shared`, and it states none of them.
>
> Solving for them (`scripts/sampling/solve_sizing.py`, analytic counts
> validated against built models) shows each row is individually reachable
> within ~3% — but by **three mutually incompatible recipes**: Small implies
> 8 experts / top-1, Mini implies 24 / top-8, Base-v2 implies 64 / top-6. That
> is not a model family. The giveaway is the active fraction, which any fixed
> recipe holds roughly constant as `d` and depth scale: v0.1's rows imply
> **40.9%, 44.8% and 19.2%**, while the adopted recipe gives 26.2%, 27.2% and
> 25.3%.
>
> **What survives is the active column**, and that is the column that matters:
> active parameters drive training FLOPs and therefore the ~78 A100-h estimate.
> The adopted recipe reproduces it to **2.8% (Small)** and **0.4% (Base-v2)**.
> The totals were wrong — Small is 1.6x and Base-v2 0.76x what was printed —
> and totals drive memory, not cost, so the cost claims stand and the
> memory-footprint statements should be re-read against the new column.

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
| Mg²⁺ site head | ion coordinates | **available** (raw PDB, 816,270 Mg sites) |
| Rigidity head | X-ray B-factors | **available** (3.86M nt) |
| **[Mg²⁺] → structure *response*** | titration series | **available [v0.2]** — 24 RMDB ladders, 527 points |

The PDB is survivorship-biased on ionic conditions: recorded Mg²⁺ spans only
5–15 mM because nobody deposits unfolded RNA. So the closed-form terms and the
site/rigidity heads are fully supervised, but the **learned response to changing
ionic strength cannot be learned from the PDB at all**, at any scale. It needs
experiments where the concentration was deliberately varied.

#### Those experiments are acquired **[v0.2, open action 2 closed]**

v0.1 recorded "RMDB exposes no bulk endpoint". That was the old Django site.
RMDB is now a static site on GitHub Pages and its **1,024 RDATs are release
assets** — 11.9 GB across five releases, `DasLab/rmdb.github.io`.

Finding the ladders took one correction. Entry-level annotations record only
**three** distinct Mg²⁺ levels across 712 entries — 0, 10 and 40 mM, with 682 at
the standard 10 — and no series at all, because the entry-level `chemical` field
records the condition *common to the whole experiment*. **A titration varies its
concentration per data row**, inside the RDAT's `ANNOTATION_DATA` lines. So the
ladders are found by reading the files, and the files worth reading are the
small ones: a classic titration is one construct at tens of conditions, while
the multi-gigabyte assets are Eterna and Ribonanza libraries at a single
condition. 865 files under 2 MB, 202 MB fetched.

| | |
|---|---|
| Mg²⁺ titration files | **24** |
| total concentration points | **527** |
| ladders with 32 levels | 14 |
| widest range | 0 – 100 mM (TODS1/TODS7) |
| deepest ladder | 32 levels over 0 – 50 mM (MTTR series, 12 files) |
| **ladders reaching below 1 mM** | **24 of 24** |

That last row is the one that matters. Every ladder crosses the sub-millimolar
regime where RNA is unfolded — exactly the region the PDB cannot contain,
because unfolded RNA is not deposited. The MTTR series alone gives 12
constructs × 32 concentrations of the same folding transition. Four ATP
titrations came with them, which supervise ligand response rather than ionic.

#### And the physics is measurably in it **[v0.2]**

`build_ionic_dataset.py` turns the ladders into **701 examples over 75,706
residues** — one per (construct, concentration), never averaged, because the
quantity of interest is how reactivity at each position *changes* with [Mg²⁺]
and averaging a ladder destroys exactly that.

The build then checks the physics rather than assuming it. As Mg²⁺ rises, RNA
folds and becomes less reactive to SHAPE and DMS chemistry, so mean reactivity
should fall along a ladder. Measured as Spearman(concentration, mean
reactivity):

| | |
|---|---|
| ladders showing the transition (ρ < −0.3) | **16 of 24** |
| median ρ | **−0.65** |
| strongest | **−0.98** (MTTR5: 0.604 → 0.180) |

That is the folding transition PHAROS's ionic conditioning claims to model,
present in the supervision before any training. The eight ladders that do not
show it are kept and flagged, not dropped: a construct that does not fold under
Mg²⁺ is a real negative, and discarding it would leave a corpus in which the
effect is true by selection.

**One parser defect on the way.** RDAT 0.34 separates fields with tabs and 0.24
with spaces. Splitting only on tab turned every line of a 0.24 file into one
unrecognised key, so the file parsed to *nothing* rather than to an error, and
SRPDIV_DMS_0001's 16 Mg²⁺ levels silently vanished — 24 ladders reported as 23.
Concentrations are now recovered by scanning each annotation line rather than
tokenising it, because annotation values contain spaces (`MgCl2:0.04 mM`) and
tokenising splits them apart.

**Scope statement, updated:** PHAROS accepts ionic condition as an input and its
*physics* terms respond correctly by construction. The *learned* response is now
supervisable, and whether it tracks a titration is a training result rather than
a data gap.

---

## 7. Hierarchical Pair Track

### 7.1 Why not flat top-K — **re-derived at scale** **[v0.2]**

This is the claim the whole pair track rests on, and v0.1 stated it from 43
chains, incorrectly:

> *"A flat top-K proposer keeping K = 32L recovers 0.200 of true contacts on
> 500–1200 nt chains, where a random scorer gets 0.746 across the same set."*

**The two figures are different populations.** 0.2003 is the 500–1200 nt bin —
**five chains**. 0.7458 is the random baseline over all 43 chains, 29 of them
32–100 nt, where K = 32L already covers **94%** of every valid pair and so a
random scorer recovers nearly all of them by construction. "Across the same set"
is not true, and as written the sentence says a learned scorer does worse than
chance, which is backwards.

Re-measured on **2,994 chains** with every scorer on the same chains at the same
budgets. Recall at K = 32·L:

| chain length | n | random | separation prior | complementarity | K/dense |
|---|---|---|---|---|---|
| 32–100 | 1,205 | 0.941 | 0.914 | 0.952 | 0.941 |
| 100–200 | 766 | 0.530 | 0.703 | 0.617 | 0.530 |
| 200–500 | 97 | 0.203 | 0.522 | 0.300 | 0.204 |
| 500–1,200 | 66 | 0.084 | 0.452 | 0.220 | 0.083 |
| **1,200–3,000** | **860** | **0.034** | **0.377** | **0.182** | 0.034 |

Random recall tracks `K/dense` to three decimals in every bin, which is the
check that the measurement is sound. It also explains v0.1's 0.746: on short
chains a linear budget *is* most of the quadratic map, so every scorer looks
good and the corpus average is meaningless.

**The conclusion survives, and is now quantitative.** On the long chains the
architecture exists for, flat pair ranking within a linear budget:

| | budget | recall |
|---|---|---|
| separation prior | c = 32 | 0.377 |
| separation prior | c = 64 | 0.504 |
| **occupied b=4 blocks** | **c = 16.9** | **1.000** |

**Block selection reaches total recall at c ≈ 17, while flat pair ranking at
nearly four times that budget recovers half the contacts.** That is the
argument, and it is an order-of-magnitude argument rather than the ~4× one v0.1
was reaching for.

Two further things fall out. **Sequence complementarity, applied flat, is worse
than |i−j| alone** on long chains — 0.182 against 0.377 — so the baseline a
learned selector must beat is the separation prior, not random; that is why
`train_block_scorer.py` ablates it explicitly. And a banded fallback does not
rescue flat ranking either: at L 1500–3000 even ±512 misses ~15%
**[v0.2: 0.8545 captured]**.

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

### 7.3 `target_c` = 24 — **closed on raw PDB** **[v0.2]**

v0.1 set `target_c = 20` against a maximum of 19.04 observed on 180 structures.
The derivative corpora broke it at 21.14, and 24 was adopted as an interim
value because those derivatives carry **0.025%** modified residues against raw
PDB's **1.005%** — a 40× under-representation of exactly the quantity that
pushes a chain's effective `c` up. The budget therefore had never been measured
on data containing the thing it depends on. It has now been.

| | v0.1 | derivatives | **raw PDB** |
|---|---|---|---|
| chains measured | 180 | 20,266 | **14,106** (from 10,520 entries) |
| modified residues in them | 1.05% | 0.025% | **0.476%** |
| mean effective c | 17.2 | 17.14 | **11.79** (all lengths) |
| p99 / p99.9 | — | 19.13 / 19.52 | **19.19 / 23.08** |
| **max** | 19.04 | 21.14 | **23.30** |
| chains over 20 | 0 | 1 | **34** (0.24%) |
| **chains over 24** | — | — | **0** |
| `target_c` | 20 | 24 | **24, confirmed** |

**Decision: 24 stands, and D9 closes.** Zero of 14,106 chains breach it across
the complete archive. The maximum did rise, 21.14 → **23.30**, and 24 absorbed
it.

#### The hedge was right; the mechanism it named was not

D9 predicted the tail would worsen because the derivatives under-represent
modified residues 40×. Splitting the raw chains by whether their entry appears
in RNA3DB or gRNAde/RNASolo settles what actually happened:

| | chains | max effective c |
|---|---|---|
| entries the derivatives cover | 12,573 | **21.14** |
| entries they do not (§11.4) | 1,533 | **23.30** |

**21.14 is exactly the published derivative maximum.** The derivative
measurement was not wrong, and it was not distorted by missing modifications —
it was *correct for the entries it contained*. The whole of the difference is
coverage.

It is sharper than that. **The top 30 chains in the entire PDB are one
deposition campaign**, and all 30 are in the uncovered set: the 9T-series
*E. coli* ribosome peptidyl-transferase-centre focused refinements (9T1E, 9T1A,
9T1F, …), ~596-nt fragments of 23S rRNA at 2.28 Å, in rRNA-methyltransferase
knockout strains. 9T1E was deposited **2025-10-21** — after RNASolo's 2023-11
snapshot. The first chain from any other structure is **rank 31**, at 21.14.

Modification enrichment is real but secondary: the top 34 carry 1.98% modified
residues against the corpus's 0.48%, but against a **length-matched** control
(chains of 500–700 nt) the enrichment is only **1.9×**, not 40×. These are
methylation-variant ribosome structures, so they are modification-rich by
design; that is a property of what happens to sit at the top, not the reason the
derivatives missed it.

**The operative lesson is not about modified residues. It is that the curated
derivatives are a stale and partial view of the archive, and the extremes live
in precisely what they omit.** A parameter set from a derivative corpus inherits
that corpus's snapshot date. This is a stronger argument for open action 8 than
the chain counts in §11.4.

**The headroom is 2.9%, not the 12% v0.2 claimed.** The maximum has risen at
every re-measurement: 19.04 → 21.14 → 23.30, and unlike the chain-length maximum
(D20) this one is not a closed population — it is a property of whatever gets
deposited next, as the 9T series demonstrates by having been deposited during
this project. So:

**The overflow path is load-bearing, not defensive.** The track must truncate to
budget and record the miss, and the recorded miss-rate is a metric to watch in
training, not a debug counter. A budget with 2.9% headroom against a statistic
that has grown every time it was looked at is a budget that will be exceeded by
some future deposition; the design is correct only because it does not assume
otherwise.

Raising to 28 would buy headroom for **34 chains in 14,106** and cost
proportionally in L3 refinement on all of them. That trade is not worth making
while the overflow path exists.

> **The effective sample size at the top is 1, not 14,106.** Ranks 1–30 are the
> same ~596-nt molecule solved thirty times in one campaign. Ranks 31–34 are
> 7PAS (21.14), 9Z80 (20.97, a 119-nt chain in which **all 119 residues are
> modified**), and 4V6O/4V6P (20.11/20.01). Four independent molecules exceed
> 20 in the entire Protein Data Bank. A budget validated against that tail is
> validated against very little, which is the argument for the overflow path
> and against reading 23.30 as a ceiling.

### 7.4a R1 — **the scorer is trained, and it works** **[v0.2, measured]**

Every recall figure in §7 before this came from random weights. The question R1
asks is whether a model can *find* the occupied blocks from sequence, and it is
the assumption the entire pair track rests on. Trained on the family-disjoint
split (11,723 train / 1,450 test chains), evaluated against three baselines at
the **identical budget**:

| scorer | L1 recall | L2 recall | L2 precision |
|---|---|---|---|
| random | 0.135 | 0.543 | 0.202 |
| separation prior | 0.231 | 0.633 | 0.242 |
| learned, prior ablated | 0.271 | 0.833 | 0.337 |
| **learned** | **0.272** | **0.840** | 0.338 |

**Sequence gain over the separation prior: +0.207 at L2.** The answer to R1 is
yes: the sequence carries real information about which blocks are occupied,
beyond the fact that contacts cluster near the diagonal.

**And the gain is concentrated exactly where the architecture needs it.** By
chain length, L2 recall:

| chain length | separation prior | learned | |
|---|---|---|---|
| < 128 | 0.717 | **0.926** | |
| 128–512 | 0.475 | **0.523** | |
| 512–1,500 | 0.360 | 0.349 | no gain |
| **≥ 1,500** | **0.278** | **0.997** | **3.6×** |

On chains of 1,500 nt and up — the regime where §7.1 shows flat pair ranking
collapses to 0.377 even at c=32 — the learned selector recovers **99.7%** of
occupied blocks. That is the case for the Hierarchical Pair Track, measured
rather than assumed.

**The 512–1,500 band is the honest weak spot**: the learned scorer matches the
separation prior there and beats it nowhere. It is also the thinnest band in the
family-disjoint test set (66 chains), so the estimate is noisy, but it is not a
band to claim anything about.

**Capacity does not help.** The experiment was run twice, at 4.4M and 61.9M
parameters with matched optimiser steps. The larger model is better on
validation (0.890 vs 0.871) and *not* better on the held-out test set (0.835 vs
0.840) — a tie, with a hint of overfitting. A third attempt at 153M with a 14×
larger batch was much worse (val L2 0.666, loss stuck at 2.37) because the batch
cut optimiser steps from ~11,700 to 2,448; that is a step-count failure, not a
capacity ceiling, and raising the learning rate to 1e-3 did not rescue it.

This is direct evidence for §12.2's position that the structure task is
**data-limited, not capacity-limited**, and for keeping PHAROS-Small.

> **What is not measured here.** L1 and L2 are each evaluated standalone, at
> their own budget, against their own labels. The track *cascades* — L2 refines
> only inside surviving L1 blocks — so end-to-end recall is bounded by L1's
> 0.272 and is not reported. The cascade number is the one a deployed track
> would deliver, and measuring it is the next thing this experiment owes.

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

**Implemented [v0.2]** — `scripts/build_motif_bank.py` compiles atlas release
4.12 into the bank and `pharos.model.motif_bank` retrieves from it. The counts
are the atlas's own: **667 classes = 413 internal-loop + 254 hairpin-loop**,
4,992 instances, 21 interaction families. Each class carries an
interaction-family histogram, a position-specific base profile computed across
its instances, and size/type/instance scalars. Those are facts about RNA, so
they are **frozen buffers**; what trains is only the projections that turn them
into keys and values, and the query projection from the pair track — 31,810
learned parameters against 82,041 frozen ones.

Three details that are load-bearing rather than incidental:

* **There is no sequence query path, by construction.** `MotifBank.forward`
  takes a pair-derived query and nothing else, and `test_motif_bank.py` asserts
  the module contains no k-mer machinery. The negative result below is not a
  caution to remember; it is enforced.
* **Rare classes are distrusted, not dropped.** Only **231 of 667** have five or
  more instances. Instance count enters as a log-prior on the retrieval logits,
  so a class seen twice cannot outrank one seen 300 times on equal evidence,
  and a motif with one instance is still reachable when the evidence is strong.
* **The gate starts closed** (bias −3, gate ≈ 0.04 at init), so an untrained
  bank injects nothing into the pair track and has to earn its way in.

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

#### The four that are free are now extracted **[v0.2]**

Heads 5, 6 and 10, and §10's disorder output, are supervised by labels that are
already inside every deposited mmCIF. `mmcif_entities.residue_labels()` reads
all four in one pass and `build_dataset.py` stores them alongside the tokens:

| head | label | source field |
|---|---|---|
| 5 Mg²⁺ sites | residue within 3 Å of a magnesium | `_atom_site` |
| 6 rigidity | per-residue mean B, z-scored within the chain | `_atom_site.B_iso_or_equiv` |
| 10 base identity | which residues are `N_struct` | `label_comp_id` |
| §10 disorder | positions absent from the coordinates | `_pdbx_unobs_or_zero_occ_residues` |

**D12 is now enforced by the data rather than by a note.** Every example carries
`rigidity_valid`, and the batch builder turns it into a per-residue
`rigidity_mask`. The reason is visible in the raw numbers: 7PAS (cryo-EM) has a
mean B-factor of **987** against 1VY7's (X-ray) **72** — per-atom B in cryo-EM
is a fitted display parameter, not a measured one. A training loop can no longer
mix the two by accident.

**The disorder label cannot be a mask over the modelled residues**, which is how
it was first written. An unobserved residue has no atoms, so it never appears in
`_atom_site` and a coordinate-aligned mask is necessarily all zeros. It is a
statement about the *polymer*, so it travels as `unobserved_seq_id` against
`n_polymer`: 1VY7 chain AA is 1,498 modelled of 1,521, with 23 unobserved —
a real label the aligned version reported as none.

**And a third of it is not disorder at all.** Over the built set the raw count
is **1,379,892** unobserved positions, against the 46,448 v0.1 quoted. Most of
the difference is scale, but not all of it: 7ANE chain 2 declares an 18,998-nt
polymer and models 604 of it, and the other 18,394 positions are not disordered
— they are the rest of a viral genome that was never in the crystal.
**250 chains of 16,604 (1.5%) are truncations like this, and they hold 472,825
positions — 34% of the signal, all of it the wrong kind.** A disorder head
trained on them would learn that RNA is mostly disordered. `disorder_valid`
excludes a chain whose declared polymer exceeds 3× its modelled length, leaving
**907,067 usable positions in 16,354 chains**.

Yields over the whole built set:

| head | supervised residues | share |
|---|---|---|
| 1 contact | 63,298,800 pairs | — |
| 5 Mg²⁺ sites | 712,033 | 5.41% |
| 6 rigidity (X-ray only) | 4,274,596 in 5,808 chains | 32.4% |
| 10 base identity | 2,586 | 0.02% |
| §10 disorder | 907,067 usable of 1,379,892 raw | 6.9% |

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

### 10.1 Implemented **[v0.2]**

`pharos.model.dynamics`, 21 properties in `test_dynamics.py`. Three decisions
came out of building it rather than out of the spec.

**Fluctuations are computed in O(L), not O(L³).** The assembled stiffness is
`6(L-1)` square; at L = 4,000 a dense inverse is 24,000 × 24,000, per chain, per
step. Only the *diagonal* blocks of the inverse are needed for the amplitudes,
and those follow from a forward-backward recursion over Schur complements —
one 6×6 solve per step. Measured: 4× the length costs **4.6×** the time, where
cubic would cost ~64×. Checked against `torch.linalg.inv` to 1e-13.

**Positive-definiteness is structural, and the obvious way to get it does not
work.** A stiffness matrix that is not positive definite has imaginary normal
modes and negative "variances". The first implementation put the eigenvalue
floor on the diagonal of the Cholesky factor, which bounds the diagonal of
`LLᵀ` and bounds nothing about its eigenvalues — driving the raw factor to −30
gave a minimum eigenvalue of **0.0000** against a nominal floor of 1e-2. The fix
is `LLᵀ + floor·I`, which raises every eigenvalue by exactly `floor`.

**The coupling bound has to be on the spectral norm.** Scaling the
nearest-neighbour block by the geometric mean of the two blocks' *diagonal
entries* bounds the wrong quantity, and the assembled matrix reached a minimum
eigenvalue of **−1.92**. For a block-tridiagonal matrix with
‖C_i‖ ≤ γ·√(λ_min(A_i)·λ_min(A_{i+1})) and γ < ½,
xᵀMx ≥ (1−2γ)·Σ λ_min(A_i)|x_i|², so the matrix is positive definite with
margin. Each block is normalised by its Frobenius norm — which dominates the
spectral norm — and scaled to γ = 0.45.

Outputs: per-step 6×6 stiffness, per-nucleotide fluctuation amplitude,
per-residue **disorder** (supervised by the 46,448 RNA
`_pdbx_unobs_or_zero_occ_residues` records, a flexibility label present in every
deposited structure and used by no RNA structure predictor), and K=3 state
weights kept separate from the structure head's.

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

### 11.2 G2 — the largest generalisation hazard **[v0.2, now measured at scale]**

G2 and G3 are properties of *entries*, not chains, so neither is computable on
RNA3DB or RNASolo — those distribute per-chain extracts with the protein
stripped out. Both have therefore stood on the 180-structure sample since they
were written. The raw corpus closes them: **10,520 of 10,527 acquired entries
contain an RNA polymer chain, carrying 13,348,166 RNA residues** — 43x the
309,197 residues G2 and G3 were originally measured on.

| | v0.1 (n=180) | **raw PDB (10,520 entries)** | |
|---|---|---|---|
| RNA residues in entries containing protein | 98.95% | **97.15%** | holds, −1.8 pts |
| entries containing protein | 88.3% (158/179) | **76.92%** (8,092/10,520) | −11.4 pts |
| isolated RNA, share of residues | 1.05% | **2.83%** | **2.7x larger** |
| isolated RNA, share of entries | 11.7% | **22.34%** (2,350) | **1.9x larger** |
| median protein chains, when present | — | **8** | |

**The hazard is real and it is smaller than v0.1 claimed.** 97.15% of structural
RNA still sits inside a ribonucleoprotein, so the headline stands. But the
isolated-RNA stratum the mitigation has to be evaluated on is **2.7x** the size
v0.1 measured — 2,350 entries and 378 thousand residues, not a rounding error.
That is enough to train on, not merely enough to report.

Not fixable by tuning. Mandatory mitigation, unchanged: **evaluation must be
reported split by isolated vs in-complex**, and the in-complex flag is a model
input.

### 11.2a G3 — ribosome skew, also re-measured **[v0.2]**

| | v0.1 (n=180) | **raw PDB (10,520 entries)** | |
|---|---|---|---|
| ribosome-like entries | 33.5% (60/179) | **20.88%** (2,197) | −12.6 pts |
| RNA residues they contribute | 92.65% | **85.94%** | holds, −6.7 pts |

Same definition as v0.1 (>2,000 RNA residues and >500 protein residues in the
entry). The skew survives — a fifth of the entries still carry six sevenths of
the residues — but it is **6.7 points lower** than the number every
residue-weighted statement in v0.1 was qualified with, and the non-ribosomal
remainder is 1.88M residues rather than 22.7k.

> **Both corrections move in the same direction: the corpus is less
> ribosome-dominated and less protein-bound than the 180-structure sample said.**
> The sample over-stated both skews because BGSU representative sets deliberately
> over-weight large assemblies. Every residue-weighted figure inherited from
> v0.1 is therefore slightly pessimistic about diversity, not optimistic — the
> safe direction, but it should be restated rather than left standing.

### 11.3 Scope statement

PHAROS predicts **single RNA chains**. Coverage of the corpus at a 4,096 context
is **99.95%** of entries **[v0.2, closed on raw PDB]**.

**The longest-chain maximum is now settled rather than provisional.** Scanned
across all 10,520 RNA-bearing PDB entries, the longest single RNA chain in the
world's structural corpus is **4,450 nt (6HRM)**, and exactly **five** entries
exceed 4,096 — 6HRM 4,450, 7UPH 4,438, 4V6X 4,298, 8TOC 4,269, 7LHD 4,217, all
large ribosomal rRNA. These are the same five the derivative corpora found. A
maximum that reproduces exactly when the sample grows from 29,807 derived chains
to the entire archive is not a sample artefact: **4,096 leaves five structures
uncovered and 4,608 covers every RNA chain that has ever been solved.** This is
the one extreme quantile in this project that has stopped moving, and the
context-window decision (D10) can now be closed on it.

It is **not** a whole-transcript or whole-ribosome model, and the raw corpus
makes that limit sharper than v0.1 stated. Total RNA per *entry* reaches
**22,345 residues** (4V4G, a 70S ribosome with two copies in the asymmetric
unit) — **1.95x** the 11,478 measured on 180 structures — and **1,584 of 10,520
entries (15.06%)** hold more than 4,096 RNA residues in total, carrying **72.2%
of all structural RNA** between them. A single-chain model sees the world's RNA
structures one chain at a time; on the residue-weighted majority of entries that
is a genuine loss of context, not a formality. lncRNAs (XIST ~19 knt) and viral
genomes (SARS-CoV-2 ~30 knt) remain an order of magnitude beyond any context
considered.

### 11.4 The curated derivatives are not the archive **[v0.2]**

Every structural number in this project before the raw pass came from RNA3DB or
gRNAde/RNASolo. Matching both corpora's entry identifiers against every
RNA-bearing entry in the PDB:

| | entries |
|---|---|
| RNA3DB | 5,389 |
| gRNAde / RNASolo (2023-11 snapshot) | 6,156 |
| union | 7,943 |
| **RNA-bearing entries in the PDB** | **10,520** |
| **in neither** | **2,581 (24.5%)** |

All 2,581 are already on disk in `data/structures/raw_pdb_entries/`. Their
median longest chain is 22 nt — the uncovered population is dominated by small
crystallographic oligos, which is presumably why the derivative pipelines
dropped them — but the tail is not small, and it points at exactly what this
corpus is shortest of:

| | count |
|---|---|
| RNA residues held | 1,372,786 (10.3% of all structural RNA) |
| entries with a chain ≥ 64 nt | 693 |
| **entries with a chain inside the 64–3,000 training window** | **608** |
| **protein-free entries** | **925**, of which 159 have a chain ≥ 64 nt |
| ribosome-sized (>2,000 RNA residues) | 237 |

Two reasons this is not a housekeeping item.

**It contains the extremes.** §7.3 shows the top 30 chains in the archive by
effective `c` are all in this uncovered set, and that the maximum over the
*covered* entries is 21.14 — precisely the published derivative figure. A
parameter fitted to a derivative corpus inherits that corpus's snapshot date,
and the 9T series that now tops the distribution was deposited 2025-10-21,
during this project.

**It contains the stratum we are shortest of.** D21 measures isolated RNA at
2.83% of residues and makes it the stratum the mandatory G2 evaluation split
runs on. 925 of these entries are protein-free — clean, high-resolution,
small-molecule RNA of exactly the kind ribosomal cryo-EM does not provide.

Four entries appear in a derivative but not in the archive; they are obsoleted
depositions, not a gap.

---

## 12. Training

### 12.1 Curriculum

1. **Pretrain** on sequence (MLM + span masking), 5B → 25B tokens staged.
   **Implemented [v0.2]:** `scripts/pretrain_mlm.py`, over the 48-shard /
   12M-sequence elDORS parquet. Five symbols, not twenty-five (D6) — elDORS is
   pre-normalised to ACGUN, and a wider input vocabulary is dead embedding rows
   and dead softmax mass, which is the NucleicBERT vocab-25 mistake. Spans are
   geometric with mean 3 rather than single tokens: a helix is locally periodic,
   so a single masked base is fillable from its neighbours without learning
   anything about structure. Chemistry is available here too — every dim but the
   shifted-pKa flag is computable from sequence — so stage 1 trains the same
   projection stage 5 uses rather than leaving it cold.

   > **The tied MLM head needed the embedding init fixed.** The output
   > projection is tied to the token embedding, so logits are `h · Wᵀ` over `d`
   > dims; with `nn.Embedding`'s default N(0,1) they inherit a standard
   > deviation of ~√d, and the initial masked-token loss measured **39.19**
   > against the **log 13 = 2.565** a fresh model should start from. The run
   > would have opened by unlearning its own initialisation. With std-0.02 init
   > it starts at 2.43–2.53, and `test_pharos.py` pins that at two widths.
2. **2D** on bpRNA + pdb_hunter dot-bracket. **Implemented [v0.2]:**
   `scripts/train_sequence_stages.py`, bpRNA-SPOT, 10,934 train sequences.
3. **Probing** on Ribonanza — **335,616 profiles acquired** (2A3_MaP and
   DMS_MaP, 206 reactivity positions, ~177 nt sequences): **499× the 673 clean
   3D sequences**, and the largest labelled channel the model will see.
   **Implemented [v0.2]**, co-trained with stage 2 rather than sequenced after
   it — the curriculum orders them because the representation matures in that
   order, not because the losses conflict.

   > **Reactivity is a masked target, and the mask is most of it.** Profiles are
   > 206 columns against sequences of median 177 nt, and positions outside the
   > read-out window are `NaN`. Measured over 4,096 rows: **only 48.8% of
   > reactivity cells are finite**. A zero in the other 51.2% is a claim that
   > the base is unreactive, which is a different statement from "not
   > measured", and training on it would teach the model that the ends of every
   > construct are protected. Every reactivity loss is masked to the finite
   > entries.

   Because probing is 31× the size of the 2D set, the two losses carry explicit
   weights (0.5 / 1.0). Summed unweighted, 2D would dominate the gradient by
   being read more often per epoch rather than by carrying more information.
4. **Physics** terms active throughout; they are closed-form and need no stage.
5. **3D** last and briefest, on 673 clean sequences, with the ensemble head.
6. **Function** heads (fitness, splicing) as auxiliary tasks, co-trained.

### 12.1a Stage 5 trained — and the two splits disagree by 8x **[v0.2, measured]**

PHAROS-Small, 239.6M total / 63.4M active, 8 epochs over the 3D set. The
headline is not the absolute numbers, which are weak; it is the **gap between
the two evaluation splits D25 insists are never averaged**:

| | `test` (family-disjoint) | `test_ribosomal` (entry-disjoint only) |
|---|---|---|
| Mg²⁺ base rate | 1.42% | 2.72% |
| Mg²⁺ average precision | 0.0257 | **0.1089** |
| **Mg²⁺ lift over chance** | **1.81×** | **4.0×** |
| **rigidity Pearson r** | **0.049** (n=53,673) | **0.399** (n=36,076) |

**Rigidity correlates at 0.399 on entries whose homologues are in training and
at 0.049 on genuinely unseen folds — an eight-fold difference.** A single
blended figure would have landed near 0.2 and described neither population.
This is the generalisation hazard of §11.2 and D25, measured rather than
argued, and it is the reason the protocol reports the two separately.

The absolute numbers are honest and weak: a 1.81× lift on Mg²⁺ sites and
r = 0.049 on rigidity are barely-there signals on unseen folds, from a model
trained for 8 epochs on 7,653 chains with no pretrained initialisation. Stage 5
is the *last and briefest* stage of §12.1 by design, and it was run here on a
randomly initialised trunk because stages 1–3 have not been run. What it
establishes is that the pipeline trains end to end and that the evaluation
protocol discriminates; it does not establish that the heads work.

> **`mg_precision_at_calibrated_thr` is 0.0 everywhere.** The model never
> pushes a logit past `log(pos_weight)`, so it is *under*-confident rather than
> over-confident — which is why average precision, not a thresholded precision,
> is the metric reported.

### 12.2 Budget **[v0.1 arithmetic, verified exact]**

**25B tokens**, not 323B. At 61M active, 323B is 5,295 tok/param = 265×
Chinchilla, the worst corner of the quantisation-degradation law, for no
information gain — RNA entropy is **2.0165 bits/nt [v0.2]**, so the corpus is
redundant rather than rich. 25B gives 410 tok/param ≈ 20× Chinchilla.

**BF16 first, FP8 on Hopper, not 4-bit.** A100 has no FP8 tensor cores, so
"~79 A100-hours" mixed an Ampere baseline with a Hopper-only lever. Honest
figures: **~78 h on A100 bf16, ~12 h on H100 fp8**.

### 12.3 Quality weighting — the corpus is saturated, so weight what you have

3D examples cannot be added (§1.1). They can be **weighted**. 10,073 entries
carry a MolProbity clashscore, median **7.52**, p90 **24.76**:

| clashscore | entries | training weight |
|---|---|---|
| 0–10 (good at any resolution) | **6,365** | 1.00 |
| 10–20 | 2,304 | 0.80 |
| 20–40 | 902 | 0.55 |
| > 40 (poor) | **490** | 0.30 |

A model fit on 673 clean sequences should not treat a clashscore-60 structure
the same as a clashscore-2 one. The weights are a stated starting point, not a
measured optimum. Unscored entries take the 0.80 band.

### 12.4 Splits must be family-disjoint, and now can be

A random split leaks. The corpus is rRNA-dominated — G3 measures **85.94%** of
residues as ribosomal (§11.2a), and the five most common Rfam families in the
pdb_hunter index are `LSU_rRNA_bacteria`, `SSU_rRNA_bacteria`,
`LSU_rRNA_eukarya`, `SSU_rRNA_eukarya`, `tRNA` — so a random split puts close
homologues of the test set into training and reports a number that means
nothing.

**6,316 entries carry an Rfam family label across 585 families**, which makes a
family-disjoint split constructible for the first time.

#### But not at 90/5/5 — that split does not exist **[v0.2, measured on the built set]**

Building the set (16,604 chains, 13,172,991 residues, 63,298,800 contacts) made
the obstruction concrete. Four families are **84.4%** of all structural RNA:

| family | residues | share |
|---|---|---|
| SSU_rRNA_bacteria | 3,678,386 | 27.9% |
| LSU_rRNA_bacteria | 3,579,582 | 27.2% |
| LSU_rRNA_eukarya | 2,278,889 | 17.3% |
| SSU_rRNA_eukarya | 1,584,595 | 12.0% |

A 5% quota is 658,650 residues, so the *smallest* of the four is 2.4× an entire
held-out bucket. Keeping a family whole means placing it somewhere, and wherever
it goes the target fractions are gone — a balanced greedy packing returned
55/27/17. **The protocol is therefore two splits, not one** (D25):

| split | chains | residues | disjointness | what it measures |
|---|---|---|---|---|
| train | 11,723 | 10,093,574 | — | — |
| val | 1,189 | 410,303 | **family** | generalisation to unseen folds |
| test | 1,450 | 410,303 | **family** | same, held back |
| `test_ribosomal` | 2,242 | 2,258,811 | entry only | the corpus bulk, homolog-rich |

`test_ribosomal` holds out whole *entries* from the four giant families.
Homologues of them are in training by necessity, so it carries its own name and
is **never averaged with `test`** — a single blended number would be 84% a
homolog-leaking measurement wearing a family-disjoint label.

Combined with the mandatory isolated-vs-in-complex stratification (§11.2) and
the blind sets (CASP15/16, RNA-Puzzles, never trained on), the protocol is:

1. hold out whole **Rfam families** wherever a family is small enough to hold
   out, and say so explicitly wherever one is not;
2. report **isolated vs in-complex** separately;
3. report `test_ribosomal` separately from `test`;
4. report blind-set performance separately again.

### 12.5 Optimiser

Muon (2× lever). Keep embeddings, norms, router and attention softmax in higher
precision regardless — this is what DeepSeek's own FP8 recipe does.

---

## 13. Open actions

| # | Action | Blocks |
|---|---|---|
| ~~1~~ | ~~Re-derive `target_c` on raw PDB entries~~ | **CLOSED** — max 23.30 on 14,106 chains, 24 confirmed (§7.3) |
| ~~2~~ | ~~Acquire RMDB titration ladders~~ | **CLOSED** — 24 Mg²⁺ ladders, 527 points, all crossing sub-mM (§6.4) |
| 3 | Complete Ribonanza — 335,616 of 2.1M acquired | head 7 coverage. **Confirmed blocked on Kaggle credentials**: the public mirrors are the same file. `multimolecule/ribonanza` is gated; `TerminatorJ/RNA_chemical_ribonanza` is public but is `train_data_QUICK_START.csv`, 550,743,596 bytes / 335,616 rows — byte-for-byte what we already hold. The remaining ~1.76M profiles exist only behind the competition login. |
| 4 | Train the block-detection scorer | §7.4, the largest unvalidated assumption |
| ~~5~~ | ~~Re-measure G2/G3 on raw whole entries~~ | **CLOSED** — 97.15% / 85.94% on 10,520 entries (§11.2, §11.2a) |
| ~~6~~ | ~~Re-derive every remaining max/min at scale~~ | **CLOSED** — the last load-bearing one was §7.1's flat-top-K result, re-derived on 2,994 chains and corrected (it compared two different populations) |
| ~~7~~ | ~~Restore sample structures to the server~~ | **CLOSED** — 8,043 present; all six test suites run under `verify_claims.py` |
| ~~8~~ | ~~Ingest the 2,581 entries no derivative covers~~ | **CLOSED** — `build_dataset.py` reads raw entries directly, so all 10,520 are in the built set (§12.4) |
| ~~9~~ | ~~Extract the disorder labels~~ | **CLOSED** — `residue_labels()` reads them, along with Mg sites, B-factors and `N_struct` (§9 note below) |

---

## 14. Provenance

Every **[v0.2]** number traces to a script and a JSON under
`data/samples/analysis/`:

```
recheck_block_sparsity_fullcorpus.py  -> block_sparsity_fullcorpus.json
recheck_contact_tails_fullcorpus.py   -> contact_tails_fullcorpus.json
recheck_ions_rigidity_rawpdb.py       -> ions_rigidity_rawpdb.json
acquire_raw_pdb_entries.py            -> raw_pdb_manifest.json
recheck_targetc_g2g3_rawpdb.py        -> targetc_g2g3_rawpdb.json        (D9/D23)
                                      -> targetc_g2g3_rawpdb_chains.json (per chain)
entry_composition_rawpdb.py           -> entry_composition_rawpdb.json   (G1/G2/G3)
                                      -> entry_composition_rawpdb_table.json
derivative_coverage.py                -> derivative_coverage.json        (§11.4)
                                      -> uncovered_entries.json          (the 2,581)
modification_census.py                -> modification_census.json        (§4.1)
resolve_ccd_parents.py                -> ccd_parents.json                (§4.1)
```

`scripts/sampling/verify_claims.py` re-derives every one of them from those
JSONs and fails the build on drift; it currently pins **283** checks. Counting
rules for entries and chains live once, in
`src/pharos/data/mmcif_entities.py`, and `test_mmcif_entities.py` asserts the
entry counter and the geometry resolver agree on every sampled entry — the two
defects this revision found (C15, C16) were both a second definition drifting
from that one.

Full comparison against v0.1 in `plans/rigor-recheck/fullcorpus-validation.md`.
