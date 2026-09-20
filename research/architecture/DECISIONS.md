# PHAROS — Decided Parameters and Formats (cycle 3)

> Every decision here traces to a measurement in `data/samples/analysis/` or to
> a source verified by search this session. Where a prior decision is overturned
> it says so.

## The conflict that forced these decisions

The design simultaneously wanted three things that **cannot all hold**:

| | Want | From |
|---|---|---|
| (a) | the **smallest viable model** (61M active) | cycle-2 right-sizing |
| (b) | the **full 323B-token corpus** (one epoch of elDORS) | data availability |
| (c) | **4-bit** training/deployment | cycle-3 precision work |

**"Low-Bit Quantization Favors Undertrained LLMs"** (ACL 2025, arXiv 2411.17691,
1500+ checkpoints) finds quantization-induced degradation (QiD) is **worst for
small models trained on many tokens**, and states directly that *"smaller models
require higher precision at high training token counts."*

PHAROS-Small at 61M active × 323B tokens is **5,295 tokens/parameter = 265×
Chinchilla-optimal**. That is the worst-case corner of exactly that law.

| Model | active | tok/param @323B | × Chinchilla | 4-bit verdict |
|---|---|---|---|---|
| PHAROS-Micro | 12M | 26,917 | 1,346× | **worst case — avoid** |
| PHAROS-Mini | 30M | 10,767 | 538× | **worst case — avoid** |
| **PHAROS-Small** | **61M** | **5,295** | **265×** | **worst case — avoid** |
| Base-v2 | 269M | 1,201 | 60× | risky — validate QiD first |

RNA makes this **worse, not better**. Measured sequence entropy is **2.0167
bits/nt** against ~11 bits for an English token, so 323B RNA tokens carry
~0.1 TB of information where 323B LM tokens carry ~0.4 TB. The effective
over-training is larger than the raw ratio suggests.

---

## D1 — Model size: **KEEP PHAROS-Small (61M active, 149M total)**

Unchanged. The right-sizing argument survives: the 3D task carries ~2.8 MB of
information and is data-limited, not capacity-limited. Ladder retained
(Micro/Mini/Small/Base-v2) with Small as default.

## D2 — Token budget: **CUT 323B → staged 25B, stop when metrics flatten** [REVERSED]

**Overturns** the earlier "323B tokens, deliberate over-training is standard".
That defence was argued for **269M** active (1,201 tok/param). At **61M** it
becomes 5,295 tok/param and puts the model in the QiD worst case for no
information gain — the corpus is redundant, not rich.

- 25B tokens → **410 tok/param ≈ 20× Chinchilla**. Still deliberately
  over-trained for inference efficiency, without entering the degradation corner.
- The staged schedule (5B/25B/100B/323B) already exists; the change is that
  **25B is now the expected stopping point, not a waypoint.**
- The 151-nt read down-weighting is promoted from a cost lever to a
  **redundancy-removal requirement**: 9 of 20 chunks have median length exactly
  151 nt and carry 16.2% of nucleotide mass with little long-range signal.

## D3 — Training format: **BF16 first, FP8 on Hopper. NOT 4-bit.** [REVERSED]

| Format | Decision | Reason |
|---|---|---|
| **BF16** | **train here** | precision is not the constraint (139× over-provisioned); a correct baseline matters more than speed |
| **FP8** (Hopper+) | **adopt after baseline** | `<0.25%` loss error vs BF16, validated at DeepSeek V2/V2-Lite scale (~1T tokens), *not* at 671B |
| **NVFP4** (Blackwell) | **do NOT adopt for this configuration** | the pretraining recipe was validated at **8B/1T ≈ 125 tok/param**. PHAROS-Small is **5,295** tok/param — a different regime entirely. The "no measurable loss" result does not transfer. |
| **MXFP4** | no | needs ~36% more tokens than NVFP4 for equal loss |
| **NF4** | **never** | W4A16 with a **frozen** 4-bit base; gradients reach only LoRA adapters. It is a finetuning/inference format. PHAROS trains from scratch. |

Keep in higher precision regardless: **embeddings, norms, router, attention
softmax** — this is what DeepSeek's own FP8 recipe does.

## D4 — Quantization-aware training: **defer, and only for deployment**

QAT is a *deployment* technique here, not a training one. It is worth it only if
4-bit inference is actually required.

- QAT beats PTQ substantially at INT4; LLM-QAT reaches within 1–2% of FP.
- But those results are on **large** models. At 61M active and 265× Chinchilla
  the QiD law predicts the largest degradation, so the recovery task is hardest
  exactly here.
- **Decision**: no QAT in stage 1. If 4-bit inference is later required, run QAT
  (not PTQ) **and measure QiD at the actual (size, token) point** rather than
  assuming a published number transfers.
- Straight-through estimator caveat recorded: gradients after QAT tend toward
  zero, biasing STE and reducing convergence stability.

## D5 — Where the freed budget goes: **attributes, not mantissa** [user's insight, corrected]

Measured: a fully attributed nucleotide carries **58.9 bits**; a 512-dim bf16
token holds **8,192 bits** — **139× over-provisioned**. So `d_model` is *compute*
space, not storage.

**The correct form of "more attributes instead of more precision"**:
attributes enter as **input features through one projection**. They do **not**
justify a wider `d_model`, because training FLOPs go as `6·N_active` and
widening `d` is quadratic in the FFN. Attributes are nearly free; width is not.

At inference, only **26.0 bits of 58.9** are available for an arbitrary sequence
(identity, position, local GC, recycled pairing state, ionic condition,
in-complex flag). The rest are **supervision targets, not inputs** — feeding
B-factor or Mg-distance in would leak the answer. `data/attributes.py` keeps the
two sets physically separate.

## D6 — Vocabulary: **two vocabularies, one model** [changed]

Finding G7: **8.90%** of structural residues fall outside `{A,C,G,U}`.

- **Pretraining**: 5 symbols. elDORS is pre-normalised; widening adds dead
  tokens — NucleicBERT's vocab-25 mistake.
- **Structural**: 5 + DNA + inosine + UNK + a learned `MOD` embedding keyed by
  mmCIF `comp_id`. Until built, the honest statement is that modified residues
  map to `N` and their geometry is not predicted.

> **The premise is source-dependent, measured Sep 2026.** The 8.90% figure comes
> from raw BGSU mmCIFs. The structural data actually on disk — RNA3DB and
> RNASolo — is **pre-normalised**: over 1,800 sampled structures / 1.26M polymer
> residues it carries **0.025%** non-ACGU (top species `N`, 245 of 310
> instances), a **42x** under-representation. So the second vocabulary is
> required only if training consumes **raw PDB entries**; against RNA3DB/RNASolo
> a 5-symbol vocabulary is very nearly exact. Decide the vocabulary and the data
> source together, not separately.
>
> Related, and worth splitting: `N` means two different things. In sequences
> (elDORS) it is an **ambiguous base call** — measured **0.203%** of nucleotides
> pooled over 238M nt, and strongly chunk-dependent (0.51% in chunk 001 vs
> 0.0077% in chunk 006), against the ~0.1% the data README states. In structures
> it is a residue whose **geometry is fully modelled but whose base identity the
> depositor left unassigned** — e.g. `8vvt_ZA` residue 1248 carries a complete
> ribose including O2' plus a partial base ring. The first has no identity and
> no geometry; the second has full geometry and no identity, so it can train the
> geometry heads normally and makes base-identity imputation a free auxiliary
> task. One shared token conflates them.

## D7 — Cost, hardware-explicit [defect #15 corrected]

A100 (Ampere) has **no FP8 tensor cores**, so the published "~79 A100-hours"
mixed an Ampere baseline with a Hopper-only lever and existed on no real machine.

At the **revised 25B-token budget**, PHAROS-Small:

| Hardware + format | GPU-hours (with Muon + read down-weighting, 2.38×) |
|---|---|
| A100 bf16 | **~78** |
| H100 fp8 | **~12** |

> **Corrected in cycle 4 (defect #17).** These were ~10 h and ~1.6 h before the
> refinement loops were counted. PHAROS-Small runs **8** loops with deep
> supervision at every segment, so compute scales 8×. Two independently written
> cost paths now agree at 78 h.

(323B-token figures, for reference: ~1,011 h A100 bf16.)

**The token-budget cut remains the largest single lever** — 12.9× against FP8's
1.6×, and larger than the whole right-sizing step (1.65× by effective compute) — and it costs no accuracy, because the tokens being dropped
are redundant rather than informative.

---

## Summary of reversals

| # | Was | Now | Why |
|---|---|---|---|
| REV-3 | 323B tokens, over-training is standard | **25B staged, stop on plateau** | argued at 269M; at 61M it is 265× Chinchilla and the QiD worst case |
| REV-4 | FP8 1.60× lever on "A100-hours" | **hardware-explicit; A100 has no FP8** | defect #15 — not a real unit |
| REV-5 | 4-bit as a goal (NVFP4/NF4) | **BF16 → FP8; no 4-bit at this size** | NF4 cannot train from scratch; NVFP4 validated at 125 tok/param, we are at 5,295 |

---

# v0.2 decisions (Sep 20, 2026) — set against the corpus, not a 180-structure sample

> Context: ten v0.1 claims were re-derived on 29,807 chains, 8,041 raw PDB
> entries, 10,424 pdb_hunter entries and 238M nt. Every central estimate held;
> every extreme quantile failed. Full comparison in
> `plans/rigor-recheck/fullcorpus-validation.md`; specification in
> `research/architecture/ARCHITECTURE_v0.2.md`.

## D8 — Parameters may not be set from small-sample maxima **[new, process]**

Four of four extreme quantiles measured on n=180 failed at scale, while six of
six means held. `target_c` and the context window had both been set from
maxima.

**Rule:** no parameter is set from an extreme quantile measured on fewer than
10³ structures. Where the corpus cannot supply that, set it from a mean plus an
explicit safety factor and state the factor.

**Enforcement suggestion:** `verify_claims.py` guards 155 *values*; it does not
guard *provenance*. A guard recording "this number is a maximum from n=180"
would have caught all four before a human did.

## D9 — `target_c`: **20 → 24** [REVERSED]

Measured maximum on 20,266 chains is **21.14**, breaching 20 by 5.7%. 24 clears
it by 12% and costs proportionally more only in L3 refinement.

Interim, not final: the measurement used derivatives carrying 0.025% modified
residues against raw PDB's 1.005%, a 40x under-representation of the quantity
that drives the tail. The pair track must also **handle overflow** rather than
assume the budget suffices.

> **CLOSED by D23.** Re-measured on 14,106 chains from the raw archive, with
> modified residues present at 19x the derivative rate: max **23.30**, zero
> chains over 24. The hedge was right and 24 absorbed it.

## D10 — Context window: coverage restated, not silently raised

8 chain files / 5 unique structures exceed 4,096 (6HRM 4,450; 7UPH; 4V6X; 8TOC;
7LHD — all large ribosomal rRNA). v0.1's table read 0.

**Decision:** keep 4,096 and state coverage as **99.97%**, or raise to 4,608 if
whole-ribosome chains matter. Do not claim 100%.

> **RESOLVED by D20.** The choice was left open because the longest-chain
> maximum had moved every time the sample grew. It has now been measured over
> the entire archive and it stops at 4,450 nt. **4,608.**

## D11 — `N` is two tokens [changed]

`N_seq` (ambiguous base call, 0.203% of nt, no geometry) and `N_struct`
(identity unassigned, 0.025% of residues, **full ribose modelled**). The second
trains geometry heads normally and makes base-identity recovery a free
auxiliary task. One token discards that.

## D12 — Rigidity head trains on X-ray B-factors only **[new, measured]**

The Mg-rigidity gradient is **1.523 sigma, monotonic, 1,535 X-ray structures**.
On 1,885 cryo-EM structures it is **0.934 sigma and non-monotonic**. v0.1
excluded cryo-EM on an assumption; it is now a measurement, and pooling would
dilute the signal by a third.

## D13 — Chemistry enters as a 24-dim per-residue vector **[new]**

H-bond capacity per edge, pKa, sugar-pucker propensity, stacking priors,
backbone charge state, modification class, local GC. Through **one projection**,
not a wider `d_model`: a nucleotide carries 58.9 bits against a 512-dim bf16
token's 8,192, so width is compute, not storage, and FFN cost is quadratic in d.

Structural observables (Saenger/LW class, B-factor, Mg distance, reactivity)
remain **targets, never inputs**.

## D14 — Four heads added: fitness, splicing, base-identity, and probing at scale

All sit on data already on disk and unused by v0.1: fitness 620,372 records,
splicing 8.6 GB across 147 species, base-identity free from `N_struct`, and
**335,616 Ribonanza reactivity profiles acquired for v0.2** — 499x the 673 clean
3D sequences.

## D15 — 3D is the smallest channel, by design [reframing]

RCSB returns **10,399** entries containing an RNA polymer entity; pdb_hunter
held 10,424. **The 3D corpus is saturated** — 673 unique sequences at <= 2.5 A
is the world supply and no acquisition will grow it. Every other component
exists to move learning off a channel that cannot grow.

## D16 — Weight 3D examples by structure quality **[new]**

The 3D corpus is saturated (D15), so examples cannot be added. They can be
weighted. `pdb_hunter` supplies **10,073 MolProbity clashscores**, median 7.52,
p90 24.76:

| clashscore | entries | weight |
|---|---|---|
| 0–10 | 6,365 | 1.00 |
| 10–20 | 2,304 | 0.80 |
| 20–40 | 902 | 0.55 |
| > 40 | 490 | 0.30 |

Fitting on 673 clean sequences and treating a clashscore-60 structure like a
clashscore-2 one wastes the little supervision there is. Weights are a stated
starting point, not a measured optimum; unscored entries take 0.80.

## D17 — Splits are family-disjoint, not random **[new]**

The corpus is rRNA-dominated: G3 puts 92.65% of residues in ribosomes, and the
five commonest Rfam families in the pdb_hunter index are LSU/SSU rRNA
(bacteria and eukarya) and tRNA. A random split therefore places close
homologues of the test set in training, and reports a meaningless number.

**6,316 entries carry an Rfam family across 585 families**, which makes a
family-disjoint split constructible for the first time. Protocol: hold out whole
families; report isolated vs in-complex separately (§11.2); report the blind
sets separately again.

## D18 — MARS is not acquired **[new, measured]**

NucleicBERT's pretraining corpus, 1.73B sequences / 1,571 GB / 413 GB to
download. A 40 MB HTTP range-probe of its first tarball shows 41.4% coding
mRNA, 16.9% genomic DNA, 10.9% ncRNA-like -- and of that ncRNA, **94.6% is rRNA
or tRNA** gene and amplicon records. Projected diverse structured ncRNA: ~3M, a
0.17% yield, for 413 GB against 290 GB of free disk. We already hold 46.2M
curated RNAcentral ncRNA, more than the 30M NucleicBERT extracted from MARS.

It would also deepen the ribosome skew D17 exists to control. MARS is the right
corpus for homology search, which is what it was built for; it is the wrong one
for a structure model. `plans/14-mars-acquisition-assessment.md`.

## D19 — pdb_hunter is referenced, not copied **[new]**

`pdb_hunter/RNA_Database` is 154 GB and already organised per entry.
`scripts/integrate_pdb_hunter.py` indexes it into
`data/catalog/pdb_hunter_index.json` with paths into that tree. Copying would
duplicate 154 GB on a volume with 290 GB free, for no gain.

## D20 — Context window: **4,608, and the question is now closed** [D10 resolved]

D10 left this open: keep 4,096 and state 99.97% coverage, or raise to 4,608. It
was left open because the longest-chain maximum had moved every time the sample
grew — 3,764 nt on 180 structures, 4,450 nt on 29,807 derived chains — and there
was no reason to think 4,450 was the end of it.

It is the end of it. Scanning **every RNA polymer chain in all 10,520
RNA-bearing PDB entries** — the whole archive, not a derivative of it — the
longest chain is **4,450 nt (6HRM)** and exactly **five** entries exceed 4,096:
6HRM 4,450, 7UPH 4,438, 4V6X 4,298, 8TOC 4,269, 7LHD 4,217, all large ribosomal
rRNA. These are the same five the derivative corpora found, at the same lengths.

**Decision: 4,608.** It covers every RNA chain that has ever been solved, with
158 nt of headroom, and it is the one extreme quantile in this project that has
stopped moving under a 58x increase in sample size. Choosing 4,096 to save 11%
of the position budget would leave the five largest rRNA chains — the hardest
and most-studied folds in the corpus — permanently outside the model.

This does not violate D8. D8 forbids setting a parameter from an extreme
quantile measured on fewer than 10³ structures; this one is measured on 10,520,
and it is not an estimate of a maximum but the maximum itself, over a closed
population. There is no larger sample to be had.

## D21 — G2 and G3 restated downward **[new, measured on raw entries]**

Both are entry-level properties, so neither was computable on RNA3DB or RNASolo,
and both stood on 180 BGSU structures until the raw corpus was acquired.
Measured on 10,520 entries / 13.3M RNA residues, 43x the original basis:

| | v0.1 (n=180) | raw PDB | |
|---|---|---|---|
| **G2** RNA residues in entries with protein | 98.95% | **97.15%** | holds, −1.8 pts |
| **G2** isolated RNA, share of residues | 1.05% | **2.83%** | **2.7x larger** |
| **G3** residues from ribosome-like entries | 92.65% | **85.94%** | holds, −6.7 pts |
| **G3** ribosome-like entries | 33.5% | **20.88%** | −12.6 pts |

**Decision: keep both mitigations, restate both numbers.** The mandatory
isolated-vs-in-complex evaluation split (§11.2) and the family-disjoint splits
(D17) were justified by these skews and both skews survive. What changes is the
size of the isolated-RNA stratum: 2,350 entries and 378k residues, **2.7x** what
v0.1 measured. That is a stratum large enough to *train* on, not merely large
enough to report a number for, which makes the D17 weighting a live lever rather
than a gesture.

Both corrections move the same way — the corpus is **less** ribosome-dominated
and **less** protein-bound than the sample said — because BGSU representative
sets deliberately over-weight large assemblies. Every residue-weighted v0.1
figure is therefore pessimistic about diversity rather than optimistic. That is
the safe direction to have been wrong in, and it is still wrong.

## D22 — G1's entry maximum is the fifth tail failure; its chain maximum is the first tail success

| | v0.1 (n=180) | raw PDB (10,520 entries) | |
|---|---|---|---|
| total RNA residues per **entry**, max | 11,478 | **22,345** (4V4G) | **1.95x breach** |
| entries over 4,096 RNA residues | 24.6% (44/179) | **15.06%** (1,584) | over-stated |
| longest single **chain** | 3,764 nt | **4,450 nt** (6HRM) | reproduces exactly |

The entry maximum is the **fifth** extreme quantile from the 180-structure
sample to fail at scale, joining `target_c`, contacts/nt, the longest chain and
the Mg:K ratio. D8 predicted it.

The chain maximum is the first to *hold* — and it holds because by the time it
was measured the second time, the sample was already 29,807 chains rather than
180. That is D8 working as intended, not an exception to it.

**Consequence for scope (§11.3):** 1,584 entries hold more than 4,096 RNA
residues in total and carry **72.2% of all structural RNA** between them. A
single-chain model is the right first target, but on the residue-weighted
majority of the archive it is predicting a part while the crystal contains the
whole. That belongs in the scope statement, not in a footnote.

## Defect C15 — a private counter keyed on the wrong chain id

`entity_poly_types()` returns **auth** chain ids. Two analysis scripts had each
grown a private `entry_composition()` that joined to it on **`label_asym_id`**.
Where an entry's two labellings differ — overwhelmingly older depositions, e.g.
1ARJ, whose sole RNA chain is `label A` / `auth N` — every atom row missed its
entity and the entry reported **zero polymer residues of any kind**. It affected
**3,254 of 10,527** raw entries, 686 of them reporting no polymer at all, and it
deflated G2 to 89.2% and G3 to 33.6% before it was caught.

This is **defect #22 recurring**: two scripts rolling their own definition of a
thing the canonical resolver already defines. The fix is the same fix — counting
now lives once, in `pharos.data.mmcif_entities.entry_composition`, beside the
resolver whose keys it has to match, and test property 10 asserts that its RNA
counts equal `rna_chain_coords`' on every sampled entry, with 1ARJ pinned as the
regression case.

**What caught it:** not a test. The scan reported 3,254 RNA-free entries in a
corpus selected for containing RNA, and 1ARJ — a textbook TAR RNA structure —
was in the list. The number was implausible on its face. Property 10 exists so
that the next one does not need to be.


## D23 — `target_c` stays at 24, and D9 closes **[measured on the raw archive]**

D9 raised the budget 20 -> 24 and said so provisionally, because the 21.14 that
forced the raise came from corpora carrying **0.025%** modified residues against
raw PDB's **1.005%** — a 40x under-representation of the quantity that drives
the tail. Re-measured on 14,106 chains from 10,520 raw entries, at **0.476%**
modified residues:

| | v0.1 (n=180) | derivatives (20,266) | **raw (14,106)** |
|---|---|---|---|
| max effective c | 19.04 | 21.14 | **23.30** |
| p99.9 | — | 19.52 | **23.08** |
| chains over 20 | 0 | 1 | **34** (0.24%) |
| **chains over 24** | — | — | **0** |

**Decision: 24, confirmed, not raised.** The maximum did rise and 24 still
contains it.

**But D9's stated mechanism was wrong, and the right one matters more.** D9
blamed modified residues. Splitting the raw chains by whether their entry is in
RNA3DB or gRNAde/RNASolo:

| | chains | max effective c |
|---|---|---|
| entries the derivatives cover | 12,573 | **21.14** |
| entries they do not | 1,533 | **23.30** |

21.14 is *exactly* the published derivative figure. The derivative measurement
was right for the population it had; the entire difference is **coverage**. The
top 30 chains in the whole archive are one deposition campaign — the 9T-series
E. coli ribosome PTC focused refinements, ~596 nt of 23S rRNA at 2.28 A in
methyltransferase-knockout strains — and all 30 are entries no derivative holds.
9T1E was deposited **2025-10-21**, after RNASolo's 2023-11 snapshot. The first
chain from any other structure is rank 31, at 21.14.

Modification enrichment is real but secondary: 1.98% in the top 34 against the
corpus's 0.48%, but only **1.9x** against a length-matched control. These are
methylation-variant structures, so they are modification-rich by design. That
describes what sits at the top; it does not explain why the derivatives missed
it.

**The lesson generalises past `target_c`: a parameter fitted to a curated
derivative inherits that derivative's snapshot date, and the extremes live in
what it omits.**

**Two things are true at once and both belong in the record.** Zero of 14,106
chains breach the budget, over the complete archive. And the headroom is
**2.9%**, not the 12% §7.3 claimed, against a statistic that has risen at every
single re-measurement: 19.04 -> 21.14 -> 23.30, about +2.1 per pass. Unlike the
chain-length maximum (D20), this is not a closed population — it is a property
of whatever is deposited next.

So the **overflow path is load-bearing**: truncate to budget, record the miss,
and treat the miss-rate as a training metric rather than a debug counter. 24 is
the right number *given* that path exists. Without it, 24 would be a bet.

Raising to 28 would buy headroom for 34 chains in 14,106 and cost proportionally
in L3 refinement on all of them. Not worth it.

**The effective sample size at the top is 1, not 14,106.** Ranks 1-30 are the
same ~596-nt molecule solved thirty times in one campaign. Ranks 31-34 are 7PAS
(21.14), 9Z80 (20.97 -- a 119-nt chain in which **all 119 residues are
modified**), and 4V6O/4V6P (20.11/20.01). **Four independent molecules exceed 20
in the entire Protein Data Bank.** A budget validated against that tail is
validated against very little; that is the argument for the overflow path, and
against reading 23.30 as a ceiling.

## Defect C16 — NMR ensembles were stacked into single residues

`rna_chain_coords()` keyed atoms by `(chain, label_seq_id, ins_code)` and never
read `pdbx_PDB_model_num`. An NMR entry deposits an ensemble — 20 models by
convention — and every model repeats every atom, so all 20 collapsed into one
residue: **1ARJ came back with 424 heavy atoms per residue** against a
nucleotide's real ~21.

Geometry built from that is the **union of the ensemble's contacts**: two
residues count as touching if they touch in *any* model. Contact counts and
`effective_c` inflate for every NMR chain. It also made one worker of the
raw-corpus scan allocate **28 GB** (52 GB peak) and stop making progress, which
is how it was noticed.

**Blast radius, checked rather than assumed:** the four chains that set the
published maxima — 7PAS, 6Q95, 1VY7, 6XHV — are all single-model cryo-EM or
X-ray, so no published tail statistic moves. Sampling 400 files from each
derivative corpus found **0 multi-model files** in either RNA3DB or
gRNAde/RNASolo, so the corpus-scale numbers derived from them are unaffected
too. The defect only ever bit the raw-entry path, which is new.

Fixed: first model only. Asserted by test property 11, which bounds heavy atoms
per residue at 50 — chosen to pass legitimate alternate conformations (4X4T
models 670 A and 669 B altlocs, so a fully dual-occupancy nucleotide is ~42
atoms, and both conformers are real positions ~1 A apart that add no contacts)
while catching 20 stacked models at ~424 with an order of magnitude to spare.

## Performance note — the contact builder was O(atom pairs) in Python

`contact_pairs` iterated `cKDTree.query_pairs` output and accumulated `(i, j)`
tuples in a Python set. On a 1,758-nt rRNA chain (37,455 heavy atoms, millions
of atom pairs within 8 A) that is tens of millions of tuple objects at ~100
bytes of overhead each. Vectorised — map atom indices to residue owners in
numpy, filter `|i-j| >= 4`, dedup with one `np.unique` over a packed key — the
same chain takes **0.4 s and 0.29 GB** instead of 28 GB and no completion. The
vectorised output was checked bit-identical to the original on 26 chains before
the old code was removed.
