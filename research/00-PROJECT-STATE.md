# Project State — RNA MoE Structure-Prediction Architecture

> **Continuation file.** Read this first in any new session. Update at the end of
> every working block. This program spans multiple sessions by design.

## Goal (from user, 2026-09-12)

Design — not yet train — a **Mixture-of-Experts architecture for RNA structure
prediction and contact-map modeling** that is materially stronger than
NucleicBERT and current RNA LMs. Required ingredients:

1. **MoE routing** over heterogeneous input sources (the 25 catalogued datasets).
2. **Motif-awareness**: exploit that certain RNA motifs are locally *rigid*
   (physics-derived structural priors).
3. **Physics/Hamiltonian terms**: ion-mediated effects, Mg²⁺ coordination,
   Manning counterion condensation.
4. **Coevolution**: MSA/DCA signal for spatial proximity in 3D.
5. **Modern attention stack**: interleaved self / linear / cross attention,
   token-level KV caching for fast retrieval over a corpus that is "just data".
6. Optionally **dynamics** (folding pathway / conformational ensemble), not just
   a single static structure.

**Deliverables**: (a) LaTeX research report, (b) very detailed architectural
blueprint diagram (HTML/CSS, LaTeX/TikZ, and/or Mermaid), all re-checked.

## Environment facts (verified 2026-09-12)

- Running on the user's **personal laptop**, NOT the server. Working dir
  `/home/shuvam/codes/sbs-rna`.
- The repo is **catalog-only**: `data/catalog/` (sqlite, 47,471 files indexed)
  and `data/exploration/`. The 256.6 GB bulk corpus is on the server and is
  **absent here**; the original root `/store/shuvam/E-motioner-X-SBS/sbs-rna`
  does not exist on this machine.
- Available: python 3.14.7, torch 2.11.0+cu130, numpy/scipy/pandas/pyarrow/
  sklearn/networkx/matplotlib, pdflatex + xelatex (TeX Live 2026), node 26.8.2.
  **Biopython is MISSING** (parse mmCIF manually or install).
- Network works: RCSB, BGSU, Rfam, RNAcentral all reachable.

## Repo issues found during orientation (not yet fixed)

1. `MANIFEST.json` record counts are capped at 50,000 for elDORS/RNAcentral
   (`build_rna_database.py:38` caps `count_fasta`; exact counts are merged only
   into SQLite at lines 547-556, after the manifest dict was built at 537).
   SQLite is correct; the manifest understates elDORS ~1,300x.
2. `RNADatabase.total_sequences()` sums *all* `n_records` — mixing sequences,
   3D chains, CSV line counts and the 10.1M derived Parquet rows that are
   re-encoded copies of elDORS. Reports 1,380,841,746 vs the true 1,369,926,204.
3. All six scripts hardcode the dead absolute path `/store/shuvam/...`.
   `build_rna_database.py` is destructive (DROP TABLE then rescan) — running it
   here would wipe the good catalog, and the exact counts it re-merges come from
   `seq_counts.txt` files that live in the ignored data tree.
4. README says elDORS median length 245 nt; `data/README.md` says 730. Both are
   real (cross-chunk sample vs chunk 001 alone) but neither states which.

## Key data facts that constrain the architecture

- elDORS chunks are **source-partitioned, not homogeneous**: 9 of 20 chunks have
  median length *exactly* 151 nt (unassembled Illumina reads); chunks 6 and 9 are
  long transcripts (median 2,193-2,389 nt); the rest assembled/medium. GC ranges
  40.7%-61.4%. => corpus weighting and a length-aware router matter.
- Alphabet is only **5 symbols** (A,C,G,T,N), no IUPAC ambiguity. Contrast
  ERNIE-RNA, which uses 11+ degenerate symbols.
- The supervision pyramid: 1.37B sequences (unlabeled) / ~160k 2D structures /
  27,452 3D chains = **6,661 unique sequences** with experimental 3D (0.0005%).
  3D data is extremely rRNA-redundant (5S rRNA alone = 5,976 clusters).
- Only **100 of 4,227 Rfam families** have any 3D representative (2.4%).

## Session log

### Session 1 (2026-09-12)

**Orientation.** Found four repo issues (above). Repo is catalog-only; bulk data absent.

**Data acquired** -> `data/samples/` (gitignored except `analysis/`):
- 180 BGSU non-redundant representative mmCIF structures (<=3.0 A), 344 MB
- 12 Rfam seed alignments (families with 3D representatives)
- 200 RNAcentral sequences
- Two fetch bugs found and fixed: BGSU `/nrlist/release/...` returns HTML not CSV
  (correct path is `/rna3dhub/nrlist/download/current/3.0A/csv`; the script now aborts
  if the feed is HTML), and RNAcentral's API stalls at `page_size=200` — paginate at 100
  via curl, since urllib stalls on that endpoint regardless.

**Prior art** -> `research/prior-art/01-05`. The pivotal finding: ERNIE-RNA (86M) beats
RiNALMo (650M) and nearly doubles cross-family F1 (0.646 vs 0.355) on the strength of one
hand-set 3-valued base-pairing bias. Inductive bias, not scale, is the lever.

**Measurements made** (scripts in `scripts/sampling/`, outputs in `data/samples/analysis/`):

| Finding | Result |
|---|---|
| Contact scaling | contacts/nt saturates 4.4-4.9; density falls 6.52% -> 0.353%. **O(L), not O(L²)** |
| Ion inventory | 17,428 Mg²⁺ vs 1,868 K⁺ (9:1); 83% of inner-sphere coordination to OP1/OP2 |
| Ion-rigidity coupling | monotonic **1.76 sigma** B-factor gradient vs Mg²⁺ distance, 24,623 nt |
| GNRA k-mers (negative) | 0.073 sigma — motif ID needs the interaction graph, not n-grams |
| **Flat top-K recall (negative)** | **0.200 on 500-1200 nt chains; random scorer gets 0.746 mean** |
| Contact separation | median 78 nt on long chains; +/-512 band still misses 19% |
| **Block occupancy** | **1.34% at b=4 on long chains; falls as L grows; effective c = 17.2** |
| **Ionic metadata (R2)** | **~36% coverage; cryo-EM 23.9% structured, X-ray 58.7% free text** |
| **PDB ionic survivorship bias** | **recorded Mg²⁺ spans only 5-15 mM — nobody deposits unfolded RNA** |
| **Coevolution signal** | APC-MI recovers curated base pairs at prec@L/5 **0.670**, rec@L 0.768 |
| **Coevolution depth gate** | **Neff/L >= 1: 0.975 precision; Neff/L < 1: 0.609.** tRNA = 1.000/1.000 |
| **HPT reference impl.** | L=4096 in **0.45 s**, 0.96% of dense; dense **not runnable** past L=1024 here |

**Architecture** -> `research/architecture/ARCHITECTURE.md` (PHAROS v0.1).
The efficiency claim was tested rather than assumed, and **the original flat sparse pair
track failed and was replaced** by a 3-level hierarchical coarse-to-fine track
(2.202% of dense at L=2861, 100% recall ceiling). This is the most important result of
the session: a load-bearing design decision was falsified by its own validation.

**Deliverables produced**:
- `research/report/main.pdf` — 18pp LaTeX report, builds clean (no over/underfull boxes)
- `research/architecture/diagrams/` — 6 Mermaid diagrams, all verified to render
  (mermaid-cli needs `-p puppeteer-config.json` pointing at `/usr/bin/chromium`)
- `research/architecture/blueprint.html` — published Artifact (v2):
  https://claude.ai/code/artifact/1756a459-45c7-4908-9cef-bcde72aab33e

### Session 2 (2026-09-13) — rigor recheck of every empirical claim

State in `plans/rigor-recheck/`. Every headline number was re-derived by an
**independent reimplementation** rather than by re-running the original script
(which would reproduce its own bugs). Verifier: `scripts/sampling/verify_claims.py`.

**Defects found in session-1 work:**

| ID | Defect | Direction |
|---|---|---|
| T16 | **PHYSICS WRONG.** Manning's `b` is the *axial projection*, not the P-P contour distance. Stated b ~ 5.9-7.0 A was wrong and inconsistent with the stated theta. Correct A-RNA: **b = 1.40 A, xi = 5.11, theta = 0.804**. theta ~ 0.76 is the **B-DNA** figure. | claim was wrong |
| T8d | **OVERCLAIM.** The coevolution depth split cannot carry the weight placed on it: exact permutation p = 0.0455 on a *post-hoc* threshold, **Spearman only +0.224** (n=12), deep arm **n=2**, and a *shallow* family (THF, Neff/L=0.279) reaches precision 1.000. "The decisive pattern" / "strongest argument for MoE" **retracted**. | claim overstated |
| T8a/b | `ss_pairs()` omits WUSS pseudoknot brackets (Aa/Bb/Cc/Dd) — **53 of 537 pairs, 9.87%** of ground truth. Deflates precision, so **0.670 is a lower bound**. | conservative |
| DISC-3 | All three deliverables said "X-ray only (31 structures)" and attributed the exclusion to cryo-EM ADP comparability alone. A **>=30-RNA-residue guard** was also applied and went unstated. | under-disclosed |
| DISC-6 | The Mg gradient is measured over **15** X-ray structures (the Mg-containing subset), not 31. 24,623 nt is correct for the 15. | mis-stated sample |

**Claims that strengthened:**
- Mg/rigidity confound **controlled**: partial corr(z_B, Mg dist | density) = **+0.372**
  vs +0.420 raw. The signal is largely *independent* of packing — it does not merely
  restate that folded cores are ordered. Within-structure effect **+1.514 sigma**,
  positive in **4/4** structures, bootstrap 95% CI [+1.256, +1.661].
- 30 A centroid prefilter **provably safe**: bound 22.20 A, observed max 18.59 A,
  **zero** contacts missed.
- `effective_c` naive accounting is **2.9% conservative** (17.61 vs 17.12 strict).
- All parameter/attention/memory arithmetic reproduces **exactly** (910.5M / 382.0M).

**Result**: 1.76 sigma gradient reproduces to 3 dp; report now 22pp, zero
over/underfull boxes; `verify_claims.py` reports ALL CLAIMS REPRODUCE across
ARCHITECTURE.md / main.tex / blueprint.html.

## Next actions (session 2+)

Highest value first:

- [x] ~~Prototype the hierarchical pair track in torch~~ **DONE (structure + cost).**
      `research/architecture/reference/` implements HPT and benchmarks it against a dense
      AF3-style baseline: L=1024 HPT 0.10 s vs dense 8.26 s / 5.15 GB (~83x); dense is
      **not runnable** at L=2048 (12.9 GB predicted) or L=4096 (51.5 GB) on this 14 GB
      machine, while HPT does L=4096 in 0.45 s at 0.96% of dense. Two silent indexing/
      budget bugs found and fixed (see ARCHITECTURE.md §5.4).
- [x] ~~OQ-1: does the >=30-residue exclusion bias the Mg-rigidity gradient?~~
      **CLOSED — NO BIAS.** Recomputing at guards 30/20/15/10/5 and with no guard at
      all moves the span only from **1.760 to 1.757 sigma** (change 0.003), monotonic
      at every threshold. Removing the guard adds 4 structures and 43 nucleotides
      (+0.17%) because the excluded structures hold 2-28 residues each. It is a
      variance control, not a bias. `scripts/sampling/test_residue_guard_bias.py`.
- [ ] Remaining rigor TODOs: T2 (ion inventory re-derivation), T4b, T6, T9, T12,
      T17 (verify cited literature numbers), T19. See `plans/rigor-recheck/todo.md`.
- [ ] **STILL OPEN (R1): train the block-detection scorer** and measure block recall with
      learned weights. The reference impl. proves the *cost and structure*; it does not
      prove a model can find the occupied blocks. This needs the probing/3D data pipeline.
- [x] ~~Audit ionic metadata availability~~ **DONE, and it rescoped the headline claim.**
      Cryo-EM stores buffers as *structured* `_em_buffer_component` (concentration + units),
      X-ray as free-text `_exptl_crystal_grow.pdbx_details`. Coverage ~36%. The deeper
      problem is **survivorship bias**: recorded Mg²⁺ spans only 5-15 mM because nobody
      deposits unfolded RNA, so the [Mg²⁺]->structure *response* cannot be learned from the
      PDB at all. The closed-form `B_elec` term and the Mg²⁺ site head are unaffected;
      training stage 5 was rewritten to condition on RMDB titration series instead.
      See ARCHITECTURE.md §7b and report §6.
- [ ] **Acquire RMDB Mg²⁺ titration series** — now a *prerequisite* for the headline
      ion-conditioning claim, not merely a nice-to-have. Ribonanza does NOT supply this;
      the titration ladders in RMDB do.
- [ ] **Acquire chemical probing data** (Ribonanza 2.1M DMS/SHAPE). ~315x more supervised
      examples than the 6,661 unique 3D sequences; the single largest missing asset.
- [x] ~~Write `research/prior-art/06-coevolution.md`~~ **DONE, then CORRECTED by the
      rigor recheck (see `plans/rigor-recheck/`).** APC-corrected MI on the 12 Rfam seeds
      recovers curated base pairs at mean prec@L/5 = 0.670 / rec@L = 0.768; tRNA is
      perfect (1.000/1.000). **The depth split was OVERCLAIMED and has been walked back**:
      exact permutation p = 0.0455 on a *post-hoc* threshold, Spearman only +0.224 over
      n=12, deep arm n=2, and a *shallow* family (THF, Neff/L = 0.279) reaches 1.000.
      The architectural case for depth-gated routing now rests on the literature and on
      between-family variance (0.333-1.000), **not** on this split. `Neff/L` remains a
      defensible router feature; the language calling it "the decisive pattern" / "the
      strongest argument for MoE" was unsupported and is removed.
      *Two cautions recorded*: (a) the first run reported 0.039 precision due to a
      broadcast bug in the MI outer product ((C,q,1) instead of (C,q,q)) plus missing
      sequence reweighting - fixing it moved the number 17x; (b) `ss_pairs()` does not
      parse WUSS pseudoknot brackets (Aa/Bb/Cc/Dd), omitting 53 of 537 ground-truth pairs
      (9.87%), which DEFLATES precision - so 0.670 is a lower bound.
- [ ] Extract Mg²⁺-site and B-factor labels at scale from the server's 27,452 chains to
      size the two "free" supervision channels properly (180 structures is a pilot).
- [ ] Fix the four repo issues listed above (MANIFEST caps, `total_sequences()`,
      hardcoded paths, README median-length inconsistency).

## Raw-archive pass (2026-09-21) — three claims closed, two defects found

The 10,527 raw PDB entries acquired for v0.2 were finally used for what they
were acquired for. **10,520 of them hold an RNA polymer chain**, 13,348,166 RNA
residues, 29,038 RNA chains — 43x the 309,197 residues the entry-level claims
had rested on since they were written on 180 BGSU structures.

| claim | v0.1 (n=180) | raw archive | outcome |
|---|---|---|---|
| **D9** `target_c` max effective c | 19.04 | **23.30** (14,106 chains) | **24 holds**, 0 breaches, 2.9% headroom |
| **G1** longest single RNA chain | 3,764 nt | **4,450 nt** (6HRM) | **reproduces exactly** -> context = 4,608 (D20) |
| **G1** total RNA per entry, max | 11,478 | **22,345** (4V4G) | 1.95x breach, fifth tail failure |
| **G2** residues in entries with protein | 98.95% | **97.15%** | holds; isolated RNA is 2.7x larger than thought |
| **G3** residues from ribosome-like entries | 92.65% | **85.94%** | holds, 6.7 pts lower |

Two things changed the design rather than just the numbers:

1. **The chain-length maximum stopped moving.** 4,450 nt, with the same five
   structures over 4,096, at 29,807 derived chains and again over the entire
   archive. That is the first extreme quantile in this project to close rather
   than merely rise, and it settles the context window at **4,608** (D20).
2. **`target_c`'s maximum did not stop moving**: 19.04 -> 21.14 -> **23.30**.
   24 absorbed it with 2.9% to spare, so the **overflow path is load-bearing**
   (D23) — but D9's stated mechanism (modified residues) was wrong. Chains from
   entries the derivatives cover max out at **21.14**, exactly the published
   derivative figure; the extra 2.16 comes entirely from **entries no derivative
   holds**. The top 30 chains in the archive are one deposition campaign
   (9T-series E. coli ribosome PTC refinements, deposited 2025-10-21, after
   RNASolo's 2023-11 snapshot); rank 31 is 21.14. Only **four independent
   molecules in the whole PDB** exceed effective c = 20.

**Two defects, both the same shape as defect #22 — a second definition drifting
from the canonical one:**

- **C15**: two scripts had private entry counters joining to `entity_poly_types`
  (which returns **auth** chain ids) on **`label_asym_id`**. On entries whose
  labellings differ — 1ARJ is `label A` / `auth N` — every atom missed its
  entity and the entry reported zero polymer residues. **3,254 of 10,527**
  entries affected; it deflated G2 to 89.2% and G3 to 33.6% before being caught.
  Counting now lives once, in `mmcif_entities.entry_composition`.
- **C16**: `rna_chain_coords` never read `pdbx_PDB_model_num`, so an NMR
  ensemble's 20 models stacked into one residue — **1ARJ at 424 heavy atoms per
  residue** against a nucleotide's ~21 — making every NMR contact map the union
  over the ensemble. Published tail statistics are unaffected (the four chains
  that set them are single-model, and 400-file samples of RNA3DB and
  gRNAde/RNASolo found zero multi-model files), but it would have corrupted the
  raw-entry pass. Fixed to first-model-only, asserted by test property 11.

**Data finding — and it is the same finding as (2).** Matching both derivative
corpora's entry ids against the archive: RNA3DB covers 5,389 entries, gRNAde/RNASolo 6,156, union **7,943** —
so **2,581 entries (24.5%) of the world's RNA structures are in neither**, and
all of them are already on disk. 608 have a chain inside the 64-3,000 training
window (693 have one at least 64 nt long) and **925 are protein-free**, which is precisely the stratum D21 showed
is under-represented. `plans/13-sequence-structure-gap-strategy.md`.

`verify_claims.py` now pins **257** checks and passes.

## Reproducing everything

```bash
python3 scripts/sampling/fetch_samples.py 180          # data
python3 scripts/sampling/analyze_ions_motifs.py        # ion_summary.json
python3 scripts/sampling/analyze_rigidity.py           # rigidity_summary.json
python3 scripts/sampling/analyze_contact_sparsity.py   # contact_sparsity.json
python3 scripts/sampling/validate_proposal_recall.py   # proposal_recall.json
python3 scripts/sampling/analyze_contact_separation.py # contact_separation.json
python3 scripts/sampling/analyze_block_sparsity.py     # block_sparsity.json
python3 scripts/sampling/audit_ionic_metadata.py       # ionic_metadata_audit.json
python3 scripts/sampling/audit_em_buffers.py           # em_buffer_audit.json
python3 scripts/sampling/measure_coevolution.py        # coevolution_signal.json
cd research/report && pdflatex main.tex                # 21pp report
python3 research/architecture/reference/benchmark_pair_track.py  # pair_track_benchmark.json
```
