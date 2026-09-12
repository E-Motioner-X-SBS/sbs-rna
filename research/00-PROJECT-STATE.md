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

## Next actions (session 2+)

Highest value first:

- [ ] **Prototype the hierarchical pair track in torch** and measure block-detection
      recall with a *learned* scorer, not the sequence-only heuristic. This is the one
      unproven load-bearing claim (risk R1).
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
- [ ] Write `research/prior-art/06-coevolution.md` (material gathered, not yet written:
      RNAcmap/rMSA, CoCoNet, DIRECT, MSA-depth limits, CS-Fold).
- [ ] Extract Mg²⁺-site and B-factor labels at scale from the server's 27,452 chains to
      size the two "free" supervision channels properly (180 structures is a pilot).
- [ ] Fix the four repo issues listed above (MANIFEST caps, `total_sequences()`,
      hardcoded paths, README median-length inconsistency).

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
cd research/report && pdflatex main.tex                # 18pp report
```
