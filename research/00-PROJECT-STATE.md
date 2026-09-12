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
- Oriented in the repo; found the four issues above.
- Built `scripts/sampling/fetch_samples.py`; downloading ~180 BGSU
  non-redundant representative mmCIF structures + 12 Rfam seed alignments +
  RNAcentral sequence sample into `data/samples/`.
  - First run failed silently: BGSU `/nrlist/release/...` returns HTML, not CSV.
    Correct endpoint is `/rna3dhub/nrlist/download/current/3.0A/csv`. Script now
    validates PDB-id shape and aborts if the feed is HTML.
- Prior-art research in progress -> `research/prior-art/`.

## Next actions

- [ ] Finish prior-art survey (see `research/prior-art/`)
- [ ] Quantify motif rigidity + ion coordination from the downloaded structures
- [ ] Draft architecture spec -> `research/architecture/ARCHITECTURE.md`
- [ ] Blueprint diagrams -> `research/architecture/diagrams/`
- [ ] LaTeX report -> `research/report/main.tex`
