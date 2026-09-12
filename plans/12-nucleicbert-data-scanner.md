# Task 12: NucleicBERT Data Scanner — Training Data Inventory for Future Model

## Restated Problem
User provided https://github.com/KIT-MBS/NucleicBERT. Tasks:
1. Identify ALL databases used for NucleicBERT pre-training + downstream fine-tuning.
2. Use this as a scanner/starting point to find MORE databases we can harness.
3. Goal: collect a huge amount of training data (sequences + 3D structures) for our future model.

## Success Criteria
- [ ] Complete inventory of NucleicBERT's databases with sizes, URLs, formats
- [ ] Expanded universe: every major RNA/nucleic-acid sequence + structure database
- [ ] For each: content size, download URL, format, license, relevance to our model
- [ ] Prioritized acquisition plan (what to download first and why)
- [ ] Written to plans/12-nucleicbert-data-scanner.md + data inventory files

## Key Facts Gathered (from NucleicBERT paper + repo, fetched 2026-09-09)
- Pre-training: MARS database, ~30M ncRNA sequences extracted from ~1.7B total (keyword "ncRNA")
- Model: 404M params, 32 layers, 32 heads, 1024 hidden, vocab 25, maxlen 1024
- Pretraining: 80/20 train/val, 300 epochs, 192 A100 GPUs, MLM 83.1% acc
- Secondary structure: RNAStrAlign (37,149 structures, 8 families) + bpRNA-1m TR0 (10,814) → tested on ArchiveII600 (3,975) + TS0 (1,305)
- Contact/distance maps: BGSU RNA 3D representative set (release 3.368, Jan 2025, 467 structures after filtering 32-1024nt, CD-HIT-EST 80%) + NucleoSeeker (408 structures)
- Splice sites: Spliceator benchmarks (zebrafish, fly, worm, plant; 10k pos + 10k neg each, 600nt windows)
- Fitness: CPEB3 ribozyme mutation dataset (Beck et al. 2022)
- Shuffle detection: same as secondary structure data + Rfam 15.0 (4,178 families)
- PDB has only 8,446 annotated RNA 3D structures (as of Jan 2025)
- Tools used: PDBFixer, CD-HIT-EST, Chemical Component Dictionary

## Open Questions
- What exactly is MARS? (need to verify: metagenomic RNA sequences database, Genomics Proteomics Bioinformatics journal)
- What other databases exist beyond NucleicBERT's choices?
- RNA-Puzzles, CASP-RNA benchmarks?
- Noncoding RNA databases: Rfam, Ensembl ncRNA, GENCODE, lncRNAdb, RNAcentral?
- Structure: PDB, BGSU NR sets, RNASolo, RNA-3D-Motif, 3D motifs?
- Alignment/MSA: Rfam full alignments, covariance models?

## Acquisition Log (2026-09-09 → 2026-09-11)

| Resource | Status | Size | Verification |
|---|---|---|---|
| elDORS_v1 (20 chunks) | ✅ COMPLETE | 182.36 GB | All 20 SHA256 OK |
| BGSU nrlist 4.56 (7 cutoffs) | ✅ COMPLETE | 1.7 MB | 5,161 NR classes (all) |
| Rfam 15.1 (seed/CM/3D/full/PDBmap) | ✅ COMPLETE | 178 MB | files present |
| RNA3DB (jsons/cmscans/mmcifs) | ✅ COMPLETE | 2.24 GB | 2026-01-05 full release |
| CASP15 (15 targets) | ✅ COMPLETE | 47 MB | mmCIF downloaded |
| CASP16 (10 targets) | ✅ COMPLETE | 9.5 MB | mmCIF downloaded |
| RNAGym (zip + repo) | ✅ COMPLETE | 310 MB | >1M measurements |
| NABench | ✅ COMPLETE | 100 MB | 162 assays |
| Spliceator | ✅ COMPLETE | 494 MB | 4-species benchmark |
| G3PO | ✅ COMPLETE | 20 MB | 147 species |
| RNA-Puzzles (std + site) | ✅ COMPLETE | 2.89 GB | 15,555 files |
| Secondary structure suite | ✅ COMPLETE | 168 MB | ArchiveII 3,975; bpRNA-1m 102k; spot splits; RNAStrAlign; bpRNA-new 5,401 |
| BGSU motif atlas 4.12 | ✅ COMPLETE | 1.8 MB | IL + HL (csv+json) |
| RNAcentral 27 active | 🔄 DOWNLOADING | 1.83/10.9 GB | EBI FTP (slow ~0.4MB/s) |
| SpliceBERT data | 🔄 DOWNLOADING | 1.32/8.6 GB | Zenodo (slow) |
| elDORS exact seq counts | 🔄 RUNNING | 8 parallel workers | results → count_results/ |

## Database artifacts created

- `data/rna_training_db/catalog.sqlite` — 23 sources, 17,610 files, 192 GB indexed
- `data/rna_training_db/MANIFEST.json` — machine-readable
- `data/rna_training_db/README.md` — full documentation
- `data/rna_training_db/rna_db.py` — Python loader API
- `data/rna_training_db/splits/pretrain_split.json` — md5-based 80/10/10 split convention
- `data/rna_training_db/samples/eldors_001_first5000.fasta` — dev sample + stats
- `scripts/acquire_benchmarks.py` — reproducible acquisition pipeline
- `scripts/build_rna_database.py` — catalog builder
- `scripts/corpus_tools.py` — split/sample/stats utilities
- `scripts/eldors_to_parquet.py` — training-ready ETL (verified 2.5k seq/s, T→U)

## Addendum (RNA-DB-2)

| Item | Result |
|---|---|
| elDORS exact counts | 1,323,715,880 seqs total (all 20 chunks), matches 1.32B advertised |
| 2M-seq census | alphabet A/T/G/C/N only; len 10-4096 (median 730, mean 1026) |
| Starter pack | data/parquet/eldors_starter: 10M seqs, 40 shards, 2.02GB |
| RNA3DB extraction | 15,441 CIF chain structures (23GB) in rna3db_extracted/ |
| gRNAde RNASolo raw | 14,369 PDB structure files (HF chaitjo/gRNAde_datasets) |
| pdb_seqres | all PDB seqres FASTA (66.9MB) for sequence<->structure joins |
| VERIFICATION.md | complete audit trail in data/rna_training_db/ |
| Resilient downloads | scripts/resilient_dl.sh watchdogs for RNAcentral + SpliceBERT |

### Model-design data guidance (from census)
- Tokenizer: 5 symbols suffice for elDORS (ACGTN); T->U mapping at load time
- Context: 4096 covers everything; 2048 covers median+mean; NucleicBERT's 1024 truncates ~40-50%
- Length outliers: 0.03% below 20nt; recommend min-length filter >= 16 or 20

## DATA REORGANIZATION (2026-09-12)

### New structure
```
data/
├── rna/                    ← RNA collection (256.6 GB, 47,471 files)
│   ├── README.md           master index
│   ├── catalog/            catalog.sqlite + docs + samples + splits
│   ├── sequences/          elDORS_v1 (20 chunks) + rnacentral
│   ├── structures/         databases/ + blind_tests/ + indices/
│   ├── families/           Rfam 15.1
│   ├── benchmarks/         secondary_structure/ + fitness/ + splicing/
│   ├── derived/            parquet_starter (10M seqs) + parquet_demo
│   └── exploration/        figures/ (8 PNGs) + reports/ (summary + stats + CSV)
└── protein_legacy/         old protein K-map campaign data (moved, not deleted)
```

### Decisions
- Moved (not copied) all RNA data into `data/rna/` and protein campaign data
  into `data/protein_legacy/` (same filesystem, instant rename; nothing deleted).
- Scripts updated: build_rna_database.py (DATA + DB_DIR + all globs),
  corpus_tools.py, eldors_to_parquet.py.
- .gitignore: `/data/*` with `!/data/rna/` plus re-ignores for the bulk
  subdirs; tracks only README + catalog + exploration (small, valuable).
- DEVIATION NOTE: a stray `data/rna/rna_training_db/` directory (duplicate
  builder output created before DB_DIR was updated) was removed with rm -rf
  after verifying its contents were regenerable and identical in schema.
  This is the only non-quarantined removal this session; it contained only
  builder-generated files (catalog.sqlite, MANIFEST.json), fully reproducible
  via `python3 scripts/build_rna_database.py`.

### Exploration findings (→ plans/14 and exploration report)
- elDORS chunks are source-partitioned into 3 regimes (read-length ~151nt,
  assembled medium, long transcripts 2200+nt); GC 40.7-61.4%.
- 3D: median resolution 3.10 Å; cryo-EM 62%/X-ray 38%; 100 Rfam families.
- All 8 exploration figures + JSON stats + per-chunk CSV generated.
