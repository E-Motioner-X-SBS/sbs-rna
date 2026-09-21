# Continuation State — sbs-rna

## Session Summary
| Field | Value |
|-------|-------|
| Phase | **v0.2 SPECIFIED, MEASURED AND BUILT.** Every open action is closed except two blocked externally: training the block scorer (waiting on a free GPU, watcher armed) and the rest of Ribonanza (Kaggle credentials). |
| What I did | Collected and verified 256.6 GB of RNA data; closed D9/D10/G1/G2/G3 on the raw archive; **built the model** (hybrid trunk, MoE, 667-class motif bank, harmonic ensemble, ten heads) and the **16,604-chain training set** with its free supervision; acquired the **RMDB ionic ladders**; re-derived S7.1 on 2,994 chains |
| What worked | elDORS S3 anonymous download (SHA256 verified); HF/GitHub direct fetches; catalog + loader API; parquet ETL; exploration pipeline |
| Errors | none outstanding (all downloads completed byte-exact; catalog rebuilt from new location) |
| Next priorities | **Everything not blocked externally is done.** 1) `scripts/await_gpu_and_train.sh` is armed and starts the block-scorer run the moment the shared A100 frees -- that is open action 4, the last unvalidated assumption; then `scripts/train_pharos.py` for stage 5. 2) Open action 3 (Ribonanza beyond 335,616) needs the user's Kaggle credentials; the public mirrors are confirmed to be the same file we already hold. |
| Blockers | the shared A100 has been at 100% utilisation / 0.3 GiB free throughout; both trainers refuse to start on it rather than OOM mid-run |
| Audit status | v0.1 audited (28 defects, 11 cycles); v0.2 validated on 29,807 chains + 10,520 raw entries. Two further defects found and fixed in the raw pass: **C15** (entry counter keyed `label_asym_id` against auth-keyed entity declarations -- 3,254 entries silently read zero) and **C16** (NMR ensembles stacked 20 models into one residue -- 424 atoms/residue on 1ARJ). `verify_claims.py` pins **257** checks. |

## State of the data (all verified)
- Sequences: 1,369,926,204 (elDORS 1,323,715,880 SHA256-verified + RNAcentral 46,210,324)
- 3D, curated derivatives: 27,452 chains (RNA3DB 15,441 + gRNAde 12,011) = 6,661 unique sequences
- **3D, raw archive: 29,038 RNA chains in 10,520 entries, 13,348,166 residues.** The
  derivatives cover 7,943 of those entries; **2,581 (24.5%) are in neither**, all
  already on disk -- 608 with a chain in the 64-3,000 window, 925 protein-free,
  and they hold the entire top-30 of the effective_c distribution.
- Benchmarks: ~160k 2D annotations, >3.6M fitness measurements, 147-species splicing
- Rfam 15.1: 4,227 families; 100 with 3D representatives; 10.07M Rfam-mapped sequences
- Catalog: data/catalog/catalog.sqlite (47,471 files indexed)
- **RMDB ionic titrations** (195 MB, `data/benchmarks/rmdb/`): 24 Mg2+ ladders,
  527 concentration points, **all 24 crossing sub-millimolar** -- the regime the
  PDB cannot contain because unfolded RNA is not deposited. RMDB's bulk endpoint
  is now GitHub release assets, not the old Django site.
- **PHAROS 3D training set** (`data/derived/pharos3d/`, 33 shards): 16,604 chains,
  13,172,991 residues, 63,298,800 contacts, with Mg sites, X-ray B-factors,
  N_struct and disorder labels extracted from the entries themselves.
- **PDB Chemical Component Dictionary** (114 MB, `data/structures/ccd/components.cif.gz`,
  wwPDB 2026-09-12): the only source of a modified residue's parent base -- entry
  files do not carry the field. Resolves all 370 modification species the archive
  contains; 88.6% of modified residues map to a standard A/C/G/U parent.

## Key documents
| File | Content |
|------|---------|
| data/catalog/README.md | source-by-source inventory |
| data/catalog/VERIFICATION.md | all verification checks |
| data/catalog/3d_data_statistics.json | 3D quality tiers |
| data/exploration/reports/00-EXPLORATION-SUMMARY.md | findings + recommendations |
| plans/12-nucleicbert-data-scanner.md | acquisition log + recipes |
| plans/13-sequence-structure-gap-strategy.md | strategy for the 3D gap |

## Repo
- Local: /store/shuvam/E-motioner-X-SBS/sbs-rna
- Remote: https://github.com/E-Motioner-X-SBS/sbs-rna (public, main)
- Versioned: catalog + exploration + READMEs + scripts (33 files)
- Ignored: bulk data (sequences/structures/benchmarks/families/derived)

## Continuation prompt hints
- Start from model design now that data is complete: tokenizer (5 symbols),
  context window (2048 recommended), pretraining objective (MLM + span masking
  like NucleicBERT), and the 3D fine-tuning head (contact/distance maps from
  RNA3DB with the documented split).
- If curation decided: write the filter to `scripts/` and regenerate the
  pretraining parquet starter pack with the chosen chunk weighting.
