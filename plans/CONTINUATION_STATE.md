# Continuation State — sbs-rna

## Session Summary
| Field | Value |
|-------|-------|
| Phase | **ARCHITECTURE v0.2 COMPLETE, raw-archive pass done** — D9, D10 and G1/G2/G3 all closed on 10,520 entries; ready for implementation |
| What I did | Collected, verified, organized, and catalogued 256.6 GB of RNA data (25 sources); built exploration reports; created this repo |
| What worked | elDORS S3 anonymous download (SHA256 verified); HF/GitHub direct fetches; catalog + loader API; parquet ETL; exploration pipeline |
| Errors | none outstanding (all downloads completed byte-exact; catalog rebuilt from new location) |
| Next priorities | 1) **Train the block-detection scorer and run stage 5** -- both trainers are written, GPU-only, and `scripts/await_gpu_and_train.sh` fires when the shared A100 frees; 2) complete Ribonanza beyond the 335,616 acquired (needs Kaggle credentials); 3) re-derive the remaining max/min at scale. Old items 1, 2, 5, 7, 8 and 9 are closed. |
| Blockers | none |
| Audit status | v0.1 audited (28 defects, 11 cycles); v0.2 validated on 29,807 chains + 10,520 raw entries. Two further defects found and fixed in the raw pass: **C15** (entry counter keyed `label_asym_id` against auth-keyed entity declarations -- 3,254 entries silently read zero) and **C16** (NMR ensembles stacked 20 models into one residue -- 424 atoms/residue on 1ARJ). `verify_claims.py` pins **249** checks. |

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
