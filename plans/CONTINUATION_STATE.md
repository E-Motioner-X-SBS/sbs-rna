# Continuation State — sbs-rna

## Session Summary
| Field | Value |
|-------|-------|
| Phase | **ARCHITECTURE v0.2 COMPLETE** — validated at corpus scale; ready for implementation |
| What I did | Collected, verified, organized, and catalogued 256.6 GB of RNA data (25 sources); built exploration reports; created this repo |
| What worked | elDORS S3 anonymous download (SHA256 verified); HF/GitHub direct fetches; catalog + loader API; parquet ETL; exploration pipeline |
| Errors | none outstanding (all downloads completed byte-exact; catalog rebuilt from new location) |
| Next priorities | 1) Re-derive target_c on raw PDB (D9 is interim); 2) acquire RMDB titrations (gates the learned ionic response); 3) complete Ribonanza beyond the 335,616 acquired; 4) train the block-detection scorer -- the largest unvalidated assumption; 5) re-measure G2/G3 on raw whole entries |
| Blockers | none |
| Audit status | v0.1 audited (28 defects, 11 cycles); v0.2 validated on 29,807 chains + 8,041 raw entries |

## State of the data (all verified)
- Sequences: 1,369,926,204 (elDORS 1,323,715,880 SHA256-verified + RNAcentral 46,210,324)
- 3D: 27,452 chains (RNA3DB 15,441 + gRNAde 12,011) = 6,661 unique sequences, 6,846 PDB entries
- Benchmarks: ~160k 2D annotations, >3.6M fitness measurements, 147-species splicing
- Rfam 15.1: 4,227 families; 100 with 3D representatives; 10.07M Rfam-mapped sequences
- Catalog: data/catalog/catalog.sqlite (47,471 files indexed)

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
