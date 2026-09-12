# Verification Report (2026-09-11)

## elDORS_v1 (primary pretraining corpus)
| Check | Result |
|---|---|
| Chunks downloaded | 20/20 |
| SHA256 (manifest) | ✅ all 20 OK |
| Total size | 182.36 GB |
| **Exact sequence count** | **1,323,715,880** (per-chunk counts in `data/elDORS_v1/seq_counts.txt`) |
| Advertised count | 1.32B (matches) |
| Chunk range | 63.8M-74.8M seqs/chunk (last chunk 33.7M) |

## 2M-sequence character census (chunk 001)
| Property | Value |
|---|---|
| Sequences analyzed | 2,000,000 |
| Nucleotides | 2,052,343,440 |
| Alphabet | A 27.98%, T 26.81%, G 22.80%, C 22.10%, N 0.31% (5 symbols total) |
| Length min / median / mean / max | 10 / 730 / 1026.2 / 4096 nt |
| p99.9 length | 4,071 nt |
| Sequences < 20 nt | 0.029% |

**Model implications**: vocabulary of 5 (A,C,G,T,N; T→U for RNA) suffices for
elDORS; context window of 4096 nt is fully lossless (median 730).

## Benchmarks
| Dataset | Verified count |
|---|---|
| ArchiveII (test) | 3,975 rows (sample row inspected: 16S rRNA, dot-bracket with pseudoknots) |
| bpRNA-1m (HF conversion) | 66,715 entries |
| bpRNA-spot splits | train 10,934 / val 1,330 / test 1,411 |
| RNAStrAlign | 27,125 entries (matches post-dedup count in literature) |
| bpRNA-new | 5,401 rows |
| CASP15 targets | 15 mmCIF files |
| CASP16 targets | 10 mmCIF files |
| RNAGym zip | 71 files (30+ assay CSVs) |
| Motif Atlas IL | 3,385 CSV lines (2,972 loops) |
| RNA3DB split (jsons) | 1,539 train / 660 test unique chains, zero overlap |
| RNA3DB mmcifs | 15,441 chain CIF files extracted and counted (rna3db_extracted/rna3db-mmcifs/) |

## Pending at time of writing
- RNAcentral 27 active set: downloading (2.8/10.9 GB), resumable
- SpliceBERT Zenodo: downloading (2.1/8.6 GB), resumable
- RNA3DB mmcif extraction: in progress (~15.4k/17.8k files)

## Starter parquet pack (`data/parquet/eldors_starter/`)
| Check | Result |
|---|---|
| Sequences | 10,000,000 (1.25M each from chunks 001-008) |
| Shards | 40 x 250k-row zstd parquet |
| Size | 2.02 GB parquet (vs ~10 GB gzip equivalent) |
| Alphabet | verified clean ACGTUN (T preserved; use --t2u for RNA) |
| Throughput | 8 parallel workers, ~3 min wall for 10M sequences |
| Demo pack | `data/parquet/eldors_v1_demo/` = 100k seqs, T->U mapped |
