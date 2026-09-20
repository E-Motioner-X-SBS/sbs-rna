# Can we acquire more RNA data? — MARS assessed, and the answer is mostly no

**Date:** Sep 20, 2026
**Question:** NucleicBERT pretrains on MARS (~1.7B sequences). Should we acquire it?
**Answer:** **No.** Measured, not assumed — see §3. What *is* worth acquiring is in §5.

---

## 1. What MARS is

*MARS and RNAcmap3: The Master Database of All Possible RNA Sequences*
(GPB 2024, [doi:10.1093/gpbjnl/qzae018](https://doi.org/10.1093/gpbjnl/qzae018)).

| | |
|---|---|
| Sequences | **1,727,789,860** |
| Bases | 1,592,396,862,523 |
| Uncompressed | **1,571 GB** |
| Download | **413 GB**, 30 tarballs of 11.9–15.6 GB |
| Location | `https://download.cncb.ac.cn/OMIX/OMIX003037/` (also FTP) |
| Composition | NCBI nt/env_nt/tsa_nt/pat_nt + RNAcentral + MG-RAST + GWH + MGnify |

NucleicBERT used **~30M** of it, extracted by ncRNA keyword — a 1.7% yield.

## 2. The disk reality

Free on `/store`: **290 GB**. The compressed download alone is 413 GB. Streaming
part-by-part (download 14 GB → filter → delete) would work, but only if the
extracted subset is worth the bandwidth. So the question reduces to: what is
actually in there?

## 3. What is actually in there — measured

Pulled the first 40 MB of `OMIX003037-01.tgz` by HTTP range request and
classified **26,250 headers**:

| Class | Count | Share |
|---|---|---|
| mRNA / coding | 10,864 | **41.4%** |
| other (unclassified) | 8,076 | 30.8% |
| genomic DNA | 4,441 | 16.9% |
| ncRNA-like | 2,869 | **10.9%** |

Representative records: *"Giant Panda satellite 1 DNA"*, *"Bos taurus mRNA for
bone Gla protein"*, *"B.taurus mRNA for cyclin A"*. This is the NCBI **nt**
bulk — genomic DNA and protein-coding transcripts.

**Now break down the 10.9% that is ncRNA-like**, which is the only part that
could matter to an RNA structure model:

| Class | Count | Share of ncRNA |
|---|---|---|
| **rRNA gene / amplicon** | 1,998 | **69.6%** |
| tRNA | 716 | 25.0% |
| sn/snoRNA | 109 | 3.8% |
| **other ncRNA** | **46** | **1.6%** |

Typical hits: *"B.physalus gene for large subunit rRNA"*, *"Bos taurus 5S rRNA
gene"*, *"Bovine tRNA-Ser(UGA) gene"*. These are rRNA/tRNA **genes and survey
amplicons** — the same handful of molecules repeated across thousands of
species.

## 4. Why that settles it

Projected over the full database: ~189M ncRNA-like sequences, of which
**94.6% are rRNA or tRNA** and roughly **3M** are other structured ncRNA.

Three reasons not to acquire:

1. **We already have better ncRNA.** RNAcentral gives us **46,210,324** curated
   non-coding sequences — more than the 30M NucleicBERT extracted from MARS, and
   curated rather than keyword-matched out of nt.
2. **It would worsen the bias we already documented.** Finding G3 records that
   **92.65% of RNA residues** in our structural corpus come from ribosomes, and
   G2 that 98.95% are in complex. Adding ~130M more rRNA sequences makes the
   pretraining distribution *more* ribosome-skewed, against a design that is
   already trying to generalise away from it.
3. **The cost is real and the yield is not.** 413 GB of download against 290 GB
   free, to extract perhaps 3M sequences of genuinely diverse structured ncRNA —
   a 0.17% yield on the full database.

**MARS is the right corpus for a homology-search tool, which is what it was
built for (RNAcmap3). It is the wrong corpus for a structure model.**

## 5. The other "gaps" are not gaps

`pdb_hunter/plans/21-nucleicbert-data-scan.md` lists five databases we lack.
Four of them are already on disk in `pdb_hunter/nucleicbert_data/`:

| Listed as missing | Actually present | Size |
|---|---|---|
| bpRNA-1m | `sec_str_data/bprna_1m.csv` | **83,667 rows** |
| ArchiveII | `sec_str_data/archiveii.csv` | 3,911 rows |
| TR0 | `sec_str_data/bprna_tr0.csv` | 10,934 rows (+ ts0 1,411, vl0 1,330) |
| RNAStralign | in `data/benchmarks/secondary_structure/rnastralign` | 27,125 |
| **MARS** | **absent — and should stay absent** | — |

Also present and previously uncounted: **`contact_map_data/` with 11,893
input/target pairs** (sequences + N×N `.npy` contact maps) and `splice_data/`,
`mutation_data/`. That contact set is larger than our 10,399 PDB entries and is
already curated into model-ready form.

**Action: update the gap list in `pdb_hunter/plans/21`, which is stale.**

## 6. What is actually worth acquiring

Ranked by value per unit effort:

| # | Target | Why | Blocker |
|---|---|---|---|
| 1 | **Rest of Ribonanza** (2.1M; we hold 335,616) | largest labelled channel; 499× the clean 3D supervision | Kaggle credentials |
| 2 | **RMDB titration ladders** | the only thing that can supervise the *learned* ionic response — the design's headline novelty | no bulk endpoint; needs scraping or author contact |
| 3 | Rfam **full** alignments (we hold seed) | deeper MSAs raise Neff/L, which gates the coevolution expert | 170 MB → larger; straightforward |
| 4 | RNA-MSM's curated MSA set | alternative alignment depth for the same purpose | check licence |

None of these is bulk sequence. **The sequence channel is not our constraint** —
1.37B sequences already carry ~80 GB of information against a 37 MB model
capacity. The constraints are *labelled* channels and *ionic-response* data.

---

## 7. Method note

The assessment cost one HTTP range request of 40 MB. Downloading 413 GB to
discover the same thing would have cost days of bandwidth and more disk than the
machine has. Range-probe before bulk-acquiring is worth making a habit.
