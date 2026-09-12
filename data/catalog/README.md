# RNA Training Database (rna_training_db)

Consolidated, verified, queryable catalog of the RNA/nucleic-acid training
resources acquired for the future RNA foundation model. Built 2026-09/10.

## Contents at a glance

| Layer | Sources | Scale |
|---|---|---|
| **Pre-training sequences** | elDORS_v1 (20 chunks, SHA256-verified), RNAcentral 27 | 1.32B + 55M seqs |
| **Families / alignments** | Rfam 15.1 (seed, CM, 3D seeds, full region) | 4,227 families |
| **3D structure indices** | BGSU nrlist 4.56 (7 cutoffs), motif atlas 4.12, RNA3DB (mmCIF + splits) | 5,161 NR classes, 2,972+2,020 motifs, all PDB RNA chains |
| **Benchmarks: 2D** | ArchiveII, bpRNA-1m, bpRNA-spot, RNAStrAlign, bpRNA-new | ~160k structures |
| **Benchmarks: 3D** | RNA-Puzzles (std submissions + site), CASP15 (15 targets), CASP16 (10 targets) | 25 experimental targets + puzzle models |
| **Benchmarks: fitness** | RNAGym (>30 assays, >1M measurements), NABench (162 assays) | >3.6M mutations |
| **Benchmarks: splicing** | Spliceator, SpliceBERT (72 vertebrates), G3PO (147 species) | 4-species test + 2M pre-mRNAs |

## Directory layout

```
data/
├── catalog/            ← THIS catalog (sqlite + manifest + docs)
├── sequences/          elDORS_v1 (20 chunks) + rnacentral
├── structures/         databases/ (rna3db, grnade_rnasolo)
│                       blind_tests/ (casp15, casp16, rna_puzzles)
│                       indices/ (bgsu nrlist, motifs, rfam map, pdb seqres)
├── families/           Rfam 15.1
├── benchmarks/         secondary_structure/, fitness/, splicing/
├── derived/            parquet_starter/, parquet_demo/
└── exploration/        figures/ + reports/
```

## Querying the catalog

```python
import sqlite3
con = sqlite3.connect("data/catalog/catalog.sqlite")

# All sources with sizes
con.execute("""
  SELECT source_name, COUNT(*), ROUND(SUM(size_bytes)/1e9,2)
  FROM files GROUP BY source_name ORDER BY 3 DESC
""").fetchall()

# Files with record counts for a source
con.execute("""
  SELECT path, n_records FROM files
  WHERE source_name='bpRNA_spot' AND n_records IS NOT NULL
""").fetchall()

# Declared splits
con.execute("SELECT name, purpose, description FROM splits").fetchall()
```

## Python loader API

```python
from rna_db import RNADatabase

db = RNADatabase("data/catalog/catalog.sqlite")

db.sources()                       # list of source dicts
db.files("elDORS_v1")              # files for a source
db.splits()                        # declared splits
db.count_sequences("elDORS_v1")    # record counts where known
db.iter_fasta("rnacentral")        # stream sequences from a source
```

## Sample statistics (first 5,000 sequences of chunk 001)

| Metric | Value |
|---|---|
| Sequences | 5,000 |
| Nucleotides | 4,524,059 |
| Composition | A 27.51%, T 25.15%, G 24.49%, C 22.71%, N 0.15% |
| Length min / median / mean / max | 16 / 642 / 904.8 / 4,080 nt |

**Alphabet note**: elDORS/MARS store sequences in the DNA alphabet
(A,C,G,T + N). RNA models should map T->U during preprocessing
(`scripts/eldors_to_parquet.py --t2u` does this automatically).

## Training-ready conversion (FASTA -> Parquet)

`scripts/eldors_to_parquet.py` streams gzipped FASTA into zstd-compressed
Parquet shards with columns `(id, sequence)`:

```bash
# demo run (verified: 100k seqs -> 2 shards, ~2.5k seq/s single-threaded)
python3 scripts/eldors_to_parquet.py --chunk 001 --limit 100000 \
    --shard-size 50000 --out data/derived/parquet_demo --t2u

# full corpus (run 8 chunks in parallel; ~7 h/chunk single-threaded)
for i in 001 002 ... 020; do
  python3 scripts/eldors_to_parquet.py --chunk $i \
      --out data/derived/parquet_full --t2u &
done; wait
```

## Pretraining split convention

`splits/pretrain_split.json` defines the homology-independent split:

- train: `md5(header) % 100 < 80` (80%)
- validation: `80 <= md5(header) % 100 < 90` (10%)
- test: `90 <= md5(header) % 100 < 100` (10%)

No data movement is required; implement the rule in the data loader.

## Integrity


- elDORS_v1: all 20 chunks verified against `elDORS_v1_manifest.sha256`
  (`sha256sum -c` reports OK for every chunk).
- All downloads are idempotent and resumable; logs kept next to data
  (`download.log` per area, `acquisition_summary.json` for benchmarks).

## Not acquired (documented for future scale-up)

| Resource | Size | Why deferred |
|---|---|---|
| MARS full | 1.57TB | elDORS supersedes; disk-limited |
| elDORS_v1_raw | 1.2TB | clustered version sufficient for pretraining |
| RNAcmap3-optimized build | 861GB | MSA pipeline branch, later phase |
| rMSA-optimized build | ~1.0TB | MSA pipeline branch, later phase |
| ENSEMBL all-species ncRNA | ~100GB+ | RNAcentral covers the same sequences |
| MGnify/MG-RAST genomes | TBs | already distilled into elDORS |

## Citations

When using these resources, cite the primary sources:
- elDORS: Dutta & Vicens 2026, bioRxiv 10.64898/2026.07.10.737016
- MARS/RNAcmap3: Chen et al. 2024, GPB, doi:10.1093/gpbjnl/qzae018
- RNAcentral: RNAcentral Consortium, NAR (release 27)
- Rfam 15.1: Ontiveros-Palacios et al., NAR 2025
- BGSU RNA 3D Hub: Leontis lab, CC BY 4.0
- RNA3DB: Szi-Marci et al., JMB 2024
- ArchiveII: Sloma & Mathews 2016; bpRNA: Danaee et al. 2018;
  RNAStrAlign: Tan et al. 2017
- RNA-Puzzles: Magnus et al.; CASP15/16: predictioncenter.org
- RNAGym: Marks & Das labs 2025; NABench 2025
- Spliceator: Scalzitti et al. 2021; SpliceBERT: Chen et al. 2024;
  G3PO: Scalzitti et al. 2020

## 3D structural data summary

| Level | Count |
|---|---|
| Chain-level structures (union) | 27,452 |
| **Unique sequences with 3D data** | **6,661** |
| PDB entries covered | 6,846 |
| Non-redundant BGSU classes (all / ≤4Å / ≤2.5Å / ≤2Å) | 5,161 / 3,953 / 1,410 / 604 |
| RNA3DB documented ML split (train/test) | 1,539 / 660 chains |
| gRNAde ML clusters | 4,223 |

Quality tiers (RNA3DB filtered set of 15,441 chains): see
`3d_data_statistics.json`. For contact-map training the practical window is
length 32-1024 nt with a resolution filter: ~2,370 (length only) to ~2,884
(≤4Å) unique sequences; the strict tier (≤2.5Å) is 673.

**Comparison**: NucleicBERT trained its 3D tasks on ~875 structures total
(408 NucleoSeeker + 467 BGSU-filtered). Our collection provides ~6,661 unique
sequences (7.6x) and 27,452 chains (31x).
