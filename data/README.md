# RNA Data Repository

Consolidated, verified data collection for the RNA foundation model.
**Total: 256.6 GB · 47,471 files · 25 sources** (indexed in `catalog/`).

```
data/
├── catalog/         queryable metadata + docs + samples + splits   (25 MB)
├── sequences/       pretraining corpora                            (193 GB)
│   ├── eldors_v1/       20 chunks · 1,323,715,880 seqs · SHA256 ✓
│   └── rnacentral/      RNAcentral 27 · 46,210,324 seqs
├── structures/      3D structural data                             (~72 GB)
│   ├── databases/
│   │   ├── rna3db/         15,441 filtered chains (26,532 parsed)
│   │   └── grnade_rnasolo/ 14,369 PDBs · 4,223 ML clusters
│   ├── blind_tests/        CASP15 (15) · CASP16 (10) · RNA-Puzzles
│   └── indices/            BGSU nrlist (7 cutoffs) · motif atlas 4.12 ·
│                           Rfam↔PDB map · PDB seqres (all)
├── families/        Rfam 15.1: seed + CM + full + 3D seeds         (170 MB)
├── benchmarks/      task-specific eval sets                        (~12 GB)
│   ├── secondary_structure/  ArchiveII, bpRNA (1m/spot/new), RNAStrAlign
│   ├── fitness/              RNAGym (>1M measurements), NABench
│   └── splicing/             Spliceator, SpliceBERT, G3PO
├── derived/         ML-ready conversions                           (2 GB)
│   ├── parquet_starter/      10M sequences · 40 shards
│   └── parquet_demo/         100k sequences · T→U mapped
└── exploration/     data analysis reports + figures (this phase)
    ├── reports/
    └── figures/
```

## Quick start

```python
import sys; sys.path.insert(0, "data/catalog")
from rna_db import RNADatabase

db = RNADatabase("data/catalog/catalog.sqlite")
db.sources()                      # 25 datasets with sizes
db.files("elDORS_v1")             # file listing
db.iter_fasta("rnacentral", limit=5)   # stream sequences
```

```bash
# rebuild the catalog after adding data (idempotent)
python3 scripts/build_rna_database.py

# convert a chunk to training-ready parquet
python3 scripts/eldors_to_parquet.py --chunk 001 --out data/derived/my_parquet --t2u

# corpus utilities (split convention, sampling, stats)
python3 scripts/corpus_tools.py split|sample|stats
```

## What each layer is for

| Layer | Content | Role in model development |
|---|---|---|
| `sequences/` | 1.37B seqs | self-supervised pretraining (MLM) |
| `derived/parquet_*` | 10.1M seqs as parquet | fast training iterations |
| `structures/databases/` | 29,810 chains | supervised 3D fine-tuning (contact/distance maps) |
| `structures/indices/` | NR classes, motifs, maps | filtering, annotation joins, eval split selection |
| `structures/blind_tests/` | 25 + puzzle models | honest evaluation (CASP/RNA-Puzzles) |
| `benchmarks/` | ~160k 2D + 2.6M mutations + 80k splice | task-specific fine-tuning + eval |
| `families/` | 4,227 Rfam families | MSA/coevolution features; fold inheritance |

## Key numbers (verified; see catalog/VERIFICATION.md)

| Metric | Value |
|---|---|
| Sequences total | 1,369,926,204 |
| Unique seqs w/ experimental 3D | 6,661 |
| Chain-level 3D structures | 27,452 |
| Non-redundant BGSU classes | 5,161 (≤4Å: 3,953; ≤2.5Å: 1,410) |
| Rfam families w/ 3D reps | 100 of 4,227 |
| Rfam-mapped sequences | 10,070,931 |
| Alphabet | A,G,C,T + N (5 symbols; T→U optional) |
| Sequence lengths (elDORS) | 10-4096 nt, median 730 |

## Provenance and licensing

Every source is documented in `catalog/README.md` with its URL, license, and
version. Highlights: elDORS (CC BY 4.0), Rfam (CC0), RNAcentral (CC0),
BGSU (CC BY 4.0), RNA3DB (repo license), gRNAde (repo license),
RNAGym/NABench/Spliceator/G3PO (repo licenses). RNA-Puzzles/CASP structure
files follow PDB terms.

## Legacy data

The protein K-map campaign data (PSICOV150, DeepMSA, EVmutation, GPCRdb,
Pfam, structures, unit tests) was moved to `data/protein_legacy/` and is NOT
part of the RNA collection.
