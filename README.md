# sbs-rna

RNA data repository and tooling for the RNA foundation model.
**256.6 GB · 47,471 files · 25 sources** (all verified; catalog queryable).

## Layout

```
sbs-rna/
├── data/                the data collection (separate data folder)
│   ├── catalog/         queryable metadata: catalog.sqlite, MANIFEST,
│   │                    VERIFICATION, loader API, samples/, splits/
│   ├── sequences/       elDORS v1 (1.32B seqs, 20 chunks) + RNAcentral 27 (46.2M)
│   ├── structures/
│   │   ├── databases/   RNA3DB (15,441 chains) + RNASolo/gRNAde (14,369 PDBs)
│   │   ├── blind_tests/ CASP15, CASP16, RNA-Puzzles
│   │   └── indices/     BGSU nrlist + motif atlas, Rfam<->PDB map, PDB seqres
│   ├── families/        Rfam 15.1 (seed, covariance models, 3D seeds, full)
│   ├── benchmarks/      secondary_structure/, fitness/, splicing/
│   ├── derived/         10M-sequence parquet starter pack (2 GB)
│   └── exploration/     figures/ (8) + reports/ (summary, stats, profiles)
├── scripts/             acquisition, cataloging, ETL, exploration
└── plans/               data-scanning + strategy documents
```

## Key numbers (all verified, see `data/catalog/VERIFICATION.md`)

| Metric | Value |
|---|---|
| Sequences | 1,369,926,204 (elDORS 1,323,715,880 + RNAcentral 46,210,324) |
| Nucleotides (elDORS est.) | ~1.25 trillion |
| Alphabet | 5 symbols (A, C, G, T, N; ~0.1% N) |
| Lengths | 10 to 4,096 nt; median 245 nt cross-chunk |
| 3D chain structures | 27,452 (RNA3DB 15,441 + gRNAde 12,011) |
| Unique sequences with 3D | 6,661 across 6,846 PDB entries |
| 3D resolution | median 3.10 A; 13,750 chains <= 4 A; 1,991 <= 2.5 A |
| Non-redundant 3D classes | 5,161 (BGSU 4.56, all cutoffs) |
| 2D structure annotations | ~160k (bpRNA, ArchiveII, RNAStrAlign) |
| Fitness measurements | >3.6M (RNAGym + NABench) |
| Splicing benchmarks | 147 species (G3PO/Spliceator/SpliceBERT) |
| Rfam families | 4,227 (100 with 3D representatives) |

## Quick start

```python
import sys; sys.path.insert(0, "data/catalog")
from rna_db import RNADatabase

db = RNADatabase("data/catalog/catalog.sqlite")
db.sources()                              # 25 datasets with sizes
db.files("elDORS_v1")                     # file listing
for header, seq in db.iter_fasta("rnacentral", limit=5):
    print(header, seq[:40])
```

```bash
# rebuild the catalog after adding data (idempotent)
python3 scripts/build_rna_database.py

# training-ready parquet conversion (T->U for RNA alphabet)
python3 scripts/eldors_to_parquet.py --chunk 001 --out data/derived/my_parquet --t2u

# corpus utilities: split convention, sampling, stats
python3 scripts/corpus_tools.py split
python3 scripts/corpus_tools.py sample 5000
python3 scripts/corpus_tools.py stats

# regenerate exploration figures + reports
python3 scripts/explore_rna_data.py --sample-per-chunk 50000
python3 scripts/explore_eldors_chunks.py
```

## Version-control policy

Only lightweight, high-value artifacts are tracked: the catalog (sqlite +
manifests + docs), the exploration figures/reports, the READMEs, and the
scripts. The bulk data (sequences, structures, benchmarks, families,
derived parquet) is ignored and regenerable via `scripts/acquire_benchmarks.py`
plus the download recipes in `plans/12-nucleicbert-data-scanner.md`.

## Provenance

Every source is documented with URL, license, and version in
`data/catalog/README.md`. Highlights: elDORS (CC BY 4.0), Rfam (CC0),
RNAcentral (CC0), BGSU (CC BY 4.0). Structures from PDB follow PDB terms.
