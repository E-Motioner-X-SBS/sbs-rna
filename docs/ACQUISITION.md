# Acquisition Inventory — every source, link and count

> **Branch:** `docs/acquisition-inventory` (from `main` @ `64739a6`)
> **Date:** 2026-09-21
> **Scope:** every dataset downloaded for the PHAROS training corpus on this
> machine, with the exact URL it came from, where it landed, its measured
> size, its record count, and its verification status.
>
> This is an inventory, not a design document. Architecture lives in
> `history_of_failed_attempts/ARCHITECTURE_v0.2_revision_trail.md`; the corpus narrative is in
> `data/catalog/README.md`.

---

## 1. Tooling

| Item | Detail |
|---|---|
| Package manager | `uv 0.12.17` (`~/.local/bin/uv`, installed via `astral.sh/uv/install.sh`) |
| Project file | `pyproject.toml` — `pdb-hunter` installed editable from `../pdb_hunter` (v2.0.16) |
| pdb_hunter clone | `git@github.com:E-Motioner-X-SBS/pdb_hunter.git` → `/home/susmitaroy1_iiserk/sbs/pdb_hunter` |
| Orchestrator | `scripts/acquire_all.py` — 147 jobs, byte-range resume, per-job logs |
| Watchdog | `scripts/watch_downloads.py` — restarts groups until `state/ALL_DONE` |
| Status | `scripts/download_status.py` |
| Integrity | `scripts/verify_downloads.py` (10,488 archives), `scripts/audit_data.py` (record counts) |
| Specialised | `scripts/acquire_rmdb.py`, `scripts/acquire_rfam_full.py`, `scripts/acquire_raw_pdb_entries.py` |

Local additions not yet committed to `main`: `pyproject.toml`, `uv.lock`
and the `scripts/` files above.

---

## 2. Where the data lives

```
data/                                         694.6 GB, 33,677 files
├── sequences/
│   ├── elDORS_v1/      20 chunks + sha256 manifest       182.36 GB
│   ├── mars/           30 parts (OMIX003037)             427.29 GB
│   └── rnacentral/     active / inactive / species-specific 23.94 GB
├── families/rfam/      Rfam 15.1 files + full_alignments/   3.48 GB
├── structures/
│   ├── raw_pdb_entries/ 10,526 atomic mmCIFs              16.03 GB
│   ├── databases/       rna3db, grnade_rnasolo            11.97 GB
│   ├── blind_tests/     casp15, casp16, rna_puzzles*       4.06 GB
│   └── indices/         bgsu nrlist+motifs, seqres, rfam map 0.07 GB
├── benchmarks/
│   ├── secondary_structure/                               1.42 GB
│   ├── fitness/                                           0.48 GB
│   ├── splicing/                                         10.33 GB
│   └── probing/         rmdb, eternabench, ribonanza      13.10 GB
└── catalog/             queryable metadata (tracked on main)
pdb_hunter/RNA_Database/  10,452 per-entry packages      138.75 GB
```

Total on disk: **833.3 GB** across `data/` and `pdb_hunter/RNA_Database`.

---

## 3. Sequence corpora

| Dataset | Source URL | Records | Size | Status |
|---|---|---|---|---|
| **elDORS v1** (20 chunks) | `https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_0NN.fasta.gz` | **1,323,715,880** | 182.36 GB | all 20 SHA256 verified |
| elDORS manifest | `…/elDORS_v1_chunks/elDORS_v1_manifest.sha256` | 20 hashes | 1,780 B | OK |
| **MARS** (30 parts) | `https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-NN.tgz` | **1,727,789,860** | 427.29 GB | sizes + `tar -tzf` OK (30/30) |
| **RNAcentral active** | `https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz` | **46,210,324** | 10.92 GB | gzip OK |
| RNAcentral inactive | `…/sequences/rnacentral_inactive.fasta.gz` | 12,322,545 | 1.33 GB | gzip OK |
| RNAcentral species-specific | `…/sequences/rnacentral_species_specific_ids.fasta.gz` | 56,419,280 | 11.69 GB | gzip OK |

Registry entry: `https://registry.opendata.aws/eldors_v1/` (S3 bucket
`eldors-v1-database`, us-east-1, no auth).
MARS record: NGDC OMIX003037 (`https://download.cncb.ac.cn/OMIX/OMIX003037/`).

---

## 4. 3D structures

| Dataset | Source URL | Content | Size |
|---|---|---|---|
| **Raw PDB entries** | `https://files.rcsb.org/download/{PDB_ID}.cif.gz` | **10,526 atomic** + 9A0D integrative (excluded) | 16.03 GB |
| **pdb_hunter RNA_Database** | harvest via `pdb-hunter rna harvest --nr-release 4.57` | **10,452 entries**: raw pdb/cif/fasta, clean per-chain pdb/cif/fasta/dbn/bpseq, `_bb.xyz`, `_meta.json` | 138.75 GB |
| **RNA3DB** | `https://github.com/marcellszi/rna3db/releases/download/2026-01-05-full-release/rna3db-jsons.tar.gz`, `…-cmscans.tar.gz`, `…-mmcifs.tar.xz` | **15,441 chains** + split/cluster JSONs | 2.15 GB |
| **gRNAde / RNASolo** | `https://huggingface.co/datasets/chaitjo/gRNAde_datasets/resolve/main/RNASolo_31102023_raw.tar.gz` (and `_processed.pt`, `_processed_df.csv`) | **14,366 cleaned PDBs** + metadata | 6.9 GB |
| BGSU NR list 4.57 | `pdb-hunter rna bgsu-nrlist` (endpoint `https://rna.bgsu.edu/rna3dhub/nrlist/download/NR/4.57/{res}/csv`) | 7 cutoffs, 1.5A→20.0A | <1 MB |
| BGSU Motif Atlas 4.14 | `pdb-hunter rna motifs-download --type il|hl --release 4.14` | IL 412 + HL 256 motifs | <1 MB |
| BGSU chain→Rfam map | `https://rna.bgsu.edu/data/pdb_chain_to_best_rfam.txt` | chain→family mapping | 872 KB |
| PDB seqres | `https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz` | all PDB sequences | 67 MB |

**Harvest source endpoints** (used by `pdb-hunter rna harvest`):
- RCSB RNA query: `https://search.rcsb.org/rcsbsearch/v2/query`
  (`entity_poly.rcsb_entity_polymer_type = RNA`, 10,399 entries)
- BGSU NR members: `https://rna.bgsu.edu/rna3dhub/nrlist/download/NR/4.57/{res}/{fmt}`
- RNAsolo bundles: `https://rnasolo.cs.put.poznan.pl/` (BGSU "M" set,
  pdb/cif/fasta/dbn/bpseq archives)

**Annotation enrichment** (`pdb-hunter rna enrich`):
- RCSB validation — **10,449 entries** (clashscore, R-free, resolution)
- RNA3DB metadata — **41,467 chain records**
- PDBe validation/secondary structure — **0/10,452; upstream API now returns 404** (documented gap)

**Raw-PDB entry list** is the union of `data/catalog/pdb_hunter_index.json`,
the fresh harvest, RNA3DB `.cif` filenames and gRNASolo `.pdb` filenames →
**10,527 requested, 10,526 atomic, 0 failed, 1 excluded (9A0D, integrative)**.

---

## 5. Families and MSA

| Dataset | Source URL | Content | Size |
|---|---|---|---|
| Rfam 15.1 seed | `https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1/Rfam.seed.gz` | **4,227 families** | 5.9 MB |
| Rfam covariance models | `…/Rfam.cm.gz` | CMs | 45.8 MB |
| Rfam region table | `…/Rfam.full_region.gz` | full annotations | 125.4 MB |
| Rfam 3D seeds | `…/Rfam.3d.seed.gz` | 3D-curated seeds | 377 KB |
| Rfam PDB map | `…/Rfam.pdb.gz` | family↔PDB | 87 KB |
| Rfam seed tree | `…/Rfam.seed_tree.tar.gz` | tree | 2.6 MB |
| Rfam bundle | `…/Rfam.tar.gz` | all-in-one | 46.1 MB |
| **Rfam full alignments** | `https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1/full_alignments/RFNNNNN.sto` | **4,077 Stockholm files** | 3.25 GB |

---

## 6. Benchmarks and supervision channels

### 6.1 Secondary structure (2D)

| Dataset | Source URL | Records |
|---|---|---|
| bpRNA-1m (5 formats) | `https://bprna.cgrb.oregonstate.edu/bpRNA_1m/{dbn,bpseq,fasta,ct,st}Files.zip` | **102,318** |
| bpRNA-1m (Zenodo) | `https://zenodo.org/api/records/16730061/files/bpRNA_1m.tar.gz/content` | — |
| ArchiveII | `https://huggingface.co/datasets/multimolecule/archiveii` (`test.parquet`) | **3,975** |
| ArchiveII (pickles) | `https://zenodo.org/api/records/16730061/files/ArchiveII.tar.gz/content` | 3,986 |
| RNAStrAlign | `https://huggingface.co/datasets/multimolecule/rnastralign` (`train.parquet`) | **37,149** |
| RNAStrAlign BPSEQ | `https://ndownloader.figshare.com/files/45555225` | — |
| RNAStrAlign JSON | `https://huggingface.co/datasets/rouskinlab/RNAstralign` (`data.json`) | — |
| bpRNA-spot (parquet) | `https://huggingface.co/datasets/multimolecule/bprna-spot` | train 10,934 / val 1,330 / test 1,411 |
| bpRNA-1m JSON | `https://huggingface.co/datasets/rouskinlab/bpRNA-1m` (`data.json`) | — |
| bpRNA-new | `https://huggingface.co/datasets/multimolecule/bprna-new` (`test.parquet`) | **5,401** |
| SPOT-RNA splits | `https://www.dropbox.com/s/w3kc4iro8ztbf3m/bpRNA_dataset.zip?dl=1`, `…/vnq0k9dg7vynu3q/PDB_dataset.zip?dl=1` | TR0/VL0/TS0, TR1/VL1/TS1 |

### 6.2 Chemical probing / mapping

| Dataset | Source URL | Content | Size |
|---|---|---|---|
| **RMDB (complete)** | `https://rmdb.stanford.edu/manifest.json` + assets under `https://github.com/DasLab/rmdb.github.io/releases/download/{data-rna-structures,data-riboswitches,data-puzzle,data-general,data-eterna}/*.rdat` | **1,024 constructs, 516.4M reactivity values, 21 Mg²⁺ titration ladders** | 12.51 GB |
| EternaBench | `https://huggingface.co/datasets/multimolecule/eternabench-{cm,switch,external.300,external.600,external.900,external.1200}` | chemical mapping benchmarks (cm: 12,293 train / 1,260 test) | 29 MB |
| Ribonanza quick-start | `https://huggingface.co/datasets/TerminatorJ/RNA_chemical_ribonanza/resolve/main/train_data_QUICK_START.csv` | **335,616 profiles** (2A3 + DMS) | 550.7 MB |
| PDB-RibonanzaNet | `https://huggingface.co/datasets/TerminatorJ/PDB_Ribonanzanet/resolve/main/{train,val,test}.csv` | PDB-derived RibonanzaNet training rows | 0.6 MB |

### 6.3 Fitness and function

| Dataset | Source URL | Content |
|---|---|---|
| RNAGym processed | `https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_processed_assays.zip` | 70 assays |
| RNAGym raw | `https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_raw_data.zip` | raw measurements |
| RNAGym repo | `https://github.com/MarksLab-DasLab/RNAGym` | baselines |
| NABench | `https://github.com/mrzzmrzz/NABench` | 162 assays, 2.6M mutations |
| CPEB3 | `https://huggingface.co/datasets/Marks-lab/RNAgym/resolve/main/fitness_prediction/assays/Zhang_2020_cpeb3_ribozyme.parquet` | 111,417 ribozyme variants |

### 6.4 Splicing

| Dataset | Source URL | Content |
|---|---|---|
| Spliceator | `https://bigest-icube.fr/spliceator/static/data/data.tar.gz` | 4-species benchmark (uncompressed tar despite `.tar.gz`) |
| SpliceBERT | `https://zenodo.org/records/7995778/files/data.tar.gz?download=1` | 72 vertebrates, 8.6 GB |
| G3PO | `https://github.com/BiGEst-ICube/g3po` | 147 species, G3PO.csv |

### 6.5 Blind tests

| Dataset | Source URL | Content |
|---|---|---|
| CASP15 | `https://files.rcsb.org/download/{7QR4,7QR3,8S95,8FZA,8TVZ,8BTZ,7ZJ4,7PTK,7PTL,8UYS,8UYE,8UYG,8UYJ,7YR7,7YR6}.cif` | 15 targets |
| CASP16 | `https://files.rcsb.org/download/{8UO6,9CFN,9C2K,9DCF,9B0L,9ELY,9BZC,9BZ1,9CBU,9CBX}.cif` | 10 targets |
| RNA-Puzzles std | `https://github.com/mmagnus/RNA-Puzzles-Standardized-Submissions` | PZ1–PZ21+ |
| RNA-Puzzles site | `https://github.com/rnapuzzles/rnapuzzles.github.io` | targets + models |

---

## 7. Verification summary

- **elDORS**: `sha256sum -c` → 20/20 OK.
- **Archives**: 10,488 tar/gzip/xz/zip/parquet files checked → 0 failures
  (`data/acquisition/verification_report.json`). One upstream mislabel noted:
  Spliceator's `data.tar.gz` is an uncompressed tar.
- **Record counts** (`data/acquisition/data_audit.json`), measured vs documented:

| Corpus | Measured | Documented |
|---|---|---|
| elDORS sequences | 1,323,715,880 | 1,323,715,880 |
| MARS sequences | 1,727,789,860 | ~1.73 B |
| RNAcentral active | 46,210,324 | 46,210,324 |
| Rfam seed families | 4,227 | 4,227 |
| RMDB constructs | 1,024 (21 Mg titrations) | 1,024 |
| raw PDB atomic entries | 10,526 (+9A0D excluded) | 10,526 |
| pdb_hunter entries | 10,452 complete | 10,452 |
| pdb_hunter dot-bracket entries | 9,438 | 9,347 |
| pdb_hunter backbone xyz | 10,358 | 10,280 |
| RNA3DB chains | 15,441 | 15,441 |
| gRNAde PDBs | 14,366 | 14,369 |
| bpRNA-1m | 102,318 | 102,318 |
| ArchiveII | 3,975 | 3,975 |
| RNAStrAlign | 37,149 | 37,149 |
| bpRNA-new | 5,401 | 5,401 |
| CASP15 / CASP16 | 15 / 10 | 15 / 10 |

Known deltas: gRNAde counts 14,366 vs 14,369 (counting convention);
bpRNA-spot parquet splits differ from the paper's TR0/VL0/TS0
(10,934/1,330/1,411 vs 10,815/1,301/1,306 — mirror revision); BGSU
indices are 4.57/4.14 vs docs' 4.56/4.12; RMDB value count is 99.2% of the
manifest figure (53 legacy `DATA:` constructs now included).

Two fixes made during acquisition:
1. `scripts/acquire_raw_pdb_entries.py` — IHM detection scanned only the
   first 2 MB, but 9A0D's `_ihm_sphere_obj_site` tag is at 5.8 MB; now the
   full stream is scanned on the failure path.
2. Raw-PDB entry list rebuilt as the union described in §4, extending the
   archive from 10,423 to the full 10,527-request corpus.

---

## 8. Not acquired (and why)

| Source | Link | Size | Blocker |
|---|---|---|---|
| Ribonanza full (2.1M) | `https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data` | ~tens of GB | Kaggle credentials |
| Kaggle Ribonanza mirrors | `iafoss/stanford-ribonanza-rna-folding-converted` (648 MB), `alexandervc/…-data` (282 MB), `shlomoron/srrf-tfrecords-ds` (583 MB), `shujun717/ribonanza-3d-coords` (8.0 GB) | — | Kaggle auth |
| elDORS rMSA build | `s3://eldors-v1-database/rMSA_optimized_elDORS/` | 1,097 GB | deferred (MSA branch) |
| elDORS RNAcmap3 build | `s3://eldors-v1-database/RNAcmap3_optimized_elDORS/` | 924 GB | deferred (MSA branch) |
| elDORS raw (unclustered) | `s3://eldors-v1-database/elDORS_v1_raw/` | 1,225 GB | superseded by clustered v1 |
| RNA–RNA contacts | PARIS `GSE74353`, LIGR-seq `GSE80167`, RISE DB, InterRRact supp. | small tables | candidate channel, not yet acquired |
| MODOMICS | `https://www.genesilico.pl/modomics/api/modifications-all?format=csv` (verified live) | ~250 KB+ | candidate channel |
| RMBase v3 / m6A-Atlas v2 | `https://rna.sysu.edu.cn/rmbase3/download.php`, `https://rnamd.org/m6a/repository.php` | GBs | JS portals |
| RNAmd / SASBDB | `https://huggingface.co/datasets/LlewynLuo/RNAmd-v1`, `https://www.sasbdb.org/` | 70 MB / GBs | candidate dynamics channel |
| ENCODE eCLIP | `https://www.encodeproject.org/metadata/?type=Experiment&assay_title=eCLIP&…&format=tsv` (verified) | ~1.4k files | candidate channel |
| R-BIND | `https://rbind.chem.duke.edu/` | small | candidate ligand channel |
| PseudoBase++ | `https://rnavlab.utep.edu/database` | ~300–400 records | site flaky |
| PDBe enrichment | `https://www.ebi.ac.uk/pdbe/api/…` | — | upstream 404 at acquisition time |

---

## 9. Methods — how the 833 GB was acquired

### 9.1 Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # uv 0.12.17
git clone git@github.com:E-Motioner-X-SBS/pdb_hunter.git ../pdb_hunter
uv sync                                                  # .venv + editable pdb-hunter
```

`pyproject.toml` pins the Python range (3.10–3.13) and lists `pdb-hunter`
as an editable path dependency; `uv.lock` records the resolved graph.

### 9.2 Orchestrator design (`scripts/acquire_all.py`)

- Every source is a `Job`: `http` (resumable byte-range download), `git`
  (shallow clone) or `cmd` (a script/CLI run whose directory output is
  recorded by a marker).
- Downloads write `<dest>.part` and rename only when the expected byte
  size is reached, so the existence of `<dest>` means complete and
  re-running costs one `stat`.
- Transient failures retry with backoff; **any byte progress resets the
  retry budget** (NGDC closes long connections every ~10–150 MB).
- Jobs run in a thread pool; each has its own log under
  `data/acquisition/logs/<job>.log`.

```bash
uv run python scripts/acquire_all.py --list                 # 147 jobs, sizes
uv run python scripts/acquire_all.py --group sequence --workers 20
uv run python scripts/acquire_all.py --group catalog  --workers 12
uv run python scripts/acquire_all.py --group long     --workers 2
uv run python scripts/acquire_all.py --only mars-04 --workers 1
```

`scripts/watch_downloads.py` is the unattended supervisor: every 3–5
minutes it checks each group's process, restarts a group whose jobs are
pending, and writes `data/acquisition/state/ALL_DONE` when nothing
remains. Directory-producing `cmd` jobs get explicit `.done` markers
(`rna-harvest`, `raw-pdb-entries`, `rmdb`, `rfam-full-alignments`).
`scripts/download_status.py` prints per-job byte progress.

### 9.3 Sequence corpora

- elDORS: 20 chunks as individual jobs (S3 anonymous), then
  `cd data/sequences/elDORS_v1 && sha256sum -c elDORS_v1_manifest.sha256`.
- MARS: 30 parts interleaved with the elDORS jobs in the `sequence` group
  so both made progress; sizes for 29 parts are pinned in the script,
  part 04 was verified against its upstream `Content-Length` separately.
- RNAcentral: active / inactive / species-specific as plain `http` jobs.
- One deliberate restart: the first `sequence` run ordered elDORS before
  MARS; it was killed and relaunched with the interleaved order (partial
  `.part` files resumed).

### 9.4 pdb_hunter steps

```bash
# 154 GB per-entry database (raw + clean + dbn/bpseq/xyz), resumable
uv run pdb-hunter rna harvest --out ../pdb_hunter/RNA_Database \
    --workers 12 --nr-release 4.57

# annotation layer: RCSB validation (clashscore), RNA3DB chain metadata
tar -xzf data/structures/databases/rna3db/rna3db-jsons.tar.gz \
    -C data/structures/databases/rna3db/jsons
uv run pdb-hunter rna enrich --out ../pdb_hunter/RNA_Database \
    --jsons-dir data/structures/databases/rna3db/jsons/rna3db-jsons \
    --workers 12

# BGSU indices (7 nrlist cutoffs + IL/HL motif atlases)
uv run pdb-hunter rna bgsu-nrlist --resolution 3.0A --format csv --out …
uv run pdb-hunter rna motifs-download --type il --release 4.14 --out …
```

### 9.5 Raw-PDB union entry list

`scripts/build_raw_pdb_entrylist.py` unions pdb_hunter's index, the fresh
harvest, RNA3DB filenames and gRNASolo filenames into
`data/structures/raw_pdb_entrylist.txt` (10,527 IDs), then

```bash
uv run python scripts/acquire_raw_pdb_entries.py --workers 12
```

fetches whatever is missing, validates every file by streaming to its
`_atom_site` tag, and classifies integrative/hybrid models (9A0D) as
`excluded_ihm` rather than failed.

### 9.6 RMDB and Rfam full alignments

- `scripts/acquire_rmdb.py` reads `https://rmdb.stanford.edu/manifest.json`,
  enumerates all five `DasLab/rmdb.github.io` data releases via the GitHub
  API, and downloads every `.rdat` with resume (1,024 files, 12.5 GB).
- `scripts/acquire_rfam_full.py` parses the EBI FTP directory listing and
  downloads all 4,077 `full_alignments/*.sto` files (3.25 GB).

Both were smoke-tested with `--limit 3` before the full run, and both are
wired into `acquire_all.py` as `cmd` jobs.

### 9.7 Verification

```bash
uv run python scripts/verify_downloads.py --workers 12   # 10,488 archives
uv run python scripts/audit_data.py                      # record counts
cd data/sequences/elDORS_v1 && sha256sum -c elDORS_v1_manifest.sha256
```

Reports land in `data/acquisition/verification_report.json` and
`data/acquisition/data_audit.json`; per-download logs in
`data/acquisition/logs/`.

### 9.8 Quick start

```bash
uv sync
uv run python scripts/acquire_all.py --list
uv run python scripts/acquire_all.py --group sequence
uv run python scripts/acquire_all.py --group catalog
uv run python scripts/acquire_all.py --group long
uv run python scripts/watch_downloads.py
uv run python scripts/verify_downloads.py
uv run python scripts/audit_data.py
```
