# Sources — every link, every program

**Generated** by `scripts/sampling/emit_source_inventory.py` from `build_jobs()` in `scripts/acquire_all.py`. Do not edit by hand: regenerate it, so the documentation cannot drift from the downloader that is actually used.

147 jobs in 3 groups. 69 carry a verified byte size, totalling **646.34 GB**; the rest are directory or repository fetches whose size is not known before the transfer.

## How to fetch

```bash
uv run python scripts/acquire_all.py --list            # the plan
uv run python scripts/acquire_all.py --group sequence  # ~1.3B seqs
uv run python scripts/acquire_all.py --group catalog   # everything else
uv run python scripts/acquire_all.py --only mars       # the 427 GB D18 skips
```

> **On this machine `--list` under-reports.** The corpus was downloaded on another host and only `data/` was copied across, not `data/acquisition/`, so jobs whose completion is recorded by a state marker read as pending while their output is on disk. `scripts/sampling/audit_inventory_gap.py` measures the tree as it is and is the authority: 17 of 18 sources present, 0 missing, 1 absent by decision.

## The programs, and what they need

| program | role |
|---|---|
| `scripts/acquire_all.py` | the plan and the driver: 147 jobs, resumable, idempotent |
| `scripts/acquire_raw_pdb_entries.py` | every RNA-containing PDB entry as mmCIF |
| `scripts/acquire_rfam_full.py` | Rfam full-region alignments |
| `scripts/acquire_rmdb.py` | RMDB Mg2+ titration ladders (GitHub release assets; the old Django bulk endpoint is gone) |
| `scripts/build_raw_pdb_entrylist.py` | the RCSB entry list the above walks |
| `scripts/watch_downloads.py` | progress against expected sizes |
| `scripts/sampling/audit_inventory_gap.py` | what is actually on disk, per source |
| `scripts/eldors_to_parquet.py` | FASTA chunks -> the parquet the trainer reads |

**One external dependency.** The `bgsu-*` jobs shell out to `pdb-hunter`, a
sibling repository, declared in `pyproject.toml` as
`pdb-hunter = { path = "../pdb_hunter", editable = true }`:

```bash
git clone https://github.com/E-Motioner-X-SBS/pdb_hunter.git ../pdb_hunter
```

It is also where the largest single corpus lives: `pdb_hunter/RNA_Database`,
**244 GiB**, referenced rather than copied under decision D19 — which is why
`du -sh data/` on this machine reports 292 GB while the inventory totals 567 GB.

## `sequence` — 54 jobs, 633.60 GB known

| job | host | size | source | lands in |
|---|---|---|---|---|
| `eldors-001` | aws-s3 | 9.38 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_001.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_001.fasta.gz` |
| `eldors-002` | aws-s3 | 9.39 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_002.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_002.fasta.gz` |
| `eldors-003` | aws-s3 | 9.46 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_003.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_003.fasta.gz` |
| `eldors-004` | aws-s3 | 9.38 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_004.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_004.fasta.gz` |
| `eldors-005` | aws-s3 | 9.39 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_005.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_005.fasta.gz` |
| `eldors-006` | aws-s3 | 9.46 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_006.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_006.fasta.gz` |
| `eldors-007` | aws-s3 | 9.38 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_007.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_007.fasta.gz` |
| `eldors-008` | aws-s3 | 9.39 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_008.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_008.fasta.gz` |
| `eldors-009` | aws-s3 | 9.46 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_009.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_009.fasta.gz` |
| `eldors-010` | aws-s3 | 9.38 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_010.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_010.fasta.gz` |
| `eldors-011` | aws-s3 | 9.40 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_011.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_011.fasta.gz` |
| `eldors-012` | aws-s3 | 9.44 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_012.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_012.fasta.gz` |
| `eldors-013` | aws-s3 | 9.39 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_013.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_013.fasta.gz` |
| `eldors-014` | aws-s3 | 9.43 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_014.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_014.fasta.gz` |
| `eldors-015` | aws-s3 | 9.42 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_015.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_015.fasta.gz` |
| `eldors-016` | aws-s3 | 9.39 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_016.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_016.fasta.gz` |
| `eldors-017` | aws-s3 | 9.45 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_017.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_017.fasta.gz` |
| `eldors-018` | aws-s3 | 9.40 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_018.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_018.fasta.gz` |
| `eldors-019` | aws-s3 | 9.38 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_019.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_019.fasta.gz` |
| `eldors-020` | aws-s3 | 3.58 GB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_020.fasta.gz> | `data/sequences/elDORS_v1/elDORS_v1_020.fasta.gz` |
| `eldors-manifest` | aws-s3 | 1.78 kB | <https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks/elDORS_v1_manifest.sha256> | `data/sequences/elDORS_v1/elDORS_v1_manifest.sha256` |
| `mars-01` | ngdc | 13.38 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-01.tgz> | `data/sequences/mars/OMIX003037-01.tgz` |
| `mars-02` | ngdc | 12.63 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-02.tgz> | `data/sequences/mars/OMIX003037-02.tgz` |
| `mars-03` | ngdc | 14.17 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-03.tgz> | `data/sequences/mars/OMIX003037-03.tgz` |
| `mars-04` | ngdc | 11.94 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-04.tgz> | `data/sequences/mars/OMIX003037-04.tgz` |
| `mars-05` | ngdc | 12.34 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-05.tgz> | `data/sequences/mars/OMIX003037-05.tgz` |
| `mars-06` | ngdc | 12.61 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-06.tgz> | `data/sequences/mars/OMIX003037-06.tgz` |
| `mars-07` | ngdc | 12.67 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-07.tgz> | `data/sequences/mars/OMIX003037-07.tgz` |
| `mars-08` | ngdc | 13.02 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-08.tgz> | `data/sequences/mars/OMIX003037-08.tgz` |
| `mars-09` | ngdc | 13.07 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-09.tgz> | `data/sequences/mars/OMIX003037-09.tgz` |
| `mars-10` | ngdc | 13.45 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-10.tgz> | `data/sequences/mars/OMIX003037-10.tgz` |
| `mars-11` | ngdc | 14.41 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-11.tgz> | `data/sequences/mars/OMIX003037-11.tgz` |
| `mars-12` | ngdc | 14.40 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-12.tgz> | `data/sequences/mars/OMIX003037-12.tgz` |
| `mars-13` | ngdc | 13.37 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-13.tgz> | `data/sequences/mars/OMIX003037-13.tgz` |
| `mars-14` | ngdc | 15.64 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-14.tgz> | `data/sequences/mars/OMIX003037-14.tgz` |
| `mars-15` | ngdc | 15.51 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-15.tgz> | `data/sequences/mars/OMIX003037-15.tgz` |
| `mars-16` | ngdc | 15.56 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-16.tgz> | `data/sequences/mars/OMIX003037-16.tgz` |
| `mars-17` | ngdc | 15.49 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-17.tgz> | `data/sequences/mars/OMIX003037-17.tgz` |
| `mars-18` | ngdc | 15.45 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-18.tgz> | `data/sequences/mars/OMIX003037-18.tgz` |
| `mars-19` | ngdc | 15.57 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-19.tgz> | `data/sequences/mars/OMIX003037-19.tgz` |
| `mars-20` | ngdc | 15.40 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-20.tgz> | `data/sequences/mars/OMIX003037-20.tgz` |
| `mars-21` | ngdc | 15.40 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-21.tgz> | `data/sequences/mars/OMIX003037-21.tgz` |
| `mars-22` | ngdc | 15.42 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-22.tgz> | `data/sequences/mars/OMIX003037-22.tgz` |
| `mars-23` | ngdc | 15.44 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-23.tgz> | `data/sequences/mars/OMIX003037-23.tgz` |
| `mars-24` | ngdc | 14.87 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-24.tgz> | `data/sequences/mars/OMIX003037-24.tgz` |
| `mars-25` | ngdc | 14.81 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-25.tgz> | `data/sequences/mars/OMIX003037-25.tgz` |
| `mars-26` | ngdc | 15.11 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-26.tgz> | `data/sequences/mars/OMIX003037-26.tgz` |
| `mars-27` | ngdc | 14.29 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-27.tgz> | `data/sequences/mars/OMIX003037-27.tgz` |
| `mars-28` | ngdc | 14.42 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-28.tgz> | `data/sequences/mars/OMIX003037-28.tgz` |
| `mars-29` | ngdc | 14.27 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-29.tgz> | `data/sequences/mars/OMIX003037-29.tgz` |
| `mars-30` | ngdc | 13.17 GB | <https://download.cncb.ac.cn/OMIX/OMIX003037/OMIX003037-30.tgz> | `data/sequences/mars/OMIX003037-30.tgz` |
| `rnacentral-active` | ebi | 10.92 GB | <https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz> | `data/sequences/rnacentral/rnacentral_active.fasta.gz` |
| `rnacentral-inactive` | ebi | 1.33 GB | <https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_inactive.fasta.gz> | `data/sequences/rnacentral/rnacentral_inactive.fasta.gz` |
| `rnacentral-species-specific` | ebi | 11.69 GB | <https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_species_specific_ids.fasta.gz> | `data/sequences/rnacentral/rnacentral_species_specific_ids.fasta.gz` |

## `catalog` — 91 jobs, 12.74 GB known

| job | host | size | source | lands in |
|---|---|---|---|---|
| `archiveii-ct` | zenodo | — | <https://zenodo.org/api/records/16730061/files/ArchiveII.tar.gz/content> | `data/benchmarks/secondary_structure/archiveii/ArchiveII.tar.gz` |
| `bgsu-motifs-hl` | bgsu | — | `run pdb-hunter rna motifs-download --type hl --release 4.14 --out` | `data/structures/indices/bgsu_motifs/motif_atlas_4.14_hl.csv` |
| `bgsu-motifs-il` | bgsu | — | `run pdb-hunter rna motifs-download --type il --release 4.14 --out` | `data/structures/indices/bgsu_motifs/motif_atlas_4.14_il.csv` |
| `bgsu-nrlist-1.5A` | bgsu | — | `run pdb-hunter rna bgsu-nrlist --resolution 1.5A --format csv --out` | `data/structures/indices/bgsu_nrlist/nrlist_4.57_1.5A.csv` |
| `bgsu-nrlist-2.0A` | bgsu | — | `run pdb-hunter rna bgsu-nrlist --resolution 2.0A --format csv --out` | `data/structures/indices/bgsu_nrlist/nrlist_4.57_2.0A.csv` |
| `bgsu-nrlist-2.5A` | bgsu | — | `run pdb-hunter rna bgsu-nrlist --resolution 2.5A --format csv --out` | `data/structures/indices/bgsu_nrlist/nrlist_4.57_2.5A.csv` |
| `bgsu-nrlist-20.0A` | bgsu | — | `run pdb-hunter rna bgsu-nrlist --resolution 20.0A --format csv --out` | `data/structures/indices/bgsu_nrlist/nrlist_4.57_20.0A.csv` |
| `bgsu-nrlist-3.0A` | bgsu | — | `run pdb-hunter rna bgsu-nrlist --resolution 3.0A --format csv --out` | `data/structures/indices/bgsu_nrlist/nrlist_4.57_3.0A.csv` |
| `bgsu-nrlist-3.5A` | bgsu | — | `run pdb-hunter rna bgsu-nrlist --resolution 3.5A --format csv --out` | `data/structures/indices/bgsu_nrlist/nrlist_4.57_3.5A.csv` |
| `bgsu-nrlist-4.0A` | bgsu | — | `run pdb-hunter rna bgsu-nrlist --resolution 4.0A --format csv --out` | `data/structures/indices/bgsu_nrlist/nrlist_4.57_4.0A.csv` |
| `bgsu-rfam-map` | bgsu | — | <https://rna.bgsu.edu/data/pdb_chain_to_best_rfam.txt> | `data/structures/indices/pdb_chain_to_best_rfam.txt` |
| `bprna-bpseq` | osu | — | <https://bprna.cgrb.oregonstate.edu/bpRNA_1m/bpseqFiles.zip> | `data/benchmarks/secondary_structure/bprna_full/bpRNA_1m_bpseq.zip` |
| `bprna-ct` | osu | — | <https://bprna.cgrb.oregonstate.edu/bpRNA_1m/ctFiles.zip> | `data/benchmarks/secondary_structure/bprna_full/bpRNA_1m_ct.zip` |
| `bprna-dbn` | osu | — | <https://bprna.cgrb.oregonstate.edu/bpRNA_1m/dbnFiles.zip> | `data/benchmarks/secondary_structure/bprna_full/bpRNA_1m_dbn.zip` |
| `bprna-fasta` | osu | — | <https://bprna.cgrb.oregonstate.edu/bpRNA_1m/fastaFiles.zip> | `data/benchmarks/secondary_structure/bprna_full/bpRNA_1m_fasta.zip` |
| `bprna-st` | zenodo | — | <https://zenodo.org/api/records/16730061/files/bpRNA_1m.tar.gz/content> | `data/benchmarks/secondary_structure/bprna_full/bpRNA_1m.tar.gz` |
| `bprna-st` | osu | — | <https://bprna.cgrb.oregonstate.edu/bpRNA_1m/stFiles.zip> | `data/benchmarks/secondary_structure/bprna_full/bpRNA_1m_st.zip` |
| `casp15-7PTK` | rcsb | — | <https://files.rcsb.org/download/7PTK.cif> | `data/structures/blind_tests/casp15/7PTK.cif` |
| `casp15-7PTL` | rcsb | — | <https://files.rcsb.org/download/7PTL.cif> | `data/structures/blind_tests/casp15/7PTL.cif` |
| `casp15-7QR3` | rcsb | — | <https://files.rcsb.org/download/7QR3.cif> | `data/structures/blind_tests/casp15/7QR3.cif` |
| `casp15-7QR4` | rcsb | — | <https://files.rcsb.org/download/7QR4.cif> | `data/structures/blind_tests/casp15/7QR4.cif` |
| `casp15-7YR6` | rcsb | — | <https://files.rcsb.org/download/7YR6.cif> | `data/structures/blind_tests/casp15/7YR6.cif` |
| `casp15-7YR7` | rcsb | — | <https://files.rcsb.org/download/7YR7.cif> | `data/structures/blind_tests/casp15/7YR7.cif` |
| `casp15-7ZJ4` | rcsb | — | <https://files.rcsb.org/download/7ZJ4.cif> | `data/structures/blind_tests/casp15/7ZJ4.cif` |
| `casp15-8BTZ` | rcsb | — | <https://files.rcsb.org/download/8BTZ.cif> | `data/structures/blind_tests/casp15/8BTZ.cif` |
| `casp15-8FZA` | rcsb | — | <https://files.rcsb.org/download/8FZA.cif> | `data/structures/blind_tests/casp15/8FZA.cif` |
| `casp15-8S95` | rcsb | — | <https://files.rcsb.org/download/8S95.cif> | `data/structures/blind_tests/casp15/8S95.cif` |
| `casp15-8TVZ` | rcsb | — | <https://files.rcsb.org/download/8TVZ.cif> | `data/structures/blind_tests/casp15/8TVZ.cif` |
| `casp15-8UYE` | rcsb | — | <https://files.rcsb.org/download/8UYE.cif> | `data/structures/blind_tests/casp15/8UYE.cif` |
| `casp15-8UYG` | rcsb | — | <https://files.rcsb.org/download/8UYG.cif> | `data/structures/blind_tests/casp15/8UYG.cif` |
| `casp15-8UYJ` | rcsb | — | <https://files.rcsb.org/download/8UYJ.cif> | `data/structures/blind_tests/casp15/8UYJ.cif` |
| `casp15-8UYS` | rcsb | — | <https://files.rcsb.org/download/8UYS.cif> | `data/structures/blind_tests/casp15/8UYS.cif` |
| `casp16-8UO6` | rcsb | — | <https://files.rcsb.org/download/8UO6.cif> | `data/structures/blind_tests/casp16/8UO6.cif` |
| `casp16-9B0L` | rcsb | — | <https://files.rcsb.org/download/9B0L.cif> | `data/structures/blind_tests/casp16/9B0L.cif` |
| `casp16-9BZ1` | rcsb | — | <https://files.rcsb.org/download/9BZ1.cif> | `data/structures/blind_tests/casp16/9BZ1.cif` |
| `casp16-9BZC` | rcsb | — | <https://files.rcsb.org/download/9BZC.cif> | `data/structures/blind_tests/casp16/9BZC.cif` |
| `casp16-9C2K` | rcsb | — | <https://files.rcsb.org/download/9C2K.cif> | `data/structures/blind_tests/casp16/9C2K.cif` |
| `casp16-9CBU` | rcsb | — | <https://files.rcsb.org/download/9CBU.cif> | `data/structures/blind_tests/casp16/9CBU.cif` |
| `casp16-9CBX` | rcsb | — | <https://files.rcsb.org/download/9CBX.cif> | `data/structures/blind_tests/casp16/9CBX.cif` |
| `casp16-9CFN` | rcsb | — | <https://files.rcsb.org/download/9CFN.cif> | `data/structures/blind_tests/casp16/9CFN.cif` |
| `casp16-9DCF` | rcsb | — | <https://files.rcsb.org/download/9DCF.cif> | `data/structures/blind_tests/casp16/9DCF.cif` |
| `casp16-9ELY` | rcsb | — | <https://files.rcsb.org/download/9ELY.cif> | `data/structures/blind_tests/casp16/9ELY.cif` |
| `cpeb3` | huggingface | — | <https://huggingface.co/datasets/Marks-lab/RNAgym/resolve/main/fitness_prediction/assays/Zhang_2020_cpeb3_ribozyme.parquet> | `data/benchmarks/fitness/rnagym/Zhang_2020_cpeb3_ribozyme.parquet` |
| `eternabench-cm-test` | huggingface | — | <https://huggingface.co/datasets/multimolecule/eternabench-cm/resolve/main/test.parquet> | `data/benchmarks/probing/eternabench-cm/test.parquet` |
| `eternabench-cm-train` | huggingface | — | <https://huggingface.co/datasets/multimolecule/eternabench-cm/resolve/main/train.parquet> | `data/benchmarks/probing/eternabench-cm/train.parquet` |
| `eternabench-external.1200-test` | huggingface | — | <https://huggingface.co/datasets/multimolecule/eternabench-external.1200/resolve/main/test.parquet> | `data/benchmarks/probing/eternabench-external.1200/test.parquet` |
| `eternabench-external.300-test` | huggingface | — | <https://huggingface.co/datasets/multimolecule/eternabench-external.300/resolve/main/test.parquet> | `data/benchmarks/probing/eternabench-external.300/test.parquet` |
| `eternabench-external.600-test` | huggingface | — | <https://huggingface.co/datasets/multimolecule/eternabench-external.600/resolve/main/test.parquet> | `data/benchmarks/probing/eternabench-external.600/test.parquet` |
| `eternabench-external.900-test` | huggingface | — | <https://huggingface.co/datasets/multimolecule/eternabench-external.900/resolve/main/test.parquet> | `data/benchmarks/probing/eternabench-external.900/test.parquet` |
| `eternabench-switch-test` | huggingface | — | <https://huggingface.co/datasets/multimolecule/eternabench-switch/resolve/main/test.parquet> | `data/benchmarks/probing/eternabench-switch/test.parquet` |
| `eternabench-switch-train` | huggingface | — | <https://huggingface.co/datasets/multimolecule/eternabench-switch/resolve/main/train.parquet> | `data/benchmarks/probing/eternabench-switch/train.parquet` |
| `g3po-repo` | github | — | <https://github.com/BiGEst-ICube/g3po> | `data/benchmarks/splicing/g3po` |
| `grnade-rnasolo_31102023_processed` | huggingface | 2.79 GB | <https://huggingface.co/datasets/chaitjo/gRNAde_datasets/resolve/main/RNASolo_31102023_processed.pt> | `data/structures/databases/grnade_rnasolo/RNASolo_31102023_processed.pt` |
| `grnade-rnasolo_31102023_processed_df` | huggingface | 3.70 MB | <https://huggingface.co/datasets/chaitjo/gRNAde_datasets/resolve/main/RNAsolo_31102023_processed_df.csv> | `data/structures/databases/grnade_rnasolo/RNAsolo_31102023_processed_df.csv` |
| `grnade-rnasolo_31102023_raw` | huggingface | 6.86 GB | <https://huggingface.co/datasets/chaitjo/gRNAde_datasets/resolve/main/RNASolo_31102023_raw.tar.gz> | `data/structures/databases/grnade_rnasolo/RNASolo_31102023_raw.tar.gz` |
| `hf-archiveii-test.parquet` | huggingface | — | <https://huggingface.co/datasets/multimolecule/archiveii/resolve/main/test.parquet> | `data/benchmarks/secondary_structure/archiveii/test.parquet` |
| `hf-bprna_full-data.json` | huggingface | — | <https://huggingface.co/datasets/rouskinlab/bpRNA-1m/resolve/main/data.json> | `data/benchmarks/secondary_structure/bprna_full/data.json` |
| `hf-bprna_new-test.parquet` | huggingface | — | <https://huggingface.co/datasets/multimolecule/bprna-new/resolve/main/test.parquet> | `data/benchmarks/secondary_structure/bprna_new/test.parquet` |
| `hf-bprna_spot-test.parquet` | huggingface | — | <https://huggingface.co/datasets/multimolecule/bprna-spot/resolve/main/test.parquet> | `data/benchmarks/secondary_structure/bprna_spot/test.parquet` |
| `hf-bprna_spot-train.parquet` | huggingface | — | <https://huggingface.co/datasets/multimolecule/bprna-spot/resolve/main/train.parquet> | `data/benchmarks/secondary_structure/bprna_spot/train.parquet` |
| `hf-bprna_spot-validation.parquet` | huggingface | — | <https://huggingface.co/datasets/multimolecule/bprna-spot/resolve/main/validation.parquet> | `data/benchmarks/secondary_structure/bprna_spot/validation.parquet` |
| `hf-rnastralign-data.json` | huggingface | — | <https://huggingface.co/datasets/rouskinlab/RNAstralign/resolve/main/data.json> | `data/benchmarks/secondary_structure/rnastralign/data.json` |
| `hf-rnastralign-train.parquet` | huggingface | — | <https://huggingface.co/datasets/multimolecule/rnastralign/resolve/main/train.parquet> | `data/benchmarks/secondary_structure/rnastralign/train.parquet` |
| `nabench-repo` | github | — | <https://github.com/mrzzmrzz/NABench> | `data/benchmarks/fitness/nabench` |
| `pdb-ribonanzanet-test` | huggingface | — | <https://huggingface.co/datasets/TerminatorJ/PDB_Ribonanzanet/resolve/main/test.csv> | `data/benchmarks/probing/ribonanzanet/test.csv` |
| `pdb-ribonanzanet-train` | huggingface | — | <https://huggingface.co/datasets/TerminatorJ/PDB_Ribonanzanet/resolve/main/train.csv> | `data/benchmarks/probing/ribonanzanet/train.csv` |
| `pdb-ribonanzanet-val` | huggingface | — | <https://huggingface.co/datasets/TerminatorJ/PDB_Ribonanzanet/resolve/main/val.csv> | `data/benchmarks/probing/ribonanzanet/val.csv` |
| `pdb-seqres` | rcsb | 66.98 MB | <https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz> | `data/structures/indices/pdb_seqres.txt.gz` |
| `rfam-3d` | ebi | 377.40 kB | <https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1/Rfam.3d.seed.gz> | `data/families/rfam/Rfam.3d.seed.gz` |
| `rfam-cm` | ebi | 45.75 MB | <https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1/Rfam.cm.gz> | `data/families/rfam/Rfam.cm.gz` |
| `rfam-full-alignments` | ebi | — | `run scripts/acquire_rfam_full.py --workers 12` | `data/families/rfam/full_alignments` |
| `rfam-full_region` | ebi | 125.36 MB | <https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1/Rfam.full_region.gz> | `data/families/rfam/Rfam.full_region.gz` |
| `rfam-pdb` | ebi | 87.29 kB | <https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1/Rfam.pdb.gz> | `data/families/rfam/Rfam.pdb.gz` |
| `rfam-seed` | ebi | 5.93 MB | <https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1/Rfam.seed.gz> | `data/families/rfam/Rfam.seed.gz` |
| `rfam-seed_tree` | ebi | 2.63 MB | <https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1/Rfam.seed_tree.tar.gz> | `data/families/rfam/Rfam.seed_tree.tar.gz` |
| `rfam-tar` | ebi | 46.15 MB | <https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1/Rfam.tar.gz> | `data/families/rfam/Rfam.tar.gz` |
| `ribonanza-quickstart` | huggingface | 550.74 MB | <https://huggingface.co/datasets/TerminatorJ/RNA_chemical_ribonanza/resolve/main/train_data_QUICK_START.csv> | `data/benchmarks/probing/ribonanza/train_data_QUICK_START.csv` |
| `rmdb` | github | — | `run scripts/acquire_rmdb.py --workers 8` | `data/benchmarks/probing/rmdb` |
| `rna-puzzles-io` | github | — | <https://github.com/rnapuzzles/rnapuzzles.github.io> | `data/structures/blind_tests/rnapuzzles_github_io` |
| `rna-puzzles-std` | github | — | <https://github.com/mmagnus/RNA-Puzzles-Standardized-Submissions> | `data/structures/blind_tests/rna_puzzles_std` |
| `rna3db-cmscans` | github | 88.16 MB | <https://github.com/marcellszi/rna3db/releases/download/2026-01-05-full-release/rna3db-cmscans.tar.gz> | `data/structures/databases/rna3db/rna3db-cmscans.tar.gz` |
| `rna3db-jsons` | github | 6.85 MB | <https://github.com/marcellszi/rna3db/releases/download/2026-01-05-full-release/rna3db-jsons.tar.gz> | `data/structures/databases/rna3db/rna3db-jsons.tar.gz` |
| `rna3db-mmcifs` | github | 2.15 GB | <https://github.com/marcellszi/rna3db/releases/download/2026-01-05-full-release/rna3db-mmcifs.tar.xz> | `data/structures/databases/rna3db/rna3db-mmcifs.tar.xz` |
| `rnagym-processed` | harvard | — | <https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_processed_assays.zip> | `data/benchmarks/fitness/rnagym/fitness_processed_assays.zip` |
| `rnagym-raw` | harvard | — | <https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_raw_data.zip> | `data/benchmarks/fitness/rnagym/fitness_raw_data.zip` |
| `rnagym-repo` | github | — | <https://github.com/MarksLab-DasLab/RNAGym> | `data/benchmarks/fitness/rnagym_repo` |
| `rnastralign-bpseq` | figshare | — | <https://ndownloader.figshare.com/files/45555225> | `data/benchmarks/secondary_structure/rnastralign/RNAStrAlign_bpseq.zip` |
| `spliceator-bigest` | bigest | — | <https://bigest-icube.fr/spliceator/static/data/data.tar.gz> | `data/benchmarks/splicing/spliceator/data.tar.gz` |
| `splicebert-zenodo` | zenodo | — | <https://zenodo.org/records/7995778/files/data.tar.gz?download=1> | `data/benchmarks/splicing/splicebert/data.tar.gz` |
| `spot-rna-bprna` | dropbox | — | <https://www.dropbox.com/s/w3kc4iro8ztbf3m/bpRNA_dataset.zip?dl=1> | `data/benchmarks/secondary_structure/bprna_spot/splits/bpRNA_dataset.zip` |
| `spot-rna-pdb` | dropbox | — | <https://www.dropbox.com/s/vnq0k9dg7vynu3q/PDB_dataset.zip?dl=1> | `data/benchmarks/secondary_structure/bprna_spot/splits/PDB_dataset.zip` |

## `long` — 2 jobs, — known

| job | host | size | source | lands in |
|---|---|---|---|---|
| `raw-pdb-entries` | rcsb | — | `run scripts/acquire_raw_pdb_entries.py --workers 6` | `data/structures/raw_pdb_entries` |
| `rna-harvest` | mixed | — | `run pdb-hunter rna harvest --out --workers 12 --nr-release 4.57` | `/store/shuvam/E-motioner-X-SBS/pdb_hunter/RNA_Database` |

