#!/usr/bin/env python3
"""Build the RNA training database catalog.

Scans data/ and produces:
  - data/rna_training_db/catalog.sqlite  (queryable metadata database)
  - data/rna_training_db/MANIFEST.json   (machine-readable manifest)
  - data/rna_training_db/README.md       (documentation)

The catalog records every acquired source with size, format, record counts,
license, and provenance URL, plus declared train/val/test splits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import time
from pathlib import Path

ROOT = Path("/store/shuvam/E-motioner-X-SBS/sbs-rna")
DATA = ROOT / "data"
DB_DIR = DATA / "catalog"
DB_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def count_fasta(path: Path, sample: int = 50_000) -> tuple[int | None, bool]:
    """Count FASTA records. Returns (count, exact). For .gz streams whole file."""
    import gzip

    n = 0
    exact = True
    try:
        with gzip.open(path, "rt", errors="ignore") as fh:
            for line in fh:
                if line.startswith(">"):
                    n += 1
                    if sample and n >= sample:
                        exact = False
                        break
    except Exception:
        return None, False
    return n, exact


def count_json_records(path: Path) -> int | None:
    try:
        with open(path) as fh:
            obj = json.load(fh)
        if isinstance(obj, list):
            return len(obj)
        if isinstance(obj, dict):
            # common HF format: {"data": [...]} or nested
            for key in ("data", "records", "sequences"):
                if key in obj and isinstance(obj[key], list):
                    return len(obj[key])
            return len(obj)
    except Exception:
        return None
    return None


def count_parquet_rows(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def count_csv_rows(path: Path) -> int | None:
    try:
        with open(path, errors="ignore") as fh:
            return sum(1 for _ in fh)
    except Exception:
        return None


def measure(path: Path) -> dict:
    info = {"size_bytes": path.stat().st_size, "format": path.suffix.lstrip(".")}
    name = path.name.lower()
    if name.endswith((".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz")):
        n, exact = count_fasta(path)
        info["n_records"] = n
        info["count_exact"] = exact
    elif name.endswith(".json"):
        info["n_records"] = count_json_records(path)
    elif name.endswith(".parquet"):
        info["n_records"] = count_parquet_rows(path)
    elif name.endswith(".csv"):
        info["n_records"] = count_csv_rows(path)
    return info


# Source definitions: (name, category, task, description, url, license, version, glob)
SOURCES = [
    (
        "elDORS_v1",
        "sequence",
        "pretraining",
        "1.32B RNA sequences clustered at 80% identity (10-4096 nt), MARS successor, designed for RNA LM pretraining",
        "https://registry.opendata.aws/eldors_v1/",
        "CC BY 4.0",
        "v1",
        "sequences/elDORS_v1/elDORS_v1_*.fasta.gz",
    ),
    (
        "RNAcentral",
        "sequence",
        "pretraining",
        "Consolidated ncRNA sequences from 50+ expert databases (active set)",
        "https://rnacentral.org/",
        "CC0",
        "27",
        "sequences/rnacentral/rnacentral_active.fasta.gz",
    ),
    (
        "Rfam",
        "alignment",
        "msa",
        "ncRNA family seed alignments, covariance models, 3D-curated seeds, full region table",
        "https://rfam.org/",
        "CC0",
        "15.1",
        "families/rfam/Rfam*",
    ),
    (
        "BGSU_motifs",
        "structure",
        "motifs",
        "RNA 3D Motif Atlas release 4.12: internal loops (2,972 loops / 413 motifs) and hairpin loops (2,020 / 254)",
        "https://rna.bgsu.edu/rna3dhub/motifs",
        "CC BY 4.0",
        "4.12",
        "structures/indices/bgsu_motifs/*",
    ),
    (
        "BGSU_nrlist",
        "structure",
        "tertiary",
        "Non-redundant RNA 3D equivalence classes at 7 resolution cutoffs (release 4.56)",
        "https://rna.bgsu.edu/rna3dhub/nrlist",
        "CC BY 4.0",
        "4.56",
        "structures/indices/bgsu_nrlist/*.csv",
    ),
    (
        "BGSU_rfam_map",
        "structure",
        "annotation",
        "PDB chain to best Rfam family mapping",
        "https://rna.bgsu.edu/data/pdb_chain_to_best_rfam.txt",
        "CC BY 4.0",
        "2026-09",
        "structures/indices/pdb_chain_to_best_rfam.txt",
    ),
    (
        "ArchiveII",
        "benchmark",
        "secondary_structure",
        "Standard 2D structure benchmark, 3,975 structures, 10 families",
        "https://huggingface.co/datasets/multimolecule/archiveii",
        "CC BY 4.0",
        "2026",
        "benchmarks/secondary_structure/archiveii/*",
    ),
    (
        "bpRNA_spot",
        "benchmark",
        "secondary_structure",
        "bpRNA-1m 80% identity splits (train/val/test) for supervised 2D prediction",
        "https://huggingface.co/datasets/multimolecule/bprna-spot",
        "CC BY 4.0",
        "2026",
        "benchmarks/secondary_structure/bprna_spot/*",
    ),
    (
        "bpRNA_1m",
        "benchmark",
        "secondary_structure",
        "Full bpRNA-1m database, 102,318 structures, 7 sources",
        "https://huggingface.co/datasets/rouskinlab/bpRNA-1m",
        "CC BY 4.0",
        "1m",
        "benchmarks/secondary_structure/bprna_full/*",
    ),
    (
        "RNAStrAlign",
        "benchmark",
        "secondary_structure",
        "37,149 structurally aligned RNA families (TurboFold II source)",
        "https://huggingface.co/datasets/rouskinlab/RNAstralign",
        "CC BY 4.0",
        "2026",
        "benchmarks/secondary_structure/rnastralign/*",
    ),
    (
        "bpRNA_new",
        "benchmark",
        "secondary_structure",
        "Cross-family generalization test set from Rfam 14.2 distinct families",
        "https://huggingface.co/datasets/multimolecule/bprna-new",
        "CC BY 4.0",
        "2026",
        "benchmarks/secondary_structure/bprna_new/*",
    ),
    (
        "RNA-Puzzles_std",
        "benchmark",
        "tertiary",
        "Standardized submissions for RNA-Puzzles PZ1-PZ21+ (blind 3D prediction benchmark)",
        "https://github.com/mmagnus/RNA-Puzzles-Standardized-Submissions",
        "see repo",
        "2026",
        "structures/blind_tests/rna_puzzles_std/*",
    ),
    (
        "RNA-Puzzles_site",
        "benchmark",
        "tertiary",
        "RNA-Puzzles official website data (puzzle targets and models)",
        "https://github.com/rnapuzzles/rnapuzzles.github.io",
        "see repo",
        "2026",
        "structures/blind_tests/rnapuzzles_github_io/*",
    ),
    (
        "CASP15_RNA",
        "benchmark",
        "tertiary",
        "CASP15 RNA targets as experimental mmCIF structures (15 files)",
        "https://predictioncenter.org/casp15/",
        "PDB terms",
        "2022",
        "structures/blind_tests/casp15/*.cif",
    ),
    (
        "CASP16_RNA",
        "benchmark",
        "tertiary",
        "CASP16 RNA targets as experimental mmCIF structures (10 files)",
        "https://predictioncenter.org/casp16/",
        "PDB terms",
        "2024",
        "structures/blind_tests/casp16/*.cif",
    ),
    (
        "RNAGym",
        "benchmark",
        "fitness",
        "RNA fitness benchmark: >30 DMS assays, >1M measurements + repo with baselines",
        "https://github.com/MarksLab-DasLab/RNAGym",
        "see repo",
        "2025",
        "benchmarks/fitness/rnagym*",
    ),
    (
        "NABench",
        "benchmark",
        "fitness",
        "162 DMS assays, 2.6M mutated sequences, standardized splits, 29 model baselines",
        "https://github.com/mrzzmrzz/NABench",
        "see repo",
        "2025",
        "benchmarks/fitness/nabench/*",
    ),
    (
        "Spliceator",
        "benchmark",
        "splicing",
        "Multi-species splice-site benchmarks (zebrafish, fly, worm, plant)",
        "https://bigest-icube.fr/spliceator/",
        "see site",
        "v1",
        "benchmarks/splicing/spliceator/*",
    ),
    (
        "SpliceBERT_data",
        "benchmark",
        "splicing",
        "Pre-mRNA sequences from 72 vertebrates + splice-site eval data",
        "https://zenodo.org/records/7995778",
        "CC BY 4.0",
        "2024",
        "benchmarks/splicing/splicebert/*",
    ),
    (
        "G3PO",
        "benchmark",
        "splicing",
        "Gene prediction benchmark: 147 eukaryote species, >20k validated genes",
        "https://github.com/BiGEst-ICube/g3po",
        "see repo",
        "2026",
        "benchmarks/splicing/g3po/*",
    ),
    (
        "RNA3DB",
        "benchmark",
        "tertiary",
        "DL-ready RNA 3D dataset: PDB chains, Rfam labels, NR train/test split",
        "https://github.com/marcellszi/rna3db",
        "see repo",
        "2026-01-05",
        "structures/databases/rna3db/*",
    ),
    (
        "training_assets",
        "derived",
        "pretraining",
        "Derived assets: sample FASTAs, split convention, demo parquet shards",
        "local",
        "n/a",
        "2026-09",
        "catalog/samples/*",
    ),
    (
        "training_assets_parquet",
        "derived",
        "pretraining",
        "Demo Parquet shards (100k sequences, T->U mapped) proving the ETL",
        "local",
        "n/a",
        "2026-09",
        "derived/**/*.parquet",
    ),
    (
        "gRNAde_RNASolo",
        "structure",
        "tertiary",
        "RNASolo raw structures snapshot (14,369 PDB files) + processed metadata CSV "
        "(Rfam labels, equivalence classes, clusters, RMSDs)",
        "https://huggingface.co/datasets/chaitjo/gRNAde_datasets",
        "see repo",
        "2023-11",
        "structures/databases/grnade_rnasolo/*",
    ),
    (
        "PDB_seqres",
        "structure",
        "annotation",
        "All PDB seqres sequences (FASTA) for sequence<->structure joins",
        "https://files.rcsb.org/pub/pdb/derived_data/",
        "PDB terms",
        "weekly",
        "structures/indices/pdb_seqres.txt.gz",
    ),
]

SPLITS = [
    (
        "eldors_chunks",
        "pretraining",
        "elDORS_v1",
        "20 chunks x ~9GB, concatenate to full 1.32B-sequence corpus",
        "sequences/elDORS_v1/elDORS_v1_*.fasta.gz",
    ),
    (
        "bprna_spot_train",
        "train",
        "bpRNA_spot",
        "10,934 structures (HF variant of TR0)",
        "benchmarks/secondary_structure/bprna_spot/train.parquet",
    ),
    (
        "bprna_spot_val",
        "validation",
        "bpRNA_spot",
        "1,330 structures (HF variant of VL0)",
        "benchmarks/secondary_structure/bprna_spot/validation.parquet",
    ),
    (
        "bprna_spot_test",
        "test",
        "bpRNA_spot",
        "1,411 structures (HF variant of TS0)",
        "benchmarks/secondary_structure/bprna_spot/test.parquet",
    ),
    (
        "archiveii_test",
        "test",
        "ArchiveII",
        "3,975 structures, standard 2D benchmark",
        "benchmarks/secondary_structure/archiveii/test.parquet",
    ),
    (
        "bprna_new_test",
        "test",
        "bpRNA_new",
        "5,401 structures, cross-family generalization",
        "benchmarks/secondary_structure/bprna_new/test.parquet",
    ),
    (
        "casp15_targets",
        "test",
        "CASP15_RNA",
        "15 experimental mmCIF target structures",
        "structures/blind_tests/casp15/*.cif",
    ),
    (
        "casp16_targets",
        "test",
        "CASP16_RNA",
        "10 experimental mmCIF target structures",
        "structures/blind_tests/casp16/*.cif",
    ),
]


def main() -> None:
    con = sqlite3.connect(DB_DIR / "catalog.sqlite")
    cur = con.cursor()
    cur.executescript(
        """
        DROP TABLE IF EXISTS files;
        DROP TABLE IF EXISTS sources;
        DROP TABLE IF EXISTS splits;
        CREATE TABLE sources (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            category TEXT,
            task TEXT,
            description TEXT,
            url TEXT,
            license TEXT,
            version TEXT
        );
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            source_name TEXT REFERENCES sources(name),
            path TEXT UNIQUE NOT NULL,
            size_bytes INTEGER,
            format TEXT,
            n_records INTEGER,
            count_exact INTEGER,
            sha256 TEXT,
            note TEXT
        );
        CREATE TABLE splits (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            purpose TEXT,
            source_name TEXT REFERENCES sources(name),
            files_glob TEXT,
            description TEXT,
            n_files INTEGER,
            total_bytes INTEGER
        );
        """
    )
    for s in SOURCES:
        cur.execute(
            "INSERT INTO sources (name, category, task, description, url, license, version) VALUES (?,?,?,?,?,?,?)",
            s[:7],
        )

    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(DATA),
        "sources": [],
    }
    n_files_total = 0
    bytes_total = 0

    for s in SOURCES:
        name, glob = s[0], s[7]
        matches = sorted(DATA.glob(glob))
        entry = {"name": name, "files": []}
        for p in matches:
            if p.is_dir():
                # include directory contents (skip .git, .cache)
                children = [
                    c
                    for c in p.rglob("*")
                    if c.is_file() and ".git" not in c.parts and ".cache" not in c.parts
                ]
            else:
                children = [p]
            for c in children:
                try:
                    m = measure(c)
                except Exception as exc:  # noqa: BLE001
                    m = {
                        "size_bytes": c.stat().st_size,
                        "format": "?",
                        "n_records": None,
                        "count_exact": 0,
                        "error": str(exc),
                    }
                rel = str(c.relative_to(DATA))
                digest = sha256_file(c) if m["size_bytes"] < 50_000_000 else None
                cur.execute(
                    "INSERT OR IGNORE INTO files (source_name, path, size_bytes, format, n_records, count_exact, sha256) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        name,
                        rel,
                        m.get("size_bytes"),
                        m.get("format"),
                        m.get("n_records"),
                        int(bool(m.get("count_exact", False))),
                        digest,
                    ),
                )
                entry["files"].append(
                    {
                        "path": rel,
                        "size": m.get("size_bytes"),
                        "n_records": m.get("n_records"),
                    }
                )
                n_files_total += 1
                bytes_total += m.get("size_bytes") or 0
        manifest["sources"].append(entry)

    # Merge exact counts produced by background counters
    for counts_file, prefix in [
        (DATA / "sequences" / "elDORS_v1" / "seq_counts.txt", "sequences/elDORS_v1/"),
        (DATA / "sequences" / "rnacentral" / "seq_count.txt", "sequences/rnacentral/"),
    ]:
        if counts_file.exists():
            for line in counts_file.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    cur.execute(
                        "UPDATE files SET n_records=?, count_exact=1 WHERE path=?",
                        (int(parts[1]), prefix + parts[0]),
                    )
        con.commit()

    for sp in SPLITS:
        files = sorted(DATA.glob(sp[4]))
        total = sum(f.stat().st_size for f in files if f.is_file())
        cur.execute(
            "INSERT INTO splits (name, purpose, source_name, files_glob, description, n_files, total_bytes) VALUES (?,?,?,?,?,?,?)",
            (sp[0], sp[1], sp[2], sp[4], sp[3], len(files), total),
        )

    con.commit()

    manifest["totals"] = {
        "files": n_files_total,
        "bytes": bytes_total,
        "gb": round(bytes_total / 1e9, 2),
    }
    with open(DB_DIR / "MANIFEST.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    # quick stats printout
    cur.execute("SELECT COUNT(*), SUM(size_bytes) FROM files")
    nf, sb = cur.fetchone()
    print(f"catalog.sqlite written: {nf} files, {(sb or 0) / 1e9:.1f} GB")
    cur.execute(
        "SELECT source_name, COUNT(*), SUM(size_bytes) FROM files GROUP BY source_name ORDER BY 3 DESC"
    )
    for row in cur.fetchall():
        print(f"  {row[0]:24s} {row[1]:5d} files {(row[2] or 0) / 1e9:8.2f} GB")
    con.close()


if __name__ == "__main__":
    main()
