#!/usr/bin/env python3
"""Python loader API for the RNA training database.

Usage:
    from rna_db import RNADatabase
    db = RNADatabase("data/rna_training_db/catalog.sqlite")
    db.sources()
    db.files("elDORS_v1")
    for header, seq in db.iter_fasta("rnacentral"):
        ...
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path
from typing import Iterator


class RNADatabase:
    def __init__(self, catalog: str | Path):
        self.catalog = Path(catalog)
        self.data_root = self.catalog.parent.parent
        self.con = sqlite3.connect(self.catalog)
        self.con.row_factory = sqlite3.Row

    # ---------- metadata ----------
    def sources(self) -> list[dict]:
        cur = self.con.execute("""
            SELECT s.*, COUNT(f.id) n_files, COALESCE(SUM(f.size_bytes),0) total_bytes
            FROM sources s LEFT JOIN files f ON f.source_name = s.name
            GROUP BY s.name ORDER BY total_bytes DESC
        """)
        return [dict(r) for r in cur.fetchall()]

    def files(self, source: str) -> list[dict]:
        cur = self.con.execute(
            "SELECT * FROM files WHERE source_name=? ORDER BY path", (source,)
        )
        return [dict(r) for r in cur.fetchall()]

    def splits(self) -> list[dict]:
        cur = self.con.execute("SELECT * FROM splits ORDER BY purpose, name")
        return [dict(r) for r in cur.fetchall()]

    def manifest(self) -> dict:
        with open(self.catalog.parent / "MANIFEST.json") as fh:
            return json.load(fh)

    # ---------- data access ----------
    def path(self, rel: str) -> Path:
        return self.data_root / rel

    def iter_fasta(self, source: str, limit: int | None = None) -> Iterator[tuple]:
        """Yield (header, sequence) from every FASTA file of a source."""
        n = 0
        for f in self.files(source):
            p = self.path(f["path"])
            if not p.name.lower().endswith(
                (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz")
            ):
                continue
            opener = gzip.open if p.suffix == ".gz" else open
            with opener(p, "rt", errors="ignore") as fh:
                header, chunks = None, []
                for line in fh:
                    if line.startswith(">"):
                        if header is not None:
                            yield header, "".join(chunks)
                            n += 1
                            if limit and n >= limit:
                                return
                        header, chunks = line[1:].strip(), []
                    elif header is not None:
                        chunks.append(line.strip())
                if header is not None:
                    yield header, "".join(chunks)
                    n += 1
                    if limit and n >= limit:
                        return

    def counts(self, source: str) -> dict:
        """Record counts per file where measured."""
        rows = self.con.execute(
            "SELECT path, n_records, count_exact FROM files WHERE source_name=? AND n_records IS NOT NULL",
            (source,),
        ).fetchall()
        return {
            r["path"]: {"n_records": r["n_records"], "exact": bool(r["count_exact"])}
            for r in rows
        }

    def total_sequences(self) -> int:
        row = self.con.execute(
            "SELECT SUM(n_records) FROM files WHERE n_records IS NOT NULL"
        ).fetchone()
        return int(row[0] or 0)

    def close(self) -> None:
        self.con.close()


if __name__ == "__main__":
    import sys

    db = RNADatabase(
        sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "catalog.sqlite"
    )
    print(f"Sources ({len(db.sources())}):")
    for s in db.sources():
        print(
            f"  {s['name']:24s} {s['n_files']:6d} files  "
            f"{s['total_bytes'] / 1e9:8.2f} GB  [{s['category']}] {s['task']}"
        )
    print(f"\nSplits ({len(db.splits())}):")
    for sp in db.splits():
        print(
            f"  {sp['name']:20s} {sp['purpose']:10s} "
            f"{sp['n_files']:4d} files {sp['total_bytes'] / 1e9:7.2f} GB"
        )
    print(f"\nTotal measured records: {db.total_sequences():,}")
