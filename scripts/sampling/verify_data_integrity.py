#!/usr/bin/env python3
"""Does every acquired file actually OPEN? Presence is not integrity.

`audit_data_presence.py` checks that catalogued files exist at their recorded
size. This checks the stronger thing: that each one parses as what its extension
claims. A download that returns an HTML error page with a 200 has the right
name and a plausible size, and fails only when something tries to read it --
which, for a training corpus, is hours into a run.

Two upstream labelling quirks are encoded here rather than worked around,
because both look like corruption and neither is:

  * **RDAT has two header forms.** 0.34 opens with `RDAT_VERSION`, the older
    0.24/0.32 files with a bare `VERSION` (ETE3D_*). Both parse.
  * **Some archives are mislabelled at the source.** figshare serves
    RNAStrAlign's bpseq bundle as a ZIP named `.tar.gz`, and Spliceator's
    `data.tar.gz` is an uncompressed tar -- the inventory already records the
    latter. The check sniffs the magic bytes instead of trusting the suffix.

Usage:
    /store/shuvam/.venv/bin/python scripts/sampling/verify_data_integrity.py
"""
from __future__ import annotations

import gzip
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/samples/analysis/data_integrity.json"


def gz_ok(p: Path) -> bool:
    with gzip.open(p, "rb") as f:
        f.read(1 << 16)
    return True


def zip_ok(p: Path) -> bool:
    return zipfile.is_zipfile(p)


def archive_ok(p: Path) -> bool:
    """tar.gz, plain tar, or a zip wearing a .tar.gz name -- sniffed, not assumed."""
    if zipfile.is_zipfile(p):
        return True
    for flags in ("-tzf", "-tf"):
        if subprocess.run(["tar", flags, str(p)], capture_output=True).returncode == 0:
            return True
    return False


def sto_ok(p: Path) -> bool:
    return p.open(errors="ignore").readline().startswith("# STOCKHOLM")


def parquet_ok(p: Path) -> bool:
    import pyarrow.parquet as pq
    pq.ParquetFile(p)
    return True


def rdat_ok(p: Path) -> bool:
    # 0.34 says RDAT_VERSION; 0.24 and 0.32 say VERSION. Both are real.
    return p.open(errors="ignore").readline().lstrip().startswith(
        ("RDAT_VERSION", "VERSION"))


def cif_ok(p: Path) -> bool:
    with gzip.open(p, "rt", errors="ignore") as f:
        return f.readline().startswith("data_")


CHECKS = [
    ("Rfam full alignments", "data/families/rfam/full_alignments/*.sto", sto_ok, 300),
    ("Rfam core", "data/families/rfam/*.gz", gz_ok, None),
    ("bpRNA-1m raw", "data/benchmarks/secondary_structure/bprna_1m_raw/*.zip", zip_ok, None),
    ("SPOT-RNA splits", "data/benchmarks/secondary_structure/spot_rna_splits/*.zip", zip_ok, None),
    ("2D archives", "data/benchmarks/secondary_structure/*/*.tar.gz", archive_ok, None),
    ("2D zips", "data/benchmarks/secondary_structure/*/*.zip", zip_ok, None),
    ("2D parquet", "data/benchmarks/secondary_structure/*/*.parquet", parquet_ok, None),
    ("EternaBench", "data/benchmarks/eternabench/*/*.parquet", parquet_ok, None),
    ("RMDB rdat", "data/benchmarks/rmdb/rdat/*.rdat", rdat_ok, 400),
    ("splicing archives", "data/benchmarks/splicing/*/*.tar.gz", archive_ok, None),
    ("RNAcentral", "data/sequences/rnacentral/*.fasta.gz", gz_ok, None),
    ("elDORS", "data/sequences/elDORS_v1/*.fasta.gz", gz_ok, 4),
    ("raw PDB entries", "data/structures/raw_pdb_entries/*.cif.gz", cif_ok, 400),
    ("CCD", "data/structures/ccd/*.cif.gz", cif_ok, None),
]


def main() -> None:
    rows: List[Dict] = []
    for label, pattern, fn, limit in CHECKS:
        paths = sorted(ROOT.glob(pattern))
        sample = paths[:limit] if limit else paths
        bad: List[str] = []
        for p in sample:
            try:
                if not fn(p):
                    bad.append(p.name)
            except Exception as e:                               # noqa: BLE001
                bad.append(f"{p.name}: {type(e).__name__}")
        rows.append({"check": label, "found": len(paths), "tested": len(sample),
                     "bad": bad, "ok": not bad})
        print(f"  {'OK  ' if not bad else 'FAIL'} {label:26s} "
              f"tested {len(sample):5d} of {len(paths):6d}"
              f"{'' if not bad else '   ' + str(bad[:3])}")

    ok = all(r["ok"] for r in rows)
    res = {"all_ok": ok, "n_checks": len(rows),
           "n_files_found": sum(r["found"] for r in rows),
           "n_files_tested": sum(r["tested"] for r in rows),
           "n_bad": sum(len(r["bad"]) for r in rows), "checks": rows}
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\n{res['n_files_tested']:,} files tested of {res['n_files_found']:,} found; "
          f"{res['n_bad']} unreadable")
    print("ALL INTEGRITY CHECKS PASS" if ok else "SOME FILES ARE NOT READABLE")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
