#!/usr/bin/env python3
"""Rebuild the raw-PDB entry list as the union of every corpus that names PDB IDs.

The v0.2 raw archive is the union of

  1. `data/catalog/pdb_hunter_index.json` — the catalogued pdb_hunter entries
  2. the freshly harvested `pdb_hunter/RNA_Database` tree
  3. RNA3DB chain filenames (rna3db-mmcifs.tar.xz)
  4. gRNASolo/RNASolo cleaned filenames (RNASolo_31102023_raw.tar.gz)

Using only (1) under-counts by 104 entries (10,423 vs the documented 10,527);
this union reproduces the documented corpus, and
`scripts/acquire_raw_pdb_entries.py` then fetches whatever is missing.

    uv run python scripts/build_raw_pdb_entrylist.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUNTER = ROOT.parent / "pdb_hunter"
INDEX = ROOT / "data" / "catalog" / "pdb_hunter_index.json"
OUT = ROOT / "data" / "structures" / "raw_pdb_entrylist.txt"
RNA3DB = (ROOT / "data/structures/databases/rna3db/rna3db-mmcifs.tar.xz")
GRNADE = (ROOT / "data/structures/databases/grnade_rnasolo/"
          "RNASolo_31102023_raw.tar.gz")


def ids_from_tar(path: Path, suffix: str, mode: str) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        print(f"  ! missing {path} — skipped", file=sys.stderr)
        return ids
    with tarfile.open(path, mode) as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(suffix):
                pid = Path(member.name).name[:4].upper()
                if len(pid) == 4 and pid.isalnum():
                    ids.add(pid)
    return ids


def main() -> int:
    ids: set[str] = set()

    if INDEX.exists():
        with open(INDEX) as fh:
            index_ids = {e["pdb_id"].upper() for e in json.load(fh)["entries"]}
        ids |= index_ids
        print(f"  pdb_hunter_index: {len(index_ids):,}")
    else:
        print(f"  ! missing {INDEX}", file=sys.stderr)

    database = HUNTER / "RNA_Database"
    if database.is_dir():
        harvest_ids = {d.name.upper() for d in database.iterdir() if d.is_dir()}
        ids |= harvest_ids
        print(f"  RNA_Database:     {len(harvest_ids):,}")
    else:
        print(f"  ! missing {database}", file=sys.stderr)

    n3 = ids_from_tar(RNA3DB, ".cif", "r|xz")
    ids |= n3
    print(f"  rna3db files:     {len(n3):,}")

    ng = ids_from_tar(GRNADE, ".pdb", "r|gz")
    ids |= ng
    print(f"  grnade files:     {len(ng):,}")

    ordered = sorted(ids)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(ordered) + "\n")
    print(f"  UNION:            {len(ordered):,} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
