#!/usr/bin/env python3
"""Complete the secondary-structure corpus against docs/ACQUISITION.md.

This machine held only the HuggingFace mirrors -- one parquet or JSON per
dataset, 164 MB against a documented 1.42 GB. The mirrors are convenient and
lossy: `bprna_full/data.json` is one serialisation of bpRNA-1m, not the five
formats the source publishes, and the SPOT-RNA **splits** are absent entirely.

The splits are the part that matters beyond byte count. TR0/VL0/TS0 and
TR1/VL1/TS1 are the partition the secondary-structure literature reports on, so
without them stage 2's numbers cannot be compared with anything published --
they would be a number on a partition of our own devising.

Every source is taken from the URL `docs/ACQUISITION.md` records, and each file
is integrity-checked on arrival rather than assumed: a zip that downloads as an
HTML error page is a common failure for Dropbox and figshare links, and it is
silent unless you look.

Usage:
    /store/shuvam/.venv/bin/python scripts/acquire_2d_complete.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/benchmarks/secondary_structure"

#: (subdir, filename, url)
SOURCES: List[Tuple[str, str, str]] = [
    # bpRNA-1m, the five formats the source publishes
    ("bprna_1m_raw", "dbnFiles.zip",
     "https://bprna.cgrb.oregonstate.edu/bpRNA_1m/dbnFiles.zip"),
    ("bprna_1m_raw", "bpseqFiles.zip",
     "https://bprna.cgrb.oregonstate.edu/bpRNA_1m/bpseqFiles.zip"),
    ("bprna_1m_raw", "fastaFiles.zip",
     "https://bprna.cgrb.oregonstate.edu/bpRNA_1m/fastaFiles.zip"),
    ("bprna_1m_raw", "ctFiles.zip",
     "https://bprna.cgrb.oregonstate.edu/bpRNA_1m/ctFiles.zip"),
    ("bprna_1m_raw", "stFiles.zip",
     "https://bprna.cgrb.oregonstate.edu/bpRNA_1m/stFiles.zip"),
    # SPOT-RNA splits -- the partition the field reports on
    ("spot_rna_splits", "bpRNA_dataset.zip",
     "https://www.dropbox.com/s/w3kc4iro8ztbf3m/bpRNA_dataset.zip?dl=1"),
    ("spot_rna_splits", "PDB_dataset.zip",
     "https://www.dropbox.com/s/vnq0k9dg7vynu3q/PDB_dataset.zip?dl=1"),
    # archival copies of the two HF-mirrored sets
    ("archiveii", "ArchiveII.tar.gz",
     "https://zenodo.org/api/records/16730061/files/ArchiveII.tar.gz/content"),
    ("bprna_full", "bpRNA_1m.tar.gz",
     "https://zenodo.org/api/records/16730061/files/bpRNA_1m.tar.gz/content"),
    ("rnastralign", "RNAStrAlign_bpseq.tar.gz",
     "https://ndownloader.figshare.com/files/45555225"),
]


def fetch(sub: str, name: str, url: str) -> Dict:
    out = DEST / sub
    out.mkdir(parents=True, exist_ok=True)
    dest = out / name
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return {"file": f"{sub}/{name}", "status": "already present",
                "bytes": dest.stat().st_size}
    r = subprocess.run(["curl", "-sSL", "--fail", "--retry", "3",
                        "--max-time", "3600", "-o", str(dest), url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        dest.unlink(missing_ok=True)
        return {"file": f"{sub}/{name}", "status": f"FAILED rc={r.returncode}",
                "bytes": 0, "error": (r.stderr or "")[:120]}
    n = dest.stat().st_size
    # integrity, not just presence: a Dropbox or figshare link that has moved
    # returns an HTML page with a 200, and it is silent unless checked
    ok = "unchecked"
    if name.endswith(".zip"):
        ok = "valid zip" if zipfile.is_zipfile(dest) else "NOT A ZIP"
    elif name.endswith((".tar.gz", ".tgz")):
        t = subprocess.run(["tar", "-tzf", str(dest)], capture_output=True)
        ok = "valid tar.gz" if t.returncode == 0 else "NOT A TAR.GZ"
    if ok.startswith("NOT"):
        head = dest.open("rb").read(200)
        return {"file": f"{sub}/{name}", "status": ok, "bytes": n,
                "head": head[:80].decode("utf-8", "replace")}
    return {"file": f"{sub}/{name}", "status": ok, "bytes": n}


def main() -> None:
    rows = []
    for sub, name, url in SOURCES:
        r = fetch(sub, name, url)
        rows.append(r)
        print(f"  {r['file']:42s} {r['bytes']/1e6:9.1f} MB  {r['status']}",
              flush=True)
        if "head" in r:
            print(f"       served: {r['head']!r}")
    total = sum(r["bytes"] for r in rows)
    good = sum(1 for r in rows if r["status"].startswith(("valid", "already")))
    out = ROOT / "data/samples/analysis/acquire_2d.json"
    out.write_text(json.dumps({"n": len(rows), "n_ok": good,
                               "total_bytes": total, "files": rows}, indent=1))
    print(f"\n{good}/{len(rows)} usable, {total/1e9:.2f} GB")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
