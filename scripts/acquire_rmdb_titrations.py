#!/usr/bin/env python3
"""Acquire RMDB titration ladders — open action 2.

§7b: the PDB's recorded Mg2+ spans only 5-15 mM because nobody deposits
unfolded RNA, so the **[Mg2+] -> structure response cannot be learned from the
PDB**. Training stage 5 conditions predictions on ionic condition and therefore
needs experiments where the concentration was deliberately varied.

Two things this script establishes that the entry metadata could not.

**RMDB has a bulk endpoint after all.** An earlier pass recorded "no bulk
endpoint"; that was the old Django site. RMDB is now a static site on GitHub
Pages and its 1,024 RDATs are release assets, 11.9 GB across five releases.

**A titration is one file, not a series of them.** Scanning the entry-level
annotations found only three distinct Mg2+ levels across 712 entries -- 0, 10
and 40 mM, with 682 of them at the standard 10 mM -- and no ladders at all. That
is because the entry-level `chemical` field records the condition *common to the
whole experiment*, and a titration varies its concentration **per data row**,
inside the RDAT's `ANNOTATION_DATA` lines. So the ladders have to be found by
reading the files.

Downloading 11.9 GB to find them would be backwards. RDATs holding a classic
titration are small -- one construct, tens of rows -- while the multi-gigabyte
files are Eterna and Ribonanza-scale libraries at a single condition. This
fetches the files below `--max-bytes` and reports which of them are ladders.

Usage:
    /store/shuvam/.venv/bin/python scripts/acquire_rmdb_titrations.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/benchmarks/rmdb"
RAW = OUT / "rdat"
REPO = "DasLab/rmdb.github.io"
RELEASES = ("data-rna-structures", "data-riboswitches", "data-puzzle",
            "data-general", "data-eterna")

_ROWCHEM = re.compile(r"chemical:([A-Za-z0-9_+\-]+):([0-9.]+)\s*(mM|uM|nM|M)\b", re.I)
_SCALE = {"m": 1.0, "u": 1e-3, "n": 1e-6}


def to_mM(v: float, unit: str) -> float:
    u = unit.lower()
    return v * 1000.0 if u == "m" else v * _SCALE.get(u[0], 1.0)


def list_assets() -> List[Tuple[str, str, int]]:
    """(release, filename, size) for every RDAT, via the GitHub API."""
    out: List[Tuple[str, str, int]] = []
    for tag in RELEASES:
        r = subprocess.run(
            ["gh", "api", f"repos/{REPO}/releases/tags/{tag}",
             "-q", '.assets[] | "\\(.name)\\t\\(.size)"'],
            capture_output=True, text=True, timeout=300)
        for line in r.stdout.splitlines():
            if "\t" in line:
                name, size = line.rsplit("\t", 1)
                out.append((tag, name, int(size)))
    return out


def fetch(job: Tuple[str, str]) -> Optional[Path]:
    tag, name = job
    dest = RAW / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"https://github.com/{REPO}/releases/download/{tag}/{name}"
    r = subprocess.run(["curl", "-sSLf", "--max-time", "300", "-o", str(dest), url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        dest.unlink(missing_ok=True)
        return None
    return dest


def row_conditions(path: Path) -> Dict[str, List[float]]:
    """Per-row ion concentrations, from the RDAT's ANNOTATION_DATA lines."""
    per: Dict[str, List[float]] = defaultdict(list)
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return {}
    for line in text.splitlines():
        if not line.startswith("ANNOTATION_DATA"):
            continue
        for m in _ROWCHEM.finditer(line):
            per[m.group(1)].append(to_mM(float(m.group(2)), m.group(3)))
    return dict(per)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-bytes", type=int, default=2_000_000,
                    help="skip the library-scale files; a titration is small")
    ap.add_argument("--min-levels", type=int, default=3)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    assets = list_assets()
    small = [(t, n) for t, n, s in assets if s <= args.max_bytes]
    print(f"[rmdb] {len(assets):,} assets, {len(small):,} under "
          f"{args.max_bytes/1e6:.1f} MB -- fetching those", flush=True)

    got: List[Path] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, p in enumerate(ex.map(fetch, small), 1):
            if p is not None:
                got.append(p)
            if i % 100 == 0:
                print(f"[rmdb]   {i}/{len(small)}", flush=True)
    print(f"[rmdb] fetched {len(got):,} files, "
          f"{sum(p.stat().st_size for p in got)/1e6:.1f} MB")

    ladders: Dict[str, Dict] = {}
    n_multi = 0
    for p in got:
        cond = row_conditions(p)
        if not cond:
            continue
        varying = {ion: sorted(set(round(v, 4) for v in vals))
                   for ion, vals in cond.items()
                   if len(set(round(v, 4) for v in vals)) >= args.min_levels}
        if not varying:
            continue
        n_multi += 1
        ladders[p.stem] = {
            "file": p.name,
            "n_rows": max(len(v) for v in cond.values()),
            "varying": varying,
            "ions": {i: {"levels": lv, "min": lv[0], "max": lv[-1],
                         "n_levels": len(lv)} for i, lv in varying.items()},
        }

    by_ion: Dict[str, int] = defaultdict(int)
    for v in ladders.values():
        for ion in v["varying"]:
            by_ion[ion] += 1
    mg = {k: v for k, v in ladders.items() if "MgCl2" in v["varying"]}

    res = {
        "source": f"https://github.com/{REPO} release assets",
        "n_assets_total": len(assets),
        "n_fetched": len(got),
        "max_bytes": args.max_bytes,
        "min_levels": args.min_levels,
        "n_titration_files": len(ladders),
        "n_mg_titrations": len(mg),
        "by_ion": dict(sorted(by_ion.items(), key=lambda kv: -kv[1])),
        "series": ladders,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "titrations.json").write_text(json.dumps(res, indent=1))

    print(f"\nfiles whose ROWS vary an ion concentration "
          f"(>= {args.min_levels} levels): {len(ladders):,}")
    for ion, n in list(res["by_ion"].items())[:10]:
        print(f"  {ion:14s} {n:4d} files")
    if mg:
        best = max(mg.items(), key=lambda kv: kv[1]["ions"]["MgCl2"]["n_levels"])
        lv = best[1]["ions"]["MgCl2"]
        print(f"\nlargest Mg2+ ladder: {best[0]}  {lv['n_levels']} levels, "
              f"{lv['min']}-{lv['max']} mM")
        print(f"  {lv['levels']}")
        tot = sum(v["ions"]["MgCl2"]["n_levels"] for v in mg.values())
        print(f"Mg2+ titration files: {len(mg)},  {tot:,} concentration points total")
    print(f"\n[rmdb] -> {OUT / 'titrations.json'}")


if __name__ == "__main__":
    main()
