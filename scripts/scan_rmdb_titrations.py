#!/usr/bin/env python3
"""Find the ionic titration series in RMDB — open action 2.

ARCHITECTURE v0.2 §7b records the reason this matters. The PDB's recorded Mg2+
spans only 5-15 mM because nobody deposits unfolded RNA, so the
**[Mg2+] -> structure response cannot be learned from the PDB at all**. The
closed-form electrostatics (§6.1, §6.2) are unaffected -- they need no data --
but training stage 5, which conditions a prediction on ionic condition, needs
experiments where the ion concentration was deliberately varied. Those are
titration ladders, and RMDB is where they are.

A previous pass recorded "no bulk endpoint" for RMDB. That was true of the old
Django site; RMDB is now a static site on GitHub Pages and its RDATs are
**release assets** -- 1,024 files, 11.9 GB across five releases. Downloading all
of that to find the titrations would be backwards: every entry's annotations,
including its chemical conditions, live in a 1.7 kB markdown file in the repo.
This reads those, finds the series, and reports what to fetch.

A titration series is identified structurally, not by name: two or more entries
that share a construct and a modifier but differ in the concentration of one
ion, with everything else held fixed. That is what a titration *is*, and
matching on the word "titration" would find only the ones whose depositor
happened to use it.

Usage:
    /store/shuvam/.venv/bin/python scripts/scan_rmdb_titrations.py \
        --entries /tmp/.../rmdb.github.io-main/_entries
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/benchmarks/rmdb"

#: ions whose concentration is worth tracking as a condition
IONS = ("MgCl2", "NaCl", "KCl", "KOAc", "NaOAc", "MgOAc", "CaCl2", "spermidine",
        "LiCl", "NH4Cl")
_CONC = re.compile(r"^([A-Za-z0-9_+\-]+)\s*:\s*([0-9.]+)\s*(mM|uM|M|nM)\b", re.I)
_SCALE = {"m": 1.0, "u": 1e-3, "n": 1e-6}


def to_mM(value: float, unit: str) -> float:
    u = unit.lower()
    if u == "m":
        return value * 1000.0
    return value * _SCALE.get(u[0], 1.0)


def parse_entry(path: Path) -> Optional[Dict]:
    """The front-matter fields this scan needs, without a YAML dependency."""
    text = path.read_text(errors="ignore")
    if not text.startswith("---"):
        return None
    fm = text.split("---", 2)[1]

    def scalar(key: str) -> str:
        m = re.search(rf"^{key}:\s*\"?([^\"\n]*)\"?\s*$", fm, re.M)
        return (m.group(1).strip() if m else "")

    chem: List[str] = []
    m = re.search(r"^\s*chemical:\s*\[(.*?)\]", fm, re.M | re.S)
    if m:
        chem = [c.strip().strip("\"'") for c in m.group(1).split(",") if c.strip()]
    mods: List[str] = []
    m = re.search(r"^\s*modifier:\s*\[(.*?)\]", fm, re.M | re.S)
    if m:
        mods = [c.strip().strip("\"'") for c in m.group(1).split(",") if c.strip()]
    temps: List[str] = []
    m = re.search(r"^\s*temperature:\s*\[(.*?)\]", fm, re.M | re.S)
    if m:
        temps = [c.strip().strip("\"'") for c in m.group(1).split(",") if c.strip()]

    conc: Dict[str, float] = {}
    other: List[str] = []
    for c in chem:
        mm = _CONC.match(c)
        if mm:
            conc[mm.group(1)] = to_mM(float(mm.group(2)), mm.group(3))
        else:
            other.append(c)

    seq = scalar("sequence")
    return {
        "rmdb_id": scalar("rmdb_id"), "name": scalar("name"),
        "category": scalar("category"),
        "modifier": mods, "temperature": temps,
        "chemical_raw": chem, "conc_mM": conc, "chemical_other": other,
        "sequence_len": len(seq), "sequence_head": seq[:40],
        "data_points": scalar("data_points"),
        "rdat": scalar("rdat") or None,
    }


def series_key(e: Dict, ion: str) -> Tuple:
    """Everything that must be held FIXED for a change in `ion` to be a titration."""
    others = tuple(sorted((k, round(v, 4)) for k, v in e["conc_mM"].items() if k != ion))
    return (e["name"], e["sequence_head"], e["sequence_len"],
            tuple(sorted(e["modifier"])), tuple(sorted(e["temperature"])),
            tuple(sorted(e["chemical_other"])), others)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entries", type=Path, required=True)
    ap.add_argument("--min-points", type=int, default=3,
                    help="distinct concentrations needed to call it a ladder")
    args = ap.parse_args()

    entries = [e for e in (parse_entry(p) for p in sorted(args.entries.glob("*.md")))
               if e]
    print(f"[rmdb] parsed {len(entries):,} entries")

    with_conc = [e for e in entries if e["conc_mM"]]
    ion_counts = defaultdict(int)
    for e in with_conc:
        for k in e["conc_mM"]:
            ion_counts[k] += 1

    ladders: Dict[str, List[Dict]] = {}
    for ion in IONS:
        groups: Dict[Tuple, List[Dict]] = defaultdict(list)
        for e in with_conc:
            if ion in e["conc_mM"]:
                groups[series_key(e, ion)].append(e)
        for key, members in groups.items():
            levels = sorted({round(m["conc_mM"][ion], 4) for m in members})
            if len(levels) < args.min_points:
                continue
            sid = f"{ion}|{key[0]}|{key[3]}|{len(levels)}"
            ladders[sid] = [{
                "rmdb_id": m["rmdb_id"], "name": m["name"],
                "conc_mM": m["conc_mM"][ion], "modifier": m["modifier"],
                "rdat": m["rdat"], "sequence_len": m["sequence_len"],
                "all_conc": m["conc_mM"],
            } for m in sorted(members, key=lambda x: x["conc_mM"][ion])]

    by_ion = defaultdict(list)
    for sid, mem in ladders.items():
        by_ion[sid.split("|")[0]].append((sid, mem))

    total_files = sum(len(m) for m in ladders.values())
    res = {
        "n_entries": len(entries),
        "n_with_concentration": len(with_conc),
        "ions_seen": dict(sorted(ion_counts.items(), key=lambda kv: -kv[1])[:15]),
        "min_points": args.min_points,
        "n_series": len(ladders),
        "n_files_in_series": total_files,
        "by_ion": {ion: {"n_series": len(v),
                         "n_files": sum(len(m) for _, m in v),
                         "max_levels": max((len(set(x["conc_mM"] for x in m))
                                            for _, m in v), default=0)}
                   for ion, v in sorted(by_ion.items())},
        "series": ladders,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "titration_index.json").write_text(json.dumps(res, indent=1))

    print(f"[rmdb] {len(with_conc):,} entries record an ion concentration")
    print(f"[rmdb] ions seen: {dict(list(res['ions_seen'].items())[:8])}")
    print(f"\ntitration series with >= {args.min_points} levels: "
          f"{len(ladders)}  ({total_files} RDAT files)")
    for ion, v in sorted(res["by_ion"].items(), key=lambda kv: -kv[1]["n_files"]):
        print(f"  {ion:12s} {v['n_series']:3d} series  {v['n_files']:4d} files  "
              f"up to {v['max_levels']} levels")
    mg = by_ion.get("MgCl2", [])
    if mg:
        sid, mem = max(mg, key=lambda x: len(x[1]))
        print(f"\nlargest Mg2+ ladder: {mem[0]['name']} "
              f"({len(mem)} points, {mem[0]['conc_mM']}-{mem[-1]['conc_mM']} mM)")
        print(f"  {[m['conc_mM'] for m in mem]}")
    print(f"\n[rmdb] -> {OUT / 'titration_index.json'}")


if __name__ == "__main__":
    main()
