#!/usr/bin/env python3
"""What the acquisition inventory documents vs what is on THIS machine.

`docs/ACQUISITION.md` (branch `docs/acquisition-inventory`) records an **833.3
GB** corpus assembled under `/home/susmitaroy1_iiserk/sbs/`. This machine holds
a different, smaller set. Some of the difference is a recorded decision; some of
it is a real gap, and the two must not be confused -- "we decided not to" and
"we forgot" look identical in a size comparison.

Each row is classified:

    present      here, at or above the documented size
    partial      here, but materially smaller than documented
    decided      absent by a numbered decision, with the decision named
    MISSING      absent, and nothing says it should be

Usage:
    /store/shuvam/.venv/bin/python scripts/sampling/audit_inventory_gap.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
ORG = ROOT.parent
OUT = ROOT / "data/samples/analysis/inventory_gap.json"

#: The inventory quotes DECIMAL GB, as `du --si` and most upstream pages do.
#: Measuring in GiB and comparing against it makes every source look 7.4% short
#: -- elDORS read 169.84 against a documented 182.36 and was flagged partial
#: when 169.84 GiB IS 182.36 GB. Same number, two units, one spurious gap.
GB = 1000 ** 3

#: (name, documented GB, path relative to ORG, decision or None, note)
INVENTORY = [
    ("elDORS v1", 182.36, "sbs-rna/data/sequences/elDORS_v1", None,
     "1,323,715,880 sequences, SHA256 verified"),
    ("MARS", 427.29, "sbs-rna/data/sequences/mars", "D18",
     "1.73B sequences; 0.17% diverse structured ncRNA yield for 413 GB"),
    ("RNAcentral active", 10.92, "sbs-rna/data/sequences/rnacentral/rnacentral_active.fasta.gz",
     None, "46,210,324 sequences"),
    ("RNAcentral inactive", 1.33, "sbs-rna/data/sequences/rnacentral/rnacentral_inactive.fasta.gz",
     None, "12,322,545 sequences"),
    ("RNAcentral species-specific", 11.69,
     "sbs-rna/data/sequences/rnacentral/rnacentral_species_specific_ids.fasta.gz",
     None, "56,419,280 sequences"),
    ("Rfam 15.1 core", 0.17, "sbs-rna/data/families/rfam", None,
     "seed, CMs, region table, 3D seeds, PDB map"),
    ("Rfam full alignments", 3.25, "sbs-rna/data/families/rfam/full_alignments", None,
     "4,077 Stockholm files -- the MSA the coevolution track (§11) needs"),
    ("raw PDB entries", 16.03, "sbs-rna/data/structures/raw_pdb_entries", None,
     "10,526 atomic mmCIFs"),
    ("pdb_hunter RNA_Database", 138.75, "pdb_hunter/RNA_Database", "D19",
     "referenced, not copied -- and it is present here"),
    ("RNA3DB", 2.15, "sbs-rna/data/structures/databases/rna3db", None, "15,441 chains"),
    ("gRNAde / RNASolo", 6.9, "sbs-rna/data/structures/databases/grnade_rnasolo", None,
     "14,366 cleaned PDBs"),
    ("blind tests", 4.06, "sbs-rna/data/structures/blind_tests", None,
     "CASP15/16, RNA-Puzzles"),
    ("secondary structure", 1.42, "sbs-rna/data/benchmarks/secondary_structure", None,
     "bpRNA, ArchiveII, RNAStrAlign, SPOT splits"),
    ("fitness", 0.48, "sbs-rna/data/benchmarks/fitness", None, "RNAGym, NABench"),
    ("splicing", 10.33, "sbs-rna/data/benchmarks/splicing", None,
     "Spliceator, SpliceBERT, G3PO"),
    ("RMDB", 12.51, "sbs-rna/data/benchmarks/rmdb", None,
     "1,024 constructs; this machine fetched only the <2 MB assets"),
    ("EternaBench", 0.029, "sbs-rna/data/benchmarks/eternabench", None,
     "chemical-mapping benchmarks, 6 HF datasets"),
    ("Ribonanza quick-start", 0.55, "sbs-rna/data/benchmarks/chemical_probing", None,
     "335,616 profiles"),
]


def du_gb(p: Path) -> float:
    """Decimal GB on disk, matching the units the inventory quotes."""
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / GB
    r = subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True)
    try:
        return int(r.stdout.split()[0]) / GB
    except (ValueError, IndexError):
        return 0.0


def main() -> None:
    rows: List[Dict] = []
    for name, doc_gb, rel, decision, note in INVENTORY:
        here = du_gb(ORG / rel)
        if here == 0.0:
            state = "decided" if decision else "MISSING"
        elif here >= 0.85 * doc_gb:
            state = "present"
        else:
            state = "partial"
        rows.append({"source": name, "documented_gb": doc_gb,
                     "here_gb": round(here, 2), "state": state,
                     "decision": decision, "path": rel, "note": note})

    order = {"MISSING": 0, "partial": 1, "decided": 2, "present": 3}
    rows.sort(key=lambda r: (order[r["state"]], -r["documented_gb"]))
    print(f"{'source':30s} {'documented':>11s} {'here':>9s}  state")
    for r in rows:
        tag = f"  ({r['decision']})" if r["decision"] else ""
        print(f"{r['source']:30s} {r['documented_gb']:9.2f} GB {r['here_gb']:7.2f} GB  "
              f"{r['state']}{tag}")

    miss = [r for r in rows if r["state"] == "MISSING"]
    part = [r for r in rows if r["state"] == "partial"]
    res = {
        "documented_total_gb": 833.3,
        "here_gb": round(sum(r["here_gb"] for r in rows), 2),
        "n_sources": len(rows),
        "n_present": sum(1 for r in rows if r["state"] == "present"),
        "n_partial": len(part), "n_missing": len(miss),
        "n_decided": sum(1 for r in rows if r["state"] == "decided"),
        "missing_gb": round(sum(r["documented_gb"] for r in miss), 2),
        "sources": rows,
    }
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\non disk here: {res['here_gb']:.1f} GB across {res['n_sources']} sources")
    print(f"  present {res['n_present']}   partial {res['n_partial']}   "
          f"MISSING {res['n_missing']}   by decision {res['n_decided']}")
    if miss:
        print(f"\ngenuinely missing ({res['missing_gb']:.2f} GB):")
        for r in miss:
            print(f"  {r['source']:28s} {r['documented_gb']:7.2f} GB -- {r['note']}")
    if part:
        print("\npartial:")
        for r in part:
            print(f"  {r['source']:28s} {r['here_gb']:.2f} of {r['documented_gb']:.2f} GB"
                  f" -- {r['note']}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
