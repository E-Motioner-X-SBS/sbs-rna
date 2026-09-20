#!/usr/bin/env python3
"""Index pdb_hunter's derived RNA annotations into the sbs-rna catalog.

Why
---
`pdb_hunter/RNA_Database` holds 10,424 harvested PDB entries with a per-entry
annotation package that sbs-rna has no equivalent of, and which nothing in
ARCHITECTURE v0.2 accounted for until this script:

  9,347  dot-bracket secondary structure (.dbn) + base pairs (.bpseq)
 10,280  backbone coordinates (.xyz)
 10,073  MolProbity clashscore
  6,316  Rfam family assignment
  5,169  resolution
  6,981  base-pair counts

Three of these change the design rather than merely adding rows.

**Clashscore** is a per-entry quality metric. The corpus is saturated at ~10.4k
entries (v0.2 Section 1.1), so examples cannot be added -- but they can be
*weighted*. A model fit on 673 clean sequences should not treat a clashscore-60
structure the same as a clashscore-2 one, and nothing in v0.2 does anything
with this.

**Dot-bracket + bpseq** roughly doubles the 2D channel and, unlike bpRNA, is
derived from the same coordinates as the 3D labels -- so 2D and 3D supervision
are consistent per entry rather than coming from different pipelines.

**Rfam family** is what makes a family-disjoint split possible. Random splits
leak: the corpus is rRNA-dominated (G3: 92.65% of residues), so a random split
puts homologues of the test set in training. 6,316 family assignments make the
honest split constructible.

Output
------
`data/catalog/pdb_hunter_index.json` -- one record per entry, with paths into
the pdb_hunter tree so files are referenced rather than copied (154 GB stays
where it is), plus a summary the architecture can cite.

Usage
-----
    PY=/store/shuvam/.venv/bin/python
    $PY scripts/integrate_pdb_hunter.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
HUNTER = ROOT.parent / "pdb_hunter"
INVENTORY = HUNTER / "RNA_training_inventory.csv"
OUT = ROOT / "data" / "catalog" / "pdb_hunter_index.json"

#: Clashscore bands for training weights. MolProbity convention: lower is better;
#: <10 is good at any resolution, >40 is poor. Weights are a starting point to be
#: tuned, not a measured optimum -- stated so nobody reads them as derived.
CLASH_BANDS = [(0, 10, 1.00), (10, 20, 0.80), (20, 40, 0.55), (40, 1e9, 0.30)]


def clash_weight(score: float | None) -> float:
    """Training weight from clashscore; unscored entries get the median band."""
    if score is None:
        return 0.80
    for lo, hi, w in CLASH_BANDS:
        if lo <= score < hi:
            return w
    return 0.30


def _f(v: str) -> float | None:
    v = (v or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def main() -> None:
    if not INVENTORY.exists():
        raise SystemExit(f"inventory not found: {INVENTORY}")
    rows = list(csv.DictReader(open(INVENTORY)))

    records: List[Dict] = []
    for r in rows:
        pid = r["pdb_id"].lower()
        folder = r.get("relative_path", "").strip()
        clash = _f(r.get("clashscore", ""))
        rec = {
            "pdb_id": pid,
            "folder": folder,
            "resolution": _f(r.get("resolution_angstrom", "")),
            "method": (r.get("method") or "").strip() or None,
            "clashscore": clash,
            "train_weight": clash_weight(clash),
            "rfam_family": (r.get("rfam_family") or "").strip() or None,
            "rfam_accession": (r.get("rfam_accession") or "").strip() or None,
            "sequence_length": _f(r.get("sequence_length", "")),
            "n_base_pairs_cww": _f(r.get("base_pairs_cww", "")),
            "has": {k[4:]: (r.get(k, "").strip() == "True")
                    for k in ("has_dbn", "has_bpseq", "has_bb_xyz",
                              "has_clean_cif", "has_clean_fasta", "has_rna3db")},
        }
        records.append(rec)

    def n_with(pred) -> int:
        return sum(1 for x in records if pred(x))

    clashes = [x["clashscore"] for x in records if x["clashscore"] is not None]
    res = [x["resolution"] for x in records if x["resolution"] is not None]
    fams = Counter(x["rfam_family"] for x in records if x["rfam_family"])

    summary = {
        "n_entries": len(records),
        "source": str(INVENTORY),
        "note": ("files are REFERENCED in the pdb_hunter tree, not copied; "
                 "that tree is 154 GB"),
        "annotations": {
            "dot_bracket_2D": n_with(lambda x: x["has"]["dbn"]),
            "bpseq": n_with(lambda x: x["has"]["bpseq"]),
            "backbone_xyz": n_with(lambda x: x["has"]["bb_xyz"]),
            "clashscore": len(clashes),
            "resolution": len(res),
            "rfam_family": sum(fams.values()),
        },
        "quality": {
            "clashscore_median": round(st.median(clashes), 2) if clashes else None,
            "clashscore_p90": round(sorted(clashes)[int(0.9 * len(clashes))], 2)
                              if clashes else None,
            "resolution_median": round(st.median(res), 2) if res else None,
            "weight_bands": [{"lo": lo, "hi": (None if hi > 1e8 else hi), "weight": w}
                             for lo, hi, w in CLASH_BANDS],
            "entries_per_band": {
                f"{lo}-{'inf' if hi > 1e8 else int(hi)}":
                    n_with(lambda x, lo=lo, hi=hi: x["clashscore"] is not None
                           and lo <= x["clashscore"] < hi)
                for lo, hi, _ in CLASH_BANDS},
        },
        "rfam": {
            "entries_with_family": sum(fams.values()),
            "distinct_families": len(fams),
            "top_families": fams.most_common(10),
            "note": ("family labels are what make a family-disjoint split "
                     "constructible; a random split leaks homologues because the "
                     "corpus is rRNA-dominated (G3: 92.65% of residues)"),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "entries": records}, indent=1))

    a, q = summary["annotations"], summary["quality"]
    print(f"indexed {summary['n_entries']:,} pdb_hunter entries -> {OUT}")
    print(f"\nannotations sbs-rna did not have:")
    for k, v in a.items():
        print(f"  {k:<20}{v:>8,}")
    print(f"\nquality: clashscore median {q['clashscore_median']}, "
          f"p90 {q['clashscore_p90']}; resolution median {q['resolution_median']} A")
    print(f"  entries per weight band: {q['entries_per_band']}")
    print(f"\nRfam: {summary['rfam']['entries_with_family']:,} entries across "
          f"{summary['rfam']['distinct_families']:,} families")
    print(f"  top: {[f for f, _ in summary['rfam']['top_families'][:5]]}")


if __name__ == "__main__":
    main()
