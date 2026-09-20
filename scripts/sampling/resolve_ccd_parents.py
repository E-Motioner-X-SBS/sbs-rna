#!/usr/bin/env python3
"""Resolve every modified RNA residue to its standard parent base, from the CCD.

`VOCAB.md` specifies a `MOD` embedding keyed by mmCIF `comp_id` "falling back to
the parent base", and `CHEMISTRY.md` dims 20-22 classify modifications into
methylation / pseudouridylation / other. Both need to know what a residue *is*.

`modification_census.py` established that the entry files cannot supply this:
their `_chem_comp` category carries only
`id / type / mon_nstd_flag / name / pdbx_synonyms / formula / formula_weight`.
The parent declaration lives in the **PDB Chemical Component Dictionary**, one
119 MB file covering every component the archive has ever used, which is both
the authoritative source and cheaper than one API call per species.

This reads the CCD once and emits, for every modification the census found:

    comp_id -> {parent, name, type, class}

where `parent` is `_chem_comp.mon_nstd_parent_comp_id` (A/C/G/U), and `class` is
the CHEMISTRY.md dim 20-22 bucket. **The class is derived from the CCD's own
`name` field, not from a hand-written list of comp_ids** -- that is the
distinction defect #22 turned on. A residue whose parent or class cannot be
established is reported as such rather than guessed.

Usage:
    /store/shuvam/.venv/bin/python scripts/sampling/resolve_ccd_parents.py
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
CCD = ROOT / "data/structures/ccd/components.cif.gz"
OUT = ROOT / "data/samples/analysis"
ACGU = {"A", "C", "G", "U"}

#: CHEMISTRY.md dims 20-22. Matched against the CCD `name`, which is a
#: systematic chemical name ("5-METHYLURIDINE 5'-MONOPHOSPHATE"), so these are
#: chemistry terms rather than a list of component codes.
CLASS_PATTERNS = [
    ("pseudouridine", re.compile(r"PSEUDOURIDINE|PSEUDO-URIDINE", re.I)),
    ("methylation", re.compile(r"\bMETHYL|\bDIMETHYL|\bTRIMETHYL|"
                               r"\bO2'-METHYL|\bM\d?[ACGU]\b", re.I)),
]


def classify(name: str) -> str:
    """CHEMISTRY.md dim 20-22 bucket, from the component's systematic name."""
    for label, pat in CLASS_PATTERNS:
        if pat.search(name):
            return label
    return "other"


def read_ccd(wanted: set[str]) -> Dict[str, Dict[str, str]]:
    """Parse only the `data_<id>` blocks we care about.

    The CCD is one concatenated mmCIF per component, so the block header is the
    index: skip to the next `data_` whenever the current id is not wanted, which
    turns a 119 MB parse into a scan.
    """
    out: Dict[str, Dict[str, str]] = {}
    cur: Optional[str] = None
    keep = False
    rec: Dict[str, str] = {}
    semi_tag: Optional[str] = None
    semi: list[str] = []
    semi_open = False

    with gzip.open(CCD, "rt", errors="ignore") as fh:
        for line in fh:
            if line.startswith("data_"):
                if keep and cur:
                    out[cur] = rec
                cur = line[5:].strip()
                keep = cur in wanted
                rec = {}
                semi_tag, semi, semi_open = None, [], False
                continue
            if not keep:
                continue
            s = line.rstrip("\n")
            if semi_tag is not None:
                # A `;`-delimited value opens with `;` on its own line and
                # closes with a lone `;`. Treating the OPENING delimiter as the
                # terminator silently yields an empty value -- which is what
                # made 17,078 modified residues classify as "unknown" despite
                # the CCD naming every one of them.
                if not semi_open:
                    semi_open = True
                    rest0 = s[1:].strip()
                    if rest0:
                        semi.append(rest0)
                    continue
                if s.startswith(";"):
                    rec[semi_tag] = " ".join(semi).strip()
                    semi_tag, semi, semi_open = None, [], False
                else:
                    semi.append(s.strip())
                continue
            if not s.startswith("_chem_comp."):
                continue
            tag, _, rest = s.strip().partition(" ")
            tag = tag.split(".", 1)[1]
            rest = rest.strip()
            if tag not in {"name", "type", "mon_nstd_parent_comp_id", "formula"}:
                continue
            if rest:
                rec[tag] = rest.strip("'\"")
            else:
                semi_tag, semi, semi_open = tag, [], False
    if keep and cur:
        out[cur] = rec
    return out


def main() -> None:
    if not CCD.exists():
        raise SystemExit(f"CCD not found at {CCD}; download components.cif.gz first")
    census = json.loads((OUT / "modification_census.json").read_text())
    # `counts` holds every species; `top_modifications` is a truncated view of
    # it kept for readability and must not be summed over.
    counts = census.get("counts") or dict(census["top_modifications"])
    wanted = set(counts)
    print(f"[ccd] resolving {len(wanted):,} modification species")

    ccd = read_ccd(wanted)
    print(f"[ccd] found {len(ccd):,} of them in the dictionary")

    resolved: Dict[str, Dict] = {}
    for cid in sorted(wanted):
        r = ccd.get(cid, {})
        parent = (r.get("mon_nstd_parent_comp_id") or "?").strip("'\"")
        parent = parent if parent in ACGU else "?"
        name = r.get("name", "")
        resolved[cid] = {
            "parent": parent,
            "name": name,
            "type": r.get("type", "?"),
            "class": classify(name) if name else "unknown",
            "count": counts.get(cid, 0),
        }

    n_res = sum(v["count"] for v in resolved.values())
    with_parent = sum(v["count"] for v in resolved.values() if v["parent"] in ACGU)
    by_class: Dict[str, int] = {}
    for v in resolved.values():
        by_class[v["class"]] = by_class.get(v["class"], 0) + v["count"]
    by_parent: Dict[str, int] = {}
    for v in resolved.values():
        by_parent[v["parent"]] = by_parent.get(v["parent"], 0) + v["count"]

    res = {
        "n_species": len(resolved),
        "n_species_in_ccd": len(ccd),
        "n_modified_residues": n_res,
        "residues_with_parent": with_parent,
        "frac_residues_with_parent": round(with_parent / max(n_res, 1), 4),
        "species_without_parent": sorted(
            (c for c, v in resolved.items() if v["parent"] == "?"),
            key=lambda c: -resolved[c]["count"])[:20],
        "by_class_residues": by_class,
        "by_parent_residues": by_parent,
        "components": resolved,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ccd_parents.json").write_text(json.dumps(res, indent=1))

    print(f"\nmodified residues       : {n_res:,}")
    print(f"resolvable to A/C/G/U   : {with_parent:,} "
          f"({100*res['frac_residues_with_parent']:.1f}%)")
    print(f"by parent               : {by_parent}")
    print(f"by CHEMISTRY class      : {by_class}")
    if res["species_without_parent"]:
        print(f"unresolved species (top): {res['species_without_parent'][:10]}")
    print(f"\n[ccd] -> {OUT / 'ccd_parents.json'}")


if __name__ == "__main__":
    main()
