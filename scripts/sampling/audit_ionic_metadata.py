#!/usr/bin/env python3
"""Close risk R2: is there enough recorded ionic metadata to train on?

PHAROS's headline novelty is accepting ionic conditions as an input. That
requires structures labelled with the ionic environment they were solved in.
mmCIF records crystallisation conditions in _exptl_crystal_grow.pdbx_details as
FREE TEXT, inconsistently. This audits how much is actually recoverable:

  - how many structures carry any crystal-growth details at all
  - how many mention Mg / K / Na / NaCl / KCl / MgCl2 / spermine etc.
  - how many carry a PARSEABLE concentration (e.g. "10 mM MgCl2")
  - pH and temperature availability

Outputs data/samples/analysis/ionic_metadata_audit.json
"""
from __future__ import annotations
import gzip, json, re, statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMP = ROOT / "data" / "samples" / "structures"
OUT = ROOT / "data" / "samples" / "analysis"

# tags that can carry solution conditions
TAGS = ["_exptl_crystal_grow.pdbx_details", "_exptl_crystal_grow.pH",
        "_exptl_crystal_grow.temp", "_exptl_crystal_grow.method",
        "_exptl.method"]

SPECIES = {
    "Mg": r"\bMg\s*(?:Cl2|SO4|\(OAc\)2|acetate)?\b|\bmagnesium\b",
    "K":  r"\bKCl\b|\bK\s*acetate\b|\bpotassium\b|\bKOAc\b",
    "Na": r"\bNaCl\b|\bsodium\b|\bNa\s*cacodylate\b|\bNaOAc\b",
    "Ca": r"\bCaCl2\b|\bcalcium\b",
    "Li": r"\bLiCl\b|\blithium\b",
    "spermine": r"\bspermin[e]?\b|\bspermidine\b",
    "Ir/Os/hexammine": r"\bhexammine\b|\biridium\b|\bosmium\b",
}
# "10 mM MgCl2" / "MgCl2 10 mM" / "0.1 M magnesium"
CONC = re.compile(
    r"(\d+(?:\.\d+)?)\s*(m?M|mM|M|%)\s*([A-Za-z][A-Za-z0-9\(\)]{1,18})"
    r"|([A-Za-z][A-Za-z0-9\(\)]{1,18})[\s,]+(\d+(?:\.\d+)?)\s*(m?M|mM|M|%)", re.I)


def header_fields(path: Path) -> dict:
    """Pull selected single-value mmCIF tags (values may be quoted or on the next line)."""
    op = gzip.open if path.suffix == ".gz" else open
    got, pending = {}, None
    with op(path, "rt", errors="ignore") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if s.startswith("_atom_site."):
                break
            if pending is not None:
                if s.startswith(";"):
                    got[pending] = s[1:].strip()
                    pending = None
                    continue
                got[pending] = s.strip().strip("'\"")
                pending = None
                continue
            t = s.strip().split(None, 1)
            if t and t[0] in TAGS:
                if len(t) > 1:
                    got[t[0]] = t[1].strip().strip("'\"")
                else:
                    pending = t[0]
    return got


def main():
    files = sorted(SAMP.glob("*.cif.gz"))
    rows, sp_count, meth = [], Counter(), Counter()
    for f in files:
        try:
            g = header_fields(f)
        except Exception:
            continue
        det = g.get("_exptl_crystal_grow.pdbx_details", "") or ""
        det = "" if det.strip() in {"?", ".", ""} else det
        ph = g.get("_exptl_crystal_grow.pH", "")
        temp = g.get("_exptl_crystal_grow.temp", "")
        m = (g.get("_exptl.method") or "").upper()
        meth[m or "UNKNOWN"] += 1
        found = [k for k, pat in SPECIES.items() if re.search(pat, det, re.I)]
        for k in found:
            sp_count[k] += 1
        concs = CONC.findall(det) if det else []
        rows.append({
            "pdb": f.name.split(".")[0], "method": m,
            "has_details": bool(det), "details_len": len(det),
            "species": found,
            "n_conc_tokens": len(concs),
            "pH": ph if ph not in {"?", ".", ""} else None,
            "temp": temp if temp not in {"?", ".", ""} else None,
            "details_excerpt": det[:180],
        })

    xr = [r for r in rows if "X-RAY" in r["method"]]
    em = [r for r in rows if "MICROSCOPY" in r["method"]]

    def frac(sub, key):
        return round(sum(1 for r in sub if r[key]) / max(len(sub), 1), 4)

    summary = {
        "structures": len(rows),
        "by_method": dict(meth.most_common()),
        "overall": {
            "with_crystal_growth_details": frac(rows, "has_details"),
            "with_pH": frac(rows, "pH"),
            "with_temp": frac(rows, "temp"),
            "with_any_ion_mentioned": round(sum(1 for r in rows if r["species"])/max(len(rows),1), 4),
            "with_parseable_concentration": round(sum(1 for r in rows if r["n_conc_tokens"])/max(len(rows),1), 4),
        },
        "xray_only": {
            "n": len(xr),
            "with_crystal_growth_details": frac(xr, "has_details"),
            "with_pH": frac(xr, "pH"),
            "with_any_ion_mentioned": round(sum(1 for r in xr if r["species"])/max(len(xr),1), 4),
            "with_parseable_concentration": round(sum(1 for r in xr if r["n_conc_tokens"])/max(len(xr),1), 4),
        },
        "cryoem_only": {
            "n": len(em),
            "with_crystal_growth_details": frac(em, "has_details"),
            "with_any_ion_mentioned": round(sum(1 for r in em if r["species"])/max(len(em),1), 4),
        },
        "species_mentions": dict(sp_count.most_common()),
        "examples": [r["details_excerpt"] for r in rows if r["n_conc_tokens"] >= 2][:5],
    }
    (OUT / "ionic_metadata_audit.json").write_text(
        json.dumps({"summary": summary, "structures": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
