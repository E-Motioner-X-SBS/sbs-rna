#!/usr/bin/env python3
"""Second pass on risk R2: cryo-EM records buffers as STRUCTURED data.

The first audit only looked at _exptl_crystal_grow.pdbx_details (free text,
X-ray only) and concluded ionic metadata was scarce. But cryo-EM entries carry
_em_buffer_component -- a machine-readable loop with concentration, units,
formula and name, plus _em_buffer.pH. Since 65% of our sampled structures are
cryo-EM, this materially changes the answer.

Measures, for EM structures: buffer-component coverage, which ionic species
appear with numeric concentrations, and how many yield a usable
([Mg2+], [K+], [Na+], pH) conditioning vector.
"""
from __future__ import annotations
import gzip, json, re, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMP = ROOT / "data" / "samples" / "structures"
OUT = ROOT / "data" / "samples" / "analysis"

ION_PATTERNS = {
    "Mg": r"mgcl|magnesium|mg\(oac\)|mgso4|mg2\+|^mg$",
    "K":  r"kcl|potassium|koac|k-acetate|^k$|kglutamate",
    "Na": r"nacl|sodium|naoac|^na$",
    "Ca": r"cacl|calcium",
    "Zn": r"zncl|zinc",
    "spermine": r"spermi",
}


def parse_em_buffer(path: Path):
    """Return (pH, [components]) from the _em_buffer / _em_buffer_component tags."""
    op = gzip.open if path.suffix == ".gz" else open
    ph, comps, cols, in_loop = None, [], [], False
    with op(path, "rt", errors="ignore") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if s.startswith("_atom_site."):
                break
            t = s.strip().split(None, 1)
            if t and t[0] == "_em_buffer.pH" and len(t) > 1:
                v = t[1].strip().strip("'\"")
                if v not in {"?", "."}:
                    try: ph = float(v)
                    except ValueError: pass
            if s.startswith("_em_buffer_component."):
                if not in_loop:
                    in_loop, cols = True, []
                cols.append(s.strip().split(".", 1)[1])
                continue
            if in_loop:
                st_ = s.strip()
                if st_.startswith(("#", "_", "loop_")) or not st_:
                    in_loop = False
                    continue
                # mmCIF rows may quote values containing spaces
                parts = re.findall(r"'[^']*'|\"[^\"]*\"|\S+", st_)
                if len(parts) >= len(cols):
                    comps.append(dict(zip(cols, [p.strip("'\"") for p in parts])))
    return ph, comps


def classify(comp: dict):
    blob = f"{comp.get('name','')} {comp.get('formula','')}".lower()
    for ion, pat in ION_PATTERNS.items():
        if re.search(pat, blob):
            return ion
    return None


def to_mM(val: str, unit: str):
    try:
        v = float(val)
    except (ValueError, TypeError):
        return None
    u = (unit or "").strip().lower()
    if u == "mm": return v
    if u == "m":  return v * 1000
    if u in {"um", "uM".lower(), "µm"}: return v / 1000
    if u == "nm": return v / 1e6
    return None


def main():
    files = sorted(SAMP.glob("*.cif.gz"))
    rows, ion_hits, ion_conc = [], Counter(), defaultdict(list)
    n_em = 0
    for f in files:
        try:
            with gzip.open(f, "rt", errors="ignore") as fh:
                is_em = False
                for line in fh:
                    if line.startswith("_atom_site."):
                        break
                    if "ELECTRON MICROSCOPY" in line:
                        is_em = True; break
            if not is_em:
                continue
            n_em += 1
            ph, comps = parse_em_buffer(f)
        except Exception:
            continue
        ions = {}
        for c in comps:
            ion = classify(c)
            if not ion:
                continue
            mm = to_mM(c.get("concentration"), c.get("concentration_units"))
            if mm is not None:
                ions.setdefault(ion, mm)
                ion_conc[ion].append(mm)
            ion_hits[ion] += 1
        rows.append({"pdb": f.name.split(".")[0], "pH": ph,
                     "n_components": len(comps), "ions_mM": ions,
                     "component_names": [c.get("name") for c in comps][:8]})

    have_comp = [r for r in rows if r["n_components"] > 0]
    have_ion = [r for r in rows if r["ions_mM"]]
    have_mg = [r for r in rows if "Mg" in r["ions_mM"]]
    have_vec = [r for r in rows if r["ions_mM"] and r["pH"] is not None]

    def q(xs, p):
        return round(sorted(xs)[min(len(xs)-1, int(p*len(xs)))], 2) if xs else None

    summary = {
        "em_structures": n_em,
        "with_buffer_components": len(have_comp),
        "frac_with_buffer_components": round(len(have_comp)/max(n_em,1), 4),
        "with_pH": round(sum(1 for r in rows if r["pH"] is not None)/max(n_em,1), 4),
        "with_any_ion_concentration": round(len(have_ion)/max(n_em,1), 4),
        "with_Mg_concentration": round(len(have_mg)/max(n_em,1), 4),
        "with_full_conditioning_vector": round(len(have_vec)/max(n_em,1), 4),
        "ion_mentions": dict(ion_hits.most_common()),
        "concentration_mM": {
            k: {"n": len(v), "median": q(v, .5), "p10": q(v, .1), "p90": q(v, .9)}
            for k, v in sorted(ion_conc.items(), key=lambda x: -len(x[1]))},
        "examples": [{"pdb": r["pdb"], "pH": r["pH"], "ions_mM": r["ions_mM"]}
                     for r in have_vec[:6]],
    }
    (OUT / "em_buffer_audit.json").write_text(
        json.dumps({"summary": summary, "structures": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
