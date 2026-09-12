#!/usr/bin/env python3
"""Measure ion coordination + composition statistics on the sampled mmCIF set.

No Biopython: parses the mmCIF _atom_site loop directly.

Produces data/samples/analysis/ion_stats.json with:
  - per-structure ion inventory (MG/NA/K/ZN/CA/MN/SR)
  - Mg2+ coordination shells to RNA phosphate/base oxygens+nitrogens
    (inner-sphere <= 2.6 A, outer-sphere 3.5-5.0 A)
  - which atom types coordinate Mg2+ (the "coordination exchange" signal)
  - RNA chain length distribution
"""
from __future__ import annotations
import gzip, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMP = ROOT / "data" / "samples" / "structures"
OUT = ROOT / "data" / "samples" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

RNA_RES = {"A", "C", "G", "U", "PSU", "5MC", "7MG", "1MA", "2MG", "M2G", "H2U", "5MU", "OMG", "OMC", "4SU", "MIA"}
IONS = {"MG", "NA", "K", "ZN", "CA", "MN", "SR", "CD", "CO", "NI", "CU", "BA"}
# electronegative RNA atoms that can coordinate a cation
COORD_ATOMS = {"OP1", "OP2", "OP3", "O5'", "O3'", "O2'", "O4'",
               "N1", "N3", "N7", "O2", "O4", "O6", "N6", "N2", "N4"}

INNER = 2.6      # direct / inner-sphere coordination cutoff (A)
OUTER_LO, OUTER_HI = 3.5, 5.0   # water-mediated / outer-sphere shell


def parse_atom_site(path: Path):
    """Yield dicts for each _atom_site row of an mmCIF file."""
    op = gzip.open if path.suffix == ".gz" else open
    cols, rows, in_loop, header = [], [], False, False
    with op(path, "rt", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("_atom_site."):
                if not in_loop:
                    in_loop, cols = True, []
                cols.append(s.split(".", 1)[1])
                header = True
                continue
            if header and in_loop:
                if s.startswith("#") or s.startswith("_") or s.startswith("loop_"):
                    break
                if not s:
                    continue
                parts = s.split()
                if len(parts) < len(cols):
                    continue
                rows.append(dict(zip(cols, parts)))
    return cols, rows


def analyze(path: Path) -> dict | None:
    try:
        cols, rows = parse_atom_site(path)
    except Exception as e:
        return {"pdb": path.name.split(".")[0], "error": str(e)}
    if not rows:
        return None
    need = {"group_PDB", "label_atom_id", "label_comp_id", "Cartn_x", "Cartn_y", "Cartn_z"}
    if not need.issubset(set(cols)):
        return None

    rna_atoms, ion_atoms = [], []
    chains = defaultdict(set)
    resnames = Counter()
    for r in rows:
        comp = r["label_comp_id"].strip('"')
        atom = r["label_atom_id"].strip('"')
        try:
            xyz = (float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"]))
        except ValueError:
            continue
        if comp in IONS and r.get("group_PDB") == "HETATM":
            ion_atoms.append((comp, xyz))
        elif comp in RNA_RES:
            resnames[comp] += 1
            ch = r.get("label_asym_id", "?")
            seq = r.get("label_seq_id", "?")
            chains[ch].add(seq)
            if atom in COORD_ATOMS:
                rna_atoms.append((atom, xyz))

    if not rna_atoms:
        return None

    # spatial hash for neighbour search
    CELL = 5.0
    grid = defaultdict(list)
    for atom, (x, y, z) in rna_atoms:
        grid[(int(x // CELL), int(y // CELL), int(z // CELL))].append((atom, x, y, z))

    inner_counts, outer_counts = [], []
    inner_partner, outer_partner = Counter(), Counter()
    for ion, (x, y, z) in ion_atoms:
        if ion != "MG":
            continue
        ci, cj, ck = int(x // CELL), int(y // CELL), int(z // CELL)
        n_in = n_out = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    for atom, ax, ay, az in grid.get((ci + di, cj + dj, ck + dk), ()):
                        d = math.dist((x, y, z), (ax, ay, az))
                        if d <= INNER:
                            n_in += 1; inner_partner[atom] += 1
                        elif OUTER_LO <= d <= OUTER_HI:
                            n_out += 1; outer_partner[atom] += 1
        inner_counts.append(n_in); outer_counts.append(n_out)

    return {
        "pdb": path.name.split(".")[0],
        "n_rna_residues": sum(len(v) for v in chains.values()),
        "n_rna_chains": len(chains),
        "chain_lengths": sorted((len(v) for v in chains.values()), reverse=True)[:10],
        "ions": dict(Counter(i for i, _ in ion_atoms)),
        "n_mg": sum(1 for i, _ in ion_atoms if i == "MG"),
        "mg_inner_counts": inner_counts,
        "mg_outer_counts": outer_counts,
        "inner_partners": dict(inner_partner),
        "outer_partners": dict(outer_partner),
        "modified_residues": {k: v for k, v in resnames.items() if k not in {"A", "C", "G", "U"}},
    }


def main() -> None:
    files = sorted(SAMP.glob("*.cif.gz"))
    print(f"analyzing {len(files)} structures ...", flush=True)
    results = []
    for i, f in enumerate(files, 1):
        r = analyze(f)
        if r:
            results.append(r)
        if i % 25 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    (OUT / "ion_stats.json").write_text(json.dumps(results, indent=2))

    # ---- aggregate report ----
    ok = [r for r in results if "error" not in r and r.get("n_rna_residues")]
    tot_ions = Counter()
    for r in ok:
        tot_ions.update(r.get("ions", {}))
    all_inner = [c for r in ok for c in r.get("mg_inner_counts", [])]
    all_outer = [c for r in ok for c in r.get("mg_outer_counts", [])]
    inner_p, outer_p = Counter(), Counter()
    for r in ok:
        inner_p.update(r.get("inner_partners", {}))
        outer_p.update(r.get("outer_partners", {}))
    lens = [r["n_rna_residues"] for r in ok]

    def pct(xs, q):
        if not xs: return 0
        s = sorted(xs); return s[min(len(s) - 1, int(q * len(s)))]

    summary = {
        "structures_parsed": len(ok),
        "structures_with_any_ion": sum(1 for r in ok if r.get("ions")),
        "structures_with_mg": sum(1 for r in ok if r.get("n_mg", 0) > 0),
        "total_ions_by_type": dict(tot_ions.most_common()),
        "total_mg_ions": len(all_inner),
        "mg_with_inner_sphere_contact": sum(1 for c in all_inner if c > 0),
        "frac_mg_inner_sphere": round(sum(1 for c in all_inner if c > 0) / max(len(all_inner), 1), 4),
        "mean_inner_contacts_per_mg": round(sum(all_inner) / max(len(all_inner), 1), 3),
        "mean_outer_contacts_per_mg": round(sum(all_outer) / max(len(all_outer), 1), 3),
        "top_inner_sphere_partners": dict(inner_p.most_common(10)),
        "top_outer_sphere_partners": dict(outer_p.most_common(10)),
        "rna_residues_total": sum(lens),
        "chain_len_median": pct(lens, 0.5),
        "chain_len_p10": pct(lens, 0.1),
        "chain_len_p90": pct(lens, 0.9),
        "chain_len_max": max(lens) if lens else 0,
    }
    (OUT / "ion_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
