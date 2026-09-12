#!/usr/bin/env python3
"""Measure the true sparsity of RNA contact maps on the sampled structures.

Decides the budget c in "keep K = c*L candidate pairs" for a sparse pair track.
Contact definition: two nucleotides are in contact if any heavy-atom pair is
within 8.0 A (standard for RNA contact maps), excluding |i-j| < 4 along the chain.
"""
from __future__ import annotations
import gzip, json, math, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMP = ROOT / "data" / "samples" / "structures"
OUT = ROOT / "data" / "samples" / "analysis"
RNA = {"A", "C", "G", "U"}
CUT, SEQ_SEP = 8.0, 4


def residues(path: Path):
    op = gzip.open if path.suffix == ".gz" else open
    cols, in_loop, header = [], False, False
    res = defaultdict(list)
    with op(path, "rt", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("_atom_site."):
                if not in_loop:
                    in_loop, cols = True, []
                cols.append(s.split(".", 1)[1]); header = True; continue
            if header and in_loop:
                if s.startswith(("#", "_", "loop_")):
                    in_loop = header = False; continue
                p = s.split()
                if len(p) < len(cols):
                    continue
                r = dict(zip(cols, p))
                if r["label_comp_id"].strip('"') not in RNA:
                    continue
                if r.get("type_symbol") == "H":
                    continue
                try:
                    xyz = (float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"]))
                except (ValueError, KeyError):
                    continue
                try:
                    sq = int(r.get("label_seq_id", "0"))
                except ValueError:
                    continue
                res[(r.get("label_asym_id", "?"), sq)].append(xyz)
    return res


def analyze(path: Path):
    res = residues(path)
    # analyse the single longest chain (well-defined L)
    by_chain = defaultdict(list)
    for (ch, sq), atoms in res.items():
        by_chain[ch].append((sq, atoms))
    if not by_chain:
        return None
    ch = max(by_chain, key=lambda c: len(by_chain[c]))
    items = sorted(by_chain[ch])
    L = len(items)
    if L < 32:
        return None
    cents = [(sum(a[0] for a in at) / len(at), sum(a[1] for a in at) / len(at),
              sum(a[2] for a in at) / len(at)) for _, at in items]
    CELL = 16.0
    grid = defaultdict(list)
    for i, c in enumerate(cents):
        grid[(int(c[0] // CELL), int(c[1] // CELL), int(c[2] // CELL))].append(i)
    contacts = 0
    for i, c in enumerate(cents):
        ci, cj, ck = int(c[0] // CELL), int(c[1] // CELL), int(c[2] // CELL)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    for j in grid.get((ci + di, cj + dj, ck + dk), ()):
                        if j <= i or (j - i) < SEQ_SEP:
                            continue
                        if math.dist(cents[i], cents[j]) > 30.0:
                            continue
                        hit = False
                        for a in items[i][1]:
                            for b in items[j][1]:
                                if math.dist(a, b) <= CUT:
                                    hit = True; break
                            if hit: break
                        if hit:
                            contacts += 1
    return {"pdb": path.name.split(".")[0], "L": L, "contacts": contacts,
            "contacts_per_nt": round(contacts / L, 3),
            "density": round(contacts / (L * (L - 1) / 2), 5)}


def main():
    files = sorted(SAMP.glob("*.cif.gz"))
    rows = []
    for i, f in enumerate(files, 1):
        try:
            r = analyze(f)
        except Exception:
            r = None
        if r:
            rows.append(r)
        if i % 30 == 0:
            print(f"  {i}/{len(files)}  ({len(rows)} usable)", flush=True)
    rows.sort(key=lambda r: r["L"])
    cpn = [r["contacts_per_nt"] for r in rows]
    dens = [r["density"] for r in rows]
    summary = {
        "chains_analyzed": len(rows),
        "contact_def": f"any heavy-atom pair <= {CUT} A, |i-j| >= {SEQ_SEP}",
        "L_median": st.median([r["L"] for r in rows]),
        "L_max": max(r["L"] for r in rows),
        "contacts_per_nt_median": round(st.median(cpn), 3),
        "contacts_per_nt_mean": round(st.mean(cpn), 3),
        "contacts_per_nt_p95": round(sorted(cpn)[int(0.95 * len(cpn))], 3),
        "contacts_per_nt_max": round(max(cpn), 3),
        "map_density_median": round(st.median(dens), 5),
        "by_length_bin": [],
    }
    for lo, hi in [(32, 100), (100, 200), (200, 500), (500, 1500), (1500, 10**9)]:
        sel = [r for r in rows if lo <= r["L"] < hi]
        if sel:
            summary["by_length_bin"].append({
                "L_range": f"{lo}-{hi if hi < 10**9 else '+'}", "n": len(sel),
                "median_L": st.median([r["L"] for r in sel]),
                "median_contacts_per_nt": round(st.median([r["contacts_per_nt"] for r in sel]), 3),
                "median_density_pct": round(100 * st.median([r["density"] for r in sel]), 3)})
    (OUT / "contact_sparsity.json").write_text(json.dumps({"summary": summary, "chains": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
