#!/usr/bin/env python3
"""Quantify RNA local rigidity from the sampled structures.

Rigidity proxy: crystallographic B-factor (atomic displacement parameter),
z-normalised *within* each structure so different refinements are comparable.

Tests the architectural premise that RNA has structurally distinct rigid and
flexible regimes, by relating normalised B to:
  - local packing density (neighbours within 10 A of C1')
  - Mg2+ proximity (is ion coordination associated with rigidification?)
  - sequence context (GNRA tetraloops vs other loops)

Only X-ray structures are used for the B-factor analysis; cryo-EM ADPs are not
comparable. Method is read from _exptl.method.
"""
from __future__ import annotations
import gzip, json, math, statistics as st
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMP = ROOT / "data" / "samples" / "structures"
OUT = ROOT / "data" / "samples" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
RNA = {"A", "C", "G", "U"}


def parse(path: Path):
    op = gzip.open if path.suffix == ".gz" else open
    cols, rows, in_loop, header, method = [], [], False, False, None
    with op(path, "rt", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            tok = s.split(None, 1)
            if tok and tok[0] == "_exptl.method" and len(tok) > 1:
                method = tok[1].strip().strip("'\"")
            if s.startswith("_atom_site."):
                if not in_loop:
                    in_loop, cols = True, []
                cols.append(s.split(".", 1)[1]); header = True; continue
            if header and in_loop:
                if s.startswith(("#", "_", "loop_")):
                    in_loop = header = False
                    continue
                if not s:
                    continue
                p = s.split()
                if len(p) >= len(cols):
                    rows.append(dict(zip(cols, p)))
    return method, cols, rows


def analyze(path: Path):
    method, cols, rows = parse(path)
    if not rows or "B_iso_or_equiv" not in cols:
        return None
    # one representative atom per residue (C1') + per-residue mean B
    res = {}
    mg = []
    for r in rows:
        comp = r["label_comp_id"].strip('"')
        atom = r["label_atom_id"].strip('"')
        try:
            x, y, z = float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])
            b = float(r["B_iso_or_equiv"])
        except (ValueError, KeyError):
            continue
        if comp == "MG":
            mg.append((x, y, z)); continue
        if comp not in RNA:
            continue
        key = (r.get("label_asym_id", "?"), r.get("label_seq_id", "?"))
        d = res.setdefault(key, {"comp": comp, "bs": [], "c1": None})
        d["bs"].append(b)
        if atom == "C1'":
            d["c1"] = (x, y, z)
    res = {k: v for k, v in res.items() if v["c1"] and v["bs"]}
    if len(res) < 30:
        return None

    keys = list(res)
    pts = [res[k]["c1"] for k in keys]
    bs = [st.mean(res[k]["bs"]) for k in keys]
    mu, sd = st.mean(bs), (st.pstdev(bs) or 1.0)
    zb = [(b - mu) / sd for b in bs]

    # neighbour density within 10 A via spatial hash
    CELL = 10.0
    grid = defaultdict(list)
    for i, (x, y, z) in enumerate(pts):
        grid[(int(x // CELL), int(y // CELL), int(z // CELL))].append(i)
    dens = []
    for i, (x, y, z) in enumerate(pts):
        ci, cj, ck = int(x // CELL), int(y // CELL), int(z // CELL)
        n = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    for j in grid.get((ci + di, cj + dj, ck + dk), ()):
                        if j != i and math.dist(pts[i], pts[j]) <= CELL:
                            n += 1
        dens.append(n)

    # distance to nearest Mg
    mgd = []
    for p in pts:
        mgd.append(min((math.dist(p, m) for m in mg), default=None))

    return {"pdb": path.name.split(".")[0], "method": method,
            "z_b": zb, "density": dens, "mg_dist": mgd,
            "comp": [res[k]["comp"] for k in keys],
            "chain": [k[0] for k in keys], "seq": [k[1] for k in keys]}


def main():
    files = sorted(SAMP.glob("*.cif.gz"))
    xray, allr = [], []
    for i, f in enumerate(files, 1):
        try:
            r = analyze(f)
        except Exception:
            r = None
        if r:
            allr.append(r)
            if r["method"] and "X-RAY" in r["method"].upper():
                xray.append(r)
        if i % 40 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    print(f"parsed {len(allr)} structures, {len(xray)} X-ray")

    # --- rigidity vs packing density (X-ray only) ---
    pairs = [(d, z) for r in xray for d, z in zip(r["density"], r["z_b"])]
    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 1000)]
    dens_tbl = []
    for lo, hi in bins:
        sel = [z for d, z in pairs if lo <= d < hi]
        if len(sel) >= 50:
            dens_tbl.append({"density_bin": f"{lo}-{hi if hi<1000 else '+'}",
                             "n": len(sel), "mean_zB": round(st.mean(sel), 3),
                             "median_zB": round(st.median(sel), 3)})

    # correlation
    if pairs:
        ds = [p[0] for p in pairs]; zs = [p[1] for p in pairs]
        md, mz = st.mean(ds), st.mean(zs)
        num = sum((a - md) * (b - mz) for a, b in pairs)
        den = math.sqrt(sum((a - md) ** 2 for a in ds) * sum((b - mz) ** 2 for b in zs))
        pearson = num / den if den else 0.0
    else:
        pearson = 0.0

    # --- rigidity vs Mg proximity ---
    mgb = [(d, z) for r in xray for d, z in zip(r["mg_dist"], r["z_b"]) if d is not None]
    mg_tbl = []
    for lo, hi in [(0, 4), (4, 6), (6, 8), (8, 12), (12, 20), (20, 10000)]:
        sel = [z for d, z in mgb if lo <= d < hi]
        if len(sel) >= 50:
            mg_tbl.append({"mg_dist_A": f"{lo}-{hi if hi<10000 else '+'}",
                           "n": len(sel), "mean_zB": round(st.mean(sel), 3)})

    # --- GNRA tetraloop context vs other ---
    gnra = {"GAAA", "GCAA", "GAGA", "GUGA", "GGAA", "GCGA", "GUAA", "GAAG"}
    ctx = {"GNRA": [], "other": []}
    for r in xray:
        by_chain = defaultdict(list)
        for i, ch in enumerate(r["chain"]):
            by_chain[ch].append(i)
        for ch, idxs in by_chain.items():
            comps = [r["comp"][i] for i in idxs]
            for k in range(len(idxs) - 3):
                tet = "".join(comps[k:k + 4])
                tgt = ctx["GNRA"] if tet in gnra else ctx["other"]
                tgt.extend(r["z_b"][i] for i in idxs[k:k + 4])

    summary = {
        "structures_analyzed": len(allr),
        "xray_structures": len(xray),
        "note": "z_B = B-factor z-normalised within each structure; LOWER = more rigid",
        "pearson_density_vs_zB": round(pearson, 4),
        "rigidity_by_packing_density": dens_tbl,
        "rigidity_by_mg_distance": mg_tbl,
        "gnra_tetraloop": {
            "n_GNRA": len(ctx["GNRA"]),
            "mean_zB_GNRA": round(st.mean(ctx["GNRA"]), 3) if ctx["GNRA"] else None,
            "n_other": len(ctx["other"]),
            "mean_zB_other": round(st.mean(ctx["other"]), 3) if ctx["other"] else None,
        },
    }
    (OUT / "rigidity_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
