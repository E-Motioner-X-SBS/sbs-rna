#!/usr/bin/env python3
"""Re-derive Fact 3 (Mg-rigidity) and the ion inventory on RAW PDB entries.

Why this could not be done before
---------------------------------
Both structural corpora on disk are stripped of HETATM, so the ion claims had
no data to be checked against on this machine. `acquire_raw_pdb_entries.py`
fetched the 7,943 raw entries behind them (13.7 GB, 0 failures), which restore
ions, real B-factors, modified residues and whole-entry context.

The claims under test, all from 180 BGSU structures of which only **15**
contained Mg:

  Fact 3   normalised B-factor vs distance to nearest Mg2+, six bins:
           -0.883 (0-4 A) -> -0.673 -> -0.415 -> +0.004 -> +0.756 -> +0.877 (>20 A),
           a monotonic **1.76 sigma** span over 24,623 nt.
  ions     Mg2+ outnumbers all other cations **9:1** (17,428 Mg vs 1,868 K);
           **83%** of inner-sphere coordination is to phosphate OP1/OP2.

Fact 3 is load-bearing: it is the stated reason ions and rigidity share one
expert and the reason ionic condition is a model input at all.

Method — as `analyze_rigidity.py`
---------------------------------
Per-residue mean B and C1' position; B **z-normalised within each structure**
so refinements are comparable; distance to nearest Mg; local density = residues
within 10 A. Structures with < 30 RNA residues are dropped (a variance control,
shown in cycle 11 not to bias the result).

Three deliberate differences:

1. **cKDTree** for Mg distance and density. The published spatial hash is
   O(n * 27 cells) and does not finish on ribosomes at this scale.
2. **X-ray and cryo-EM are reported separately.** The published analysis used
   X-ray only, because cryo-EM ADPs are not comparable to crystallographic
   B-factors. With 7,943 entries there is enough cryo-EM to report it as its
   own stratum rather than discard it -- and if the gradient is real it should
   appear in both, which is a stronger test than either alone.
3. **Modified residues are kept** (they carry C1' and B like any other), since
   raw entries actually contain them.

Usage
-----
    PY=/store/shuvam/.venv/bin/python
    $PY scripts/sampling/recheck_ions_rigidity_rawpdb.py --workers 10
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/structures/raw_pdb_entries"
OUT = ROOT / "data/samples/analysis"

RNA = {"A", "C", "G", "U"}
SOLVENT = {"HOH", "WAT", "DOD"}
CATIONS = {"MG", "K", "NA", "ZN", "MN", "CA", "CO", "NI", "SR", "CS", "BA", "FE"}
#: Distance bins for the gradient, matching the published six.
MG_BINS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 1e9)]
MIN_RES = 30
DENS_R = 10.0
#: Inner-sphere Mg-O coordination cutoff (first hydration shell displaced).
INNER_SPHERE = 2.6


def parse_entry(path: Path) -> Optional[Dict]:
    """Stream one gzipped mmCIF -> per-residue B/C1', cation sites, method."""
    method = "?"
    cols: List[str] = []
    in_loop = header = False
    res: Dict[tuple, dict] = {}
    cations: Dict[str, list] = defaultdict(list)
    phos: List[tuple] = []          # OP1/OP2 oxygens, for the coordination test
    other_o: List[tuple] = []
    try:
        with gzip.open(path, "rt", errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                # Exact tag, not a prefix: `_exptl.method_details` also starts
                # with `_exptl.method` and is almost always `?`, so a prefix
                # match silently overwrites the real method with a question
                # mark. Same failure family as the vacuous `tok in text` guard
                # (defect #28) -- match the token, not its beginning.
                parts = s.split(None, 1)
                if parts and parts[0] == "_exptl.method":
                    if len(parts) > 1:
                        method = parts[1].strip().strip("'\"")
                    continue
                if s.startswith("_atom_site."):
                    if not in_loop:
                        in_loop, cols = True, []
                    cols.append(s.split(".", 1)[1])
                    header = True
                    continue
                if header and in_loop:
                    if s.startswith(("#", "_", "loop_")):
                        in_loop = header = False
                        continue
                    q = s.split()
                    if len(q) < len(cols):
                        continue
                    r = dict(zip(cols, q))
                    comp = r.get("label_comp_id", "").strip('"')
                    if comp in SOLVENT:
                        continue
                    try:
                        x = float(r["Cartn_x"]); y = float(r["Cartn_y"]); z = float(r["Cartn_z"])
                    except (KeyError, ValueError):
                        continue
                    if comp in CATIONS:
                        cations[comp].append((x, y, z))
                        continue
                    atom = r.get("label_atom_id", "").strip('"')
                    # polymer positions only: mmCIF gives label_seq_id to polymers
                    sq = r.get("label_seq_id", ".")
                    if sq in (".", "?"):
                        continue
                    if atom in ("OP1", "OP2"):
                        phos.append((x, y, z))
                    elif atom.startswith("O"):
                        other_o.append((x, y, z))
                    try:
                        b = float(r.get("B_iso_or_equiv", "nan"))
                    except ValueError:
                        b = float("nan")
                    key = (r.get("label_asym_id", "?"), sq)
                    d = res.setdefault(key, {"comp": comp, "bs": [], "c1": None})
                    if not math.isnan(b):
                        d["bs"].append(b)
                    if atom == "C1'":
                        d["c1"] = (x, y, z)
    except (OSError, EOFError):
        return None
    res = {k: v for k, v in res.items() if v["c1"] and v["bs"]}
    if len(res) < MIN_RES:
        return None
    return {"pdb": path.name.split(".")[0], "method": method, "res": res,
            "cations": dict(cations), "phos": phos, "other_o": other_o}


def analyse(path_str: str) -> Optional[Dict]:
    e = parse_entry(Path(path_str))
    if e is None:
        return None
    keys = list(e["res"])
    pts = np.array([e["res"][k]["c1"] for k in keys], dtype=np.float64)
    bs = np.array([np.mean(e["res"][k]["bs"]) for k in keys], dtype=np.float64)
    sd = bs.std()
    if sd <= 0:
        return None
    zb = (bs - bs.mean()) / sd

    tree = cKDTree(pts)
    dens = np.array([len(tree.query_ball_point(p, DENS_R)) - 1 for p in pts])

    mg = np.array(e["cations"].get("MG", []), dtype=np.float64).reshape(-1, 3)
    mgd = None
    if len(mg):
        mgd = cKDTree(mg).query(pts, k=1)[0]

    # inner-sphere coordination: which oxygens sit within 2.6 A of an Mg
    inner_phos = inner_other = 0
    if len(mg):
        mgt = cKDTree(mg)
        if e["phos"]:
            inner_phos = int(sum(len(x) > 0 for x in
                                 mgt.query_ball_point(np.array(e["phos"]), INNER_SPHERE)))
        if e["other_o"]:
            inner_other = int(sum(len(x) > 0 for x in
                                  mgt.query_ball_point(np.array(e["other_o"]), INNER_SPHERE)))

    return {
        "pdb": e["pdb"], "method": e["method"], "n_res": len(keys),
        "cation_counts": {k: len(v) for k, v in e["cations"].items()},
        "inner_phos": inner_phos, "inner_other": inner_other,
        "zb": zb.tolist(), "dens": dens.tolist(),
        "mgd": (mgd.tolist() if mgd is not None else None),
        "n_modified": int(sum(1 for k in keys if e["res"][k]["comp"] not in RNA)),
    }


def gradient(rows: List[Dict]) -> Dict:
    """Mean z_B per Mg-distance bin, plus the span and monotonicity."""
    acc = [[] for _ in MG_BINS]
    for r in rows:
        if not r["mgd"]:
            continue
        for d, z in zip(r["mgd"], r["zb"]):
            for bi, (lo, hi) in enumerate(MG_BINS):
                if lo <= d < hi:
                    acc[bi].append(z)
                    break
    means = [float(np.mean(a)) if a else float("nan") for a in acc]
    ns = [len(a) for a in acc]
    finite = [m for m in means if not math.isnan(m)]
    mono = all(finite[i] <= finite[i + 1] for i in range(len(finite) - 1))
    return {"bin_means": [round(m, 4) for m in means], "bin_n": ns,
            "span_sigma": round(max(finite) - min(finite), 4) if finite else None,
            "monotonic": mono, "n_nt": sum(ns),
            "n_structures": sum(1 for r in rows if r["mgd"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="ions_rigidity_rawpdb.json")
    args = ap.parse_args()

    files = sorted(str(p) for p in RAW.glob("*.cif.gz"))
    if args.limit:
        files = files[::max(1, len(files) // args.limit)][:args.limit]
    print(f"[ions] {len(files):,} raw entries, {args.workers} workers")

    rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, r in enumerate(ex.map(analyse, files, chunksize=4), 1):
            if r:
                rows.append(r)
            if n % 500 == 0:
                print(f"[ions]   {n}/{len(files)} usable={len(rows)}", flush=True)

    xray = [r for r in rows if "X-RAY" in r["method"].upper()]
    em = [r for r in rows if "ELECTRON MICROSCOPY" in r["method"].upper()]

    cat = Counter()
    for r in rows:
        cat.update(r["cation_counts"])
    ip = sum(r["inner_phos"] for r in rows)
    io = sum(r["inner_other"] for r in rows)

    res = {
        "n_entries_usable": len(rows), "n_xray": len(xray), "n_cryoem": len(em),
        "published": {
            "bin_means": [-0.883, -0.673, -0.415, 0.004, 0.756, 0.877],
            "span_sigma": 1.76, "n_nt": 24623, "n_structures": 15,
            "mg_vs_k_ratio": 9.0, "mg_count": 17428, "k_count": 1868,
            "inner_sphere_phosphate_frac": 0.83,
        },
        "cation_inventory": dict(cat.most_common()),
        "mg_vs_all_other_cations": round(
            cat.get("MG", 0) / max(sum(v for k, v in cat.items() if k != "MG"), 1), 2),
        "mg_vs_k": round(cat.get("MG", 0) / max(cat.get("K", 1), 1), 2),
        "inner_sphere": {
            "phosphate_OP1_OP2": ip, "other_oxygen": io,
            "phosphate_frac": round(ip / max(ip + io, 1), 4),
        },
        "gradient_xray": gradient(xray),
        "gradient_cryoem": gradient(em),
        "gradient_all": gradient(rows),
        "modified_residues": {
            "total_residues": sum(r["n_res"] for r in rows),
            "modified": sum(r["n_modified"] for r in rows),
        },
    }
    res["modified_residues"]["frac"] = round(
        res["modified_residues"]["modified"] / max(res["modified_residues"]["total_residues"], 1), 5)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(res, indent=1))

    p = res["published"]
    print(f"\nentries usable {len(rows):,}  (X-ray {len(xray):,}, cryo-EM {len(em):,})")
    print(f"\nION INVENTORY   published Mg:K = 9:1 ({p['mg_count']:,} vs {p['k_count']:,})")
    print(f"                full      Mg:K = {res['mg_vs_k']}:1  "
          f"Mg vs all other cations = {res['mg_vs_all_other_cations']}:1")
    print(f"                counts: {dict(list(res['cation_inventory'].items())[:8])}")
    print(f"\nINNER SPHERE    published 83% to phosphate OP1/OP2")
    print(f"                full      {100*res['inner_sphere']['phosphate_frac']:.1f}%  "
          f"({ip:,} phosphate vs {io:,} other O)")
    print(f"\nFACT 3 gradient (z_B by Mg distance, 0-4 ... >20 A)")
    print(f"  published  {p['bin_means']}  span {p['span_sigma']} sigma  "
          f"({p['n_nt']:,} nt, {p['n_structures']} structures)")
    for name in ("gradient_xray", "gradient_cryoem", "gradient_all"):
        g = res[name]
        print(f"  {name[9:]:<8} {g['bin_means']}  span {g['span_sigma']} sigma  "
              f"monotonic={g['monotonic']}  ({g['n_nt']:,} nt, {g['n_structures']:,} structures)")
    m = res["modified_residues"]
    print(f"\nMODIFIED RESIDUES (raw PDB)  {m['modified']:,}/{m['total_residues']:,} = "
          f"{100*m['frac']:.3f}%   [derivatives: 0.025%, BGSU-180 canonical: 1.05%]")
    print(f"\n[ions] -> {OUT / args.out}")


if __name__ == "__main__":
    main()
