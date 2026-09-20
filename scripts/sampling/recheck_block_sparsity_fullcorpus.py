#!/usr/bin/env python3
"""Re-derive the block-sparsity / effective-c claims on the FULL server corpus.

Why
---
Every geometric number in ARCHITECTURE.md was measured on **180** BGSU
representative structures fetched onto the laptop. The server holds the actual
corpus: 15,441 RNA3DB per-chain mmCIFs and 14,366 gRNAde/RNASolo PDBs. OQ-7 is
explicitly deferred pending exactly this:

    "target_c = 20 has only 4.8% headroom at the worst chain (19.04). Adequate
     on these 180 structures; the quantity that pushes a chain up is
     modified-residue density, which this corpus under-represents. Widen c or
     measure the distribution on more data before fixing it."

`c = 20` is a load-bearing PHAROS-Small parameter: it sets the hierarchical pair
track's budget, hence the 2.2%-of-dense cost and the whole §5 argument. If the
distribution's tail breaches 20 on a larger sample, the budget is wrong.

Method — identical to `analyze_block_sparsity.py`
-------------------------------------------------
Longest chain per file; contact = any heavy-atom pair within 8.0 A with
|i-j| >= 4; occupancy at block size b; effective_c = occupied_blocks * b^2 / L.
Chains restricted to 64 <= L <= 3000 with >= 20 contacts, as published.

Two differences, both deliberate:

1. **Both residue definitions are run.** The published script keeps only
   {A,C,G,U}, which deletes modified residues from the *middle* of a chain and
   renumbers everything after them (defect D14: 34.1% of chains change length,
   worst +27.9%). Since OQ-7 names modified-residue density as the risk, the
   two rules are measured side by side rather than one being assumed.
     - `acgu`      keep only A/C/G/U            (published rule)
     - `polymer`   keep every polymer residue   (canonical rule)
2. **cKDTree** replaces the O(L^2) python loop, which does not terminate at this
   scale (the same fix cycle 11 applied in C16).

This is a DIFFERENT SAMPLE from the published 180, not a reproduction of it.
RNA3DB mmCIFs are single-chain extracts with HETATM and B-factors stripped, so
nothing here can speak to the Mg/rigidity or ion claims; only geometry.

Usage
-----
    PY=/store/shuvam/.venv/bin/python
    $PY scripts/sampling/recheck_block_sparsity_fullcorpus.py --limit 2000
    $PY scripts/sampling/recheck_block_sparsity_fullcorpus.py --all --workers 12
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
RNA3DB = ROOT / "data/structures/databases/rna3db/rna3db_extracted/rna3db-mmcifs"
GRNADE = ROOT / "data/structures/databases/grnade_rnasolo/extracted/raw"
OUT = ROOT / "data/samples/analysis"

ACGU = {"A", "C", "G", "U"}
CUT, SEQ_SEP = 8.0, 4
BLOCKS = [4, 8, 16, 32]
MIN_L, MAX_L, MIN_CONTACTS = 64, 3000, 20

#: Non-polymer species that share an auth chain with the RNA they solvate.
#: Excluded under BOTH rules -- defect D23 (water and ions swept in by an
#: entity-only selection) is not a definitional question, it is a bug.
SOLVENT = {"HOH", "WAT", "DOD"}


def _resname_is_polymer_nt(name: str) -> bool:
    """A nucleotide-like polymer residue, modified ones included.

    Deliberately permissive: the point of the `polymer` rule is to keep
    modified residues that the ACGU filter deletes. Solvent and monatomic ions
    are excluded by name length / membership rather than by a curated list.
    """
    n = name.strip().strip('"')
    if n in SOLVENT:
        return False
    return 1 <= len(n) <= 3


def parse_mmcif(path: Path) -> Dict[str, List[List[tuple]]]:
    """RNA3DB per-chain mmCIF -> {rule: [residue -> [(x,y,z), ...]]}."""
    cols: List[str] = []
    in_loop = header = False
    res_acgu: Dict[tuple, list] = defaultdict(list)
    res_poly: Dict[tuple, list] = defaultdict(list)
    with open(path, "rt", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
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
                p = s.split()
                if len(p) < len(cols):
                    continue
                r = dict(zip(cols, p))
                if r.get("type_symbol") == "H":
                    continue
                comp = r.get("label_comp_id", "").strip('"')
                try:
                    xyz = (float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"]))
                except (ValueError, KeyError):
                    continue
                # polymer position: mmCIF gives label_seq_id only to polymers
                sq_raw = r.get("label_seq_id", ".")
                if sq_raw in (".", "?"):
                    continue
                try:
                    sq = int(sq_raw)
                except ValueError:
                    continue
                key = (r.get("label_asym_id", "?"), sq)
                if _resname_is_polymer_nt(comp):
                    res_poly[key].append(xyz)
                if comp in ACGU:
                    res_acgu[key].append(xyz)
    return {"acgu": _longest_chain(res_acgu), "polymer": _longest_chain(res_poly)}


def parse_pdb(path: Path) -> Dict[str, List[List[tuple]]]:
    """gRNAde/RNASolo PDB -> {rule: [residue -> [(x,y,z), ...]]}."""
    res_acgu: Dict[tuple, list] = defaultdict(list)
    res_poly: Dict[tuple, list] = defaultdict(list)
    with open(path, "rt", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            elem = line[76:78].strip()
            if elem == "H":
                continue
            comp = line[17:20].strip()
            chain = line[21]
            try:
                seq = int(line[22:26])
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            key = (chain, seq)
            if _resname_is_polymer_nt(comp):
                res_poly[key].append(xyz)
            if comp in ACGU:
                res_acgu[key].append(xyz)
    return {"acgu": _longest_chain(res_acgu), "polymer": _longest_chain(res_poly)}


def _longest_chain(res: Dict[tuple, list]) -> Optional[List[List[tuple]]]:
    if not res:
        return None
    by: Dict[str, list] = defaultdict(list)
    for k in res:
        by[k[0]].append(k)
    ch = max(by, key=lambda c: len(by[c]))
    return [res[k] for k in sorted(by[ch], key=lambda k: k[1])]


def contact_pairs(atoms: List[List[tuple]]) -> List[tuple]:
    """Residue pairs with any heavy-atom distance <= CUT and |i-j| >= SEQ_SEP.

    cKDTree over the flat atom list with a residue-index label, then reduce
    atom pairs to residue pairs. Equivalent to the published double loop; the
    loop simply does not finish at corpus scale.
    """
    flat, owner = [], []
    for i, at in enumerate(atoms):
        for a in at:
            flat.append(a)
            owner.append(i)
    if not flat:
        return []
    pts = np.asarray(flat, dtype=np.float64)
    own = np.asarray(owner, dtype=np.int64)
    tree = cKDTree(pts)
    out = set()
    for a_idx, b_idx in tree.query_pairs(CUT, output_type="ndarray"):
        i, j = own[a_idx], own[b_idx]
        if j - i >= SEQ_SEP:
            out.add((int(i), int(j)))
        elif i - j >= SEQ_SEP:
            out.add((int(j), int(i)))
    return sorted(out)


def block_stats(pairs: List[tuple], L: int) -> Dict:
    rec = {}
    dense = L * (L - 1) / 2
    for b in BLOCKS:
        nb = (L + b - 1) // b
        occ = {(p[0] // b, p[1] // b) for p in pairs}
        total_blocks = nb * (nb + 1) // 2
        rec[str(b)] = {
            "occupied": len(occ),
            "occupancy_frac": round(len(occ) / max(total_blocks, 1), 5),
            "fine_vs_dense": round(len(occ) * b * b / dense, 5),
            "effective_c": round(len(occ) * b * b / L, 2),
        }
    return rec


def analyse_one(path_str: str) -> Optional[Dict]:
    path = Path(path_str)
    try:
        chains = parse_mmcif(path) if path.suffix == ".cif" else parse_pdb(path)
    except Exception:                                            # noqa: BLE001
        return None
    rec = {"file": path.name, "rules": {}}
    for rule, atoms in chains.items():
        if not atoms:
            continue
        L = len(atoms)
        if not (MIN_L <= L <= MAX_L):
            continue
        pairs = contact_pairs(atoms)
        if len(pairs) < MIN_CONTACTS:
            continue
        rec["rules"][rule] = {
            "L": L,
            "n_contacts": len(pairs),
            "contacts_per_nt": round(len(pairs) / L, 4),
            "density": round(len(pairs) / (L * (L - 1) / 2), 6),
            "blocks": block_stats(pairs, L),
        }
    return rec if rec["rules"] else None


def summarise(rows: List[Dict], rule: str) -> Dict:
    sel_all = [r["rules"][rule] for r in rows if rule in r["rules"]]
    if not sel_all:
        return {}
    out = {"n_chains": len(sel_all), "by_length_bin": [], "global": {}}
    for lo, hi in [(64, 200), (200, 500), (500, 1500), (1500, 3000)]:
        sel = [r for r in sel_all if lo <= r["L"] < hi]
        if not sel:
            continue
        e = {"L_range": f"{lo}-{hi}", "n": len(sel),
             "median_L": st.median([r["L"] for r in sel]),
             "mean_contacts_per_nt": round(st.mean(r["contacts_per_nt"] for r in sel), 3),
             "mean_density": round(st.mean(r["density"] for r in sel), 5),
             "per_block": {}}
        for b in BLOCKS:
            k = str(b)
            cs = [r["blocks"][k]["effective_c"] for r in sel]
            e["per_block"][k] = {
                "mean_occupancy_frac": round(st.mean(r["blocks"][k]["occupancy_frac"] for r in sel), 5),
                "mean_fine_vs_dense": round(st.mean(r["blocks"][k]["fine_vs_dense"] for r in sel), 5),
                "mean_effective_c": round(st.mean(cs), 2),
                "max_effective_c": round(max(cs), 2),
                "p99_effective_c": round(float(np.percentile(cs, 99)), 2),
                "n_over_20": int(sum(c > 20 for c in cs)),
            }
        out["by_length_bin"].append(e)
    for b in BLOCKS:
        k = str(b)
        cs = [r["blocks"][k]["effective_c"] for r in sel_all]
        out["global"][k] = {
            "max_effective_c": round(max(cs), 2),
            "p99_effective_c": round(float(np.percentile(cs, 99)), 2),
            "p999_effective_c": round(float(np.percentile(cs, 99.9)), 2),
            "n_over_20": int(sum(c > 20 for c in cs)),
            "frac_over_20": round(sum(c > 20 for c in cs) / len(cs), 5),
            "worst_files": [r0["file"] for r0 in sorted(
                (r for r in rows if rule in r["rules"]),
                key=lambda r: -r["rules"][rule]["blocks"][k]["effective_c"])[:5]],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="block_sparsity_fullcorpus.json")
    args = ap.parse_args()

    files = sorted(str(p) for p in RNA3DB.rglob("*.cif"))
    files += sorted(str(p) for p in GRNADE.glob("*.pdb"))
    if not args.all:
        # deterministic stride so the subsample spans both corpora and all
        # components rather than taking an alphabetical prefix
        step = max(1, len(files) // args.limit)
        files = files[::step][:args.limit]
    print(f"[recheck] {len(files)} structures, {args.workers} workers")

    rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(analyse_one, f) for f in files]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                rows.append(r)
            if n % 1000 == 0:
                print(f"[recheck]   {n}/{len(files)}  usable={len(rows)}", flush=True)

    result = {
        "n_files_scanned": len(files),
        "n_chains_usable": len(rows),
        "contact_def": {"cutoff_A": CUT, "seq_sep": SEQ_SEP,
                        "L_range": [MIN_L, MAX_L], "min_contacts": MIN_CONTACTS},
        "published_180_reference": {
            "b4_occupancy_by_bin": {"64-200": 0.1634, "200-500": 0.0687,
                                    "500-1500": 0.0317, "1500-3000": 0.0134},
            "b4_effective_c_by_bin": {"64-200": 7.9, "200-500": 12.6,
                                      "500-1500": 14.8, "1500-3000": 17.2},
            "max_effective_c_published": 19.03,
            "max_effective_c_canonical": 19.04,
            "target_c": 20,
        },
        "acgu": summarise(rows, "acgu"),
        "polymer": summarise(rows, "polymer"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(result, indent=1))

    for rule in ("acgu", "polymer"):
        s = result[rule]
        if not s:
            continue
        print(f"\n=== rule = {rule}   ({s['n_chains']} chains) ===")
        print(f"{'L bin':<12}{'n':>6}{'med L':>7}{'c/nt':>7}{'occ b=4':>10}"
              f"{'mean c':>9}{'max c':>8}{'p99 c':>8}{'>20':>6}")
        for e in s["by_length_bin"]:
            p = e["per_block"]["4"]
            print(f"{e['L_range']:<12}{e['n']:>6}{e['median_L']:>7.0f}"
                  f"{e['mean_contacts_per_nt']:>7.2f}{p['mean_occupancy_frac']:>10.4f}"
                  f"{p['mean_effective_c']:>9.2f}{p['max_effective_c']:>8.2f}"
                  f"{p['p99_effective_c']:>8.2f}{p['n_over_20']:>6}")
        g = s["global"]["4"]
        print(f"  GLOBAL b=4: max c = {g['max_effective_c']}, "
              f"p99 = {g['p99_effective_c']}, p99.9 = {g['p999_effective_c']}, "
              f"chains over c=20: {g['n_over_20']} ({g['frac_over_20']*100:.2f}%)")
    print(f"\n[recheck] -> {OUT / args.out}")


if __name__ == "__main__":
    main()
