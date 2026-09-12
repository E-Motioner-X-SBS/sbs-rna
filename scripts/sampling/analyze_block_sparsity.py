#!/usr/bin/env python3
"""Test the hierarchical (coarse-to-fine) alternative to flat top-K proposal.

Flat sequence-only top-K collapsed on long chains, and contacts are not local
enough for a banded fix. The remaining option is COARSE-TO-FINE: build a dense
pair map at block resolution b (cheap, since it is (L/b)^2), select the blocks
that contain contacts, and refine only inside those.

This works only if contacts are CLUSTERED -- helical stems are contiguous
anti-diagonal runs, so they should occupy few blocks. Measured here:
  - occupancy: fraction of blocks containing >=1 contact
  - cost of a dense coarse map at (L/b)^2
  - recall ceiling of keeping the top-N occupied blocks
"""
from __future__ import annotations
import gzip, json, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAMP = ROOT / "data" / "samples" / "structures"
OUT = ROOT / "data" / "samples" / "analysis"
RNA = {"A", "C", "G", "U"}
CUT, SEQ_SEP = 8.0, 4
BLOCKS = [4, 8, 16, 32]
MAX_L = 3000


def load_chain(path: Path):
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
                if r["label_comp_id"].strip('"') not in RNA or r.get("type_symbol") == "H":
                    continue
                try:
                    xyz = (float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"]))
                    sq = int(r.get("label_seq_id", "0"))
                except (ValueError, KeyError):
                    continue
                res[(r.get("label_asym_id", "?"), sq)].append(xyz)
    by = defaultdict(list)
    for k in res:
        by[k[0]].append(k)
    if not by:
        return None
    ch = max(by, key=lambda c: len(by[c]))
    return [res[k] for k in sorted(by[ch], key=lambda k: k[1])]


def contact_pairs(atoms):
    L = len(atoms)
    cent = np.array([[sum(a[d] for a in at)/len(at) for d in range(3)] for at in atoms])
    dm = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=-1)
    cand = np.argwhere((dm <= 30.0) & (np.triu(np.ones((L, L), bool), SEQ_SEP)))
    out = []
    for i, j in cand:
        hit = False
        for a in atoms[i]:
            for b in atoms[j]:
                if (a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2 <= CUT*CUT:
                    hit = True; break
            if hit: break
        if hit:
            out.append((int(i), int(j)))
    return out


def main():
    rows = []
    files = sorted(SAMP.glob("*.cif.gz"))
    for i, f in enumerate(files, 1):
        try:
            atoms = load_chain(f)
        except Exception:
            continue
        if not atoms or not (64 <= len(atoms) <= MAX_L):
            continue
        pairs = contact_pairs(atoms)
        if len(pairs) < 20:
            continue
        L = len(atoms)
        rec = {"pdb": f.name.split(".")[0], "L": L, "n_contacts": len(pairs), "blocks": {}}
        for b in BLOCKS:
            nb = (L + b - 1) // b
            occ = {(p[0] // b, p[1] // b) for p in pairs}
            total_blocks = nb * (nb + 1) // 2
            # contacts captured if we keep every occupied block, and the fine cost of doing so
            rec["blocks"][str(b)] = {
                "n_block_rows": nb,
                "occupied": len(occ),
                "occupancy_frac": round(len(occ) / max(total_blocks, 1), 5),
                "coarse_map_entries": total_blocks,
                "coarse_vs_dense": round(total_blocks / (L*(L-1)/2), 5),
                "fine_pairs_if_kept": len(occ) * b * b,
                "fine_vs_dense": round(len(occ)*b*b / (L*(L-1)/2), 5),
                "effective_c": round(len(occ)*b*b / L, 2),
            }
        rows.append(rec)
        if i % 30 == 0:
            print(f"  {i}/{len(files)} ({len(rows)} usable)", flush=True)

    summary = {"chains": len(rows), "block_sizes": BLOCKS, "by_length_bin": []}
    for lo, hi in [(64, 200), (200, 500), (500, 1500), (1500, 3000)]:
        sel = [r for r in rows if lo <= r["L"] < hi]
        if not sel:
            continue
        entry = {"L_range": f"{lo}-{hi}", "n": len(sel),
                 "median_L": st.median([r["L"] for r in sel]), "per_block": {}}
        for b in BLOCKS:
            k = str(b)
            entry["per_block"][k] = {
                "mean_occupancy_frac": round(st.mean(r["blocks"][k]["occupancy_frac"] for r in sel), 5),
                "mean_coarse_vs_dense": round(st.mean(r["blocks"][k]["coarse_vs_dense"] for r in sel), 5),
                "mean_fine_vs_dense": round(st.mean(r["blocks"][k]["fine_vs_dense"] for r in sel), 5),
                "mean_effective_c": round(st.mean(r["blocks"][k]["effective_c"] for r in sel), 2),
                "max_effective_c": round(max(r["blocks"][k]["effective_c"] for r in sel), 2),
            }
        summary["by_length_bin"].append(entry)
    (OUT / "block_sparsity.json").write_text(json.dumps({"summary": summary, "chains": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
