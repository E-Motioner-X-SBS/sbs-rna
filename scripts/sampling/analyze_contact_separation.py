#!/usr/bin/env python3
"""Where do RNA contacts actually live along the sequence?

The proposal-recall test showed a sequence-only top-K proposer collapses on long
chains. The fix depends on whether contacts are concentrated at short sequence
separation. If they are, a GUARANTEED local band costs a fixed c*L and captures
most contacts, leaving only a small long-range remainder for a learned top-k.

Measures the cumulative distribution of |i-j| over true contacts, by chain length.
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
BANDS = [8, 16, 32, 64, 128, 256, 512]
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
    by_chain = defaultdict(list)
    for k in res:
        by_chain[k[0]].append(k)
    if not by_chain:
        return None
    ch = max(by_chain, key=lambda c: len(by_chain[c]))
    return [res[k] for k in sorted(by_chain[ch], key=lambda k: k[1])]


def contacts(atoms):
    L = len(atoms)
    cent = np.array([[sum(a[d] for a in at)/len(at) for d in range(3)] for at in atoms])
    dm = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=-1)
    cand = np.argwhere((dm <= 30.0) & (np.triu(np.ones((L, L), bool), SEQ_SEP)))
    seps = []
    for i, j in cand:
        hit = False
        for a in atoms[i]:
            for b in atoms[j]:
                if (a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2 <= CUT*CUT:
                    hit = True; break
            if hit: break
        if hit:
            seps.append(int(j - i))
    return seps


def main():
    rows = []
    files = sorted(SAMP.glob("*.cif.gz"))
    for i, f in enumerate(files, 1):
        try:
            atoms = load_chain(f)
        except Exception:
            continue
        if not atoms or not (32 <= len(atoms) <= MAX_L):
            continue
        seps = contacts(atoms)
        if len(seps) < 10:
            continue
        L = len(atoms); n = len(seps)
        rows.append({"pdb": f.name.split(".")[0], "L": L, "n_contacts": n,
                     "cum": {str(b): round(sum(1 for s in seps if s <= b)/n, 4) for b in BANDS},
                     "median_sep": int(st.median(seps)),
                     "max_sep": max(seps)})
        if i % 30 == 0:
            print(f"  {i}/{len(files)} ({len(rows)} usable)", flush=True)

    summary = {"chains": len(rows), "bands": BANDS,
               "note": "fraction of true contacts with |i-j| <= band",
               "by_length_bin": []}
    for lo, hi in [(32, 100), (100, 200), (200, 500), (500, 1500), (1500, 3000)]:
        sel = [r for r in rows if lo <= r["L"] < hi]
        if not sel:
            continue
        summary["by_length_bin"].append({
            "L_range": f"{lo}-{hi}", "n": len(sel),
            "median_L": st.median([r["L"] for r in sel]),
            "cum_frac": {str(b): round(st.mean(r["cum"][str(b)] for r in sel), 4) for b in BANDS},
            "median_sep": st.median([r["median_sep"] for r in sel]),
        })
    summary["overall_cum_frac"] = {str(b): round(st.mean(r["cum"][str(b)] for r in rows), 4) for b in BANDS}
    (OUT / "contact_separation.json").write_text(json.dumps({"summary": summary, "chains": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
