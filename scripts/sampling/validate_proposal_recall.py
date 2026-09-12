#!/usr/bin/env python3
"""De-risk R1: does a cheap pair-proposal stage actually retain true 3D contacts?

The PHAROS sparse pair track keeps only K = c*L candidate edges. If the proposal
stage fails to rank a true contact into the top K, nothing downstream can
recover it -- an invisible accuracy ceiling. This measures that recall directly
on real structures, using a scorer that uses ONLY information available before
any 3D prediction exists (sequence + base-pairing prior + stacking context).

Scorer (deliberately weak, so this is a LOWER bound on a trained proposer):
    score(i,j) = w_c * complementarity(i,j)
               + w_s * stack_context(i,j)
               + w_d * exp(-|i-j| / tau)
complementarity uses the ERNIE-RNA prior values (CG 3, AU 2, GU 0.8).

Ground truth: heavy-atom contact <= 8.0 A, |i-j| >= 4 (same definition as
analyze_contact_sparsity.py).
"""
from __future__ import annotations
import gzip, json, math, statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAMP = ROOT / "data" / "samples" / "structures"
OUT = ROOT / "data" / "samples" / "analysis"
RNA = {"A", "C", "G", "U"}
CUT, SEQ_SEP = 8.0, 4
PAIR_SCORE = {("G", "C"): 3.0, ("C", "G"): 3.0,
              ("A", "U"): 2.0, ("U", "A"): 2.0,
              ("G", "U"): 0.8, ("U", "G"): 0.8}
W_C, W_S, W_D, TAU = 1.0, 0.6, 2.0, 40.0
MAX_L = 1200          # keep the dense O(L^2) reference scoring tractable


def load_chain(path: Path):
    """Return (sequence, per-residue heavy-atom coords) for the longest chain."""
    op = gzip.open if path.suffix == ".gz" else open
    cols, in_loop, header = [], False, False
    res = defaultdict(list)
    comp = {}
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
                c = r["label_comp_id"].strip('"')
                if c not in RNA or r.get("type_symbol") == "H":
                    continue
                try:
                    xyz = (float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"]))
                    sq = int(r.get("label_seq_id", "0"))
                except (ValueError, KeyError):
                    continue
                key = (r.get("label_asym_id", "?"), sq)
                res[key].append(xyz); comp[key] = c
    by_chain = defaultdict(list)
    for k in res:
        by_chain[k[0]].append(k)
    if not by_chain:
        return None, None
    ch = max(by_chain, key=lambda c: len(by_chain[c]))
    keys = sorted(by_chain[ch], key=lambda k: k[1])
    return "".join(comp[k] for k in keys), [res[k] for k in keys]


def true_contacts(atoms) -> set[tuple[int, int]]:
    L = len(atoms)
    cent = np.array([[sum(a[d] for a in at) / len(at) for d in range(3)] for at in atoms])
    d = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=-1)
    cand = np.argwhere((d <= 30.0) & (np.triu(np.ones((L, L), bool), SEQ_SEP)))
    out = set()
    for i, j in cand:
        hit = False
        for a in atoms[i]:
            for b in atoms[j]:
                if (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2 <= CUT*CUT:
                    hit = True; break
            if hit: break
        if hit:
            out.add((int(i), int(j)))
    return out


def proposal_scores(seq: str) -> np.ndarray:
    L = len(seq)
    comp = np.zeros((L, L), np.float32)
    for i in range(L):
        for j in range(i + SEQ_SEP, L):
            comp[i, j] = PAIR_SCORE.get((seq[i], seq[j]), 0.0)
    # stacking context: neighbours pairing too
    stack = np.zeros_like(comp)
    stack[1:-1, 1:-1] = comp[:-2, 2:] + comp[2:, :-2]
    idx = np.arange(L)
    sep = np.abs(idx[:, None] - idx[None, :]).astype(np.float32)
    dist = np.exp(-sep / TAU)
    sc = W_C * comp + W_S * stack + W_D * dist
    sc[np.tril_indices(L, SEQ_SEP - 1)] = -np.inf
    return sc


def recall_at(scores: np.ndarray, truth: set, L: int, cs) -> dict:
    flat = scores.ravel()
    order = np.argsort(-flat)
    out = {}
    for c in cs:
        K = min(c * L, int((flat > -np.inf).sum()))
        top = order[:K]
        sel = {(int(t // L), int(t % L)) for t in top}
        out[c] = len(sel & truth) / max(len(truth), 1)
    return out


def main():
    cs = [4, 8, 16, 32, 64]
    rows = []
    files = sorted(SAMP.glob("*.cif.gz"))
    for i, f in enumerate(files, 1):
        try:
            seq, atoms = load_chain(f)
        except Exception:
            continue
        if not seq or not (32 <= len(seq) <= MAX_L):
            continue
        L = len(seq)
        truth = true_contacts(atoms)
        if len(truth) < 10:
            continue
        sc = proposal_scores(seq)
        r = recall_at(sc, truth, L, cs)
        # random baseline at c=32
        rng = np.random.default_rng(0)
        rnd = rng.random((L, L)).astype(np.float32)
        rnd[np.tril_indices(L, SEQ_SEP - 1)] = -np.inf
        rb = recall_at(rnd, truth, L, [32])[32]
        rows.append({"pdb": f.name.split(".")[0], "L": L, "n_true": len(truth),
                     "contacts_per_nt": round(len(truth) / L, 3),
                     "recall": {str(k): round(v, 4) for k, v in r.items()},
                     "random_c32": round(rb, 4)})
        if i % 25 == 0:
            print(f"  scanned {i}/{len(files)} ({len(rows)} usable)", flush=True)

    if not rows:
        print("no usable chains"); return
    summary = {
        "chains": len(rows),
        "length_range": [min(r["L"] for r in rows), max(r["L"] for r in rows)],
        "scorer": "ERNIE-RNA pair prior + stacking context + exp seq-distance decay",
        "note": "sequence-only scorer => LOWER bound on a trained proposal stage",
        "mean_recall_by_c": {str(c): round(st.mean(r["recall"][str(c)] for r in rows), 4) for c in cs},
        "median_recall_by_c": {str(c): round(st.median([r["recall"][str(c)] for r in rows]), 4) for c in cs},
        "p10_recall_by_c": {str(c): round(sorted(r["recall"][str(c)] for r in rows)[max(0, len(rows)//10)], 4) for c in cs},
        "worst_recall_c32": round(min(r["recall"]["32"] for r in rows), 4),
        "random_baseline_c32_mean": round(st.mean(r["random_c32"] for r in rows), 4),
        "by_length_bin": [],
    }
    for lo, hi in [(32, 100), (100, 200), (200, 500), (500, 1200)]:
        sel = [r for r in rows if lo <= r["L"] < hi]
        if sel:
            summary["by_length_bin"].append({
                "L_range": f"{lo}-{hi}", "n": len(sel),
                "mean_recall_c32": round(st.mean(r["recall"]["32"] for r in sel), 4),
                "min_recall_c32": round(min(r["recall"]["32"] for r in sel), 4),
                "mean_frac_dense_c32": round(st.mean(min(32*r["L"], r["L"]**2)/(r["L"]**2) for r in sel), 4)})
    (OUT / "proposal_recall.json").write_text(json.dumps({"summary": summary, "chains": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
