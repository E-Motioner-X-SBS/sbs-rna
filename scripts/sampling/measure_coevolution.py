#!/usr/bin/env python3
"""Measure how much structural signal RNA coevolution actually carries.

The architecture routes a coevolution expert. This tests, on the 12 downloaded
Rfam seed alignments, how well APC-corrected mutual information recovers the
curated consensus secondary structure (#=GC SS_cons) -- and how that depends on
alignment depth, which the literature flags as RNA's key weakness.

Metric: precision of the top-L MI-ranked column pairs against SS_cons base
pairs, where L is the number of consensus columns. Also reports Neff (effective
sequence count at 80% identity) so the depth dependence is explicit.
"""
from __future__ import annotations
import json, math, re, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ALN = ROOT / "data" / "samples" / "benchmarks"
OUT = ROOT / "data" / "samples" / "analysis"
ALPH = "ACGU-"
OPEN_CH, CLOSE_CH = "(<[{Aa", ")>]}Aa"   # Stockholm WUSS brackets


def read_stockholm(path: Path):
    seqs, ss = defaultdict(str), ""
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("#=GC SS_cons"):
            ss += line.split(None, 2)[2] if len(line.split(None, 2)) > 2 else ""
        elif line.startswith("#") or line.startswith("//") or not line.strip():
            continue
        else:
            parts = line.split(None, 1)
            if len(parts) == 2:
                seqs[parts[0]] += parts[1].strip()
    return seqs, ss


def ss_pairs(ss: str):
    """Parse WUSS consensus structure into (i,j) column pairs."""
    stacks = defaultdict(list)
    pairs = []
    for i, c in enumerate(ss):
        if c in "(<[{":
            stacks[c].append(i)
        elif c in ")>]}":
            m = {")": "(", ">": "<", "]": "[", "}": "{"}[c]
            if stacks[m]:
                pairs.append((stacks[m].pop(), i))
    return pairs


def encode(seqs):
    keys = list(seqs)
    M = np.full((len(keys), len(seqs[keys[0]])), 4, np.int8)
    idx = {c: i for i, c in enumerate("ACGU")}
    for r, k in enumerate(keys):
        for c, ch in enumerate(seqs[k].upper().replace("T", "U")):
            M[r, c] = idx.get(ch, 4)
    return M


def neff(M, thresh=0.8):
    """Effective sequence count: 1/cluster size at `thresh` identity (subsampled)."""
    n = M.shape[0]
    sub = M if n <= 400 else M[np.random.default_rng(0).choice(n, 400, replace=False)]
    ident = (sub[:, None, :] == sub[None, :, :]).mean(-1)
    w = 1.0 / np.maximum((ident >= thresh).sum(1), 1)
    return float(w.sum() * (n / sub.shape[0]))


def seq_weights(M, thresh=0.8):
    """Standard DCA reweighting: w_n = 1 / (number of sequences within `thresh` identity)."""
    N = M.shape[0]
    w = np.ones(N, np.float32)
    BLK = 512
    for a in range(0, N, BLK):
        blk = M[a:a+BLK]
        cnt = np.zeros(blk.shape[0], np.int32)
        for b in range(0, N, BLK):
            ident = (blk[:, None, :] == M[b:b+BLK][None, :, :]).mean(-1)
            cnt += (ident >= thresh).sum(1)
        w[a:a+BLK] = 1.0 / np.maximum(cnt, 1)
    return w


def apc_mi(M, keep, weights=None):
    """APC-corrected, sequence-reweighted, pseudocounted mutual information."""
    q = 5
    sub = M[:, keep]
    N, C = sub.shape
    w = np.ones(N, np.float32) if weights is None else weights.astype(np.float32)
    Meff = float(w.sum())
    lam = 0.5                      # pseudocount weight (standard DCA choice)
    onehot = np.stack([(sub == a) for a in range(q)], -1).astype(np.float32)  # N,C,q

    f1 = np.einsum("n,nca->ca", w, onehot) / Meff
    f1 = (1 - lam) * f1 + lam / q

    mi = np.zeros((C, C), np.float32)
    wo = onehot * w[:, None, None]
    for i in range(C):
        fij = np.einsum("na,ncb->cab", onehot[:, i, :], wo) / Meff     # C,q,q
        fij = (1 - lam) * fij + lam / (q * q)
        outer = f1[i][None, :, None] * f1[:, None, :]                  # C,q,q  (fixed)
        mi[i] = np.sum(fij * np.log(fij / outer), axis=(1, 2))
    np.fill_diagonal(mi, 0.0)
    m = mi.mean(1)
    apc = np.outer(m, m) / max(mi.mean(), 1e-12)
    return mi - apc


def main():
    rows = []
    for f in sorted(ALN.glob("rfam_*.stk")):
        seqs, ss = read_stockholm(f)
        if not seqs or not ss:
            continue
        M = encode(seqs)
        w = seq_weights(M)
        if M.shape[0] < 20:
            continue
        # keep columns that are consensus (SS_cons not '.' gap-ish) and <50% gaps
        gapfrac = (M == 4).mean(0)
        keep = [c for c in range(min(M.shape[1], len(ss))) if gapfrac[c] < 0.5]
        if len(keep) < 20:
            continue
        pos = {c: i for i, c in enumerate(keep)}
        truth = {(pos[a], pos[b]) for a, b in ss_pairs(ss)
                 if a in pos and b in pos and abs(pos[a] - pos[b]) >= 4}
        if len(truth) < 5:
            continue
        sc = apc_mi(M, keep, w)
        L = len(keep)
        iu = np.triu_indices(L, 4)
        vals = sc[iu]
        order = np.argsort(-vals)
        ranked = [(int(iu[0][k]), int(iu[1][k])) for k in order]
        res = {"family": f.stem, "n_seqs": int(M.shape[0]), "neff": round(float(w.sum()), 1),
               "n_cols": L, "n_true_pairs": len(truth)}
        for name, k in [("topL", L), ("topL_2", L // 2), ("topL_5", L // 5)]:
            sel = set(ranked[:k])
            res[f"prec_{name}"] = round(len(sel & truth) / max(k, 1), 4)
            res[f"rec_{name}"] = round(len(sel & truth) / max(len(truth), 1), 4)
        res["neff_per_col"] = round(res["neff"] / L, 3)
        rows.append(res)
        print(f"  {f.stem:38s} N={res['n_seqs']:5d} Neff={res['neff']:7.1f} "
              f"L={L:4d} prec@L/5={res['prec_topL_5']:.3f} rec@L={res['rec_topL']:.3f}", flush=True)

    if not rows:
        print("no families usable"); return
    summary = {
        "families": len(rows),
        "metric": "APC-corrected MI vs Rfam SS_cons base pairs, |i-j|>=4",
        "mean_prec_topL_5": round(st.mean(r["prec_topL_5"] for r in rows), 4),
        "mean_prec_topL": round(st.mean(r["prec_topL"] for r in rows), 4),
        "mean_rec_topL": round(st.mean(r["rec_topL"] for r in rows), 4),
        "best": max(rows, key=lambda r: r["prec_topL_5"])["family"],
        "worst": min(rows, key=lambda r: r["prec_topL_5"])["family"],
    }
    # depth dependence
    lo = [r for r in rows if r["neff_per_col"] < 1.0]
    hi = [r for r in rows if r["neff_per_col"] >= 1.0]
    summary["depth_dependence"] = {
        "shallow_neff_per_col_lt_1": {"n": len(lo),
            "mean_prec_topL_5": round(st.mean(r["prec_topL_5"] for r in lo), 4) if lo else None},
        "deep_neff_per_col_ge_1": {"n": len(hi),
            "mean_prec_topL_5": round(st.mean(r["prec_topL_5"] for r in hi), 4) if hi else None},
    }
    (OUT / "coevolution_signal.json").write_text(json.dumps({"summary": summary, "families": rows}, indent=2))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
