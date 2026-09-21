#!/usr/bin/env python3
"""Re-derive §7.1's flat-top-K result at corpus scale — open action 6.

§7.1 is the foundational justification for selecting **blocks** rather than
ranking pairs, and it rests on 180 structures. Re-reading its source
(`proposal_recall.json`) shows it also misreads them.

    "A flat top-K proposer keeping K = 32L recovers 0.200 of true contacts on
     500-1200 nt chains, where a random scorer gets 0.746 across the same set."

The two numbers are from different populations. **0.2003** is `mean_recall_c32`
for the 500-1200 nt bin -- **five chains**. **0.7458** is
`random_baseline_c32_mean` over all **43** chains, 29 of which are 32-100 nt,
where K = 32L already covers 59% of every valid pair and so a random scorer
recovers most of them by construction. "Across the same set" is not true, and as
written the sentence says a learned scorer does worse than chance, which is
backwards.

This measures all of it on the same chains, at the same budgets:

    random         uniform over the valid pairs
    separation     exp decay in |i-j|, no sequence
    complementary  Watson-Crick/wobble potential x separation decay
    block oracle   keep every occupied b=4 block -- the ceiling the
                   Hierarchical Pair Track is aiming at

Whether §7.1's conclusion survives is the point of running it. The conclusion
is that flat pair ranking cannot reach usable recall inside a linear budget on
long chains; the evidence for it should be a matched comparison over thousands
of chains rather than an unmatched one over five.

Usage:
    /store/shuvam/.venv/bin/python scripts/sampling/recheck_proposal_recall.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/derived/pharos3d"
OUT = ROOT / "data/samples/analysis"
sys.path.insert(0, str(ROOT / "src"))

from pharos.data.vocab import SYMBOLS                             # noqa: E402

MIN_SEP = 4
BUDGETS = (4, 8, 16, 32, 64)
BINS = ((32, 100), (100, 200), (200, 500), (500, 1200), (1200, 3000))
#: Watson-Crick and wobble pairing potential, on the parent-base tokens
_PAIR = {("A", "U"): 2.0, ("U", "A"): 2.0, ("G", "C"): 3.0, ("C", "G"): 3.0,
         ("G", "U"): 1.0, ("U", "G"): 1.0}


def scores_for(tokens: np.ndarray, L: int, kind: str, rng) -> np.ndarray:
    """An (n_valid,) score vector over the valid upper-triangle pairs."""
    iu = np.triu_indices(L, MIN_SEP)
    if kind == "random":
        return rng.random(len(iu[0])).astype(np.float32)
    sep = (iu[1] - iu[0]).astype(np.float32)
    decay = np.exp(-sep / 200.0)
    if kind == "separation":
        return decay
    sym = np.array([SYMBOLS[t] if 0 <= t < len(SYMBOLS) else "N" for t in tokens])
    a, b = sym[iu[0]], sym[iu[1]]
    comp = np.array([_PAIR.get((x, y), 0.0) for x, y in zip(a, b)], dtype=np.float32)
    return comp * decay + 0.05 * decay


def recall_at(scores: np.ndarray, iu, truth: set, L: int, c: int) -> float:
    k = min(int(c * L), len(scores))
    if k <= 0 or not truth:
        return float("nan")
    top = np.argpartition(-scores, k - 1)[:k]
    got = sum(1 for t in top if (int(iu[0][t]), int(iu[1][t])) in truth)
    return got / len(truth)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-length", type=int, default=3000)
    ap.add_argument("--max-chains", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from pharos.data.loader import Pharos3DDataset
    ds = Pharos3DDataset(DATA, max_length=args.max_length)
    idx = list(range(len(ds)))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    idx = idx[:args.max_chains]
    print(f"[rec] {len(idx):,} chains of {len(ds):,} (<= {args.max_length} nt)",
          flush=True)

    rows: List[Dict] = []
    for n, i in enumerate(idx, 1):
        x = ds[i]
        L = int(x["length"])
        con = x["contacts"]
        if L < 32 or len(con) < 10:
            continue
        truth = {(int(a), int(b)) for a, b in con}
        iu = np.triu_indices(L, MIN_SEP)
        if len(iu[0]) == 0:
            continue
        r: Dict = {"pdb": x["meta"]["pdb"], "chain": x["meta"]["chain"], "L": L,
                   "n_true": len(truth),
                   "frac_dense_c32": min(32 * L, len(iu[0])) / len(iu[0])}
        for kind in ("random", "separation", "complementary"):
            sc = scores_for(x["tokens"], L, kind, np.random.default_rng(args.seed + i))
            r[kind] = {str(c): round(recall_at(sc, iu, truth, L, c), 4)
                       for c in BUDGETS}
        # the ceiling: keep every occupied b=4 block
        b = 4
        occ = {(int(a) // b, int(bb) // b) for a, bb in con}
        r["block_oracle_recall"] = 1.0        # by construction
        r["block_oracle_c"] = round(len(occ) * b * b / L, 2)
        rows.append(r)
        if n % 500 == 0:
            print(f"[rec]   {n}/{len(idx)} usable={len(rows)}", flush=True)

    by_bin: Dict = {}
    for lo, hi in BINS:
        sel = [r for r in rows if lo <= r["L"] < hi]
        if not sel:
            continue
        by_bin[f"{lo}-{hi}"] = {
            "n": len(sel),
            "frac_dense_c32": round(float(np.mean([r["frac_dense_c32"] for r in sel])), 4),
            **{kind: {c: round(float(np.mean([r[kind][c] for r in sel])), 4)
                      for c in map(str, BUDGETS)}
               for kind in ("random", "separation", "complementary")},
            "block_oracle_c_mean": round(
                float(np.mean([r["block_oracle_c"] for r in sel])), 2),
        }

    res = {
        "n_chains": len(rows),
        "min_separation": MIN_SEP,
        "budgets_c": list(BUDGETS),
        "v0_1_claim": {
            "text": ("flat top-K at K=32L recovers 0.200 on 500-1200 nt where a "
                     "random scorer gets 0.746 across the same set"),
            "defect": ("the two figures are different populations: 0.2003 is the "
                       "500-1200 bin (n=5) and 0.7458 is the random baseline over "
                       "all 43 chains, 29 of them 32-100 nt where K=32L already "
                       "covers 59% of pairs"),
        },
        "by_length_bin": by_bin,
        "overall": {kind: {c: round(float(np.mean([r[kind][c] for r in rows])), 4)
                           for c in map(str, BUDGETS)}
                    for kind in ("random", "separation", "complementary")},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "proposal_recall_fullcorpus.json").write_text(json.dumps(res, indent=1))

    print(f"\nrecall at K = c.L, mean over {len(rows):,} chains\n")
    print(f"{'bin':12s} {'n':>6s} {'c':>4s} {'random':>8s} {'separation':>11s} "
          f"{'complement':>11s}   frac of dense")
    for name, b in by_bin.items():
        for c in ("32",):
            print(f"{name:12s} {b['n']:6d} {c:>4s} {b['random'][c]:8.4f} "
                  f"{b['separation'][c]:11.4f} {b['complementary'][c]:11.4f}   "
                  f"{b['frac_dense_c32']:.4f}")
    print(f"\n[rec] -> {OUT / 'proposal_recall_fullcorpus.json'}")


if __name__ == "__main__":
    main()
