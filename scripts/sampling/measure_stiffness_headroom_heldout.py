#!/usr/bin/env python3
"""Defect #12 fix: score the nested stiffness models fairly.

`measure_stiffness_headroom.py` fits each Gaussian on the same steps it scores
(in-sample) and evaluates each model on its OWN covered subset -- M0 on 103,964
steps, M1 on 90,098, M2 on 78,076. Both biases favour the finer partition:

  * finer grouping always lowers IN-SAMPLE NLL, regardless of real signal;
  * M2's covered subset is the well-populated (more regular) steps, which are
    intrinsically easier to fit.

This rescores with two corrections:
  A. COMMON SUBSET  - every model scored on exactly the steps all models cover.
  B. HELD-OUT       - fit on a 50% split, score on the other half, both ways.

Reuses the original's feature extraction so only the scoring changes.
"""
from __future__ import annotations
import importlib.util, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "samples" / "analysis"
SRC = ROOT / "scripts" / "sampling" / "measure_stiffness_headroom.py"
MIN_N, RIDGE = 200, 1e-3


def fit(A):
    mu = A.mean(0)
    C = np.cov(A.T) + RIDGE * np.eye(A.shape[1])
    return mu, C


def nll_of(A, mu, C):
    """Mean Gaussian NLL per sample of A under N(mu, C)."""
    d = A.shape[1]
    sign, logdet = np.linalg.slogdet(C)
    if sign <= 0:
        return float("nan")
    D = A - mu
    m = np.einsum("ij,jk,ik->i", D, np.linalg.inv(C), D)
    return float(0.5 * (m + logdet + d * math.log(2 * math.pi)).mean())


def load_records():
    spec = importlib.util.spec_from_file_location("hm", SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["hm"] = m
    spec.loader.exec_module(m)
    # re-run only the extraction half of the original main()
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        recs = m.collect() if hasattr(m, "collect") else None
    if recs is None:
        raise SystemExit("original script exposes no collect(); extraction not reusable")
    return recs


def keys(r):
    return {"M0": "ALL",
            "M1": r["ctx"],
            "Mstruct": (r["canon"], r["helical"]),
            "M2": (r["ctx"], r["canon"], r["helical"])}


def main():
    recs = load_records()
    print(f"records: {len(recs):,}")
    models = ["M0", "M1", "Mstruct", "M2"]

    # which steps does each model cover (group has >= MIN_N members)?
    counts = {mm: defaultdict(int) for mm in models}
    for r in recs:
        k = keys(r)
        for mm in models:
            counts[mm][k[mm]] += 1
    covered = {mm: set() for mm in models}
    for i, r in enumerate(recs):
        k = keys(r)
        for mm in models:
            if counts[mm][k[mm]] >= MIN_N:
                covered[mm].add(i)
    for mm in models:
        print(f"  {mm:8s} covers {len(covered[mm]):>7,} steps "
              f"({100*len(covered[mm])/len(recs):5.1f}%)")
    common = set.intersection(*(covered[mm] for mm in models))
    print(f"  COMMON   {len(common):>7,} steps ({100*len(common)/len(recs):5.1f}%)\n")

    rng = np.random.default_rng(0)
    idx = np.array(sorted(common))
    perm = rng.permutation(len(idx))
    half = len(idx) // 2
    folds = [idx[perm[:half]], idx[perm[half:]]]

    res = {"n_records": len(recs),
           "coverage": {mm: len(covered[mm]) for mm in models},
           "common_subset": len(common)}

    # --- A. in-sample on the COMMON subset ---
    insample = {}
    for mm in models:
        groups = defaultdict(list)
        for i in idx:
            groups[keys(recs[i])[mm]].append(recs[i]["vec"])
        tot, n = 0.0, 0
        for g, v in groups.items():
            A = np.array(v, float)
            if len(A) < 8:
                continue
            mu, C = fit(A)
            x = nll_of(A, mu, C)
            if not math.isnan(x):
                tot += x * len(A); n += len(A)
        insample[mm] = round(tot / max(n, 1), 4)
    res["A_in_sample_common_subset"] = insample

    # --- B. held-out, 2-fold, averaged ---
    heldout = {}
    for mm in models:
        tot, n = 0.0, 0
        for tr, te in [(folds[0], folds[1]), (folds[1], folds[0])]:
            gtr = defaultdict(list)
            for i in tr:
                gtr[keys(recs[i])[mm]].append(recs[i]["vec"])
            params = {}
            for g, v in gtr.items():
                A = np.array(v, float)
                if len(A) >= 8:
                    params[g] = fit(A)
            gte = defaultdict(list)
            for i in te:
                gte[keys(recs[i])[mm]].append(recs[i]["vec"])
            for g, v in gte.items():
                if g not in params:
                    continue
                A = np.array(v, float)
                mu, C = params[g]
                x = nll_of(A, mu, C)
                if not math.isnan(x):
                    tot += x * len(A); n += len(A)
        heldout[mm] = round(tot / max(n, 1), 4)
    res["B_held_out_2fold"] = heldout

    def gains(d):
        return {"sequence_over_global": round(d["M0"] - d["M1"], 4),
                "structure_over_sequence": round(d["M1"] - d["M2"], 4),
                "structure_only_over_global": round(d["M0"] - d["Mstruct"], 4)}
    res["gains_in_sample"] = gains(insample)
    res["gains_held_out"] = gains(heldout)
    res["published_gains"] = {"sequence_over_global": 2.1443,
                              "structure_over_sequence": 3.0282,
                              "structure_only_over_global": 1.8765}
    (OUT / "stiffness_headroom_heldout.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in
                      ("A_in_sample_common_subset", "B_held_out_2fold",
                       "gains_in_sample", "gains_held_out", "published_gains")}, indent=2))


if __name__ == "__main__":
    main()
