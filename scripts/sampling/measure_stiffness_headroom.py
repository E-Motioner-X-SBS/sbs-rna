#!/usr/bin/env python3
"""Is a learned stiffness ENCODER justified, or would a lookup table do?

A 76-entry dinucleotide-context table is the obvious way to supply step
stiffness. But the GNRA result already showed sequence context alone is a weak
predictor of rigidity, so the table may be leaving signal on the floor.

This quantifies the headroom by fitting Gaussians over the 6 step-deformation
coordinates and comparing average negative log-likelihood per step:

  M0  one GLOBAL Gaussian                     (no context at all)
  M1  per-SEQUENCE-context Gaussians          (what a lookup table achieves)
  M2  per-(sequence x STRUCTURAL) Gaussians   (what an encoder could reach)

The structural split uses signals an encoder would plausibly have access to:
whether the step sits inside a long uninterrupted helical run, and the local
base-pair class (canonical vs non-canonical, from the Saenger annotation).

If M2 - M1 is a meaningful further drop in NLL, the extra information is real
and a learned encoder conditioned on structural context beats the table.
"""
from __future__ import annotations
import gzip, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_basepair_geometry import loops, fnum, STEP_COORDS

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "data" / "samples" / "structures"
OUT = ROOT / "data" / "samples" / "analysis"
MIN_N = 200
RIDGE = 1e-3          # regularise covariance so tiny groups stay invertible


def gauss_nll(A: np.ndarray, mu=None, C=None) -> tuple[float, int]:
    """Mean NLL per sample of a multivariate Gaussian fitted to A."""
    if mu is None:
        mu = A.mean(0)
    if C is None:
        C = np.cov(A.T) + RIDGE * np.eye(A.shape[1])
    sign, logdet = np.linalg.slogdet(C)
    if sign <= 0:
        return float("nan"), len(A)
    P = np.linalg.inv(C)
    d = A - mu
    maha = np.einsum("ni,ij,nj->n", d, P, d)
    k = A.shape[1]
    return float(0.5 * (maha.mean() + logdet + k * math.log(2 * math.pi))), len(A)


def collect():
    WANT = {"_ndb_struct_na_base_pair_step", "_ndb_struct_na_base_pair"}
    recs = []
    files = sorted(S.glob("*.cif.gz"))
    for k, f in enumerate(files, 1):
        pair_class = {}
        steps = []
        for cat, cols, rows in loops(f, WANT):
            if cat == "_ndb_struct_na_base_pair":
                for r in rows:
                    key = (r.get("i_label_asym_id"), r.get("i_label_seq_id"))
                    pair_class[key] = r.get("hbond_type_28", "?")
            elif cat == "_ndb_struct_na_base_pair_step":
                for r in rows:
                    vec = [fnum(r.get(c)) for c in STEP_COORDS]
                    if any(v is None for v in vec):
                        continue
                    steps.append((r, vec))
        if not steps:
            continue
        # helical-run length: consecutive steps on the same chain with consecutive seq ids
        idx = []
        for r, vec in steps:
            try:
                s1 = int(r.get("i_label_seq_id_1", "0"))
            except ValueError:
                s1 = 0
            idx.append((r.get("i_label_asym_id_1", "?"), s1))
        runlen = [1] * len(steps)
        for i in range(1, len(steps)):
            if idx[i][0] == idx[i-1][0] and idx[i][1] == idx[i-1][1] + 1:
                runlen[i] = runlen[i-1] + 1
        for i in range(len(steps) - 2, -1, -1):
            if idx[i+1][0] == idx[i][0] and idx[i+1][1] == idx[i][1] + 1:
                runlen[i] = max(runlen[i], runlen[i+1])
        for i, (r, vec) in enumerate(steps):
            ctx = (f"{r.get('i_label_comp_id_1','?')}{r.get('i_label_comp_id_2','?')}"
                   f"/{r.get('j_label_comp_id_2','?')}{r.get('j_label_comp_id_1','?')}")
            cls = pair_class.get((r.get("i_label_asym_id_1"), r.get("i_label_seq_id_1")), "?")
            canonical = cls in {"19", "20", "28"}      # WC G-C, WC A-U, wobble G-U
            helical = runlen[i] >= 4
            recs.append({"ctx": ctx, "vec": vec, "canon": canonical, "helical": helical})
        if k % 40 == 0:
            print(f"  {k}/{len(files)}  ({len(recs):,} steps)", flush=True)

    return recs


def main():
    recs = collect()
    A = np.array([r["vec"] for r in recs], float)
    print(f"\ntotal steps: {len(recs):,}")

    # M0 global
    nll0, _ = gauss_nll(A)

    # M1 per sequence context
    by_ctx = defaultdict(list)
    for r in recs:
        by_ctx[r["ctx"]].append(r["vec"])
    tot, n_used, groups1 = 0.0, 0, 0
    for c, v in by_ctx.items():
        if len(v) < MIN_N:
            continue
        nll, n = gauss_nll(np.array(v, float))
        if not math.isnan(nll):
            tot += nll * n; n_used += n; groups1 += 1
    nll1 = tot / max(n_used, 1)

    # M2 per (sequence x structural) context
    by_cs = defaultdict(list)
    for r in recs:
        by_cs[(r["ctx"], r["canon"], r["helical"])].append(r["vec"])
    tot2, n2, groups2 = 0.0, 0, 0
    for c, v in by_cs.items():
        if len(v) < MIN_N:
            continue
        nll, n = gauss_nll(np.array(v, float))
        if not math.isnan(nll):
            tot2 += nll * n; n2 += n; groups2 += 1
    nll2 = tot2 / max(n2, 1)

    # what does structure alone buy, ignoring sequence?
    by_s = defaultdict(list)
    for r in recs:
        by_s[(r["canon"], r["helical"])].append(r["vec"])
    tots, ns = 0.0, 0
    for c, v in by_s.items():
        if len(v) < MIN_N: continue
        nll, n = gauss_nll(np.array(v, float))
        if not math.isnan(nll): tots += nll*n; ns += n
    nll_s = tots / max(ns, 1)

    res = {
        "total_steps": len(recs),
        "M0_global_nll": round(nll0, 4),
        "M1_sequence_context_nll": round(nll1, 4), "M1_groups": groups1, "M1_steps": n_used,
        "M2_sequence_x_structure_nll": round(nll2, 4), "M2_groups": groups2, "M2_steps": n2,
        "Mstruct_only_nll": round(nll_s, 4),
        "gain_sequence_over_global": round(nll0 - nll1, 4),
        "gain_structure_over_sequence": round(nll1 - nll2, 4),
        "gain_structure_only_over_global": round(nll0 - nll_s, 4),
    }
    (OUT / "stiffness_headroom.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("\nInterpretation (lower NLL = better fit, nats/step):")
    print(f"  a lookup table on sequence context buys {res['gain_sequence_over_global']:.3f} nats")
    print(f"  adding STRUCTURAL context buys a further {res['gain_structure_over_sequence']:.3f} nats")
    if res["gain_structure_over_sequence"] > 0.05:
        print("  => structural context carries real extra signal; a LEARNED ENCODER is justified")
    else:
        print("  => structural context adds little; a lookup table would suffice")


if __name__ == "__main__":
    main()
