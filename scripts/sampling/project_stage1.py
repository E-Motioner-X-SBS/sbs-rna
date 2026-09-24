#!/usr/bin/env python3
"""What is the rest of stage 1 worth? The honest answer is: we cannot tell yet.

Two standard forms fit the clean held-out curve, and they disagree by more than
the entire improvement measured so far:

  power law with a floor   bits = c + a*T^-alpha   -> 1.575 bits at 4B, floor 1.540
  log-linear               bits = k + m*ln(T)      -> 1.375 bits at 4B

The residual scatter is 0.054 bits and the token range spanned is 3.2x
(0.425B to 1.377B). A power law needs one to two decades to pin its exponent;
fitted here it comes out at alpha = 1.14, which is an order of magnitude
steeper than any published neural scaling exponent and is what a floor fit does
when it is given a short, noisy span. So the floor is an artefact of the fit,
not a measurement -- and the log-linear form is equally unconstrained in the
other direction.

This script exists to say that in numbers rather than to pick a winner. What
would resolve it is more token range and less scatter, and the scatter is the
binding constraint: at 0.054 bits of noise over a 0.25-bit total descent, no
functional form is identifiable. The pool-interleaving and max-batch fixes are
therefore not only training changes; they are what makes the curve predictable.

Run: python3 scripts/sampling/project_stage1.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data/samples/analysis/runs/heldout_mlm.csv"
OUT = ROOT / "data/samples/analysis/stage1_projection.json"
CORPUS_BITS = 2.0165


def main() -> int:
    rows = {}
    with CSV.open() as f:
        for r in csv.DictReader(f):
            if r["n_seq"] == "1024" and int(r["n_masked"]) == 22375:
                rows[int(r["step"])] = (float(r["tokens"]), float(r["bits"]))
    s = np.array(sorted(rows))
    T = np.array([rows[k][0] for k in s])
    B = np.array([rows[k][1] for k in s])

    best = None
    for c in np.arange(0.0, 1.61, 0.01):
        y = B - c
        if (y <= 0).any():
            continue
        A = np.polyfit(np.log(T), np.log(y), 1)
        pred = c + np.exp(np.polyval(A, np.log(T)))
        sse = float(((B - pred) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(c), A)
    sse, c, A = best
    alpha, a = -float(A[0]), float(np.exp(A[1]))
    rms_pl = float(np.sqrt(sse / len(B)))
    A2 = np.polyfit(np.log(T), B, 1)
    rms_ll = float(np.sqrt(((B - np.polyval(A2, np.log(T))) ** 2).mean()))

    def pl(t):
        return c + a * t ** -alpha

    def ll(t):
        return float(np.polyval(A2, np.log(t)))

    print(f"n={len(s)} clean readings over {T.min()/1e9:.3f}B-{T.max()/1e9:.3f}B "
          f"tokens ({T.max()/T.min():.1f}x range)")
    print(f"  power law   bits = {c:.3f} + {a:.3g}*T^-{alpha:.3f}   rms {rms_pl:.4f}")
    print(f"  log-linear  bits = {A2[1]:.3f} {A2[0]:+.4f}*ln(T)      rms {rms_ll:.4f}")
    print(f"\n{'tokens':>9}  {'power law':>10}  {'log-linear':>10}  {'spread':>8}")
    proj = {}
    for t in (2e9, 4e9, 8e9):
        p, l = pl(t), ll(t)
        proj[f"{t:.0f}"] = {"power_law": round(float(p), 4),
                            "log_linear": round(l, 4),
                            "spread": round(float(p - l), 4)}
        print(f"{t/1e9:8.0f}B  {p:10.4f}  {l:10.4f}  {p-l:8.4f}")
    print(f"\nalpha = {alpha:.2f}. Published neural scaling exponents are 0.05-0.15;")
    print("this is what a floor fit returns on a short noisy span, not a measurement.")
    print(f"At 4B the two forms differ by {pl(4e9)-ll(4e9):.3f} bits, against a total")
    print(f"measured descent of {B.max()-B.min():.3f} bits and scatter of {rms_ll:.3f}.")
    print("The remaining budget is worth somewhere in that interval; the data")
    print("cannot narrow it further, and the scatter is why.")

    OUT.write_text(json.dumps(
        {"n": len(s), "tokens_min": float(T.min()), "tokens_max": float(T.max()),
         "best_measured_bits": float(B.min()),
         "corpus_bits": CORPUS_BITS,
         "power_law": {"floor": c, "a": a, "alpha": alpha, "rms": rms_pl},
         "log_linear": {"k": float(A2[1]), "m": float(A2[0]), "rms": rms_ll},
         "projection": proj,
         "verdict": "not identifiable: the two standard forms differ at 4B by "
                    "more than the total improvement measured so far"}, indent=1))
    print(f"\n[proj] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
