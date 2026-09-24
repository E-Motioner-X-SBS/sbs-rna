#!/usr/bin/env python3
"""Is the held-out curve's scatter noise, or is it the batch size moving?

The clean held-out curve has a residual scatter of about 0.046 bits around its
trend, and every check-in has called that measurement noise. It is not, or not
mostly. `_pack_pool` cuts a batch on the padded token budget OR on the sequence
COUNT, and at the old `--max-batch` 512 the count bound on short-sequence
shards: elDORS c020 capped 92.5% of its batches and carried 83,899 real tokens
where c001 carried 194,607. So the effective batch size is set by whichever
shard is streaming and swings 2.3x across the run, with gradient noise scaling
as 1/sqrt(batch).

This joins the per-100-step token rate out of the trainer log to the clean
held-out readings and asks whether they move together. The step number is
partialled out, because bits fall over training and the token rate rose over
training, and a raw correlation would mostly be measuring that.

The alternative explanation -- that it is the batch CONTENT, not its size, and
a model fed short sequences is simply worse -- predicts the same correlation,
so it is worth saying what separates them: the fixed held-out sample is itself
drawn from c020 and is SHORT (mean 185 nt). If content were the mechanism,
training on short-sequence shards should help on a short-sequence evaluation.
It hurts.

What this does NOT support is explaining any single reading. Step 8,500 dropped
to 1.8224 bits on a 134,000-token mean batch and was attributed to exactly this
mechanism; step 8,750 then ran on a SMALLER batch, 123,500, and recovered to
1.6602. The aggregate correlation survives that -- it is an ensemble effect at
t = -2.8 -- but the per-point story does not, and step 8,500 has no established
cause.

Run: python3 scripts/sampling/measure_batch_effect.py
"""
from __future__ import annotations

import csv
import json
import re
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "data/samples/analysis/cron/run-20260924-115630.log"
CSV = ROOT / "data/samples/analysis/runs/heldout_mlm.csv"
OUT = ROOT / "data/samples/analysis/batch_size_effect.json"


def main() -> int:
    pts = []
    for ln in LOG.read_text().splitlines():
        m = re.search(r"step (\d+) ([\d.]+)M tok", ln)
        if m:
            pts.append((int(m.group(1)), float(m.group(2)) * 1e6))
    pts.sort()
    rate = {s1: (t1 - t0) / (s1 - s0)
            for (s0, t0), (s1, t1) in zip(pts, pts[1:]) if s1 > s0}

    seen = {}
    with CSV.open() as f:
        for r in csv.DictReader(f):
            # one comparable series only: the 1024-sequence sample
            if r["n_seq"] == "1024" and r["min_len"] == "20" \
                    and int(r["n_masked"]) == 22375:
                seen[int(r["step"])] = float(r["bits"])
    steps, x, y = [], [], []
    for st in sorted(seen):
        w = [v for k, v in rate.items() if st - 250 < k <= st]
        if w:
            steps.append(st)
            x.append(float(np.mean(w)))
            y.append(seen[st])
    if len(steps) < 6:
        raise SystemExit("[batch] too few paired points")
    X, Y, S = np.array(x), np.array(y), np.array(steps, dtype=float)

    def resid(v):
        return v - np.polyval(np.polyfit(S, v, 1), S)

    r_raw = float(np.corrcoef(X, Y)[0, 1])
    r_par = float(np.corrcoef(resid(X), resid(Y))[0, 1])
    df = len(X) - 3
    t = r_par * sqrt(df / max(1e-9, 1 - r_par ** 2))

    print(f"{'step':>6}  {'tok/step':>9}  {'bits':>7}")
    for s, a, b in zip(steps, x, y):
        print(f"{s:6d}  {a:9.0f}  {b:7.4f}")
    print(f"\nn = {len(X)} checkpoints")
    print(f"  r(tokens per step, held-out bits)      {r_raw:+.4f}")
    print(f"  partial r, step number controlled for  {r_par:+.4f}"
          f"   t = {t:+.2f} on {df} df")
    print("  negative means: fewer real tokens in the step, worse held-out bits")

    res = {"n": len(X), "steps": steps,
           "tokens_per_step": [round(v, 1) for v in x],
           "bits": y, "r_raw": round(r_raw, 4),
           "r_partial_step_controlled": round(r_par, 4),
           "t": round(float(t), 3), "df": df,
           "note": "observational; mechanism is gradient noise ~ "
                   "1/sqrt(batch); the content confound is argued against by "
                   "the held-out sample itself being short-sequence. AGGREGATE "
                   "ONLY: it does not explain individual excursions. Step "
                   "8,500 (134,000 tok/step, 1.8224 bits) was attributed to "
                   "it and step 8,750 then ran on a SMALLER batch (123,500) "
                   "and recovered to 1.6602, so that attribution is withdrawn "
                   "and 8,500 has no established cause."}
    OUT.write_text(json.dumps(res, indent=1))
    print(f"[batch] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
