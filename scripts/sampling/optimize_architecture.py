#!/usr/bin/env python3
"""Search the PHAROS design space for cheaper configurations at equal capacity.

The cost model showed 96.3% of training FLOPs are the 6*N_active term, so the
lever is ACTIVE parameters -- not context length and not attention, which is
only 3.7% of the budget.

A specific inefficiency to test: PHAROS-Base puts MoE in every SECOND block and
leaves the other 16 blocks as dense FFN with d_ff=3072. Those dense blocks cost
113M ACTIVE params -- exactly as much as all 16 MoE blocks -- while carrying far
less total capacity. Replacing them with MoE at lower top-k should cut active
params hard while RAISING total capacity.
"""
from __future__ import annotations
import json, itertools
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data" / "samples" / "analysis"
d = 768
OTHER = 80e6                 # pair track + heads + decoder (unchanged)
TOK = 323e9                  # 1 epoch of elDORS
MFU, A100 = 0.35, 312e12


def cfg_cost(n_blocks, moe_every, dff_moe, dff_dense, n_routed, n_shared, topk):
    n_moe = n_blocks // moe_every
    n_dense = n_blocks - n_moe
    attn = n_blocks * 4 * d * d
    per_e = 3 * d * dff_moe
    moe_tot = n_moe * (n_routed + n_shared) * per_e
    moe_act = n_moe * (topk + n_shared) * per_e
    dense = n_dense * 3 * d * dff_dense
    total = attn + moe_tot + dense + OTHER
    active = attn + moe_act + dense + OTHER
    flops = 6 * active * TOK
    return total, active, flops / (A100 * MFU) / 3600


BASE = dict(n_blocks=32, moe_every=2, dff_moe=512, dff_dense=3072,
            n_routed=32, n_shared=2, topk=4)
bt, ba, bh = cfg_cost(**BASE)
print(f"BASELINE  total {bt/1e6:6.0f}M  active {ba/1e6:6.0f}M  {bh:7,.0f} A100-h\n")

cands = [
 ("all-MoE, top-2+1 shared",      dict(BASE, moe_every=1, n_shared=1, topk=2)),
 ("all-MoE, top-2+2 shared",      dict(BASE, moe_every=1, topk=2)),
 ("all-MoE, top-3+1 shared",      dict(BASE, moe_every=1, n_shared=1, topk=3)),
 ("all-MoE, top-4+2 shared",      dict(BASE, moe_every=1)),
 ("all-MoE finer (dff 256, top-4+2)", dict(BASE, moe_every=1, dff_moe=256, n_routed=64)),
 ("all-MoE finer (dff 256, top-6+2)", dict(BASE, moe_every=1, dff_moe=256, n_routed=64, topk=6)),
 ("baseline but top-2",           dict(BASE, topk=2)),
 ("24 blocks, all-MoE top-2+1",   dict(BASE, n_blocks=24, moe_every=1, n_shared=1, topk=2)),
]
print(f"{'configuration':36s} {'total':>8s} {'active':>8s} {'A100-h':>9s} {'vs base':>9s} {'capacity':>9s}")
rows=[]
for name, c in cands:
    t, a, h = cfg_cost(**c)
    rows.append({"name": name, "total_M": round(t/1e6), "active_M": round(a/1e6),
                 "a100_hours": round(h), "compute_vs_base": round(h/bh, 3),
                 "capacity_vs_base": round(t/bt, 2)})
    print(f"{name:36s} {t/1e6:7.0f}M {a/1e6:7.0f}M {h:9,.0f} {h/bh:8.2f}x {t/bt:8.2f}x")

print("\n=== stacking the wins on the best configuration ===")
best = min(rows, key=lambda r: r["a100_hours"] if r["capacity_vs_base"] >= 1.0 else 1e18)
print(f"pick: {best['name']}  ({best['active_M']}M active, {best['capacity_vs_base']}x capacity)")
h = best["a100_hours"]
steps = [("architecture (all-MoE, lower top-k)", bh / h)]
h2 = h / 2.0;   steps.append(("Muon optimiser (~2x compute efficiency)", 2.0))
h3 = h2 / 1.6;  steps.append(("FP8 on supported hardware (~1.6x realised)", 1.6))
print(f"\n{'lever':44s} {'factor':>8s} {'A100-h':>10s}")
cum = bh
print(f"{'baseline':44s} {'--':>8s} {cum:10,.0f}")
for name, f in steps:
    cum /= f
    print(f"{name:44s} {f:7.2f}x {cum:10,.0f}")
print(f"\nTOTAL SPEEDUP: {bh/cum:.1f}x   ({bh:,.0f} -> {cum:,.0f} A100-hours)")
print(f"  = {cum/24/8:.1f} days on 8 A100s   or {cum/24/4:.1f} days on 4")
json.dump({"baseline_a100_hours": round(bh), "candidates": rows,
           "stacked_a100_hours": round(cum)}, open(OUT/"architecture_optimization.json","w"), indent=2)
