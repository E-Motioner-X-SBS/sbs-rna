#!/usr/bin/env python3
"""Defect #17: refinement loops were never counted in the FLOP budget.

`training_cost_model.py` computes 6 * N_active * tokens -- the cost of ONE pass
through the blocks. PHAROS-Small runs 8 refinement loops, and section 10b.3
specifies deep supervision at EVERY segment with a one-step (detached) gradient,
so each segment is its own forward+backward. Cost therefore scales with the loop
count, and every published figure is understated by exactly that factor.

The design even states loops "cost compute but no additional activation memory"
-- then omits the compute.

This recomputes the ladder and the lever chain with loops included.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "samples" / "analysis"

# name, d, blocks, loops, total_M, active_M, published_A100h (one-pass)
LADDER = [
    ("PHAROS-Micro",  256,  8, 16,   23,  12,   61),
    ("PHAROS-Mini",   384, 12, 12,   67,  30,  148),
    ("PHAROS-Small",  512, 16,  8,  149,  61,  299),
    ("PHAROS-Base-v2",768, 32,  3, 1401, 269, 1325),
]
BASE_PUBLISHED = 1883      # "PHAROS-Base as first specified"
BASE_LOOPS = 3


def main():
    print("=== the ladder, with the loop factor restored ===")
    print(f"{'model':16s}{'active':>8s}{'loops':>7s}{'eff.compute':>13s}"
          f"{'published h':>13s}{'REAL h':>10s}{'understated':>13s}")
    rows = []
    for name, d, blk, loops, tot, act, pub in LADDER:
        eff = act * loops                      # active-param-passes per token
        real = pub * loops
        print(f"{name:16s}{act:>6d}M{loops:>7d}{eff:>11d}M{pub:>13,d}{real:>10,d}{loops:>12d}x")
        rows.append({"model": name, "active_M": act, "loops": loops,
                     "effective_compute_M": eff, "published_h": pub, "real_h": real})

    print("\n=== what this does to the size comparison ===")
    small = next(r for r in rows if r["model"] == "PHAROS-Small")
    base = next(r for r in rows if r["model"] == "PHAROS-Base-v2")
    print(f"  by ACTIVE PARAMS   Small {small['active_M']}M vs Base-v2 {base['active_M']}M"
          f"   -> Small {base['active_M']/small['active_M']:.1f}x smaller")
    print(f"  by EFFECTIVE COMPUTE Small {small['effective_compute_M']}M vs "
          f"Base-v2 {base['effective_compute_M']}M   -> Small only "
          f"{base['effective_compute_M']/small['effective_compute_M']:.2f}x cheaper")
    print("  => the '4.4x smaller' framing does not survive: 8 loops against 3 eat most of it.")

    print("\n=== depth: the comparison mixes train-time and serve-time ===")
    print(f"  {'model':16s}{'train depth':>13s}{'serve@3 depth':>15s}")
    for name, d, blk, loops, tot, act, pub in LADDER:
        print(f"  {name:16s}{blk*loops:>13d}{blk*3:>15d}")
    print("  Section 10b.3 says 'train 8-16, serve 3'. So Small's 8 is a TRAIN figure")
    print("  while Base-v2's 3 reads as a SERVE figure -- they are different quantities")
    print("  in the same column. At a matched serve-3 setting:")
    print(f"    Small  16x3 = 48 effective layers")
    print(f"    Base-v2 32x3 = 96 effective layers")
    print("  => the headline 'MORE effective depth than Base-v2 (128 vs 96)' holds only")
    print("     at training. At inference it REVERSES: Base-v2 is 2x deeper.")

    print("\n=== corrected lever chain (PHAROS-Small, loops included) ===")
    base_real = BASE_PUBLISHED * BASE_LOOPS
    print(f"  baseline PHAROS-Base, {BASE_LOOPS} loops   {base_real:>8,d} A100-h "
          f"(published {BASE_PUBLISHED:,} was one-pass)")
    steps = [("all-MoE fine-grained", 1.42), ("right-size to Small", None),
             ("Muon optimiser", 2.00), ("read down-weighting", 1.19)]
    x = base_real
    x /= 1.42
    print(f"  /1.42 all-MoE fine-grained            {x:>8,.0f}")
    # right-sizing now has to carry the loop change 3 -> 8
    rs = (269 / 61) * (3 / 8)
    x /= rs
    print(f"  /{rs:.2f} right-size to Small (8 loops)  {x:>8,.0f}"
          f"   <- was 4.43x assuming equal loops; really {rs:.2f}x")
    for nm, f in [("Muon", 2.00), ("read down-weighting", 1.19)]:
        x /= f
        print(f"  /{f:<4.2f} {nm:29s} {x:>8,.0f}")
    print(f"  TOTAL {base_real/x:.1f}x   (published claimed 24x)")

    tok_cut = 323 / 25
    print(f"\n  at the cycle-3 token budget (25B, /{tok_cut:.1f}x): {x/tok_cut:>8,.0f} A100-h bf16")
    print(f"  on H100 fp8 (/6.34x raw):                 {x/tok_cut/6.34:>8,.0f} A100-equivalent h")

    res = {"defect_17": "refinement loops omitted from the FLOP budget",
           "ladder_with_loops": rows,
           "small_vs_base_by_active": round(base["active_M"]/small["active_M"], 2),
           "small_vs_base_by_effective_compute": round(
               base["effective_compute_M"]/small["effective_compute_M"], 2),
           "depth_train": {r["model"]: None for r in rows},
           "corrected_total_factor": round(base_real/x, 1),
           "corrected_a100h_323B": round(x),
           "corrected_a100h_25B": round(x/tok_cut)}
    for name, d, blk, loops, tot, act, pub in LADDER:
        res["depth_train"][name] = {"train": blk*loops, "serve_at_3": blk*3}
    (OUT / "cost_with_loops.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
