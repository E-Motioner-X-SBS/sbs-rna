#!/usr/bin/env python3
"""Find the MoE configuration ARCHITECTURE v0.2 §5.4's sizing table implies.

§5.4 gives, for three models, `d_model`, block count, loop count, total
parameters and active parameters, and labels the row
**"[v0.1 arithmetic, verified exact]"**. It does not give `n_experts`,
`d_expert`, `top_k` or `n_shared` -- and those are what actually determine the
two numbers it quotes. So the arithmetic cannot be checked from the document:
the inputs are not in it.

This solves for them. Parameter counts are computed **analytically** and
validated against a real `Pharos` instantiation, because instantiating a model
per grid point takes minutes per configuration at Base-v2 scale.

The result decides between two possibilities that matter:

* a configuration exists that hits all three rows, in which case it should be
  written into `PharosConfig` and the table is vindicated; or
* none does, in which case "verified exact" is a claim about arithmetic nobody
  can reproduce, and the built numbers are the true ones.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

#: name -> (d_model, n_blocks, n_loops, total, active) as printed in §5.4
TARGETS: Dict[str, Tuple[int, int, int, float, float]] = {
    "PHAROS-Small": (512, 16, 8, 149e6, 61e6),
    "PHAROS-Mini": (384, 12, 12, 67e6, 30e6),
    "Base-v2": (768, 32, 3, 1401e6, 269e6),
}
N_SYMBOLS, N_MOD, D_CHEM, MAX_LEN = 13, 371, 24, 4608
ROUTER_COND = 6 + 8           # length bins + the scalar block


def analytic(d: int, nb: int, ne: int, de: int, k: int, ns: int,
             n_heads: int, d_pair: int = 128) -> Tuple[int, int]:
    """(total, active) for the implementation in `pharos.model`.

    Mirrors the modules exactly; `--verify` checks it against a built model.
    """
    h = n_heads
    embed = (N_SYMBOLS * d + N_MOD * d + (D_CHEM * d + d) + MAX_LEN * d + 2 * d)
    # one mixer of each kind, per §5.1's 8-block period tiled to nb
    gdn = 5 * d * d + d + 2 * (d * h + h)
    swa = 4 * d * d
    full = 4 * d * d + h
    pattern = ["gdn", "gdn", "swa", "gdn", "gdn", "swa", "gdn", "full"]
    kinds = [pattern[i % 8] for i in range(nb)]
    mixers = sum({"gdn": gdn, "swa": swa, "full": full}[x] for x in kinds)

    expert = 3 * d * de                      # SwiGLU: w1 2*d*de, w2 de*d
    gate = (d + ROUTER_COND) * ne
    per_block_ff_fixed = gate + 2 * d + ne   # gate, norm, expert_bias
    ff_total = nb * (per_block_ff_fixed + (ne + ns) * expert)
    ff_active = nb * (per_block_ff_fixed + (k + ns) * expert)

    trunk_fixed = nb * (2 * d) + mixers + 2 * d + 2 * d + d * d   # norms + recycle

    def mlp(di, dh, do):
        return 2 * di + (di * dh + dh) + (dh * do + do)
    heads = (mlp(d_pair, d_pair, 1) + mlp(d_pair, d_pair, 40)          # 1,2
             + mlp(d, 2 * d, 9 * 3) + mlp(d, d, 3)                      # 3
             + mlp(d, d, 8) + mlp(d, d // 2, 1) + mlp(d, d // 2, 1)     # 4,5,6
             + mlp(d, d, 2) + mlp(d, d // 2, 4)                         # 7,10
             + mlp(d, d, 1) + mlp(d, d // 2, 3))                        # 8,9
    other = (2 * d * d_pair + d_pair) + 1                               # pair_proj, elec

    fixed = embed + trunk_fixed + heads + other
    return fixed + ff_total, fixed + ff_active


def verify() -> bool:
    from pharos.model.pharos import Pharos, PharosConfig
    ok = True
    for d, nb, ne, de, k, ns, nh in [(512, 16, 16, 256, 2, 1, 8),
                                     (384, 12, 8, 192, 2, 1, 6)]:
        cfg = PharosConfig(d_model=d, n_blocks=nb, n_heads=nh, n_experts=ne,
                           d_expert=de, top_k=k, n_shared=ns)
        pc = Pharos(cfg).param_counts()
        at, aa = analytic(d, nb, ne, de, k, ns, nh)
        good = (pc["total"] == at and pc["active"] == aa)
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} d={d} nb={nb}: built "
              f"{pc['total']:,}/{pc['active']:,}  analytic {at:,}/{aa:,}")
    return ok


def main() -> None:
    print("verifying the analytic count against built models")
    if not verify():
        raise SystemExit("analytic formula disagrees with the implementation")

    grid = {"ne": [4, 8, 16, 24, 32, 48, 64, 96, 128],
            "de": [96, 128, 160, 192, 256, 320, 384, 512, 640, 768],
            "k": [1, 2, 4, 6, 8], "ns": [0, 1, 2, 3, 4]}
    out = {}
    print("\nsolving for the configuration §5.4 implies")
    for name, (d, nb, nl, tt, ta) in TARGETS.items():
        best = None
        for ne, de, k, ns in itertools.product(grid["ne"], grid["de"],
                                               grid["k"], grid["ns"]):
            if k > ne:
                continue
            t, a = analytic(d, nb, ne, de, k, ns, max(4, d // 64))
            err = abs(t - tt) / tt + abs(a - ta) / ta
            if best is None or err < best[0]:
                best = (err, ne, de, k, ns, t, a)
        err, ne, de, k, ns, t, a = best
        out[name] = {"d_model": d, "n_blocks": nb, "n_loops": nl,
                     "spec_total": tt, "spec_active": ta,
                     "n_experts": ne, "d_expert": de, "top_k": k, "n_shared": ns,
                     "fit_total": t, "fit_active": a,
                     "rel_err_total": round(abs(t - tt) / tt, 4),
                     "rel_err_active": round(abs(a - ta) / ta, 4)}
        print(f"  {name:14s} spec {tt/1e6:7.1f}M/{ta/1e6:6.1f}M  ->  "
              f"ne={ne:3d} de={de:3d} k={k} ns={ns}  "
              f"fit {t/1e6:7.1f}M/{a/1e6:6.1f}M  "
              f"err {100*abs(t-tt)/tt:4.1f}%/{100*abs(a-ta)/ta:4.1f}%")

    worst = max(max(v["rel_err_total"], v["rel_err_active"]) for v in out.values())
    out["_verdict"] = (
        "reproducible" if worst < 0.02 else
        "NOT reproducible: no configuration on the grid matches all three rows")
    print(f"\nworst relative error across the three rows: {100*worst:.1f}%")
    print(f"verdict: {out['_verdict']}")
    (ROOT / "data/samples/analysis/sizing_solution.json").write_text(
        json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
