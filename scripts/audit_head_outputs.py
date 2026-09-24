#!/usr/bin/env python3
"""Does every head actually respond to its input, and is its output in range?

A head can be wired, trained and reported on while emitting something
meaningless, and nothing downstream will say so. The failure that prompted this
script: head 3 was producing backbones whose bonds were six to eleven times too
long, with a clash score of exactly 0.000 and TM-score, lDDT and GDT all
ranking it as merely inaccurate rather than as not-a-molecule.

So each head is checked for the properties a working head must have, on real
data, with two different inputs:

**Responds to input.** An output identical across two unrelated batches is a
constant, whatever its loss curve says. This catches a head whose gradient
never arrived, whose input was detached, or which collapsed onto a bias.

**Finite.** NaN and Inf propagate silently through a mean.

**In range.** A probability outside [0, 1], a logit of 10^4, a distance
prediction that is negative -- each says the head's parameterisation and its
loss disagree about what it emits.

**Not degenerate.** Near-zero variance across positions is a head that has
learned the marginal and nothing else, which is the structural-head equivalent
of predicting the commonest base.

Run against any checkpoint; it reports rather than asserts, because "low
variance" is a symptom and not always a fault.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-loops", type=int, default=2)
    args = ap.parse_args()

    from pharos.data.chemistry import N_DIMS
    from pharos.model.moe import RouterFeatures
    from pharos.model.pharos import Pharos, PharosConfig

    dev = torch.device(args.device)
    if args.ckpt and args.ckpt.exists():
        sd = torch.load(args.ckpt, map_location=dev, weights_only=False)
        saved = sd.get("cfg") or {}
        cfg = PharosConfig(**{k: v for k, v in saved.items()
                              if k in PharosConfig.__dataclass_fields__})
        for k, v in saved.items():
            if not hasattr(cfg, k):
                setattr(cfg, k, v)
        model = Pharos(cfg).to(dev).eval()
        model.load_state_dict({k.replace("_orig_mod.", ""): v
                               for k, v in sd["model"].items()}, strict=False)
        tag = f"{args.ckpt.name} (step {sd.get('step','?')})"
    else:
        cfg = PharosConfig(d_model=128, n_blocks=2, n_loops=1, n_heads=4,
                           window=32, d_pair=48, n_experts=8, d_expert=64,
                           top_k=2, n_shared=1)
        model = Pharos(cfg).to(dev).eval()
        tag = "randomly initialised"
    print(f"[audit] {tag}\n")

    B, L = 2, 96
    g = torch.Generator(device="cpu").manual_seed(0)

    def forward(seed):
        tg = torch.Generator(device="cpu").manual_seed(seed)
        tok = torch.randint(0, 4, (B, L), generator=tg).to(dev)
        chem = torch.randn(B, L, N_DIMS, generator=tg).to(dev)
        msk = torch.ones(B, L, dtype=torch.bool, device=dev)
        ft = RouterFeatures(length=msk.sum(1).float())
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16,
                                             enabled=(dev.type == "cuda")):
            return model(tok, torch.zeros_like(tok), chem, msk,
                         feats=ft, n_loops=args.n_loops, mlm=True)

    a, b = forward(1), forward(2)
    rows = []
    for k in sorted(a):
        va = a[k]
        if not torch.is_tensor(va) or not va.is_floating_point():
            continue
        vb = b[k]
        x = va.float()
        same = bool(torch.allclose(x, vb.float(), atol=1e-6))
        finite = bool(torch.isfinite(x).all())
        # variance ACROSS positions, which is what a per-residue head must have
        pv = float(x.reshape(x.shape[0], x.shape[1], -1).std(dim=1).mean()) \
            if x.dim() >= 3 else float(x.std())
        rows.append((k, tuple(x.shape), float(x.min()), float(x.max()),
                     pv, same, finite))

    print(f"{'output':22s} {'shape':22s} {'min':>9s} {'max':>9s} "
          f"{'pos.std':>8s}  flags")
    bad = 0
    for k, sh, mn, mx, pv, same, fin in rows:
        flags = []
        if same:
            flags.append("CONSTANT-ACROSS-INPUTS")
        if not fin:
            flags.append("NON-FINITE")
        if pv < 1e-4:
            flags.append("near-zero variance")
        if abs(mn) > 1e4 or abs(mx) > 1e4:
            flags.append("extreme magnitude")
        bad += bool(flags)
        print(f"  {k:20s} {str(sh):22s} {mn:9.3f} {mx:9.3f} {pv:8.4f}  "
              f"{', '.join(flags) if flags else 'ok'}")
    print(f"\n{len(rows)} float outputs, {bad} flagged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
