#!/usr/bin/env python3
"""Does the router actually route? Width, spread and dead experts, measured.

The MoE block has always computed `mean_width`, `max_width`, `router_entropy`
and `token_router_entropy` -- four scalars at every one of 18 blocks, at every
step of every run -- and the trunk threw all four away before the trainer could
see them. Only `balance_loss` survived, and `balance_loss` is one number that
cannot distinguish a specialised router from a collapsed one without a stated
threshold that nobody had written down. So "no router collapse" was an
inference from a quantity nobody had calibrated, and the central claim about
nucleus routing -- that a confident token fires one expert and an ambiguous one
fires many, which is the whole reason for choosing it over fixed top-k -- had
never been measured on a trained model at all.

This is that measurement, and it is the same shape as the audit's other
findings: a component that exists, is measured, and whose output is never
checked for basic validity.

What the numbers mean
---------------------
`balance` is Switch's `n_experts * sum(frac * pbar)` PER BLOCK. It is 1.0 at
perfect uniformity and `n_experts` when one expert takes everything, so it has
a floor of 1 rather than 0. The trainer logs the SUM over blocks, which is why
a logged 0.197 at `balance_weight` 0.01 across 18 blocks is 1.09 per block --
9% above uniform -- and not the 20x concentration the raw figure suggests.

`width` is the claim itself: how many experts a token fires. At `max_k` it has
degenerated to fixed top-k; at 1 it has degenerated to argmax routing.

`effective experts` is `1 / sum(usage^2)`, the participation ratio: the number
of equally used experts that would give the same usage concentration.

Run: python3 scripts/audit_router.py --ckpt <checkpoint.pt>
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pretrain_mlm as pm                                       # noqa: E402
from pharos.model.moe import RouterFeatures                     # noqa: E402
from pharos.model.pharos import Pharos, PharosConfig            # noqa: E402

OUT = ROOT / "data/samples/analysis/router_audit.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--size", default="shared400")
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--n-loops", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default: this is a few thousand tokens and "
                         "the GPU is usually mid-run")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    dev = torch.device(args.device)
    cfg = getattr(PharosConfig, args.size)()
    model = Pharos(cfg).to(dev).eval()
    st = torch.load(args.ckpt, map_location=dev, weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v for k, v in st.get("model", st).items()}
    model.load_state_dict(sd, strict=False)
    E, K, NB = cfg.n_experts, cfg.max_k, cfg.n_blocks

    corpora = [ROOT / "data/derived/parquet_starter",
               ROOT / "data/derived/parquet_mars"]
    pool = list(itertools.islice(
        pm.iter_sequences(pm.heldout_files(corpora), 20, args.max_len,
                          shards=None, include_heldout=True), 4000))
    rng = np.random.default_rng(0)
    seqs = [pool[int(i)] for i in rng.choice(len(pool), args.n_seq, replace=False)]

    bc = pm.BatchChemistry(pm.SYMBOLS, dev)
    tok, msk, lens = pm.encode_batch(seqs)
    inp, _, _ = pm.apply_span_mask(tok, lens, np.random.default_rng(1234))
    inp_t = torch.as_tensor(inp, device=dev)
    bmask = torch.as_tensor(msk, device=dev)
    chem = bc(inp_t, bmask)
    n_real = bmask.sum(1)
    feats = RouterFeatures(
        length=n_real.float(),
        chem_summary=(chem.sum(1) / n_real.unsqueeze(1).clamp(min=1))[:, :5])
    with torch.no_grad():
        out = model(inp_t, torch.zeros_like(inp_t), chem, bmask,
                    feats=feats, n_loops=args.n_loops, mlm=True)

    aux = out["aux"]
    if "mean_width" not in aux:
        raise SystemExit(
            "[router] the trunk is not propagating router telemetry -- this "
            "model predates the fix and the audit cannot run")
    usage = aux["expert_usage"].detach().float().cpu()
    r = {
        "checkpoint": str(args.ckpt),
        "step": int(st.get("step", -1)),
        "tokens": int(st.get("tokens", -1)),
        "n_experts": E, "max_k": K, "n_blocks": NB,
        "threshold_rel": float(cfg.threshold_rel),
        "tokens_scored": int(bmask.sum()),
        "mean_width": float(aux["mean_width"]),
        "max_width": float(aux["max_width"]),
        "router_entropy_bits": float(aux["router_entropy"]) / float(np.log(2)),
        "max_entropy_bits": float(np.log2(E)),
        "token_router_entropy_bits":
            float(aux["token_router_entropy"]) / float(np.log(2)),
        "balance_per_block": float(aux["balance_loss"]) / 0.01 / NB,
        "dead_expert_frac": float((usage < 0.1 / E).float().mean()),
        "top_decile_share": float(usage.sort(descending=True).values[:E // 10].sum()),
        "effective_experts": 1.0 / float((usage ** 2).sum()),
    }

    # ---- the validity checks the telemetry existed for and never got -------
    bad = []
    if not 1.0 <= r["mean_width"] <= K:
        bad.append(f"mean width {r['mean_width']:.2f} outside [1, {K}]")
    if r["mean_width"] >= K - 1e-6:
        bad.append(f"mean width is pinned at max_k ({K}): nucleus routing has "
                   f"degenerated to fixed top-k and the threshold does nothing")
    if r["mean_width"] <= 1.0 + 1e-6:
        bad.append("mean width is 1: routing has degenerated to argmax and "
                   "only one expert ever receives gradient")
    if r["dead_expert_frac"] > 0.25:
        bad.append(f"{r['dead_expert_frac']:.1%} of experts take under a tenth "
                   f"of a uniform share")
    if r["balance_per_block"] > 0.1 * E:
        bad.append(f"balance {r['balance_per_block']:.1f} per block is over a "
                   f"tenth of the collapsed value {E}")
    r["warnings"] = bad

    print(f"[router] {args.ckpt.name}  step {r['step']:,}  "
          f"{r['tokens']/1e6:.1f}M tokens  {r['tokens_scored']:,} tokens scored")
    print(f"  routing width           {r['mean_width']:8.2f} mean, "
          f"{r['max_width']:.0f} max, of max_k {K}")
    print(f"  router entropy          {r['router_entropy_bits']:8.3f} bits "
          f"of {r['max_entropy_bits']:.3f}")
    print(f"  per-token entropy       {r['token_router_entropy_bits']:8.3f} bits")
    print(f"  balance per block       {r['balance_per_block']:8.3f} "
          f"(1.0 uniform, {E} collapsed)")
    print(f"  dead experts            {r['dead_expert_frac']:8.1%}")
    print(f"  effective experts       {r['effective_experts']:8.1f} of {E}")
    print(f"  top decile takes        {r['top_decile_share']:8.1%} "
          f"(10.0% at uniform)")
    for w in bad:
        print(f"  WARNING {w}")
    if not bad:
        print("  all validity checks pass")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=1))
    print(f"[router] wrote {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
