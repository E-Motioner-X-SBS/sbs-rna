#!/usr/bin/env python3
"""End-to-end recall of the Hierarchical Pair Track, which §7.4a owes.

§7.4a evaluates L1 and L2 **standalone**: each at its own budget, against its
own labels, as if the other did not exist. That is the right way to ask whether
each level can score, and the wrong way to describe what a deployed track
delivers, because the track CASCADES -- L2 only ever refines inside the L1
pairs that survived. An L2 block whose parent L1 pair was dropped is
unrecoverable no matter how well L2 scores it.

So the number that matters is:

    cascade recall = (true L2 blocks that survive L1 AND are picked by L2)
                     -------------------------------------------------------
                                  all true L2 blocks

with the denominator over EVERY valid occupied L2 pair, including the ones L1
threw away. Reported beside it:

  l1_ceiling   the fraction of true L2 blocks whose parent L1 pair survived --
               the best cascade recall that L2 could possibly reach. The gap
               between the ceiling and the cascade is L2's loss; 1 - ceiling is
               L1's, and they are different problems with different fixes.

All four scorers see the same batches, the same valid masks and the same
budgets, as in §7.4a, so the comparison is by construction.

Usage:
    python3 scripts/sampling/measure_cascade_recall.py [--max-batches N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pharos.data.loader import Pharos3DDataset                       # noqa: E402
from pharos.model.block_scorer import (BlockScorer, ScorerConfig,    # noqa: E402
                                       l1_budget, l2_budget,
                                       occupancy_labels, valid_mask)
from train_block_scorer import gpu_free_gib, to_device               # noqa: E402

CKPT = ROOT / "data/derived/checkpoints/block_scorer.pt"
DATA = ROOT / "data/derived/pharos3d"
OUT = ROOT / "data/samples/analysis"
MODES = ("random", "separation", "learned_noprior", "learned")


def l1_valid(n: int, sep_blocks: int, bmask: torch.Tensor) -> torch.Tensor:
    """L1's valid mask, with the block diagonal optionally admitted.

    `block_scorer.valid_mask` clamps the separation to at least one block:
    `max(1, min_sep // b)`. At L1 that is `max(1, 4 // 16) = 1`, so an L1 pair
    on the diagonal is never valid and never selected -- and every L2 pair
    inside it is unreachable, however well L2 scores it. L2's own separation is
    `max(1, 4 // 4) = 1`, four residues, so the cascade silently throws away
    every contact between 4 and 15 residues apart. Passing 0 here admits the
    diagonal and lets L2's separation do the filtering it was written to do.
    """
    idx = torch.arange(n, device=bmask.device)
    v = (idx[None, :] - idx[:, None]) >= sep_blocks
    return v & bmask[:, None] & bmask[None, :]


def _topk_mask(scores: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    """`(n, n)` bool: the top-`k` valid entries of `scores`."""
    n = scores.shape[0]
    k = int(max(1, min(k, int(valid.sum()))))
    flat = scores.masked_fill(~valid, float("-inf")).flatten()
    keep = torch.zeros(n * n, dtype=torch.bool, device=scores.device)
    keep[flat.topk(k).indices] = True
    return keep.view(n, n)


def _unlearned_scores(mode: str, B: int, L: int, n: int, lengths, b: int,
                      device) -> tuple:
    bm = torch.zeros(B, n, dtype=torch.bool, device=device)
    for i, Li in enumerate(lengths):
        bm[i, :((int(Li) + b - 1) // b)] = True
    if mode == "separation":
        idx = torch.arange(n, device=device, dtype=torch.float32)
        s = (-torch.log1p((idx[None, :] - idx[:, None]).abs()))[None]
        s = s.expand(B, n, n).contiguous()
    else:
        s = torch.rand(B, n, n, device=device)
    return s, bm


@torch.no_grad()
def cascade(model: Optional[BlockScorer], ds: Pharos3DDataset,
            cfg: ScorerConfig, device, mode: str,
            max_batches: Optional[int], token_budget: int, seed: int,
            keep_frac_l1: Optional[float] = None,
            l1_sep_blocks: Optional[int] = None) -> Dict:
    if model is not None:
        model.eval()
    torch.manual_seed(seed)
    kf = cfg.keep_frac_l1 if keep_frac_l1 is None else keep_frac_l1
    sep1 = cfg.sep_blocks("l1") if l1_sep_blocks is None else l1_sep_blocks
    got: Dict[str, List[float]] = {"cascade": [], "l1_ceiling": [],
                                   "l2_standalone": [], "kept_frac": [],
                                   "l2_binds": [], "pairs_per_nt": [],
                                   "diag_unreachable": []}
    by_len: Dict[str, List[float]] = {}
    n_chains = 0

    for bi, batch in enumerate(ds.iter_batches(token_budget=token_budget,
                                               shuffle=False, seed=seed)):
        if max_batches is not None and bi >= max_batches:
            break
        t = to_device(batch, device)
        B, Lpad = t["tokens"].shape
        if mode.startswith("learned"):
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                out = model(t["tokens"], t["mod_ids"], t["chem"], t["mask"],
                            use_sep_prior=(mode == "learned"))
            out = {k: (v.float() if v.dtype != torch.bool else v)
                   for k, v in out.items()}
        else:
            out = {}
            for lvl, b in (("l1", cfg.b1), ("l2", cfg.b2)):
                n = (Lpad + b - 1) // b
                s, bm = _unlearned_scores(mode, B, Lpad, n, t["lengths"], b, device)
                out[f"{lvl}_scores"], out[f"{lvl}_bmask"] = s, bm

        ratio = cfg.b1 // cfg.b2            # L2 blocks per L1 block, 4
        for i, Lt in enumerate(t["lengths"]):
            L = int(Lt)
            s1, s2 = out["l1_scores"][i], out["l2_scores"][i]
            n1, n2 = s1.shape[0], s2.shape[0]
            v1 = l1_valid(n1, sep1, out["l1_bmask"][i])
            v2 = valid_mask(n2, cfg.sep_blocks("l2"), out["l2_bmask"][i])
            if not (bool(v1.any()) and bool(v2.any())):
                continue
            lab2 = occupancy_labels(t["contacts"][i], cfg.b2, n2, device) * v2
            n_pos = float(lab2.sum())
            if n_pos == 0:
                continue
            n_chains += 1

            k1 = max(1, int(kf * float(v1.sum())))
            keep1 = _topk_mask(s1, v1, k1)
            # an L2 pair survives only inside a surviving L1 pair
            idx2 = torch.arange(n2, device=device)
            parent = (idx2 // ratio).clamp(max=n1 - 1)
            survived = keep1[parent][:, parent]
            reachable = v2 & survived

            k2 = l2_budget(L, cfg)
            keep2 = _topk_mask(s2, reachable, k2) & reachable
            hit = float((lab2 * keep2).sum())
            ceiling = float((lab2 * survived).sum()) / n_pos
            standalone = float((lab2 * _topk_mask(s2, v2, k2)).sum()) / n_pos

            got["cascade"].append(hit / n_pos)
            got["l1_ceiling"].append(ceiling)
            got["l2_standalone"].append(standalone)
            got["kept_frac"].append(float(keep2.sum()) / max(float(v2.sum()), 1.0))
            # does L2 actually have to choose? It only binds when the surviving
            # L1 region holds more L2 pairs than the pair budget allows. Below
            # ~400 nt it does not, and the cascade is exactly the L1 ceiling.
            got["l2_binds"].append(1.0 if float(reachable.sum()) > k2 else 0.0)
            got["pairs_per_nt"].append(float(keep2.sum()) * cfg.b2 ** 2 / max(L, 1))
            # L1's valid mask needs one block of separation, and an L1 block is
            # b1 residues. So an L2 pair whose two blocks fall inside the SAME
            # L1 block has no valid parent and is unreachable at any budget --
            # L2 admits contacts 4 residues apart, L1 only admits them 16 apart.
            same_l1 = (parent[:, None] == parent[None, :]) & v2
            got["diag_unreachable"].append(float((lab2 * same_l1).sum()) / n_pos)
            band = ("<128" if L < 128 else "128-512" if L < 512
                    else "512-1500" if L < 1500 else ">=1500")
            by_len.setdefault(band, []).append(hit / n_pos)

    res = {"mode": mode, "n_chains": n_chains, "keep_frac_l1": kf,
           "l1_sep_blocks": sep1}
    for k, v in got.items():
        res[k] = round(float(np.mean(v)), 4) if v else None
    res["cascade_by_length"] = {k: round(float(np.mean(v)), 4)
                                for k, v in sorted(by_len.items())}
    res["chains_by_length"] = {k: len(v) for k, v in sorted(by_len.items())}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--token-budget", type=int, default=8192,
                    help="small by default: this is an eval and it may share "
                         "the card with a training run")
    ap.add_argument("--min-free-gib", type=float, default=8.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not CKPT.exists():
        print(f"no trained scorer at {CKPT}; run train_block_scorer.py first")
        return 1
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mem = gpu_free_gib()
        if mem and mem[0] < args.min_free_gib:
            print(f"only {mem[0]:.1f} GiB free, need {args.min_free_gib}")
            return 1

    sd = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ScorerConfig(**sd["cfg"])
    model = BlockScorer(cfg).to(device)
    model.load_state_dict(sd["model"])
    te = Pharos3DDataset(DATA, split="test")
    print(f"scorer from epoch {sd.get('epoch')}, test split, "
          f"b1={cfg.b1} b2={cfg.b2} target_c={cfg.target_c}\n")

    results = {}
    for mode in MODES:
        r = cascade(model if mode.startswith("learned") else None, te, cfg,
                    device, mode, args.max_batches, args.token_budget, seed=0)
        results[mode] = r
        print(f"  {mode:16s} cascade {r['cascade']}  ceiling {r['l1_ceiling']}  "
              f"L2 standalone {r['l2_standalone']}  ({r['n_chains']} chains)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cascade_recall.json").write_text(json.dumps(results, indent=1))
    lr = results["learned"]
    print(f"\nlearned cascade {lr['cascade']} against an L1 ceiling of "
          f"{lr['l1_ceiling']} and an L2 standalone of {lr['l2_standalone']}")
    print("by length:", lr["cascade_by_length"])
    print(f"L2 binds on {100*(lr['l2_binds'] or 0):.1f}% of chains; "
          f"everywhere else the cascade IS the L1 ceiling")
    print(f"{100*(lr['diag_unreachable'] or 0):.1f}% of true L2 blocks sit inside "
          f"a single L1 block, which L1's min-separation makes unreachable "
          f"at ANY budget")

    # L1 is the bottleneck by construction, so the question the architecture
    # actually has to answer is what a wider L1 costs. Sweep it.
    print("\n== what does a wider L1 buy? (learned scorer) ==")
    sweep = {}
    for kf in (0.12, 0.25, 0.50, 1.00):
        r = cascade(model, te, cfg, device, "learned", args.max_batches,
                    args.token_budget, seed=0, keep_frac_l1=kf)
        sweep[f"{kf:.2f}"] = r
        print(f"  keep_frac_l1 {kf:.2f}  cascade {r['cascade']}  "
              f"ceiling {r['l1_ceiling']}  pairs/nt {r['pairs_per_nt']}")
    results["l1_budget_sweep"] = sweep

    # Does admitting the L1 diagonal recover the 18.9%? The scorer was trained
    # with the diagonal masked out, so it has never been asked to score one --
    # this is what the CURRENT weights give, a floor rather than the value of
    # the fix, and the honest comparison is at the same L1 budget.
    print("\n== L1 with the block diagonal admitted (same budget, same weights) ==")
    diag = {}
    for kf in (0.12, 0.25):
        r = cascade(model, te, cfg, device, "learned", args.max_batches,
                    args.token_budget, seed=0, keep_frac_l1=kf, l1_sep_blocks=0)
        diag[f"{kf:.2f}"] = r
        base = sweep[f"{kf:.2f}"]
        print(f"  keep_frac_l1 {kf:.2f}  cascade {r['cascade']} "
              f"(was {base['cascade']})  ceiling {r['l1_ceiling']} "
              f"(was {base['l1_ceiling']})  pairs/nt {r['pairs_per_nt']}")
    results["l1_diagonal_admitted"] = diag
    (OUT / "cascade_recall.json").write_text(json.dumps(results, indent=1))
    print(f"-> {OUT / 'cascade_recall.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
