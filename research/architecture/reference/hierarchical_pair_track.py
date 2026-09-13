#!/usr/bin/env python3
"""Reference implementation of the PHAROS Hierarchical Pair Track (HPT).

Purpose: turn the cost arithmetic in ARCHITECTURE.md section 5 into running code
whose peak memory can be measured, rather than asserted.

The track replaces a dense L x L pair representation with three coarse-to-fine
levels, selecting contact-bearing BLOCKS rather than ranking individual pairs:

  L1  dense pair map at block size b1 (=16)    -> (L/b1)^2 entries
  L2  refine selected b1 blocks to b2 (=4)
  L3  refine selected b2 blocks to pairs

Design constraints this encodes (all measured, see data/samples/analysis/):
  * contacts/nt saturates at 4.4-4.9 -> contacts scale O(L)
  * block occupancy at b=4 is 1.34% on 1500-3000 nt chains and FALLS with L
  * a flat top-K proposal recovers only ~20% of contacts on long chains

This is a structural/memory reference, not a trained model. Weights are random.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HPTConfig:
    d_model: int = 768
    d_pair: int = 128
    b1: int = 16          # coarse block size
    b2: int = 4           # intermediate block size
    keep_frac_l1: float = 0.12   # coarse blocks kept; measured b1=16 occupancy is 7.26%
    target_c: float = 20.0       # pair budget as K = target_c * L, in the SAME units as
                                 # the measured requirement (4.4-4.9 contacts/nt, and
                                 # effective c = 17.2 to keep every occupied b=4 block)
    min_sep: int = 4             # |i-j| below this is excluded (local chain)
    n_tri_layers: int = 2


def _block_pool(x: torch.Tensor, b: int) -> torch.Tensor:
    """Mean-pool a (B, L, D) token track into (B, ceil(L/b), D) block features."""
    B, L, D = x.shape
    pad = (-L) % b
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    return x.view(B, (L + pad) // b, b, D).mean(2)


class CoarseBlockScorer(nn.Module):
    """Dense but CHEAP: scores every block pair at resolution b, not every nucleotide pair.

    At b=16 and L=4096 this is a 256x256 map -- 65k entries -- versus 16.8M dense
    pairs, so it can be genuinely dense and therefore loses no recall at this level.
    """

    def __init__(self, cfg: HPTConfig, b: int):
        super().__init__()
        self.b = b
        self.q = nn.Linear(cfg.d_model, cfg.d_pair, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_pair, bias=False)
        self.sep = nn.Parameter(torch.tensor([1.0, -0.05]))   # learned separation prior

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        blk = _block_pool(tok, self.b)                     # B, N, D
        q, k = self.q(blk), self.k(blk)                    # B, N, d_pair
        s = torch.einsum("bnd,bmd->bnm", q, k) / math.sqrt(q.shape[-1])
        n = blk.shape[1]
        idx = torch.arange(n, device=tok.device)
        sep = (idx[None, :] - idx[:, None]).abs().float()
        s = s + self.sep[0] * torch.exp(self.sep[1] * sep)[None]
        return s


class HierarchicalPairTrack(nn.Module):
    def __init__(self, cfg: HPTConfig):
        super().__init__()
        self.cfg = cfg
        self.l1 = CoarseBlockScorer(cfg, cfg.b1)
        self.l2 = CoarseBlockScorer(cfg, cfg.b2)
        self.pair_proj = nn.Linear(2 * cfg.d_model, cfg.d_pair)
        self.tri = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(cfg.d_pair),
                          nn.Linear(cfg.d_pair, cfg.d_pair), nn.ReLU(),
                          nn.Linear(cfg.d_pair, cfg.d_pair))
            for _ in range(cfg.n_tri_layers)])
        self.out = nn.Linear(cfg.d_pair, 1)

    @staticmethod
    def _topk_mask(scores: torch.Tensor, frac: float, min_sep_blocks: int):
        """Keep the top `frac` of the upper triangle (beyond min separation)."""
        B, N, _ = scores.shape
        idx = torch.arange(N, device=scores.device)
        valid = (idx[None, :] - idx[:, None]) >= min_sep_blocks
        s = scores.masked_fill(~valid[None], float("-inf"))
        n_valid = int(valid.sum())
        k = max(1, int(frac * n_valid))
        flat = s.view(B, -1)
        kth = flat.topk(k, dim=-1).values[:, -1:]
        return (flat >= kth).view(B, N, N) & valid[None], k

    def forward(self, tok: torch.Tensor):
        cfg = self.cfg
        B, L, _ = tok.shape
        stats = {}

        # ---- L1: dense coarse map at b1 ----
        s1 = self.l1(tok)
        stats["l1_entries_scored"] = s1.numel()
        n1 = s1.shape[1]
        stats["l1_entries_needed"] = n1 * (n1 + 1) // 2   # only the triangle is required
        m1, k1 = self._topk_mask(s1, cfg.keep_frac_l1, max(1, cfg.min_sep // cfg.b1))
        stats["l1_kept_blocks"] = k1

        # ---- L2: refine survivors to b2 ----
        r = cfg.b1 // cfg.b2
        s2 = self.l2(tok)                                     # B, N2, N2
        n2 = s2.shape[1]
        # a b2 pair is eligible only if its parent b1 pair survived L1, and it
        # clears the minimum sequence separation
        parent = m1.repeat_interleave(r, 1).repeat_interleave(r, 2)[:, :n2, :n2]
        idx2 = torch.arange(n2, device=tok.device)
        sep_ok = (idx2[None, :] - idx2[:, None]) >= max(1, cfg.min_sep // cfg.b2)
        eligible = parent & sep_ok[None]
        stats["l2_candidate_blocks"] = int(eligible.sum())

        # budget stated in the same units as the measurement: K = target_c * L pairs,
        # which is K / b2^2 blocks at this level
        want_blocks = max(1, int(math.ceil(cfg.target_c * L / (cfg.b2 ** 2))))
        k2 = min(want_blocks, stats["l2_candidate_blocks"])
        stats["l2_budget_blocks"] = want_blocks
        flat = s2.masked_fill(~eligible, float("-inf")).view(B, -1)
        top = flat.topk(k2, dim=-1).indices
        m2 = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, top, True).view(B, n2, n2)
        m2 = m2 & eligible
        stats["l2_kept_blocks"] = int(m2.sum())

        # ---- L3: materialise pair features ONLY inside surviving b2 blocks ----
        bi, bj = m2[0].nonzero(as_tuple=True)
        off = torch.arange(cfg.b2, device=tok.device)
        # every (row, col) cell inside a surviving block -- the full b2 x b2 cross
        # product, not just its diagonal
        ii = (bi[:, None, None] * cfg.b2 + off[None, :, None]).expand(-1, cfg.b2, cfg.b2).reshape(-1)
        jj = (bj[:, None, None] * cfg.b2 + off[None, None, :]).expand(-1, cfg.b2, cfg.b2).reshape(-1)
        ii = ii.clamp(max=L - 1)
        jj = jj.clamp(max=L - 1)
        keep = (jj - ii) >= cfg.min_sep
        ii, jj = ii[keep], jj[keep]
        stats["l3_pairs"] = ii.numel()
        stats["dense_pairs"] = L * (L - 1) // 2
        stats["frac_of_dense"] = ii.numel() / max(stats["dense_pairs"], 1)
        stats["effective_c"] = ii.numel() / L

        z = self.pair_proj(torch.cat([tok[0, ii], tok[0, jj]], dim=-1))

        # Gate each pair feature by the score of the b2 block that selected it.
        # Without this the selector scores reach the loss only through topk, which
        # is non-differentiable -- so l1/l2 would receive NO gradient and could
        # never learn to find occupied blocks. (Caught by the correctness tests;
        # the speed benchmark ran under no_grad and never exposed it.)
        sel_b2 = s2[0, bi, bj].repeat_interleave(cfg.b2 * cfg.b2)[keep]
        sel_b1 = s1[0, bi // r, bj // r].repeat_interleave(cfg.b2 * cfg.b2)[keep]
        gate = torch.sigmoid(sel_b2 + sel_b1).unsqueeze(-1)
        z = z * gate

        for layer in self.tri:
            z = z + layer(z)
        logits = self.out(z).squeeze(-1)

        # Expose the block scores so an auxiliary block-occupancy loss can
        # supervise the selectors directly -- the primary training signal for
        # them, since topk gives none.
        aux = {"l1_scores": s1, "l2_scores": s2, "l1_mask": m1, "l2_mask": m2,
               "b2_index": (bi, bj), "b1_index": (bi // r, bj // r)}
        return logits, (ii, jj), {**stats, "aux": aux}


class DensePairTrack(nn.Module):
    """AlphaFold3-style dense baseline, for the memory comparison."""

    def __init__(self, cfg: HPTConfig):
        super().__init__()
        self.pair_proj = nn.Linear(2 * cfg.d_model, cfg.d_pair)
        self.tri = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(cfg.d_pair),
                          nn.Linear(cfg.d_pair, cfg.d_pair), nn.ReLU(),
                          nn.Linear(cfg.d_pair, cfg.d_pair))
            for _ in range(cfg.n_tri_layers)])
        self.out = nn.Linear(cfg.d_pair, 1)

    def forward(self, tok: torch.Tensor):
        B, L, D = tok.shape
        z = self.pair_proj(torch.cat(
            [tok[:, :, None].expand(B, L, L, D), tok[:, None, :].expand(B, L, L, D)], -1))
        for layer in self.tri:
            z = z + layer(z)
        return self.out(z).squeeze(-1), None, {"dense_pairs": L * L}
