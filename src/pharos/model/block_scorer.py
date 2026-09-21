#!/usr/bin/env python3
"""The block-detection scorer — the largest unvalidated assumption in PHAROS.

ARCHITECTURE v0.2 §7 replaces a dense L x L pair track with three coarse-to-fine
levels that select contact-bearing **blocks** rather than ranking pairs, and
justifies it with a measurement: block occupancy at b=4 is 1.34% on long chains
and *falls* with L, while a flat top-K proposal recovers only ~20% of contacts.
The reference implementation proved the cost and the structure. It did not prove
the thing the whole design rests on: **that a model can find the occupied blocks
from sequence alone.** Every recall number published so far came from random
weights. This is the model that answers it.

Why it is block-level rather than residue-level
-----------------------------------------------
The selector's output is defined on blocks, so that is where its capacity
belongs. Residues are encoded with dilated convolutions -- O(L), and local
complementarity is a local computation -- and the result is pooled to blocks
*before* any attention. At b1=16 and L=4608 that is 288 blocks, so attention
over them is 83k entries instead of 21M, and the quadratic term costs nothing.
This is the same argument §7 makes for the pair track itself, applied to the
thing that drives it.

What must be beaten
-------------------
A selector that scores nothing but sequence separation would already do well:
contacts concentrate near the diagonal, and `CoarseBlockScorer` in the reference
carries an explicit separation prior. So "the learned scorer achieves recall R"
is not a result on its own. `train_block_scorer.py` reports it against three
baselines at the identical budget -- random, separation-only, and this model
with its separation prior ablated -- because the only interesting quantity is
how much the *sequence* contributes over the *geometry of a chain*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ScorerConfig:
    d_model: int = 192
    d_block: int = 128
    d_pair: int = 64
    n_conv: int = 6               # dilated residue layers; receptive field ~2*3^n
    n_attn: int = 2               # block-level self-attention layers
    n_heads: int = 4
    b1: int = 16
    b2: int = 4
    min_sep: int = 4
    n_symbols: int = 13
    n_mod: int = 371              # 370 species + NO_MOD
    d_chem: int = 24
    dropout: float = 0.1
    #: operating budgets, from the measurements they came from
    keep_frac_l1: float = 0.12    # b1=16 occupancy measures 7.26%
    target_c: float = 24.0        # D23, closed on the raw archive at max 23.30
    use_sep_prior: bool = True    # ablation switch; see the module docstring
    #: Minimum block separation at L1, IN L1 BLOCKS.
    #:
    #: `max(1, min_sep // b)` gives the same physical separation in different
    #: units at the two levels: `4 // 4 = 1` block is 4 residues at L2, and
    #: `max(1, 4 // 16)` is 1 block = **16 residues** at L1. The clamp is what
    #: does it. The cascade measurement (§7.4a) shows the cost: 18.9% of true
    #: L2 blocks sit inside a single L1 block, have no valid parent, and are
    #: unreachable at any budget -- keeping every L1 pair still ceilings at
    #: 0.811.
    #:
    #: 0 admits the diagonal and lets L2's own separation filter. It is NOT the
    #: default, because flipping it under weights trained with the diagonal
    #: masked makes the cascade WORSE at the shipped budget (0.264 -> 0.247):
    #: the scorer has never been asked to score a diagonal block, and arbitrary
    #: scores there displace off-diagonal blocks it ranks well. Set it to 0 and
    #: RETRAIN; do not set it to 0 alone.
    l1_sep_blocks: int = 1

    def sep_blocks(self, level: str) -> int:
        """Minimum block separation for `level`, in that level's block units."""
        if level == "l1":
            return self.l1_sep_blocks
        return max(1, self.min_sep // self.b2)


class ResidueEncoder(nn.Module):
    """Token + modification + chemistry -> per-residue features, in O(L).

    Dilated convolutions rather than attention, for two reasons. The
    computation this stage has to do is local -- what base is this, what is
    around it, is this stretch complementary to something nearby -- and it has
    to run at L up to 4,608 on chains that are 95% shorter than that, where
    attention's constant factor dominates. Dilation 1,3,9,... reaches a
    receptive field of ~2*3^n_conv, which at n_conv=6 is the whole of a typical
    chain.
    """

    def __init__(self, cfg: ScorerConfig):
        super().__init__()
        d = cfg.d_model
        self.tok = nn.Embedding(cfg.n_symbols, d)
        # modification embedding, keyed by CCD species (VOCAB.md). Index 0 is
        # "unmodified" and is zeroed so an unmodified residue is exactly its
        # token embedding -- 99.4% of residues, and they should not pay for
        # the existence of the other 0.6%.
        self.mod = nn.Embedding(cfg.n_mod, d, padding_idx=0)
        self.chem = nn.Linear(cfg.d_chem, d)
        self.norm_in = nn.LayerNorm(d)
        self.convs = nn.ModuleList()
        for i in range(cfg.n_conv):
            dil = 3 ** min(i, 5)
            self.convs.append(nn.Sequential(
                nn.Conv1d(d, 2 * d, 3, padding=dil, dilation=dil),
                nn.GLU(dim=1),
                nn.Dropout(cfg.dropout)))
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(cfg.n_conv)])

    def forward(self, tokens: torch.Tensor, mod_ids: torch.Tensor,
                chem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.tok(tokens) + self.mod(mod_ids) + self.chem(chem)
        x = self.norm_in(x) * mask.unsqueeze(-1)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x.transpose(1, 2)).transpose(1, 2)
            x = norm(x + h) * mask.unsqueeze(-1)
        return x


def block_pool(x: torch.Tensor, mask: torch.Tensor, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masked mean-pool `(B, L, D)` into `(B, ceil(L/b), D)`, plus a block mask.

    Masked, not plain: a chain whose length is not a multiple of `b` has a
    partial final block, and averaging its padding toward zero would make the
    last block of every such chain look systematically different from the rest.
    """
    B, L, D = x.shape
    pad = (-L) % b
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
        mask = F.pad(mask, (0, pad))
    n = (L + pad) // b
    xm = (x * mask.unsqueeze(-1)).view(B, n, b, D).sum(2)
    cnt = mask.view(B, n, b).sum(2).clamp(min=1e-6)
    return xm / cnt.unsqueeze(-1), mask.view(B, n, b).any(2)


class BlockAttention(nn.Module):
    """Pre-norm self-attention over blocks. N <= 288 at b1, so it is free."""

    def __init__(self, cfg: ScorerConfig, d: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout,
                                          batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                nn.Dropout(cfg.dropout), nn.Linear(2 * d, d))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=~key_padding_mask,
                         need_weights=False)
        x = x + a
        return x + self.ff(self.n2(x))


class LevelScorer(nn.Module):
    """Scores every block pair at one resolution."""

    def __init__(self, cfg: ScorerConfig, b: int):
        super().__init__()
        self.cfg, self.b = cfg, b
        d = cfg.d_block
        self.proj = nn.Linear(cfg.d_model, d)
        self.attn = nn.ModuleList([BlockAttention(cfg, d) for _ in range(cfg.n_attn)])
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, cfg.d_pair, bias=False)
        self.k = nn.Linear(d, cfg.d_pair, bias=False)
        # A log-separation prior with learned amplitude and slope. Contacts
        # concentrate near the diagonal, and a model made to rediscover that
        # from scratch spends capacity on it; giving it two parameters and then
        # ABLATING them is how the experiment separates "learned the geometry
        # of a chain" from "learned the sequence".
        self.sep = nn.Parameter(torch.tensor([1.0, -0.5]))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, tok: torch.Tensor, mask: torch.Tensor,
                use_sep_prior: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        blk, bmask = block_pool(tok, mask, self.b)
        h = self.proj(blk)
        for layer in self.attn:
            h = layer(h, bmask)
        h = self.norm(h)
        q, k = self.q(h), self.k(h)
        s = torch.einsum("bnd,bmd->bnm", q, k) / math.sqrt(q.shape[-1]) + self.bias
        if use_sep_prior if use_sep_prior is not None else self.cfg.use_sep_prior:
            n = blk.shape[1]
            idx = torch.arange(n, device=tok.device, dtype=torch.float32)
            sep = (idx[None, :] - idx[:, None]).abs()
            s = s + self.sep[0] * torch.exp(self.sep[1] * torch.log1p(sep))[None]
        return s, bmask


class BlockScorer(nn.Module):
    """The two selectors of the Hierarchical Pair Track, trainable end to end."""

    def __init__(self, cfg: ScorerConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ResidueEncoder(cfg)
        self.l1 = LevelScorer(cfg, cfg.b1)
        self.l2 = LevelScorer(cfg, cfg.b2)

    def forward(self, tokens, mod_ids, chem, mask,
                use_sep_prior: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        tok = self.encoder(tokens, mod_ids, chem, mask.float())
        s1, m1 = self.l1(tok, mask.float(), use_sep_prior)
        s2, m2 = self.l2(tok, mask.float(), use_sep_prior)
        return {"l1_scores": s1, "l1_bmask": m1, "l2_scores": s2, "l2_bmask": m2}


# ---------------------------------------------------------------------------
# Labels, budgets and the recall that R1 asks for.
# ---------------------------------------------------------------------------

def occupancy_labels(contacts: torch.Tensor, b: int, n_blocks: int,
                     device=None) -> torch.Tensor:
    """`(nb, nb)` 0/1: does this block contain at least one true contact?"""
    lab = torch.zeros(n_blocks, n_blocks, device=device)
    if contacts is not None and len(contacts):
        bi = torch.div(contacts[:, 0], b, rounding_mode="floor").clamp(max=n_blocks - 1)
        bj = torch.div(contacts[:, 1], b, rounding_mode="floor").clamp(max=n_blocks - 1)
        lab[bi, bj] = 1.0
    return lab


def valid_mask(n: int, min_sep_blocks: int, bmask: torch.Tensor) -> torch.Tensor:
    """Upper triangle beyond the minimum separation, within the real length.

    `min_sep_blocks` is taken literally. It used to be clamped to at least 1
    here, which silently turned L1's `4 // 16 = 0` into "one 16-residue block"
    and made the L1 diagonal permanently invalid -- see `ScorerConfig
    .l1_sep_blocks`. Callers pass `cfg.sep_blocks(level)`, which states the
    separation per level instead of deriving it from a shared residue count
    that means different things at different block sizes.
    """
    idx = torch.arange(n, device=bmask.device)
    v = (idx[None, :] - idx[:, None]) >= min_sep_blocks
    return v & bmask[:, None] & bmask[None, :]


def recall_at_budget(scores: torch.Tensor, labels: torch.Tensor,
                     valid: torch.Tensor, k: int) -> Tuple[float, float, int]:
    """`(recall, precision, k)` when the top-`k` valid blocks are kept.

    This is the number R1 is about. A missed block is unrecoverable -- no later
    level can refine a block that was never selected -- while a spurious one
    only costs compute, so recall is the metric and precision is reported
    alongside to show what the recall cost.
    """
    lab = labels * valid
    n_pos = float(lab.sum())
    if n_pos == 0:
        return float("nan"), float("nan"), 0
    s = scores.masked_fill(~valid, float("-inf")).flatten()
    k = int(max(1, min(k, int(valid.sum()))))
    top = s.topk(k).indices
    hit = float(lab.flatten()[top].sum())
    return hit / n_pos, hit / k, k


def l1_budget(n_blocks: int, valid: torch.Tensor, cfg: ScorerConfig) -> int:
    """Blocks kept at L1: a fraction of the valid upper triangle."""
    return max(1, int(cfg.keep_frac_l1 * float(valid.sum())))


def l2_budget(length: int, cfg: ScorerConfig) -> int:
    """Blocks kept at L2, from the pair budget: K = target_c * L pairs."""
    return max(1, int(math.ceil(cfg.target_c * length / (cfg.b2 ** 2))))


def batched_labels(contacts, b: int, n: int, B: int, device) -> torch.Tensor:
    """`(B, n, n)` occupancy labels for a whole batch in one scatter."""
    lab = torch.zeros(B, n, n, device=device)
    for i, c in enumerate(contacts):          # one scatter per chain, no kernel
        if c is not None and len(c):          # launch per element
            bi = torch.div(c[:, 0], b, rounding_mode="floor").clamp(max=n - 1)
            bj = torch.div(c[:, 1], b, rounding_mode="floor").clamp(max=n - 1)
            lab[i, bi, bj] = 1.0
    return lab


def batched_valid(n: int, min_sep_blocks: int, bmask: torch.Tensor) -> torch.Tensor:
    """`(B, n, n)` upper triangle beyond `min_sep_blocks`, within real length."""
    idx = torch.arange(n, device=bmask.device)
    tri = (idx[None, :] - idx[:, None]) >= min_sep_blocks
    return tri[None] & bmask[:, None, :] & bmask[:, :, None]


def occupancy_loss(out: Dict[str, torch.Tensor], contacts, lengths,
                   cfg: ScorerConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Recall-weighted BCE over both levels, batched.

    Occupancy is 1.34-7.26% positive depending on level and length, and the
    asymmetry is not symmetric in consequence: a missed block is unrecoverable,
    a spurious one costs compute. Positives are therefore up-weighted by
    neg/pos **per chain**, because the ratio varies by an order of magnitude
    with length and a batch-level constant would silently re-weight long chains.

    Vectorised across the batch. The per-chain Python loop this replaces cost
    43 ms/step at batch 32 and would have cost roughly 16x that at batch 512 --
    1,024 sequential BCE launches per step, with the GPU idle between them.
    The per-chain pos_weight survives the vectorisation by being folded into a
    per-element weight, `1 + (pw_i - 1) . label`, which is what `pos_weight`
    expands to anyway.
    """
    total = torch.zeros((), device=out["l1_scores"].device)
    stats: Dict[str, float] = {}
    B = out["l1_scores"].shape[0]
    for lvl, b in (("l1", cfg.b1), ("l2", cfg.b2)):
        s_all, bm = out[f"{lvl}_scores"], out[f"{lvl}_bmask"]
        n = s_all.shape[1]
        v = batched_valid(n, cfg.sep_blocks(lvl), bm)               # B,n,n
        lab = batched_labels(contacts, b, n, B, s_all.device) * v
        nv = v.flatten(1).sum(1)                                    # B
        pos = lab.flatten(1).sum(1).clamp(min=1.0)                  # B
        pw = ((nv - pos) / pos).clamp(1.0, 500.0)                   # B
        w = (1.0 + (pw - 1.0).view(B, 1, 1) * lab) * v
        per = F.binary_cross_entropy_with_logits(
            s_all, lab, weight=w, reduction="none")
        # mean over each chain's valid entries, then mean over chains that have
        # any -- identical to the per-chain loop it replaces
        has = nv > 0
        chain_mean = per.flatten(1).sum(1) / nv.clamp(min=1.0)
        lvl_loss = chain_mean[has].sum() / max(B, 1) if bool(has.any()) \
            else torch.zeros((), device=s_all.device)
        total = total + lvl_loss
        stats[f"{lvl}_loss"] = float(lvl_loss.detach())
    return total, stats
