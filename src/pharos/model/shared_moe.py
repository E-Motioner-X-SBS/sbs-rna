#!/usr/bin/env python3
"""Many experts that share one network, and a router that picks 1 to all of them.

Two departures from the MoE in `moe.py`, for two measured reasons.

**Experts share a network.** In `moe.py` each of E experts owns a full SwiGLU:
`3 * d_model * d_expert` parameters, so experts are expensive and E stays small
(48 at d=640 is already 17.7M per block). Here every expert reads and writes
through **one shared SwiGLU** and differs only by a gain and a bias over its
hidden units. The cost of an expert drops from `3*d*d_ff` to `4*d_ff + d`,
which at d_ff=768 is 3.7k against 1.47M -- four hundred times cheaper -- so
**256 experts cost a twentieth of what 48 independent ones did**.

That is also the better inductive bias for this problem. RNA feed-forward
computation is largely shared -- the same base chemistry, the same backbone --
and what differs between a tRNA elbow and a ribosomal expansion segment is a
modulation of it, not a different function. Independent experts must relearn
the shared part E times from a corpus that cannot afford it; a shared network
learns it once and every routed token contributes gradient to it.

**The router picks a variable number.** Fixed top-k spends identical compute on
a poly-U tract and a four-way junction. Here the router takes every expert whose
probability clears a threshold, which is nucleus (top-p) selection: a token the
router is confident about fires one expert, an ambiguous one fires many, up to
all E. The argmax always fires, so the width is 1..E and never 0.

Variable width costs nothing structurally, because `GroupedExperts` already
takes an arbitrary list of (token, expert) pairs rather than a fixed-k tensor --
the permutation does not care how many pairs each token contributed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import RouterFeatures


@dataclass
class SharedMoEConfig:
    d_model: int = 640
    d_expert: int = 512          # the SHARED hidden width, now affordable
    n_experts: int = 256
    #: Kept so existing configs still load; modulation has no rank.
    rank: int = 16
    n_shared: int = 1            # always-on dense expert
    #: Nucleus routing. An expert fires when its probability clears
    #: `threshold_rel / n_experts` -- RELATIVE to uniform, not absolute. An
    #: absolute threshold is not scale-free: at init every expert sits at 1/E,
    #: so 0.02 fires all 32 experts and none of 256, and the routing width at
    #: initialisation depends on E rather than on the router. Relative to
    #: uniform, `1.0` means "fire the experts this token prefers over chance",
    #: which behaves the same at any E. `max_k` caps the width so one
    #: pathological token cannot make a step cost E times the rest.
    threshold_rel: float = 1.0
    max_k: int = 32
    n_length_bins: int = 6
    d_router_extra: int = 8
    balance_weight: float = 0.01
    dropout: float = 0.0


class SharedAdapterExperts(nn.Module):
    """One SwiGLU, E per-expert modulations, arbitrary (token, expert) pairs.

    **Why modulation and not a low-rank adapter.** A rank-r adapter gives each
    expert a `(d, r)` matrix, and applying it needs that matrix contracted
    against each pair's own token. Gathering per pair materialises `(M, d, r)`:
    at 8,192 tokens routed eight ways that is (65536, 640, 16), gigabytes for a
    single gather, four times per block. Padding the pairs into an `(E, cap, d)`
    buffer avoids the gather but allocates `E * cap` rows for `M` pairs, and at
    E=256 with uneven groups that is worse. Both OOMed an 80 GiB card at step 0
    on the smallest batch available.

    Modulation has neither failure. Each expert owns a gain and a bias
    **vector** over the shared hidden units, so the per-pair gather is
    `(M, 2f)` -- exactly the size of the activation it multiplies, which has to
    exist anyway. Peak memory becomes independent of E: 256 experts cost the
    same per step as 8, and the FLOPs are those of one dense SwiGLU over M
    pairs rather than over E*cap padded rows.

    It is also the right capacity here. The experts share a SwiGLU because RNA
    feed-forward computation is largely shared -- same chemistry, same backbone
    -- and what separates a tRNA elbow from a ribosomal expansion segment is
    which features get amplified, not a different function. A gain and a bias
    per expert say exactly that, over 2f=1536 hidden units: the gain on the
    SwiGLU's gate half changes *what the expert passes*, the gain on its value
    half changes *what it carries*, and they are learned independently.
    """

    def __init__(self, cfg: SharedMoEConfig):
        super().__init__()
        d, f, E = cfg.d_model, cfg.d_expert, cfg.n_experts
        self.E, self.f = E, f
        self.w1 = nn.Parameter(torch.empty(d, 2 * f))
        self.w2 = nn.Parameter(torch.empty(f, d))
        nn.init.normal_(self.w1, std=d ** -0.5)
        nn.init.normal_(self.w2, std=f ** -0.5)
        # Zero gain and zero bias: every expert starts as exactly the shared
        # network and differentiates only as it learns. Random modulation at
        # init hands the router E arbitrary functions and no reason to prefer
        # any of them, which is a harder problem than the one we want solved.
        self.gain = nn.Parameter(torch.zeros(E, 2 * f))
        self.bias = nn.Parameter(torch.zeros(E, 2 * f))
        self.out_gain = nn.Parameter(torch.zeros(E, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, tokens: torch.Tensor, row: torch.Tensor,
                expert_idx: torch.Tensor, weight: torch.Tensor,
                n_out: int) -> torch.Tensor:
        d = tokens.shape[-1]
        out = tokens.new_zeros(n_out, d)
        if int(row.shape[0]) == 0:
            return out
        # No sort and no permutation: every pair runs through the same shared
        # matmul, so the only per-expert lookups are three vector gathers.
        x = tokens[row]                                       # (M, d)
        h = x @ self.w1                                       # (M, 2f) shared
        h = h * (1.0 + self.gain[expert_idx]) + self.bias[expert_idx]
        a, b = h.chunk(2, dim=-1)
        y = (F.silu(a) * b) @ self.w2                         # (M, d) shared
        y = y * (1.0 + self.out_gain[expert_idx])
        return out.index_add_(0, row, self.drop(y) * weight.unsqueeze(-1))


class SharedMoEFeedForward(nn.Module):
    """Nucleus-routed shared-adapter experts plus an always-on dense expert."""

    def __init__(self, cfg: SharedMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.experts = SharedAdapterExperts(cfg)
        self.shared = nn.ModuleList([
            nn.Sequential(nn.Linear(cfg.d_model, 2 * cfg.d_expert, bias=False),
                          nn.GLU(dim=-1),
                          nn.Linear(cfg.d_expert, cfg.d_model, bias=False))
            for _ in range(cfg.n_shared)])
        d_cond = cfg.n_length_bins + cfg.d_router_extra
        self.gate = nn.Linear(cfg.d_model + d_cond, cfg.n_experts, bias=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.expert_bias = nn.Parameter(torch.zeros(cfg.n_experts))

    def route(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor]:
        """Nucleus selection -> (row, expert, weight) over a variable width.

        Every expert above the threshold fires. The argmax is forced in so the
        width is never zero, and `max_k` bounds it above. Weights renormalise
        over the chosen set, so a token firing one expert and a token firing
        twenty both contribute a convex combination.
        """
        cfg = self.cfg
        N, E = probs.shape
        keep = probs >= cfg.threshold_rel / E
        top1 = probs.argmax(-1, keepdim=True)
        keep.scatter_(1, top1, True)
        if cfg.max_k < E:
            # keep only the max_k largest among those above threshold
            kth = probs.topk(cfg.max_k, dim=-1).values[:, -1:]
            keep &= probs >= kth
            keep.scatter_(1, top1, True)
        row, expert = keep.nonzero(as_tuple=True)
        w = probs[row, expert]
        denom = torch.zeros(N, device=probs.device, dtype=probs.dtype)
        denom.index_add_(0, row, w)
        return row, expert, w / denom[row].clamp_min(1e-9)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                feats: Optional[RouterFeatures] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        B, L, D = x.shape
        h = self.norm(x)
        cond = (feats or RouterFeatures()).vector(
            B, x.device, cfg.n_length_bins, cfg.d_router_extra)
        gin = torch.cat([h, cond.unsqueeze(1).expand(B, L, -1).to(h.dtype)], dim=-1)
        probs = torch.softmax((self.gate(gin) + self.expert_bias).float(), dim=-1)

        out = torch.zeros_like(x)
        for s in self.shared:
            out = out + s(h)
        flat = h.reshape(-1, D)
        row, expert, w = self.route(probs.reshape(-1, cfg.n_experts))
        out = out + self.experts(flat, row, expert, w.to(x.dtype),
                                 flat.shape[0]).view(B, L, D)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)

        m = mask if mask is not None else torch.ones(B, L, dtype=torch.bool,
                                                     device=x.device)
        n_tok = m.sum().clamp(min=1)
        mf = m.reshape(-1).float()
        frac = torch.zeros(cfg.n_experts, device=x.device)
        frac.index_add_(0, expert, mf[row])
        width = torch.zeros(flat.shape[0], device=x.device)
        width.index_add_(0, row, torch.ones_like(w, dtype=torch.float32))
        frac = frac / (n_tok * width.mean().clamp(min=1.0))
        pbar = (probs * m.unsqueeze(-1)).sum((0, 1)) / n_tok
        balance = cfg.n_experts * (frac * pbar).sum()
        return out + x, {
            "balance_loss": cfg.balance_weight * balance,
            "expert_usage": frac.detach(),
            "router_entropy": (-(pbar.clamp_min(1e-9).log() * pbar).sum()).detach(),
            "token_router_entropy": (
                (-(probs.clamp_min(1e-9).log() * probs).sum(-1) * m).sum()
                / n_tok).detach(),
            #: what the fixed-k version could not report: how wide routing went
            "mean_width": (width * mf).sum().div(n_tok).detach(),
            "max_width": width.max().detach(),
        }

    @property
    def n_active_params(self) -> int:
        """Parameters touched by a token at the AVERAGE routing width.

        Variable width makes this a mean rather than a constant, which is the
        point: a simple token is cheaper than a hard one.
        """
        cfg = self.cfg
        shared = sum(p.numel() for e in self.shared for p in e.parameters())
        trunk = self.experts.w1.numel() + self.experts.w2.numel()
        per_adapter = 2 * 2 * cfg.d_expert + cfg.d_model   # gain, bias, out_gain
        typical = max(1, cfg.max_k // 4)
        gate = sum(p.numel() for p in self.gate.parameters())
        return shared + trunk + typical * per_adapter + gate + cfg.n_experts
