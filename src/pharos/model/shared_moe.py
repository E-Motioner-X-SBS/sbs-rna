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
    """One SwiGLU, E per-expert modulations, merged before the network runs.

    **Merge the experts, not their outputs.** The obvious way to run E experts
    over a shared network is to build the (token, expert) pair list and push
    every pair through: k times the tokens, k times the activations, and a pair
    count that changes every step because nucleus routing is variable-width.
    That last part is fatal for `torch.compile` -- it recompiled on eight
    consecutive steps and then gave up -- and a profile of the eager version
    put 70% of GPU time in gathers, scatters and elementwise multiplies against
    19% in the matmuls that do the actual work.

    Because an expert here is a **vector**, not a network, the experts can be
    merged first: take the router's convex combination of the selected experts'
    gains and biases, then run the shared SwiGLU once on the merged modulation.
    A token routed to eight experts costs exactly what a token routed to one
    costs. Every shape is now a function of the token count alone, so the whole
    block compiles, and the mixing is a dense matmul rather than a gather.

    This is soft merging (Muqeeth et al., SMEAR), and it is not a compromise:
    merging parameters by router weight gives an *exact* gradient to the router,
    where discrete top-k needs a straight-through estimator and leaves the
    unselected experts with no signal at all.

    It is also what makes "1 to all experts" real rather than nominal. Firing
    all 512 experts is now a denser weighted sum over the same matmul -- the
    same cost as firing one -- so the cap exists for specialisation, not to
    stop the step running out of memory.
    """

    def __init__(self, cfg: SharedMoEConfig):
        super().__init__()
        d, f, E = cfg.d_model, cfg.d_expert, cfg.n_experts
        self.E, self.f, self.d = E, f, d
        self.w1 = nn.Parameter(torch.empty(d, 2 * f))
        self.w2 = nn.Parameter(torch.empty(f, d))
        nn.init.normal_(self.w1, std=d ** -0.5)
        nn.init.normal_(self.w2, std=f ** -0.5)
        # gain | bias over the 2f hidden units, then a gain over the d outputs,
        # in ONE matrix so merging is a single matmul instead of three.
        # All zero: every expert starts as exactly the shared network and
        # differentiates only as it learns. Random modulation at init hands the
        # router E arbitrary functions and no reason to prefer any of them.
        self.mod = nn.Parameter(torch.zeros(E, 4 * f + d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """`tokens` is `(N, d)`; `weights` is `(N, E)`, a convex row per token."""
        f = self.f
        mod = weights @ self.mod.to(weights.dtype)        # (N, 4f + d)
        gain, bias, out_gain = mod.split([2 * f, 2 * f, self.d], dim=-1)
        h = tokens @ self.w1                              # (N, 2f), shared
        h = h * (1.0 + gain) + bias
        a, b = h.chunk(2, dim=-1)
        y = (F.silu(a) * b) @ self.w2                     # (N, d), shared
        return self.drop(y * (1.0 + out_gain))


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

    def route(self, probs: torch.Tensor) -> torch.Tensor:
        """Nucleus selection -> a dense `(N, E)` matrix of convex row weights.

        Every expert above the threshold fires; the argmax is forced in so the
        width is never zero, and `max_k` bounds it above. Weights renormalise
        over the chosen set, so a token firing one expert and a token firing
        twenty both contribute a convex combination.

        Dense rather than a `(row, expert)` pair list, even though the rows are
        mostly zero. The pair list has a length that changes with the routing
        decisions, which makes every downstream shape dynamic and defeats
        `torch.compile`; a dense mask is the same information at a fixed shape,
        and it feeds a matmul instead of a gather.
        """
        cfg = self.cfg
        E = probs.shape[-1]
        keep = probs >= cfg.threshold_rel / E
        top1 = probs.argmax(-1, keepdim=True)
        keep.scatter_(1, top1, True)
        if cfg.max_k < E:
            kth = probs.topk(cfg.max_k, dim=-1).values[:, -1:]
            keep = keep & (probs >= kth)
            keep.scatter_(1, top1, True)
        w = probs * keep
        return w / w.sum(-1, keepdim=True).clamp_min(1e-9)

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
        w = self.route(probs.reshape(-1, cfg.n_experts))
        out = out + self.experts(h.reshape(-1, D), w.to(x.dtype)).view(B, L, D)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)

        m = mask if mask is not None else torch.ones(B, L, dtype=torch.bool,
                                                     device=x.device)
        n_tok = m.sum().clamp(min=1)
        mf = m.reshape(-1).float()
        fired = (w > 0).float()
        width = fired.sum(-1)
        frac = (fired * mf.unsqueeze(-1)).sum(0) / (
            n_tok * width.mean().clamp(min=1.0))
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
        typical = max(1, min(cfg.max_k, cfg.n_experts) // 4)
        gate = sum(p.numel() for p in self.gate.parameters())
        return shared + trunk + typical * per_adapter + gate + cfg.n_experts
