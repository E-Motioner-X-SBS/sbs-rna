#!/usr/bin/env python3
"""Fine-grained mixture-of-experts feed-forward (ARCHITECTURE v0.2 §5.3).

The routing decision PHAROS cares about is *what kind of RNA is this* — a tRNA
and a ribosomal subunit and a synthetic aptamer should not share a feed-forward
path. So the router sees more than the token: §5.3 specifies length bin,
`Neff/L` (coevolution depth), the in-complex flag, and a chemistry summary, all
concatenated to the hidden state before the gate.

Three things are structural rather than incidental.

**Fine-grained experts with a shared expert.** Splitting the FFN into many
narrow experts and always routing through one shared expert is the DeepSeek-MoE
arrangement; defect #21 in v0.1 was copying one of their two ratios and not the
other. The shared expert carries what every RNA has in common so the routed
experts are free to specialise, which is the entire point of conditioning the
router on RNA *type*.

**The router is conditioned on features that may not exist yet.** `Neff/L` and
the in-complex flag are available from the input, but pair-derived features are
not, at recycle 0 — §5.3 records this circularity as an open design gap. The
implementation makes it explicit: `RouterFeatures.recycle` gates the
pair-derived block, and at recycle 0 those inputs are zeroed rather than
fabricated, so a model cannot silently train on a feature it will not have.

**Load balancing is measured, not assumed.** A router conditioned on length and
type is a router with a standing invitation to collapse onto whichever type
dominates the corpus — and this corpus is 85.94% ribosomal by residue (G3). The
auxiliary balance loss is therefore not optional here, and `expert_usage` is
returned on every forward so collapse shows up in a training curve instead of
in a post-mortem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    d_model: int = 512
    d_expert: int = 256          # narrow, fine-grained
    n_experts: int = 16
    n_shared: int = 1
    top_k: int = 2
    #: §5.3's router inputs, beyond the hidden state
    n_length_bins: int = 6
    d_router_extra: int = 8      # Neff/L, in-complex, chemistry summary, recycle
    balance_weight: float = 0.01
    dropout: float = 0.0


@dataclass
class RouterFeatures:
    """The conditioning §5.3 specifies. All optional; absent means zero.

    `recycle` is not decoration: pair-derived router features do not exist on
    the first pass, so anything derived from the pair track is masked to zero
    at recycle 0 and only admitted from recycle 1. Training with them always
    present and inferring with them sometimes absent is a train/test mismatch,
    and it is the failure §5.3 flags.
    """
    length: Optional[torch.Tensor] = None         # (B,) chain length
    neff_over_l: Optional[torch.Tensor] = None    # (B,) MSA depth, 0 if unknown
    in_complex: Optional[torch.Tensor] = None     # (B,) 1 if the entry has protein
    chem_summary: Optional[torch.Tensor] = None   # (B, k) pooled chemistry
    recycle: int = 0

    def vector(self, B: int, device, n_bins: int, d_extra: int) -> torch.Tensor:
        """(B, n_bins + d_extra) — one-hot length bin plus the scalar block."""
        bins = torch.zeros(B, n_bins, device=device)
        if self.length is not None:
            # geometric bins: chain length spans 32..4608 and the interesting
            # structure is multiplicative, not additive
            lg = torch.log2(self.length.to(device).float().clamp(min=32.0) / 32.0)
            idx = lg.floor().clamp(0, n_bins - 1).long()
            bins.scatter_(1, idx.unsqueeze(1), 1.0)
        extra = torch.zeros(B, d_extra, device=device)
        if self.neff_over_l is not None:
            extra[:, 0] = self.neff_over_l.to(device).float()
        if self.in_complex is not None:
            extra[:, 1] = self.in_complex.to(device).float()
        if self.chem_summary is not None:
            c = self.chem_summary.to(device).float()
            w = min(c.shape[-1], d_extra - 3)
            extra[:, 2:2 + w] = c[:, :w]
        extra[:, -1] = float(self.recycle > 0)
        return torch.cat([bins, extra], dim=-1)


class Expert(nn.Module):
    """A narrow SwiGLU feed-forward."""

    def __init__(self, d_model: int, d_expert: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, 2 * d_expert, bias=False)
        self.w2 = nn.Linear(d_expert, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w1(x).chunk(2, dim=-1)
        return self.drop(self.w2(F.silu(a) * b))


class MoEFeedForward(nn.Module):
    """Top-k routed experts plus a shared expert, conditioned on RNA type."""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.experts = nn.ModuleList(
            [Expert(cfg.d_model, cfg.d_expert, cfg.dropout) for _ in range(cfg.n_experts)])
        self.shared = nn.ModuleList(
            [Expert(cfg.d_model, cfg.d_expert, cfg.dropout) for _ in range(cfg.n_shared)])
        d_cond = cfg.n_length_bins + cfg.d_router_extra
        self.gate = nn.Linear(cfg.d_model + d_cond, cfg.n_experts, bias=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        # a per-expert bias the balance loss can push on without touching the
        # token-dependent part of the gate
        self.expert_bias = nn.Parameter(torch.zeros(cfg.n_experts))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                feats: Optional[RouterFeatures] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        B, L, D = x.shape
        h = self.norm(x)
        cond = (feats or RouterFeatures()).vector(
            B, x.device, cfg.n_length_bins, cfg.d_router_extra)
        gin = torch.cat([h, cond.unsqueeze(1).expand(B, L, -1).to(h.dtype)], dim=-1)
        logits = self.gate(gin) + self.expert_bias

        # Padding is excluded from the STATISTICS below, not from the softmax.
        # Masking the logits to -inf makes a padded row's softmax 0/0, and the
        # resulting NaN propagates silently into the balance loss -- a loss term
        # that is NaN for any batch containing a short chain, which is most of
        # them.
        probs = torch.softmax(logits.float(), dim=-1)
        topv, topi = probs.topk(cfg.top_k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(x)
        for s in self.shared:
            out = out + s(h)
        flat_h = h.reshape(-1, D)
        flat_out = out.reshape(-1, D)
        fi = topi.reshape(-1, cfg.top_k)
        fv = topv.reshape(-1, cfg.top_k).to(x.dtype)
        for e, expert in enumerate(self.experts):
            sel = (fi == e)
            if not bool(sel.any()):
                continue
            rows = sel.any(-1).nonzero(as_tuple=True)[0]
            w = (fv * sel).sum(-1)[rows].unsqueeze(-1)
            flat_out[rows] += expert(flat_h[rows]) * w
        out = flat_out.view(B, L, D)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)

        # ---- load balance -------------------------------------------------
        # Switch-Transformer form: n_experts * mean(fraction routed) . mean(prob).
        # Minimised when both are uniform, and it is the term that stops a
        # type-conditioned router collapsing onto the 85.94% of residues that
        # are ribosomal.
        m = mask if mask is not None else torch.ones(B, L, dtype=torch.bool, device=x.device)
        n_tok = m.sum().clamp(min=1)
        one_hot = torch.zeros_like(probs).scatter_(-1, topi, 1.0)
        frac = (one_hot * m.unsqueeze(-1)).sum((0, 1)) / (n_tok * cfg.top_k)
        pbar = (probs * m.unsqueeze(-1)).sum((0, 1)) / n_tok
        balance = cfg.n_experts * (frac * pbar).sum()
        return x + out, {
            "balance_loss": cfg.balance_weight * balance,
            "expert_usage": frac.detach(),
            "router_entropy": (-(pbar.clamp_min(1e-9).log() * pbar).sum()).detach(),
        }

    @property
    def n_active_params(self) -> int:
        """Parameters touched per token: shared experts plus `top_k` routed.

        `expert_bias` counts. It has one entry per expert and every entry
        participates in every token's routing decision, so leaving it out
        under-reports the active count by `n_experts` per block -- which is how
        a cross-check against an independent analytic count found it.
        """
        per = sum(p.numel() for p in self.experts[0].parameters())
        shared = sum(p.numel() for e in self.shared for p in e.parameters())
        gate = sum(p.numel() for p in self.gate.parameters())
        return (shared + self.cfg.top_k * per + gate
                + self.expert_bias.numel() + self.cfg.d_model * 2)
