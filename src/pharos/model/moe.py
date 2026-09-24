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

import math

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

    #: Upper edge of the length binning, in nucleotides. 0 keeps the original
    #: octave bins and is the default, so a run in flight is unaffected.
    #:
    #: The octave bins saturate. `floor(log2(L/32))` clamped to `n_bins - 1`
    #: gives bins 0..4 one octave each (32-64, 64-128, ... 512-1024) and makes
    #: bin 5 a catch-all for EVERYTHING above 1024 -- 4.5 octaves, 1,024 to the
    #: corpus's 4,608, in a single one-hot. Two consequences, both measured:
    #: in stage 1 the length filter is 20-1024, so bin 5 requires a chain of
    #: exactly 1,024 and is dead -- one sixth of the length conditioning
    #: carries no information, while bin 1 takes 58% of chains. In stage 5,
    #: where chains run to 4,608, every long chain lands in one bin, which is
    #: precisely the regime the pair track exists for.
    #:
    #: Set `length_bin_max` and the bins spread evenly in log2 over
    #: [32, length_bin_max] instead: at 6 bins and 4,608 that is 1.195 octaves
    #: each, edges 32/73/167/381/871/1989/4608, and the top bin is reachable.
    length_bin_max: int = 0

    def vector(self, B: int, device, n_bins: int, d_extra: int) -> torch.Tensor:
        """(B, n_bins + d_extra) — one-hot length bin plus the scalar block."""
        bins = torch.zeros(B, n_bins, device=device)
        if self.length is not None:
            # geometric bins: chain length spans 32..4608 and the interesting
            # structure is multiplicative, not additive
            lg = torch.log2(self.length.to(device).float().clamp(min=32.0) / 32.0)
            if self.length_bin_max > 32:
                span = math.log2(self.length_bin_max / 32.0)
                lg = lg * (n_bins / max(span, 1e-6))
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
        # `torch.as_tensor` rather than `float(self.recycle > 0)`: under
        # `torch.compile` the loop index that sets `recycle` is a SymInt, so the
        # Python comparison yields a SymBool and inductor fails on ToFloat. The
        # tensor form traces for both a plain int and a symbolic one.
        extra[:, -1] = torch.as_tensor(self.recycle, device=device).gt(0).float()
        return torch.cat([bins, extra], dim=-1)


class Expert(nn.Module):
    """A narrow SwiGLU feed-forward. Used for the always-on shared experts."""

    def __init__(self, d_model: int, d_expert: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, 2 * d_expert, bias=False)
        self.w2 = nn.Linear(d_expert, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w1(x).chunk(2, dim=-1)
        return self.drop(self.w2(F.silu(a) * b))


class GroupedExperts(nn.Module):
    """`n_experts` SwiGLU experts as stacked weights and ONE batched matmul.

    A `ModuleList` of experts is a Python loop, and fine-grained MoE makes that
    loop long: 32 experts x 16 blocks is 512 sequential launches per forward,
    each on a slice too small to occupy the GPU. Profiled at d=512, 32k tokens:
    the MoE was **12.9 ms of a 22.9 ms block**, so 55% of the trunk, almost all
    of it launch overhead rather than arithmetic.

    Here the weights live in `(E, d, 2de)` and `(E, de, d)` tensors, tokens are
    permuted into expert-contiguous order, and the whole thing is two `bmm`
    calls. The permutation pads each expert's group to the largest group, which
    is cheap precisely because the load-balancing loss exists -- the groups are
    near-uniform by construction, so the padding waste is small and bounded.
    """

    def __init__(self, n_experts: int, d_model: int, d_expert: int,
                 dropout: float = 0.0):
        super().__init__()
        self.n_experts = n_experts
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, 2 * d_expert))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_expert, d_model))
        for w in (self.w1, self.w2):
            nn.init.normal_(w, std=d_model ** -0.5)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, row: torch.Tensor,
                expert_idx: torch.Tensor, weight: torch.Tensor,
                n_out: int) -> torch.Tensor:
        """Scatter-add each routed pair's expert output into its source row.

        A "pair" is one (token, chosen expert) assignment, so a token routed to
        `top_k` experts contributes `top_k` pairs. `tokens` is the UNREPLICATED
        `(N, d)` hidden state and `row[i]` selects the token for pair `i`;
        `expert_idx[i]` is the expert it chose and `weight[i]` its gate weight.

        Taking `row` rather than a pre-replicated `(N*top_k, d)` tensor is a
        memory decision, not a style one: at 32k tokens and top-4 that replica
        is 131,072 x 512, retained for backward in each of 16 blocks, and
        building it OOMed an 80 GiB card. The permutation gathers straight from
        `tokens` instead, so the replica never exists.
        """
        M = int(row.shape[0])
        d = tokens.shape[-1]
        E = self.n_experts
        out = tokens.new_zeros(n_out, d)
        if M == 0:
            return out
        order = torch.argsort(expert_idx)
        e_sorted = expert_idx[order]
        counts = torch.bincount(e_sorted, minlength=E)
        # `int(...)` is a device synchronisation, and under `torch.compile` a
        # graph break -- once per MoE block per loop, so 32 times a step at 16
        # blocks and 2 loops. The textbook fix is a fixed capacity
        # `ceil(f * M / E)` with the overflow dropped, which needs nothing read
        # back from the device.
        #
        # MEASURED, and it is worse. Compiled, 48x512: 0.289 s/step exact
        # against 0.377 / 0.364 / 0.389 at capacity factors 1.25 / 1.5 / 2.0,
        # so 25% SLOWER at the best of them, and slower eager too. Selecting
        # the survivors is itself data-dependent -- `e_sorted[keep]`,
        # `row[order][keep]`, `weight[order][keep]` are three boolean gathers
        # with their own syncs and allocations -- so it trades one
        # synchronisation for several. And there was nothing to win: the drop
        # fraction was 0.00% at every factor tried, because the load-balance
        # loss keeps the groups near-uniform and `counts.max()` is already
        # close to `M / E`.
        #
        # So the sync stays. It is exact, no token is ever dropped, and the
        # obvious optimisation has been tried and rejected on measurement.
        cap = int(counts.max())
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.arange(M, device=tokens.device) - starts[e_sorted]

        buf = tokens.new_zeros(E, cap, d)
        buf[e_sorted, slot] = tokens[row[order]]          # gather, not replicate
        a, b = torch.bmm(buf, self.w1).chunk(2, dim=-1)
        y = self.drop(torch.bmm(F.silu(a) * b, self.w2))          # E, cap, d

        contrib = y[e_sorted, slot] * weight[order].unsqueeze(-1)
        return out.index_add_(0, row[order], contrib)


class MoEFeedForward(nn.Module):
    """Top-k routed experts plus a shared expert, conditioned on RNA type."""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.experts = GroupedExperts(cfg.n_experts, cfg.d_model, cfg.d_expert,
                                      cfg.dropout)
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
        N = flat_h.shape[0]
        fi = topi.reshape(-1, cfg.top_k)
        fv = topv.reshape(-1, cfg.top_k).to(x.dtype)
        # one (token, expert) pair per chosen slot, all routed in a single
        # batched matmul rather than one launch per expert
        rows = torch.arange(N, device=x.device).repeat_interleave(cfg.top_k)
        routed = self.experts(flat_h, rows, fi.reshape(-1), fv.reshape(-1), N)
        out = out + routed.view(B, L, D)
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
            # Entropy of the MEAN routing distribution. At perfect balance
            # this is log(n_experts) BY CONSTRUCTION -- it says the load is
            # even and nothing about whether any token is routed sharply.
            "router_entropy": (-(pbar.clamp_min(1e-9).log() * pbar).sum()).detach(),
            # Entropy of each token's OWN distribution, averaged. This is the
            # one that distinguishes a specialising router from a collapsed
            # one: both give a uniform mean, and only the collapsed one gives
            # every token a uniform distribution of its own. At log(n_experts)
            # the MoE is an expensive dense model.
            "token_router_entropy": (
                (-(probs.clamp_min(1e-9).log() * probs).sum(-1)
                 * m).sum() / n_tok).detach(),
        }

    @property
    def n_active_params(self) -> int:
        """Parameters touched per token: shared experts plus `top_k` routed.

        `expert_bias` counts. It has one entry per expert and every entry
        participates in every token's routing decision, so leaving it out
        under-reports the active count by `n_experts` per block -- which is how
        a cross-check against an independent analytic count found it.
        """
        per = (self.experts.w1[0].numel() + self.experts.w2[0].numel())
        shared = sum(p.numel() for e in self.shared for p in e.parameters())
        gate = sum(p.numel() for p in self.gate.parameters())
        return (shared + self.cfg.top_k * per + gate
                + self.expert_bias.numel() + self.cfg.d_model * 2)
