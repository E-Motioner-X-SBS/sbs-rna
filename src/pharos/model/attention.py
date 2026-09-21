#!/usr/bin/env python3
"""The three mixers of the hybrid stack (ARCHITECTURE v0.2 §5.1).

The block pattern is `[GDN, GDN, SWA, GDN, GDN, SWA, GDN, FULL]`, twice:

    GDN   Gated DeltaNet, linear, O(L)        10 of 16 blocks
    SWA   sliding-window softmax, w=128       4 of 16
    FULL  full attention with physics bias    2 of 16

The justification is a property of RNA, not a generic efficiency argument: an
A-form helix is locally periodic and low-information, so linear attention
carries it; junctions, pseudoknots and kissing loops need global mixing and are
rare. Full attention is reserved for the two blocks that feed the pair track.

§5.1 is also explicit that at L=2048 the attention terms are **3.7% of training
FLOPs**, so this stack earns its place on memory and inference latency, not on
training throughput. Nothing here should be justified by the wrong number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _heads(x: torch.Tensor, h: int) -> torch.Tensor:
    """(B, L, D) -> (B, h, L, D/h)."""
    B, L, D = x.shape
    return x.view(B, L, h, D // h).transpose(1, 2)


def _merge(x: torch.Tensor) -> torch.Tensor:
    """(B, h, L, d) -> (B, L, h*d)."""
    B, h, L, d = x.shape
    return x.transpose(1, 2).reshape(B, L, h * d)


class GatedDeltaNet(nn.Module):
    """Linear attention with a delta rule and a forget gate, O(L) in time.

    The recurrence, per head, with state `S` of shape (d_k, d_v)::

        S_t = a_t * S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T
        o_t = S_t^T q_t

    `a_t` in (0,1) is the forget gate and `b_t` in (0,1) the write strength. The
    delta term `(I - b k k^T)` is what separates this from plain linear
    attention: it *erases* the component of the state already associated with
    `k` before writing, so repeated keys overwrite rather than accumulate. On a
    helix, where the same local motif recurs every ~11 nt, accumulation is
    exactly the failure mode -- the state saturates on the periodic signal and
    stops carrying anything else.

    Implemented chunkwise: the within-chunk part is a masked matmul and the
    across-chunk part is a loop over `L / chunk` steps, which at L=4,608 and
    chunk 64 is 72 iterations of fully batched work.
    """

    def __init__(self, d_model: int, n_heads: int = 8, d_head: Optional[int] = None,
                 chunk: int = 128, dropout: float = 0.0):
        # chunk 128, measured. The cost has two terms pulling opposite ways:
        # L/chunk sequential iterations, each with O(chunk^2) within-chunk work.
        # At d=512, L=1024: 64 -> 13.4 ms, **128 -> 8.8 ms**, 256 -> 12.7,
        # 512 -> 20.5. The chunked form is exact at any size, so this is purely
        # a throughput choice and `test_attention.py` checks it at 8/16/64.
        super().__init__()
        self.h = n_heads
        self.dk = d_head or (d_model // n_heads)
        self.chunk = chunk
        inner = self.h * self.dk
        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.a = nn.Linear(d_model, self.h)        # forget gate, per head
        self.b = nn.Linear(d_model, self.h)        # write strength, per head
        self.g = nn.Linear(d_model, inner)         # output gate
        self.out = nn.Linear(inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        # start near "remember everything, write moderately"
        nn.init.constant_(self.a.bias, 3.0)
        nn.init.constant_(self.b.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                **_) -> torch.Tensor:
        """Chunkwise-parallel evaluation of the recurrence in the class docstring.

        The rearrangement, which `test_attention.py` checks against a
        step-by-step implementation to 1e-8:

        **Decay is factored out first.** Writing `S_i = g_i S~_i` with
        `g_i = prod_{j<=i} a_j` turns the gated recurrence into an ungated delta
        rule on `S~` with rescaled values `v~_i = v_i / g_i`, because the decay
        applies to the erase and the carry identically and so commutes out of
        both. What remains is the plain delta rule, which has a closed chunk
        form.

        **The chunk is then one triangular solve.** With `A = tril(diag(b) K K^T, -1)`
        strictly lower triangular, `T = (I + A)^{-1} diag(b)` is the UT transform
        of the chunk's rank-1 updates: it converts `C` sequential erase-write
        steps into a single low-rank correction `S~_C = S~_0 + K^T U'`, where
        `U' = T V~ - (T K) S~_0`. `(I + A)` is unit lower triangular, so the
        solve is exact and cheap.

        The masks are inclusive of the diagonal, because `o_i` reads `S_i`
        *after* position `i`'s own write.
        """
        B, L, _ = x.shape
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        q = _heads(self.q(x), self.h)
        k = _heads(self.k(x), self.h)
        v = _heads(self.v(x), self.h)
        # L2-normalised keys keep `I - b k k^T` contractive, so the state cannot
        # grow without bound over thousands of steps
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        a = torch.sigmoid(self.a(x)).transpose(1, 2)                    # B,h,L
        b = torch.sigmoid(self.b(x)).transpose(1, 2)
        if mask is not None:
            m = mask.unsqueeze(1).to(x.dtype)
            b = b * m
            a = a * m + (1.0 - m)          # padding neither writes nor forgets

        C = self.chunk
        pad = (-L) % C
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
            a = F.pad(a, (0, pad), value=1.0)
            b = F.pad(b, (0, pad), value=0.0)
        n_chunk = (L + pad) // C
        dv = v.shape[-1]
        q, k, v = (t.view(B, self.h, n_chunk, C, -1) for t in (q, k, v))
        a, b = (t.view(B, self.h, n_chunk, C) for t in (a, b))

        eye = torch.eye(C, device=x.device, dtype=x.dtype)
        tri_incl = torch.tril(torch.ones(C, C, device=x.device, dtype=x.dtype))
        tri_strict = torch.tril(torch.ones(C, C, device=x.device, dtype=x.dtype), -1)

        S = x.new_zeros(B, self.h, self.dk, dv)
        outs = []
        for c in range(n_chunk):
            qc, kc, vc = q[:, :, c], k[:, :, c], v[:, :, c]      # B,h,C,d
            ac, bc = a[:, :, c], b[:, :, c]                       # B,h,C
            g = torch.cumprod(ac.clamp_min(1e-8), dim=-1)         # within-chunk decay
            gi = g.unsqueeze(-1)
            vt = vc / gi.clamp_min(1e-12)                         # v~

            KK = torch.einsum("bhid,bhjd->bhij", kc, kc)
            A = tri_strict * (bc.unsqueeze(-1) * KK)
            T = torch.linalg.solve_triangular(
                eye + A, torch.diag_embed(bc), upper=False, unitriangular=True)
            W = T @ kc                                            # B,h,C,dk
            U = T @ vt                                            # B,h,C,dv
            Up = U - W @ S                                        # B,h,C,dv

            QK = tri_incl * torch.einsum("bhid,bhjd->bhij", qc, kc)
            o = (qc @ S + QK @ Up) * gi                           # undo the rescale
            outs.append(o)
            S = (S + torch.einsum("bhid,bhie->bhde", kc, Up)) * g[..., -1:].unsqueeze(-1)

        o = torch.cat(outs, dim=2)[:, :, :L]
        o = _merge(o) * torch.sigmoid(self.g(x))
        return self.drop(self.out(o))


class SlidingWindowAttention(nn.Module):
    """Softmax attention restricted to +/- w.

    **Implemented as a dense masked attention at every length**, so the compute
    is O(L^2) with a band mask, not O(L*w). The docstring here used to claim a
    chunked banded path for `L > 2*window` and there has never been one in the
    code -- the result is identical, the cost is not. At the MLM context of
    1,024 that is 4x the attention work these four blocks need; at the 4,608
    structural context it is 18x, and the `(B, 1, L, L)` mask is materialised
    on top.

    It is left as it is for now because it is correct and because the measured
    bottleneck is elsewhere -- a profile of a stage-1 step puts 18% of GPU time
    in tensor-core GEMMs and the rest in elementwise work and copies, which is
    what compiling the trunk addresses. Banding is worth doing for stage 5,
    where L is four times larger; it is not worth doing blind.
    """

    def __init__(self, d_model: int, n_heads: int = 8, window: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.h, self.w = n_heads, window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                **_) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = (_heads(t, self.h) for t in self.qkv(x).chunk(3, dim=-1))
        idx = torch.arange(L, device=x.device)
        band = (idx[None, :] - idx[:, None]).abs() <= self.w
        if mask is not None:
            band = band[None] & mask[:, None, :] & mask[:, :, None]
            attn_mask = band.unsqueeze(1)
        else:
            attn_mask = band[None, None]
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.drop if self.training else 0.0)
        return self.out(_merge(o))


class FullAttention(nn.Module):
    """Global softmax attention with an additive physics bias.

    The bias is where the Hamiltonian reaches the token track: a screened
    electrostatic term computed from the previous loop's distance estimate
    (§6.2) is added to the logits, so the ionic condition changes what attends
    to what rather than only re-scoring the output. At recycle 0 there is no
    distance estimate and the bias is zero -- the same circularity §5.3 records
    for the router.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.bias_scale = nn.Parameter(torch.zeros(n_heads))
        self.drop = dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                pair_bias: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = (_heads(t, self.h) for t in self.qkv(x).chunk(3, dim=-1))
        am = None
        if mask is not None:
            am = (mask[:, None, :] & mask[:, :, None]).unsqueeze(1)
        if pair_bias is not None:
            bias = pair_bias.unsqueeze(1) * self.bias_scale.view(1, -1, 1, 1)
            if am is not None:
                bias = bias.masked_fill(~am, float("-inf"))
            am = bias
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=am, dropout_p=self.drop if self.training else 0.0)
        return self.out(_merge(o))


#: §5.1's period-8 pattern, two cycles.
BLOCK_PATTERN: Tuple[str, ...] = ("gdn", "gdn", "swa", "gdn", "gdn", "swa", "gdn", "full")


def make_mixer(kind: str, d_model: int, n_heads: int, window: int,
               dropout: float) -> nn.Module:
    if kind == "gdn":
        return GatedDeltaNet(d_model, n_heads, dropout=dropout)
    if kind == "swa":
        return SlidingWindowAttention(d_model, n_heads, window, dropout)
    if kind == "full":
        return FullAttention(d_model, n_heads, dropout)
    raise ValueError(f"unknown mixer {kind!r}")
