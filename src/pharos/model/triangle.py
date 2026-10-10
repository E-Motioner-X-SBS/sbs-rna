"""Triangle multiplicative update on the pair representation.

WHY THIS EXISTS
---------------
The model builds locally-valid RNA backbone and places it in the wrong
global arrangement: lDDT 0.269 beside TM 0.094, bonds inside tolerance,
and a fold that is wrong. Finding 100 supplied the decoder with the
geometry it was missing. This supplies the *pair representation* with the
constraint it was missing, which is a different gap in the same failure.

Measured, before anything here existed: perturb one residue's hidden
state and look at which entries of the pair tensor move. Of 528 pairs
that do not involve that residue, **0** change. `DiffusionPairFeatures`
is an outer sum `A s_i + B s_j`, so `z[i,j]` is a function of endpoints
`i` and `j` and of nothing else. Pair `(i,j)` never hears from pair
`(i,k)` or `(k,j)`.

That is exactly the information a distogram needs in order to be
*embeddable in three dimensions*. Distances obey the triangle
inequality, `d(i,j) <= d(i,k) + d(k,j)`, and a per-pair-independent
predictor has no mechanism that can know this. It can emit a set of
pairwise distances for which no 3D structure exists at all -- every pair
locally plausible, the whole set globally incoherent. "Locally right,
globally wrong" is what that looks like from the outside.

The triangle multiplicative update (Jumper et al. 2021, Algorithms 11
and 12) is the standard remedy and the one ingredient of that
architecture this model never had: it makes `z[i,j]` a function of the
products `a[i,k] * b[j,k]` summed over all `k`, so every pair is updated
by every triangle it participates in.

COST
----
`O(L^3 c)` per direction. At the stage-5 shape -- `L = 1024`, batch 16 --
and `c = 32` that is about 1.1 TFLOP for the pair, which is single-digit
milliseconds on this card, and the `(B, L, L, c)` intermediates are
roughly 1 GiB each. Affordable, but not free, so the channel count is a
knob and the forward is checkpointable. `c = 32` against `d_pair = 160`
is deliberate: the triangle term is a *constraint* on the pair track, not
a replacement for it.

DISCIPLINE
----------
`proj_out` is zero-initialised, so a loaded checkpoint's forward pass is
bit-identical until the update earns its way in -- the same rule as
`bias_scale`, `site_scale`, `coev_proj` and `GeometricBias.proj`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class TriangleMultiplication(nn.Module):
    """One direction of the triangle multiplicative update.

    `outgoing=True` is Algorithm 11: `z[i,j] <- sum_k a[i,k] * b[j,k]`,
    which updates a pair from the two edges leaving its endpoints.
    `outgoing=False` is Algorithm 12: `z[i,j] <- sum_k a[k,i] * b[k,j]`,
    the edges arriving at them. Both are needed; neither is the transpose
    of the other once the projections differ.
    """

    def __init__(self, d_pair: int, c: int = 32, outgoing: bool = True):
        super().__init__()
        self.outgoing = outgoing
        self.c = c
        self.norm_in = nn.LayerNorm(d_pair)
        self.lin_a = nn.Linear(d_pair, c, bias=False)
        self.lin_b = nn.Linear(d_pair, c, bias=False)
        self.gate_a = nn.Linear(d_pair, c)
        self.gate_b = nn.Linear(d_pair, c)
        self.norm_out = nn.LayerNorm(c)
        self.gate_out = nn.Linear(d_pair, d_pair)
        self.proj_out = nn.Linear(c, d_pair, bias=False)
        nn.init.zeros_(self.proj_out.weight)      # inert at load

    def forward(self, z: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`z` is (B, L, L, d_pair) -> the same shape, to be added to `z`.

        `mask` is (B, L) over residues. Padding is zeroed in `a` and `b`
        rather than masked afterwards, because the sum over `k` runs over
        every position: a pad left in `a` contributes a real product to
        every real pair, which is a leak and not a rounding error.
        """
        s = self.norm_in(z)
        a = torch.sigmoid(self.gate_a(s)) * self.lin_a(s)
        b = torch.sigmoid(self.gate_b(s)) * self.lin_b(s)
        if mask is not None:
            m = (mask[:, :, None] & mask[:, None, :]).unsqueeze(-1).to(a.dtype)
            a = a * m
            b = b * m
        if self.outgoing:
            t = torch.einsum("bikc,bjkc->bijc", a, b)
        else:
            t = torch.einsum("bkic,bkjc->bijc", a, b)
        return torch.sigmoid(self.gate_out(s)) * self.proj_out(self.norm_out(t))


class TrianglePairStack(nn.Module):
    """`n_layers` rounds of outgoing-then-incoming, residually.

    Two rounds is the default rather than AlphaFold's forty-eight, because
    this is a constraint bolted onto a pair track that already works and
    not an Evoformer: the question it is here to answer is whether the
    pair representation being 3D-embeddable moves the fold at all.
    """

    def __init__(self, d_pair: int, c: int = 32, n_layers: int = 2,
                 grad_checkpoint: bool = True):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(TriangleMultiplication(d_pair, c, outgoing=True))
            self.layers.append(TriangleMultiplication(d_pair, c, outgoing=False))

    def forward(self, z: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        ck = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for layer in self.layers:
            upd = (checkpoint(layer, z, mask, use_reentrant=False) if ck
                   else layer(z, mask))
            z = z + upd
        return z
