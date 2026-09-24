#!/usr/bin/env python3
"""Head 3, rebuilt: a conditional diffusion decoder over backbone coordinates.

What it replaces. `StructureHead` was `MLP(d_model -> 2*d_model -> K*9)`:
absolute xyz for three backbone atoms, regressed directly from token features.
That formulation has three problems and they compound.

* **It is not equivariant.** Rotate the true structure and the target changes
  while the input does not, so the network must memorise an arbitrary global
  frame per example. Nothing in the architecture says structures are defined up
  to rigid motion.
* **An L2 regression to coordinates has no mode.** Where a region is genuinely
  flexible -- and RNA is full of those -- the minimiser is the mean of the
  ensemble, which is a structure that exists nowhere. Averaging two hairpin
  orientations gives a collapsed hairpin.
* **There is no mechanism for geometric consistency.** Neighbouring residues
  are predicted independently; nothing enforces a P-P distance, a sensible
  pucker, or non-overlap.

Diffusion answers all three. The model learns a *distribution* over structures
rather than a point estimate, so multimodality is representable instead of
averaged away; random rigid augmentation during training makes the frame
irrelevant; and denoising is iterative, so each step conditions on a complete
(if noisy) structure and can enforce consistency across residues.

**Formulation: EDM (Karras et al.), as AlphaFold3 uses.** Preconditioning is
what makes a diffusion model trainable across six orders of magnitude of noise
without a separate schedule per scale:

    D(x; sigma) = c_skip(sigma) * x + c_out(sigma) * F(c_in(sigma) * x, c_noise)

so the raw network F sees a unit-variance input and predicts a unit-variance
target at every noise level, and the skip connection hands it the identity for
free when sigma is small. The loss is weighted by lambda(sigma) so no noise
level dominates the gradient.

**Conditioning.** The denoiser is a transformer over residues whose attention
carries the pair representation as a bias -- which is where the pair track and
the coevolution features enter the structure prediction. Without that bias the
denoiser would be a sequence model guessing coordinates; with it, a predicted
contact is a term in the attention logits that pulls two residues together.

Equivariance is by augmentation rather than construction, again following AF3:
each training example is randomly rotated and centred, so the network sees the
same structure in every orientation and has no reason to prefer one. That is
cheaper than an equivariant architecture and, at this scale, sufficient.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Backbone atoms per residue: phosphate, C4' sugar anchor, glycosidic N.
#: Three points fix a frame, which is the minimum for an orientation-aware
#: backbone without modelling every atom.
N_ATOM = 3


@dataclass
class DiffusionConfig:
    d_model: int = 512
    d_pair: int = 128
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.0
    #: EDM noise distribution. Coordinates are in angstroms and RNA backbones
    #: span tens of angstroms, so sigma_data is set from the corpus rather than
    #: the image-diffusion default of 0.5.
    sigma_data: float = 16.0
    p_mean: float = -1.2
    p_std: float = 1.5
    sigma_min: float = 0.002
    sigma_max: float = 160.0
    rho: float = 7.0              # sampling schedule curvature
    n_steps: int = 50             # inference denoising steps


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of log-noise, the standard conditioning signal.

    Built in the dtype of `t` rather than float32: the head is exercised in
    float64 by its own tests and in bfloat16 under autocast, and a hardcoded
    float32 here silently breaks both.
    """
    dt = t.dtype if t.is_floating_point() else torch.float32
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=t.device, dtype=dt) / half)
    a = t.to(dt)[..., None] * freqs
    emb = torch.cat([a.sin(), a.cos()], dim=-1)
    return F.pad(emb, (0, dim - emb.shape[-1])) if emb.shape[-1] < dim else emb


class DenoiseBlock(nn.Module):
    """Self-attention over residues with the pair representation as bias."""

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        self.h = h
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.pair_bias = nn.Linear(cfg.d_pair, h, bias=False)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
        # zero-init the residual branches: at initialisation the block is the
        # identity, so a deep denoiser starts as a shallow one and deepens as it
        # learns. Diffusion training is unstable without this.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, x: torch.Tensor, pair: Optional[torch.Tensor],
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm1(x)
        q, k, v = (t.view(B, L, self.h, D // self.h).transpose(1, 2)
                   for t in self.qkv(h).chunk(3, dim=-1))
        bias = None
        if pair is not None:
            bias = self.pair_bias(pair).permute(0, 3, 1, 2)          # B,h,L,L
        if mask is not None:
            m = (mask[:, None, None, :] & mask[:, None, :, None])
            neg = torch.finfo(q.dtype).min
            bias = (torch.zeros(B, self.h, L, L, device=x.device, dtype=q.dtype)
                    if bias is None else bias).masked_fill(~m, neg)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.out(a.transpose(1, 2).reshape(B, L, D))
        return x + self.ff(self.norm2(x))


class CoordDenoiser(nn.Module):
    """F in the EDM preconditioning: noisy coordinates -> denoised, conditioned."""

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.in_proj = nn.Linear(N_ATOM * 3, d)
        self.cond_proj = nn.Linear(d, d)
        self.time_proj = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([DenoiseBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, N_ATOM * 3)
        nn.init.zeros_(self.out.weight)        # start by predicting c_skip * x
        nn.init.zeros_(self.out.bias)

    def forward(self, x_noisy: torch.Tensor, c_noise: torch.Tensor,
                single: torch.Tensor, pair: Optional[torch.Tensor],
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, L = x_noisy.shape[:2]
        h = self.in_proj(x_noisy.reshape(B, L, N_ATOM * 3))
        h = h + self.cond_proj(single)
        h = h + self.time_proj(_timestep_embedding(c_noise, self.cfg.d_model))[:, None, :]
        for blk in self.blocks:
            h = blk(h, pair, mask)
        return self.out(self.norm(h)).view(B, L, N_ATOM, 3)


def random_rigid(coords: torch.Tensor, mask: torch.Tensor,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Centre on the masked centroid and apply a uniform random rotation.

    This is where equivariance comes from. Without it the network has to learn a
    global frame it was never given, and the same structure in two orientations
    is two unrelated training targets.
    """
    B = coords.shape[0]
    m = mask[..., None, None].to(coords.dtype)
    # The denominator must count the VALUES summed, not the residues. `m` is
    # (B, L, 1, 1) so `m.sum((1,2))` is L, while `(coords*m).sum((1,2))` sums
    # L * N_ATOM values -- the centroid came out three times too large and the
    # "centred" output was not centred.
    m_full = m.expand_as(coords)
    centre = ((coords * m).sum((1, 2), keepdim=True)
              / m_full.sum((1, 2), keepdim=True).clamp(min=1))
    x = (coords - centre) * m
    # Uniform rotations via QR of a Gaussian matrix, forced into SO(3).
    #
    # Sign-correcting by diag(R) gives an ORTHOGONAL matrix, which is not the
    # same as a rotation: half of them come out with determinant -1, and those
    # are reflections. A reflection preserves every pairwise distance, so it
    # passes a distance check -- and it mirrors chirality. RNA backbones are
    # chiral, so augmenting with reflections would have trained the model on
    # enantiomers that cannot exist, with nothing in the loss objecting.
    dt = coords.dtype if coords.is_floating_point() else torch.float32
    g = torch.randn(B, 3, 3, device=coords.device, dtype=dt, generator=generator)
    q, r = torch.linalg.qr(g)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]
    # flip one column wherever det == -1, turning the reflection into a rotation
    flip = torch.where(torch.det(q) < 0, -1.0, 1.0).to(dt)
    q = torch.cat([q[..., :-1], q[..., -1:] * flip[:, None, None]], dim=-1)
    return torch.einsum("blad,bde->blae", x, q) * m


class DiffusionStructureHead(nn.Module):
    """Head 3. Training returns a loss; inference denoises from pure noise."""

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.net = CoordDenoiser(cfg)

    # ---- EDM preconditioning -------------------------------------------
    def _c(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        sd = self.cfg.sigma_data
        s2, d2 = sigma ** 2, sd ** 2
        c_skip = d2 / (s2 + d2)
        c_out = sigma * sd / (s2 + d2).sqrt()
        c_in = 1.0 / (s2 + d2).sqrt()
        c_noise = sigma.log() / 4.0
        return c_skip, c_out, c_in, c_noise

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, single: torch.Tensor,
                pair: Optional[torch.Tensor], mask: Optional[torch.Tensor]
                ) -> torch.Tensor:
        c_skip, c_out, c_in, c_noise = self._c(sigma)
        v = c_skip.view(-1, 1, 1, 1), c_out.view(-1, 1, 1, 1), c_in.view(-1, 1, 1, 1)
        f = self.net(v[2] * x, c_noise, single, pair, mask)
        return v[0] * x + v[1] * f

    def loss(self, coords: torch.Tensor, single: torch.Tensor,
             pair: Optional[torch.Tensor], mask: torch.Tensor,
             generator: Optional[torch.Generator] = None,
             augment: bool = True) -> Dict[str, torch.Tensor]:
        """EDM denoising loss, with sigma-weighting so no noise scale dominates.

        `augment=False` centres without rotating. Training uses the rotation;
        turning it off isolates denoising from equivariance, which is the only
        way to tell a model that cannot denoise from one that simply has not yet
        learned that structures are defined up to a rigid motion.
        """
        cfg = self.cfg
        B = coords.shape[0]
        if augment:
            x0 = random_rigid(coords, mask, generator)
        else:
            m = mask[..., None, None].to(coords.dtype)
            cen = ((coords * m).sum((1, 2), keepdim=True)
                   / m.expand_as(coords).sum((1, 2), keepdim=True).clamp(min=1))
            x0 = (coords - cen) * m
        # every tensor this head creates takes the dtype of the coordinates it
        # was handed. Hardcoding float32 breaks float64 gradient checks and
        # bfloat16 autocast alike, and the failure is a dtype error deep inside
        # a Linear rather than anywhere near the cause.
        dt = coords.dtype if coords.is_floating_point() else torch.float32
        ln_sigma = (torch.randn(B, device=coords.device, dtype=dt,
                                generator=generator) * cfg.p_std + cfg.p_mean)
        sigma = ln_sigma.exp().clamp(cfg.sigma_min, cfg.sigma_max)
        noise = torch.randn(x0.shape, device=coords.device, dtype=dt,
                            generator=generator)
        x = x0 + sigma.view(-1, 1, 1, 1) * noise
        pred = self.denoise(x, sigma, single, pair, mask)

        w = (sigma ** 2 + cfg.sigma_data ** 2) / (sigma * cfg.sigma_data) ** 2
        m = mask[..., None, None].to(pred.dtype)
        # Divide by the number of VALUES summed, not the number of residues.
        # `m` is (B, L, 1, 1) so `m.sum((1,2,3))` counts L while the numerator
        # sums L * N_ATOM * 3 terms -- the loss came out exactly 9x too large.
        # It still trained (a constant factor is absorbed by the learning rate)
        # but it reported 8.98 where EDM's weighting is designed to give ~1.0 at
        # initialisation, and it would have silently given head 3 nine times
        # the weight of every other head in the stage-5 objective.
        # This is the same mistake as the centroid in `random_rigid`, which is
        # why the divisor there is spelled the same way.
        den = m.expand_as(pred).sum((1, 2, 3)).clamp(min=1)
        se = (((pred - x0) ** 2) * m).sum((1, 2, 3)) / den
        return {"loss": (w * se).mean(),
                "mse": se.mean().detach(),
                "sigma": sigma.mean().detach()}

    # ---- sampling -------------------------------------------------------
    def sigmas(self, n: int, device, dtype=torch.float32) -> torch.Tensor:
        """Karras schedule: dense where the signal appears, sparse in pure noise."""
        cfg = self.cfg
        i = torch.arange(n, device=device, dtype=dtype)
        a, b = cfg.sigma_max ** (1 / cfg.rho), cfg.sigma_min ** (1 / cfg.rho)
        s = (a + i / max(n - 1, 1) * (b - a)) ** cfg.rho
        return torch.cat([s, s.new_zeros(1)])

    @torch.no_grad()
    def sample(self, single: torch.Tensor, pair: Optional[torch.Tensor],
               mask: torch.Tensor, n_steps: Optional[int] = None,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Heun (2nd-order) sampling. Half the steps of Euler for equal error."""
        cfg = self.cfg
        B, L = single.shape[:2]
        dev = single.device
        dt = single.dtype if single.is_floating_point() else torch.float32
        steps = self.sigmas(n_steps or cfg.n_steps, dev, dt)
        x = torch.randn(B, L, N_ATOM, 3, device=dev, dtype=dt,
                        generator=generator) * steps[0]
        for i in range(len(steps) - 1):
            s, s_next = steps[i], steps[i + 1]
            sv = s.expand(B)
            d = (x - self.denoise(x, sv, single, pair, mask)) / s
            x_next = x + (s_next - s) * d
            if s_next > 0:                       # Heun correction
                dn = (x_next - self.denoise(x_next, s_next.expand(B), single,
                                            pair, mask)) / s_next
                x_next = x + (s_next - s) * 0.5 * (d + dn)
            x = x_next
        return x * mask[..., None, None].to(x.dtype)


class DiffusionPairFeatures(nn.Module):
    """The pair representation head 3 was being denied.

    `DenoiseBlock` takes a `(B, L, L, d_pair)` tensor and turns it into an
    attention bias -- that is how a structure decoder is told which residues
    belong near each other. Stage 5 passed `None`, so the denoiser saw only
    per-residue embeddings and had to infer every geometric relationship from
    them. A decoder with no pair channel cannot be told that residue 12 pairs
    with residue 64; it can only be told what residue 12 and residue 64 are.

    Three sources, summed:

    **An outer sum, not an outer concatenation.** `proj(cat(h_i, h_j))` is the
    obvious construction and costs `O(L^2 * d_model * d_pair)` -- 64 GFLOP for
    one 512-residue chain. `a_i + b_j` gives each pair a learned function of
    both endpoints for `O(L * d_model * d_pair)` of projection and an
    `O(L^2 * d_pair)` broadcast, which is the difference between affordable and
    not. It cannot represent interactions between the two endpoints that are
    not additive, and the attention layers above it exist to supply those.

    **Relative position**, clamped to +/- `max_rel`. Without it the decoder has
    no notion of chain connectivity at all: nothing in a bag of residue
    embeddings says residue i and residue i+1 are covalently bonded, and a
    backbone that does not know that is a point cloud.

    **Coevolution**, through a zero-initialised projection, so the feature
    starts contributing exactly nothing and has to earn its way in -- the same
    discipline `coev_proj` uses on the contact path, and for the same reason:
    a model that already works must not regress the moment a feature is
    switched on.
    """

    def __init__(self, d_model: int, d_pair: int, max_rel: int = 32):
        super().__init__()
        self.max_rel = max_rel
        self.a = nn.Linear(d_model, d_pair, bias=False)
        self.b = nn.Linear(d_model, d_pair, bias=False)
        self.rel = nn.Embedding(2 * max_rel + 2, d_pair)
        self.coev = nn.Linear(1, d_pair, bias=False)
        nn.init.zeros_(self.coev.weight)
        self.norm = nn.LayerNorm(d_pair)

    def forward(self, single: torch.Tensor,
                coev: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = single.shape
        p = self.a(single).unsqueeze(2) + self.b(single).unsqueeze(1)
        idx = torch.arange(L, device=single.device)
        d = (idx[None, :] - idx[:, None]).clamp(-self.max_rel, self.max_rel)
        p = p + self.rel(d + self.max_rel)[None]
        if coev is not None:
            p = p + self.coev(coev.unsqueeze(-1).to(p.dtype))
        return self.norm(p)
