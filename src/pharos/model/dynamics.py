#!/usr/bin/env python3
"""Harmonic ensemble from a predicted stiffness field (ARCHITECTURE v0.2 §10).

Much of RNA is not one structure. §10's position is that once a per-step
stiffness field exists, an equilibrium ensemble is nearly free: assemble `F`,
take normal modes, read off fluctuation amplitudes, emit K=3 states with
weights. No sampling, no molecular dynamics -- the standard elastic-network
construction.

**What this does not claim**, and the docstring says it because the section
does: this is **equilibrium breathing, not a folding pathway**. The couplings
are local and the model never traverses a barrier. Calling it "folding
dynamics" would overstate it.

Three things the measurements dictate
-------------------------------------
**The stiffness field is conditioned on sequence AND structure.** The nested
models measured on our own corpus: global 19.7235 nats, sequence-context
17.5792, structure-context 17.8470, and sequence x structure **14.5510**.
Sequence and structure are near-equal complementary contributors (+1.7847 and
+1.7409 held out, the combination adding +1.0336 beyond sequence alone). An
earlier version claimed structure dominates; that was an in-sample artefact.
So `StiffnessField` takes both and neither is optional.

**Neighbouring steps couple.** The literature is explicit that the pentameric
scale is the minimum range of elastic couplings -- a dinucleotide-step model is
demonstrably insufficient -- so the assembled matrix is block-tridiagonal, not
block-diagonal.

**Fluctuations must be computable at L = 4,450.** The assembled stiffness is
6(L-1) square; inverting it densely is O((6L)^3), which at L=4,000 is a 24,000
x 24,000 inverse per chain per step. `block_tridiagonal_variance` computes only
the diagonal blocks of the inverse, which is all the fluctuation amplitudes
need, by the standard forward-backward recursion in **O(L)** with 6x6 blocks.
`test_dynamics.py` checks it against a dense inverse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

#: helical step coordinates: shift, slide, rise, tilt, roll, twist
N_STEP_DOF = 6


@dataclass
class DynamicsConfig:
    d_model: int = 512
    n_states: int = 3
    #: floor on the stiffness eigenvalues. A stiffness matrix that is not
    #: positive definite has imaginary normal modes, i.e. the harmonic
    #: approximation has broken down; the floor makes that impossible by
    #: construction rather than something to detect afterwards.
    eig_floor: float = 1e-2
    #: Spectral-norm bound on the coupling, as a fraction of
    #: sqrt(lambda_min(F_i) lambda_min(F_i+1)). Must be < 0.5: the assembled
    #: matrix is then positive definite with margin (1 - 2 g), which is the
    #: whole reason the fluctuation variances are guaranteed non-negative.
    max_coupling: float = 0.45
    dropout: float = 0.0


def _spd_from_cholesky(raw: torch.Tensor, floor: float) -> torch.Tensor:
    """(..., 21) -> (..., 6, 6) symmetric positive definite, lambda_min >= floor.

    Parameterised by a Cholesky factor so positive-definiteness is structural,
    not something to detect afterwards: predicting the 21 free entries of a
    symmetric matrix directly and hoping it comes out positive definite does
    not work, and imaginary normal modes are the result.

    **The floor is added AFTER the product, not to the factor's diagonal.**
    Putting it on the diagonal of `L` bounds the diagonal of `L L^T` and bounds
    nothing about its eigenvalues -- with large off-diagonal entries the matrix
    is still nearly singular. Driving the raw factor to -30 gave a minimum
    eigenvalue of 0.0000 against a nominal floor of 1e-2. `L L^T + floor . I`
    raises every eigenvalue by exactly `floor`, which is the guarantee that was
    wanted.
    """
    n = N_STEP_DOF
    idx = torch.tril_indices(n, n, device=raw.device)
    L = raw.new_zeros(*raw.shape[:-1], n, n)
    L[..., idx[0], idx[1]] = raw
    diag = torch.arange(n, device=raw.device)
    L[..., diag, diag] = F.softplus(L[..., diag, diag])
    eye = torch.eye(n, device=raw.device, dtype=raw.dtype)
    return L @ L.transpose(-1, -2) + floor * eye


class StiffnessField(nn.Module):
    """Per-step 6x6 stiffness and nearest-neighbour coupling.

    Supervised where measured `F` exists (76 contexts) and otherwise trained
    only through the fluctuation amplitudes, which have B-factor and RMSF
    labels -- X-ray B-factors only, per D12.
    """

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        n_tri = N_STEP_DOF * (N_STEP_DOF + 1) // 2      # 21
        # a step is a property of a DINUCLEOTIDE, so the input is the pair of
        # adjacent residue representations, not one of them
        self.step = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(),
            nn.Dropout(cfg.dropout))
        self.diag = nn.Linear(d, n_tri)
        self.coupling = nn.Linear(2 * d, N_STEP_DOF * N_STEP_DOF)
        nn.init.zeros_(self.coupling.weight)
        nn.init.zeros_(self.coupling.bias)

    def forward(self, tok: torch.Tensor, mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(F_diag (B, S, 6, 6), F_off (B, S-1, 6, 6))` for S = L-1 steps."""
        B, L, _ = tok.shape
        if L < 2:
            z = tok.new_zeros(B, 0, N_STEP_DOF, N_STEP_DOF)
            return z, z
        steps = self.step(torch.cat([tok[:, :-1], tok[:, 1:]], dim=-1))   # B,S,d
        Fd = _spd_from_cholesky(self.diag(steps), self.cfg.eig_floor)
        if steps.shape[1] < 2:
            return Fd, Fd.new_zeros(B, 0, N_STEP_DOF, N_STEP_DOF)
        pair = torch.cat([steps[:, :-1], steps[:, 1:]], dim=-1)
        raw = self.coupling(pair).view(B, -1, N_STEP_DOF, N_STEP_DOF)
        # Bound the coupling so the ASSEMBLED matrix is positive definite.
        #
        # For a block-tridiagonal matrix with diagonal blocks A_i (smallest
        # eigenvalue a_i) and off-diagonal blocks C_i,
        #     x^T M x >= sum a_i |x_i|^2 - 2 sum ||C_i|| |x_i||x_{i+1}|
        # so ||C_i|| <= g sqrt(a_i a_{i+1}) with g < 1/2 gives
        #     x^T M x >= (1 - 2g) sum a_i |x_i|^2 > 0.
        #
        # Scaling by the geometric mean of the DIAGONAL ENTRIES, as a first
        # version did, bounds the wrong quantity: with large coupling the
        # assembled matrix reached a minimum eigenvalue of -1.92. Normalising
        # each block by its Frobenius norm bounds the spectral norm (Frobenius
        # dominates it) by exactly g sqrt(a_i a_{i+1}).
        #
        # a_i is read from the diagonal blocks and detached -- it is a scale,
        # not a signal, and differentiating an eigendecomposition through
        # near-degenerate eigenvalues is a needless source of NaN.
        lam = torch.linalg.eigvalsh(Fd.detach())[..., 0].clamp_min(self.cfg.eig_floor)
        gm = (lam[:, :-1] * lam[:, 1:]).sqrt()                      # B, S-1
        fro = raw.flatten(-2).norm(dim=-1).clamp_min(1e-6)          # B, S-1
        Fo = raw * (self.cfg.max_coupling * gm / fro)[..., None, None]
        return Fd, Fo


def block_tridiagonal_variance(diag: torch.Tensor, off: torch.Tensor,
                               jitter: float = 1e-6) -> torch.Tensor:
    """Diagonal blocks of the inverse of a block-tridiagonal matrix, in O(S).

    The assembled stiffness is `6(L-1)` square. Fluctuation amplitudes need only
    the diagonal blocks of its inverse, and those follow from the standard
    forward-backward recursion over Schur complements -- one 6x6 solve per step
    instead of one `6L x 6L` inverse, which is what makes this affordable at
    L = 4,450 rather than a cost that grows as L^3.

    `diag` is (B, S, 6, 6) symmetric positive definite, `off` is (B, S-1, 6, 6)
    holding the block coupling step `i` to step `i+1`.
    """
    B, S, n, _ = diag.shape
    eye = torch.eye(n, device=diag.device, dtype=diag.dtype).expand(B, n, n)
    if S == 0:
        return diag.new_zeros(B, 0, n, n)

    # forward: D_i = A_i - C_{i-1}^T D_{i-1}^{-1} C_{i-1}
    Dfwd = [diag[:, 0] + jitter * eye]
    for i in range(1, S):
        C = off[:, i - 1]
        red = C.transpose(-1, -2) @ torch.linalg.solve(Dfwd[-1], C)
        Dfwd.append(diag[:, i] - red + jitter * eye)

    # backward: E_i = A_i - C_i E_{i+1}^{-1} C_i^T
    Ebwd = [None] * S
    Ebwd[S - 1] = diag[:, S - 1] + jitter * eye
    for i in range(S - 2, -1, -1):
        C = off[:, i]
        red = C @ torch.linalg.solve(Ebwd[i + 1], C.transpose(-1, -2))
        Ebwd[i] = diag[:, i] - red + jitter * eye

    # the diagonal block of the inverse combines both sweeps
    out = []
    for i in range(S):
        M = Dfwd[i] + Ebwd[i] - diag[:, i] - jitter * eye
        out.append(torch.linalg.solve(M + jitter * eye, eye))
    return torch.stack(out, dim=1)


class HarmonicEnsemble(nn.Module):
    """Stiffness -> fluctuation amplitudes -> K states, plus a disorder head.

    The disorder head is the cheapest addition in the design: a residue recorded
    in `_pdbx_unobs_or_zero_occ_residues` is one too mobile to model, which is a
    direct per-residue flexibility label present in every deposited structure
    and used by no RNA structure predictor. 46,448 RNA records, at zero
    acquisition cost.
    """

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.stiffness = StiffnessField(cfg)
        self.disorder = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(),
            nn.Linear(d // 2, 1))
        self.state_weights = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, cfg.n_states))
        # maps a step's 6-dof variance onto a per-nucleotide scalar amplitude
        self.amp = nn.Linear(N_STEP_DOF, 1)
        nn.init.constant_(self.amp.weight, 1.0 / N_STEP_DOF)
        nn.init.zeros_(self.amp.bias)

    def forward(self, tok: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, L, _ = tok.shape
        Fd, Fo = self.stiffness(tok, mask)
        out: Dict[str, torch.Tensor] = {
            "stiffness_diag": Fd, "stiffness_off": Fo,
            "disorder_logit": self.disorder(tok).squeeze(-1),
        }
        if Fd.shape[1] == 0:
            out["fluctuation"] = tok.new_zeros(B, L)
            out["state_logits"] = self.state_weights(tok.mean(1))
            return out

        cov = block_tridiagonal_variance(Fd, Fo)                 # B,S,6,6
        var = cov.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)      # B,S,6
        step_amp = self.amp(var).squeeze(-1)                     # B,S
        # a nucleotide's amplitude is the mean of the steps it participates in;
        # the two chain ends belong to one step each
        nt = tok.new_zeros(B, L)
        nt[:, :-1] = nt[:, :-1] + step_amp
        nt[:, 1:] = nt[:, 1:] + step_amp
        denom = tok.new_ones(B, L) * 2.0
        denom[:, 0] = 1.0
        denom[:, -1] = 1.0
        out["fluctuation"] = (nt / denom) * mask.to(tok.dtype)
        out["state_logits"] = self.state_weights(
            (tok * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1))
        return out
