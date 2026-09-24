#!/usr/bin/env python3
"""Muon: momentum, orthogonalised. An optimiser for the 2-D weights only.

AdamW rescales each coordinate of the gradient independently, which throws away
the fact that a weight MATRIX has a spectrum. A gradient whose energy sits in a
handful of directions produces an update that moves those directions far and
the rest hardly at all, and the effective step size per direction is whatever
the second-moment estimate happens to be.

Muon takes the momentum buffer and replaces it with the nearest semi-orthogonal
matrix -- every singular value set to 1 -- so the update moves every direction
by the same amount. The orthogonalisation is a quintic Newton-Schulz iteration
rather than an SVD, five matmuls instead of a decomposition, which is what
makes it affordable per step.

**It applies to 2-D parameters only, and that is not a detail.** Embeddings,
LayerNorm gains, biases and the per-expert modulation vectors have no spectrum
to orthogonalise -- for a vector the operation collapses to sign(x) scaled, which
is not what anyone wants. Those stay on AdamW. `muon_param_groups` does the
split, and it also does the weight-decay split that AdamW here was missing:
decaying a LayerNorm gain has no scale-invariance argument behind it, and
decaying the zero-initialised expert modulation actively pulls the MoE's
specialisation back toward the shared network it is trying to differ from.

The learning rates are NOT interchangeable. Muon's update has unit spectral
norm by construction, so its natural rate is ~0.02 where AdamW's is ~6e-4;
comparing the two at one learning rate compares nothing.

Reference: Jordan et al., "Muon: An optimizer for hidden layers in neural
networks" (2024).
"""
from __future__ import annotations

from typing import Iterable, List

import torch

#: Quintic coefficients. They do not converge to exactly 1 -- the iteration is
#: tuned to drive singular values into [~0.7, ~1.3] in five steps rather than to
#: converge slowly to 1, because the update only needs to be approximately
#: orthogonal and five matmuls is the budget.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7
                  ) -> torch.Tensor:
    """The nearest semi-orthogonal matrix to `g`, approximately.

    Runs in bfloat16 on purpose: the iteration is a fixed-point scheme whose
    error is dominated by the coefficient tuning, not by rounding, and the
    matmuls are the cost.
    """
    if g.ndim != 2:
        raise ValueError(f"newton_schulz needs a matrix, got {g.ndim} dims")
    a, b, c = _NS_COEFFS
    x = g.bfloat16()
    x = x / (x.norm() + eps)
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum SGD whose update is orthogonalised before it is applied."""

    def __init__(self, params: Iterable, lr: float = 0.02,
                 momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0):
        super().__init__(list(params), dict(lr=lr, momentum=momentum,
                                            nesterov=nesterov,
                                            ns_steps=ns_steps,
                                            weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if group["nesterov"] else buf
                flat = upd.reshape(upd.size(0), -1)
                flat = newton_schulz(flat, group["ns_steps"])
                upd = flat.view_as(p)
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                # An orthogonal update moves every direction by the same
                # amount, so a tall matrix would take a larger step in aggregate
                # than a wide one at the same lr. This restores parity.
                scale = max(1.0, p.size(0) / p.size(-1)) ** 0.5
                p.add_(upd, alpha=-lr * scale)
        return loss


def muon_param_groups(model, muon_lr: float = 0.02, adamw_lr: float = 6e-4,
                      weight_decay: float = 0.01):
    """Split a model into the tensors Muon should own and the rest.

    Returns `(muon_params, adamw_groups)`. A parameter goes to Muon when it is
    at least 2-D and is not an embedding or the output projection -- those are
    interfaces to a vocabulary rather than hidden transforms, and Muon's
    spectral argument does not apply to them.

    The AdamW side is split again into decay and no-decay, which the previous
    single-group AdamW was not doing.
    """
    muon: List[torch.nn.Parameter] = []
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lower = name.lower()
        is_embed = ("embed" in lower or "tok.weight" in lower
                    or "pos.weight" in lower or "mod.weight" in lower
                    or lower.endswith("rel.weight"))
        is_head = "mlm_head" in lower or lower.startswith("heads.")
        if p.ndim >= 2 and not is_embed and not is_head:
            muon.append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)          # norms, biases, per-expert vectors
    groups = [{"params": decay, "lr": adamw_lr, "weight_decay": weight_decay},
              {"params": no_decay, "lr": adamw_lr, "weight_decay": 0.0}]
    return muon, groups
