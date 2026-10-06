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
#: tuned to drive singular values TOWARD [~0.7, ~1.3] in five steps rather
#: than to converge slowly to 1, because the update only needs to be
#: approximately orthogonal and five matmuls is the budget.
#:
#: "Toward", not "into", and the difference is measured. Five steps reach that
#: interval for a well-conditioned input -- a random (768, 2304) lands in
#: [0.681, 1.140] -- and cannot reach it for a near-singular one. A random
#: SQUARE Gaussian has its smallest singular value at the Marchenko-Pastur
#: edge, so 768x768 goes from condition number 5,458 to 105: a 52x
#: compression, bounded above at 1.204, and nowhere near orthogonal.
#: **31% of shared400's 222 Muon tensors are square** (39.5M of 343M
#: parameters), so this is not a corner case. The upper bound always holds,
#: which is what stops the update exploding; the lower one does not.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7
                  ) -> torch.Tensor:
    """The nearest semi-orthogonal matrix to `g`, approximately.

    Runs in bfloat16 on purpose: the iteration is a fixed-point scheme whose
    error is dominated by the coefficient tuning, not by rounding, and the
    matmuls are the cost.
    """
    if g.ndim < 2:
        raise ValueError(f"newton_schulz needs a matrix, got {g.ndim} dims")
    a, b, c = _NS_COEFFS
    x = g.bfloat16()
    # Per-SLICE normalisation. A stacked expert tensor is `(E, d, f)` and each
    # expert is its own matrix, so a single Frobenius norm over the whole
    # stack would let one expert's scale set every other expert's step.
    x = x / (x.flatten(-2).norm(dim=-1)[..., None, None] + eps)
    transposed = g.size(-2) > g.size(-1)
    if transposed:
        x = x.transpose(-1, -2)
    for _ in range(steps):
        aa = x @ x.transpose(-1, -2)
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.transpose(-1, -2)
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
                # `upd.reshape(upd.size(0), -1)` for a 3-D tensor flattens
                # `(E, d, f)` to `(E, d*f)` and orthogonalises ACROSS THE
                # EXPERT AXIS -- it makes the experts mutually orthogonal
                # instead of making each expert's weight matrix orthogonal,
                # which is not what Muon means and is not what its spectral
                # argument licenses. `GroupedExperts.w1` is `(32, 512, 512)`,
                # so every independent-expert config -- mini, small, base400 --
                # trained its MoE on a transform of the update rather than the
                # update. shared400, the config stage 1 actually runs, has
                # only 2-D tensors in Muon and is unaffected.
                #
                # The last two dimensions are the matrix; anything in front is
                # a batch. `newton_schulz` handles both and the 2-D result is
                # bit-identical to before.
                upd = newton_schulz(upd if upd.ndim > 2
                                    else upd.reshape(upd.size(0), -1),
                                    group["ns_steps"]).view_as(p)
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                # An orthogonal update moves every direction by the same
                # amount, so a tall matrix would take a larger step in aggregate
                # than a wide one at the same lr. This restores parity.
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
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
        elif is_embed:
            # No decay on embeddings. The MLM head has no weight matrix of its
            # own -- it is TIED to the token embedding -- so the embedding's
            # magnitude directly sets the logit scale, and decaying it works
            # against the one quantity the model's confidence depends on.
            # Measured at 395M tokens: mean top-1 probability 0.3307 against a
            # uniform 0.25, predictive entropy 1.9446 of a possible 2.000, and
            # not one masked position in a thousand predicted above p=0.9. The
            # embedding is growing (per-row norm 0.554 -> 1.163, 2.3x) so decay
            # is not winning, but it is pulling the wrong way, and not decaying
            # embeddings is standard for exactly this reason.
            no_decay.append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)          # norms, biases, per-expert vectors
    groups = [{"params": decay, "lr": adamw_lr, "weight_decay": weight_decay},
              {"params": no_decay, "lr": adamw_lr, "weight_decay": 0.0}]
    return muon, groups
