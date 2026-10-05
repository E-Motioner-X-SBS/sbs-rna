"""The one check every training loop needs and none of them had.

A loss that goes non-finite does not stop anything. `nan.backward()` runs,
`clip_grad_norm_` passes NaN through, `opt.step()` writes NaN into every
weight, and from then on the model emits NaN, the loss is NaN, the checkpoint
written at the next `ckpt_every` boundary is NaN, and the run continues
burning the card for days producing a file that cannot be resumed from.

Nothing in this project caught that. `PharosHeads.loss` silently skipped
non-finite terms -- but it is dead code, called by nothing. The four real
trainers had no check at all, and the held-out monitor could not see it
either: `float("nan")` parses, and every comparison against NaN is False, so
a diverged run read as "better than unigram" and was reported OK.

So: refuse loudly, before the optimiser, and say which component went first.
A run that cannot compute a finite loss has nothing to save.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch


class NonFiniteLoss(RuntimeError):
    """Raised instead of stepping the optimiser on a NaN or Inf loss."""


def check_loss(loss: torch.Tensor, step: int,
               parts: Optional[Mapping[str, float]] = None,
               extra: str = "") -> None:
    """Raise `NonFiniteLoss` if `loss` is not finite. Call before `backward()`.

    `parts` is the per-head breakdown when the trainer has one. It is reported
    because "the loss is NaN" is not actionable and "the loss is NaN and
    `structure` is the only non-finite term in it" is: the heads are summed,
    so exactly one of them usually went first and the sum only records that
    something did.
    """
    if torch.isfinite(loss).all():
        return
    bad: Dict[str, float] = {}
    if parts:
        for k, v in parts.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv or fv in (float("inf"), float("-inf")):
                bad[k] = fv
    detail = (f"; non-finite parts: {bad}" if bad else
              "; no single part is non-finite, so it came from a term that is "
              "not reported in `parts` or from the sum itself"
              if parts else "")
    raise NonFiniteLoss(
        f"loss is {float(loss)} at step {step}{detail}{extra}. Refusing to "
        f"step the optimiser: one backward pass through this writes NaN into "
        f"every weight and the next checkpoint saves it.")
