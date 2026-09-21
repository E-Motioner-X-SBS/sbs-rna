#!/usr/bin/env python3
"""The 24-dim chemistry vector for a padded batch, on the GPU, in two lookups.

`chain_chemistry` is the definition (see `chemistry.py`); this is the same
function for a `(B, L)` token batch with no Python loop and no host round-trip.
It exists because the training loop was spending most of its time here:

* `encode_batch` called `chain_chemistry` once per sequence on CPU, and
  `masked_chemistry` then called it again -- per sequence, per step, after a
  `.cpu()` of the token tensor -- and discarded the first result. The chemistry
  was computed twice and used once.
* Both ran between two GPU kernels, so the card waited through all of it.

Everything except dim 23 is a pure function of the symbol, so it is a table
indexed by token id. Dim 23's +/-16 GC window is a cumulative sum, which is a
single kernel over the batch. The tables come from `chemistry_tables`, which
calls `residue_chemistry` itself, so this cannot drift from the definition.

`test_chemistry_torch.py` asserts equality against `chain_chemistry` rather than
approximate agreement: the cumulative sums are over 0/1 indicators and stay
exact in float32 well past the 4,608-token context.

Padding is not a special case. A PAD token resolves to parent `N`, so it scores
0 in both the GC numerator and its denominator, and a window that overlaps the
pad region returns exactly what the unpadded chain returns.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .chemistry import GC_WINDOW, N_DIMS, chemistry_tables, gc_lookups


class BatchChemistry:
    """Callable `(tokens, mask) -> (B, L, 24)`, tables resident on the device."""

    def __init__(self, symbols: Sequence[str], device, *,
                 gc_window: int = GC_WINDOW, deoxy: bool = False,
                 dtype: torch.dtype = torch.float32) -> None:
        interior, terminal = chemistry_tables(symbols, deoxy=deoxy)
        gc, known = gc_lookups(symbols)
        to = lambda a: torch.as_tensor(a, dtype=dtype, device=device)   # noqa: E731
        self.interior, self.terminal = to(interior), to(terminal)
        self.gc, self.known = to(gc), to(known)
        self.w = int(gc_window)
        self.n_dims = N_DIMS

    def __call__(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`tokens` `(B, L)` long, `mask` `(B, L)` bool. Sequences left-aligned."""
        if tokens.dim() != 2:
            raise ValueError(f"expected (B, L) tokens, got {tuple(tokens.shape)}")
        B, L = tokens.shape
        out = self.interior[tokens]
        # index 0 is the 5' terminus: free 5'-OH, so no phosphate charge (dim 18)
        out = torch.cat([self.terminal[tokens[:, :1]], out[:, 1:]], dim=1)

        m = mask.to(out.dtype)
        gc = self.gc[tokens] * m
        known = self.known[tokens] * m
        cg = F.pad(gc.cumsum(-1), (1, 0))
        ck = F.pad(known.cumsum(-1), (1, 0))
        idx = torch.arange(L, device=tokens.device)
        lo = (idx - self.w).clamp(min=0)
        hi = (idx + self.w + 1).clamp(max=L)
        denom = ck[:, hi] - ck[:, lo]
        num = cg[:, hi] - cg[:, lo]
        gc_frac = torch.where(denom > 0, num / denom.clamp(min=1.0),
                              torch.zeros_like(num))
        out = torch.cat([out[..., :23], gc_frac.unsqueeze(-1)], dim=-1)
        return out * m.unsqueeze(-1)
