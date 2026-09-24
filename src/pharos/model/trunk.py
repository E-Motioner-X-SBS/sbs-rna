#!/usr/bin/env python3
"""The PHAROS token trunk: hybrid blocks, MoE, and depth from loops (§5).

§5.2's claim is that **16 blocks x 8 loops = 128 effective layers** beats a
32-block stack's 96 at 9.4x fewer parameters, and that with the one-step
gradient the loops cost compute but no additional activation memory. That is a
claim about an implementation, so here is the implementation.

How the loop works, and why it costs no activation memory
---------------------------------------------------------
Loops 1..N-1 run under `torch.no_grad()`; only the final loop is differentiated.
The gradient that reaches the weights is therefore the one-step gradient at the
converged point rather than a backprop through all N unrollings, which is what
keeps activation memory at one loop's worth instead of N. Deep supervision is
applied at every loop segment, so the intermediate iterates are still trained --
through the heads, on their own detached inputs, not through the recurrence.

**The caveat that must travel with this** (§5.2): the independent ARC Prize
analysis credits HRM's outer loop but finds most of its benchmark performance
comes from memorising evaluation-time tasks. The loop is adopted on its measured
merits here; the *parameter-count* argument must not lean on HRM.

Physics and refinement are coupled, not sequential
--------------------------------------------------
The recycled signal is the previous iteration's distance estimate, and it enters
the next pass through the electrostatic pair bias on the FULL-attention blocks
(§6.2). So a change in ionic condition changes what attends to what on the next
loop, rather than only re-scoring a finished representation. At recycle 0 there
is no estimate, the bias is zero, and the router runs on sequence-only features
-- the circularity §5.3 records, made explicit rather than papered over.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.utils.checkpoint
import torch.nn as nn

from .attention import BLOCK_PATTERN, make_mixer
from .moe import MoEConfig, MoEFeedForward, RouterFeatures
from .shared_moe import SharedMoEConfig, SharedMoEFeedForward


@dataclass
class TrunkConfig:
    d_model: int = 512
    n_blocks: int = 16
    n_loops: int = 8
    n_heads: int = 8
    window: int = 128
    dropout: float = 0.0
    moe: MoEConfig = field(default_factory=lambda: MoEConfig(d_model=512))
    #: pattern is tiled to `n_blocks`; §5.1 specifies period 8, two cycles
    pattern: Tuple[str, ...] = BLOCK_PATTERN
    #: Recompute each block in the backward pass instead of storing its
    #: activations. The one-step gradient already keeps only the LAST loop's
    #: activations, so what remains is one pass through `n_blocks` -- and at
    #: 512 experts with d_expert 2304 that one pass is 12 MiB per token, which
    #: fills an 80 GiB card at 6,700 tokens. Recomputing costs one extra
    #: forward (~30% of step time) and returns roughly 8x the batch, and on
    #: this hardware the larger batch more than pays for the recompute: the
    #: bottleneck at small batch is kernel launch and memory bandwidth, not
    #: arithmetic. Measured in `scripts/bench_memory.py`.
    grad_checkpoint: bool = False

    def block_kinds(self) -> List[str]:
        return [self.pattern[i % len(self.pattern)] for i in range(self.n_blocks)]


class TrunkBlock(nn.Module):
    """One mixer plus one MoE feed-forward, both pre-norm and residual."""

    def __init__(self, cfg: TrunkConfig, kind: str):
        super().__init__()
        self.kind = kind
        self.norm = nn.LayerNorm(cfg.d_model)
        self.mixer = make_mixer(kind, cfg.d_model, cfg.n_heads, cfg.window, cfg.dropout)
        # Either flavour, chosen by the config object's type rather than by a
        # flag: a SharedMoEConfig can only mean shared-adapter experts, so there
        # is no way to pass one and get the other.
        self.ff = (SharedMoEFeedForward(cfg.moe)
                   if isinstance(cfg.moe, SharedMoEConfig)
                   else MoEFeedForward(cfg.moe))

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                pair_bias: Optional[torch.Tensor],
                feats: Optional[RouterFeatures]) -> Tuple[torch.Tensor, Dict]:
        h = self.norm(x)
        x = x + self.mixer(h, mask=mask, pair_bias=pair_bias if self.kind == "full" else None)
        return self.ff(x, mask, feats)


class TokenTrunk(nn.Module):
    """The 16-block hybrid stack, run `n_loops` times with the one-step gradient."""

    def __init__(self, cfg: TrunkConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([TrunkBlock(cfg, k) for k in cfg.block_kinds()])
        self.norm_out = nn.LayerNorm(cfg.d_model)
        # the recycled representation is injected additively, through its own
        # norm, so loop 0 (which has nothing to recycle) is not a special case
        # in the arithmetic -- it simply adds zero
        self.recycle_norm = nn.LayerNorm(cfg.d_model)
        self.recycle_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.zeros_(self.recycle_proj.weight)

    def _one_pass(self, x: torch.Tensor, mask: torch.Tensor,
                  pair_bias: Optional[torch.Tensor],
                  feats: Optional[RouterFeatures]) -> Tuple[torch.Tensor, Dict]:
        aux_sum = {"balance_loss": x.new_zeros(())}
        usage: List[torch.Tensor] = []
        #: Scalars the MoE block computes and this loop used to throw away.
        #:
        #: `SharedMoEFeedForward` reports `mean_width`, `max_width` and two
        #: router entropies at every block of every step, with a comment saying
        #: they are "what the fixed-k version could not report". They were
        #: computed 18 times a step for the whole run and never left this
        #: function: only `balance_loss` and `expert_usage` were kept, and the
        #: trainer reads only the first of those. So the central claim about
        #: nucleus routing -- that a confident token fires one expert and an
        #: ambiguous one fires many -- had never been measured, and "no router
        #: collapse" rested on a single aggregate with no stated threshold.
        #: Averaging four more scalars over the blocks costs nothing.
        SCALARS = ("mean_width", "max_width", "router_entropy",
                   "token_router_entropy")
        scal: Dict[str, List[torch.Tensor]] = {k: [] for k in SCALARS}
        ckpt = self.cfg.grad_checkpoint and torch.is_grad_enabled()
        for blk in self.blocks:
            if ckpt:
                # use_reentrant=False so the block may return a dict and so
                # that the no-grad recycle loops above are unaffected
                x, aux = torch.utils.checkpoint.checkpoint(
                    blk, x, mask, pair_bias, feats, use_reentrant=False)
            else:
                x, aux = blk(x, mask, pair_bias, feats)
            aux_sum["balance_loss"] = aux_sum["balance_loss"] + aux["balance_loss"]
            usage.append(aux["expert_usage"])
            for k in SCALARS:
                if k in aux:
                    scal[k].append(aux[k])
        aux_sum["expert_usage"] = torch.stack(usage).mean(0)
        for k in SCALARS:
            if scal[k]:
                v = torch.stack(scal[k])
                # the widest block, not the mean of the widest, for max_width
                aux_sum[k] = v.max() if k == "max_width" else v.mean()
        return self.norm_out(x), aux_sum

    def forward(self, x0: torch.Tensor, mask: torch.Tensor,
                feats: Optional[RouterFeatures] = None,
                n_loops: Optional[int] = None,
                pair_bias_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
                supervise: Optional[Callable[[torch.Tensor, int], None]] = None,
                ) -> Tuple[torch.Tensor, Dict]:
        """Run the stack `n_loops` times; differentiate only the last.

        `pair_bias_fn(h, loop)` supplies the electrostatic bias from the previous
        iterate's distance estimate and must return `None` at loop 0.
        `supervise(h, loop)` receives every intermediate iterate, **detached**,
        so deep supervision trains the heads on them without backpropagating
        through the recurrence.
        """
        cfg = self.cfg
        n = n_loops if n_loops is not None else cfg.n_loops
        h = x0
        pair_bias = None

        # ---- loops 0..n-2: no grad, no activation memory -------------------
        if n > 1:
            with torch.no_grad():
                for it in range(n - 1):
                    f = feats
                    if f is not None:
                        f = RouterFeatures(**{**f.__dict__, "recycle": it})
                    inp = x0 + self.recycle_proj(self.recycle_norm(h)) if it else x0
                    h, _ = self._one_pass(inp, mask, pair_bias, f)
                    if pair_bias_fn is not None:
                        pair_bias = pair_bias_fn(h, it)
                    if supervise is not None:
                        supervise(h.detach(), it)
            h = h.detach()
            if pair_bias is not None:
                pair_bias = pair_bias.detach()

        # ---- final loop: the only one that carries a gradient --------------
        it = n - 1
        f = feats
        if f is not None:
            f = RouterFeatures(**{**f.__dict__, "recycle": it})
        inp = x0 + self.recycle_proj(self.recycle_norm(h)) if it else x0
        h, aux = self._one_pass(inp, mask, pair_bias, f)
        if supervise is not None:
            supervise(h, it)
        aux["n_loops"] = n
        aux["effective_layers"] = n * cfg.n_blocks
        return h, aux

    # -- parameter accounting, checkable against §5.4 ----------------------
    def param_counts(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        # `.experts` holds the routed parameters in both flavours: E full
        # SwiGLUs in one, a shared trunk plus E adapters in the other.
        moe_total = sum(p.numel() for b in self.blocks for p in b.ff.experts.parameters())
        moe_active = sum(b.ff.n_active_params for b in self.blocks)
        moe_all = sum(p.numel() for b in self.blocks for p in b.ff.parameters())
        return {"total": total,
                "active": total - moe_all + moe_active,
                "experts_total": moe_total,
                "effective_layers": self.cfg.n_loops * self.cfg.n_blocks}
