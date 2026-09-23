#!/usr/bin/env python3
"""PHAROS, assembled — embeddings, trunk, pair track, heads, physics bias.

This is where ARCHITECTURE v0.2 §5.4's sizing table stops being arithmetic and
becomes a thing that can be counted:

    PHAROS-Small (default)  d=512  16 blocks  8 loops  128 eff. layers  149M / 61M active
    PHAROS-Mini             d=384  12 blocks 12 loops  144 eff. layers   67M / 30M
    Base-v2 (scale-up)      d=768  32 blocks  3 loops   96 eff. layers 1,401M / 269M

`test_pharos.py` asserts the built model reproduces those numbers. A spec whose
parameter count is off by a factor is a spec whose cost estimate is off by the
same factor, and §5.4's "[v0.1 arithmetic, verified exact]" had never been
verified against code.

The wiring the rest of the design implies
-----------------------------------------
* Chemistry enters through **one projection** into the existing width, never by
  widening `d_model` (§4): a fully attributed nucleotide is 58.9 bits against a
  512-dim bf16 token's 8,192, so width is compute, not storage.
* The pair track is fed by the two FULL-attention blocks and selects blocks
  rather than ranking pairs (§7).
* The recycled distance estimate re-enters through the electrostatic bias
  (§6.2), so physics and refinement are coupled rather than sequential.
* At recycle 0 there is no distance estimate: the bias is zero and the router
  sees sequence-only features. That is §5.3's open circularity, represented
  rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import DynamicsConfig, HarmonicEnsemble
from .heads import HeadConfig, PharosHeads
from .shared_moe import SharedMoEConfig
from .motif_bank import MotifBankConfig, load_bank
from .moe import MoEConfig, RouterFeatures
from .trunk import TokenTrunk, TrunkConfig


@dataclass
class PharosConfig:
    d_model: int = 512
    n_blocks: int = 16
    n_loops: int = 8
    n_heads: int = 8
    window: int = 128
    d_pair: int = 128
    # The MoE recipe, shared across the whole family. §5.4 quoted totals and
    # active counts for three models but never these four numbers, which are
    # what determine them -- so its "[v0.1 arithmetic, verified exact]" could
    # not be checked from the document. Solving for them
    # (`scripts/sampling/solve_sizing.py`) shows the three rows imply three
    # mutually inconsistent recipes, with active fractions of 40.9%, 44.8% and
    # 19.2%; no fixed recipe produces that spread, because a fixed recipe holds
    # the active fraction roughly constant as d and depth scale.
    #
    # This recipe reproduces §5.4's ACTIVE column -- Small 62.7M against 61M,
    # Base-v2 267.9M against 269M -- and active parameters are what the cost
    # model depends on. The totals differ, and the built ones are the true ones.
    # It is fine-grained with shared-expert isolation, which is both halves of
    # the DeepSeek-MoE arrangement rather than the one half defect #21 copied.
    n_experts: int = 32
    d_expert: int = 256          # d_model // 2 at Small
    top_k: int = 4
    n_shared: int = 2
    n_symbols: int = 13
    n_mod: int = 371
    d_chem: int = 24
    max_length: int = 4608        # D20: the longest RNA chain ever solved is 4,450
    dropout: float = 0.0

    #: When set, the trunk uses shared-adapter experts with nucleus routing
    #: instead of independent experts with fixed top-k.
    shared_experts: bool = False
    grad_checkpoint: bool = False
    rank: int = 16
    max_k: int = 32
    threshold_rel: float = 1.0

    @classmethod
    def shared400(cls) -> "PharosConfig":
        """~394M total, ~302M ACTIVE, with 512 experts that share one network.

        The same budget as `base400` spent differently, and the difference is
        in the active count. base400 buys 48 independent experts at d_expert
        192 and activates 126M of its 405M parameters; this buys **512 experts
        at d_expert 2304** and activates **302M of 394M**, because an expert
        here is a gain-and-bias over a shared SwiGLU (4*d_ff + d = 9.9k
        parameters) rather than its own network (3*d*d_ff = 5.3M). Experts stop
        being where the parameters go, so the parameters go into the network
        every token actually runs through.

        That is the whole point. 61M active was the previous configuration's
        real capacity; the rest sat idle in experts a given token never
        selected. Sharing converts dormant expert parameters into active ones
        and lets the count of experts grow at the same time -- 512 here against
        48 -- so specialisation gets *finer* while capacity gets *denser*.

        Routing is nucleus rather than top-k: width is 1..max_k per token, so a
        poly-U tract fires one expert and a four-way junction fires eight. The
        argmax always fires, so the width is never zero.

        Wide beats deep for the same parameters on this hardware. d_expert 2304
        at 18 blocks and d_model 768 counts the same as 2048 at 20 blocks but
        runs larger matmuls, which is what actually keeps the CUDA cores busy.
        """
        c = cls(d_model=768, n_blocks=18, n_loops=8, n_heads=12, window=128,
                d_pair=160, n_experts=512, d_expert=2304, top_k=6, n_shared=1)
        c.shared_experts = True
        # max_k = n_experts: a token may genuinely fire all 512. Merging
        # makes that the same matmul as firing one, so the only reason left to
        # cap the width would be to force specialisation -- and the nucleus
        # threshold already does that, adaptively, per token.
        c.max_k, c.threshold_rel = 512, 1.0
        # 12 MiB/token without it, 0.9 with; the recompute costs ~30% and buys
        # 13x the batch, and on this card the bigger batch is worth more
        c.grad_checkpoint = True
        return c

    @classmethod
    def base400(cls) -> "PharosConfig":
        """~400M total, ~126M active. The configuration the rebuild trains.

        Small's MoE was 32 experts x d_expert 256 at d_model 512: 239.6M total,
        63.4M active, and the router stayed within 3.2% of uniform for the first
        500M tokens. Two changes, for two different reasons.

        **Wider model, d_model 512 -> 640.** Measured MFU on Small was 6.6% with
        only ~18% of GPU time in tensor-core GEMMs: at d=512 the matmuls are too
        small to saturate an A100 and the stack is launch-bound. Width is the
        lever that fixes that, and it raises active parameters where they do
        the most work.

        **More, narrower experts: 32 x 256 -> 64 x 192.** Finer granularity
        gives the router more to distinguish and each token a smaller, more
        specialised slice; top_k rises 4 -> 6 so active capacity grows with it.
        This is the DeepSeek-MoE finding -- many narrow experts beat few wide
        ones at equal active parameters -- and it directly targets the router
        flatness measured at 96.8% of uniform.

        Counted, not estimated: 404.8M total, 126.2M active (31.2%), against
        Small's 258.5M / 82.3M. Active parameters -- which are what the FLOP
        budget and the cost model actually depend on -- rise 1.53x.
        """
        return cls(d_model=640, n_blocks=18, n_loops=8, n_heads=10, window=128,
                   d_pair=160, n_experts=48, d_expert=192, top_k=6, n_shared=2)

    @classmethod
    def small(cls) -> "PharosConfig":
        return cls()

    @classmethod
    def mini(cls) -> "PharosConfig":
        return cls(d_model=384, n_blocks=12, n_loops=12, d_expert=192, n_heads=6)

    @classmethod
    def base_v2(cls) -> "PharosConfig":
        return cls(d_model=768, n_blocks=32, n_loops=3, d_expert=384, n_heads=12)

    def trunk(self) -> TrunkConfig:
        return TrunkConfig(grad_checkpoint=self.grad_checkpoint, 
            d_model=self.d_model, n_blocks=self.n_blocks, n_loops=self.n_loops,
            n_heads=self.n_heads, window=self.window, dropout=self.dropout,
            moe=(SharedMoEConfig(
                    d_model=self.d_model, d_expert=self.d_expert,
                    n_experts=self.n_experts, n_shared=self.n_shared,
                    rank=self.rank, max_k=self.max_k,
                    threshold_rel=self.threshold_rel, dropout=self.dropout)
                 if self.shared_experts else
                 MoEConfig(d_model=self.d_model, d_expert=self.d_expert,
                           n_experts=self.n_experts, n_shared=self.n_shared,
                           top_k=self.top_k, dropout=self.dropout)))

    def head_cfg(self) -> HeadConfig:
        return HeadConfig(d_model=self.d_model, d_pair=self.d_pair,
                          dropout=self.dropout)


class InputEmbedding(nn.Module):
    """Token + modification + chemistry + position, into `d_model`.

    Chemistry is **one linear projection**, by design (§4). The temptation is to
    widen `d_model` to "make room" for 24 more numbers; the measurement says a
    token is already 139x over-provisioned for the information a nucleotide
    carries, and FFN cost is quadratic in width while an input projection is
    linear. So the 24 dims cost one matrix and nothing else.
    """

    def __init__(self, cfg: PharosConfig):
        super().__init__()
        d = cfg.d_model
        self.tok = nn.Embedding(cfg.n_symbols, d)
        self.mod = nn.Embedding(cfg.n_mod, d, padding_idx=0)
        self.chem = nn.Linear(cfg.d_chem, d)
        self.pos = nn.Embedding(cfg.max_length, d)
        self.norm = nn.LayerNorm(d)
        # Standard transformer init, std 0.02, and it is not cosmetic here.
        # `nn.Embedding` defaults to N(0, 1); the MLM head is TIED to this
        # matrix, so logits are h . W^T over d dims and inherit a standard
        # deviation of ~sqrt(d). At d=512 that is a softmax sharp enough to put
        # the initial masked-token loss at **39.19** against the log(13) = 2.56
        # a fresh model should start from -- the run would open by unlearning
        # its own initialisation.
        for emb in (self.tok, self.mod, self.pos):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.mod.weight[0].zero_()          # padding_idx stays exactly zero

    def forward(self, tokens: torch.Tensor, mod_ids: torch.Tensor,
                chem: torch.Tensor) -> torch.Tensor:
        L = tokens.shape[1]
        p = torch.arange(L, device=tokens.device).clamp(max=self.pos.num_embeddings - 1)
        return self.norm(self.tok(tokens) + self.mod(mod_ids)
                         + self.chem(chem) + self.pos(p)[None])


class ElectrostaticBias(nn.Module):
    """§6.2 — screened Coulomb pair bias from the previous loop's distances.

    `B_elec(r) = -A * q_eff^2 * exp(-kappa r) / r`, with `q_eff = -0.196` for
    A-RNA from Manning condensation (§6.1) and `kappa` from the ionic condition.
    The measured behaviour this must reproduce: as salt rises, the screening
    length shortens 9.61 -> 6.88 A and the bias weakens -0.0097 -> -0.0064. That
    is the mechanism by which an ionic condition changes a prediction, so it is
    computed from `physics.manning`, not learned.
    """

    def __init__(self, learn_scale: bool = True):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1), requires_grad=learn_scale)

    def forward(self, dist: torch.Tensor, kappa: float, q_eff: float = -0.196
                ) -> torch.Tensor:
        r = dist.clamp(min=2.0)
        b = -(q_eff ** 2) * torch.exp(-kappa * r) / r
        return b * torch.exp(self.log_scale)


class Pharos(nn.Module):
    """The whole model, minus training-only machinery."""

    def __init__(self, cfg: PharosConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = InputEmbedding(cfg)
        self.trunk = TokenTrunk(cfg.trunk())
        self.heads = PharosHeads(cfg.head_cfg())
        self.pair_proj = nn.Linear(2 * cfg.d_model, cfg.d_pair)
        self.elec = ElectrostaticBias()
        # §8: retrieval, not memorisation. Queried from the PAIR features --
        # i.e. after pairing is estimated -- never from sequence, because the
        # sequence-only version of this idea was measured at 0.073 sigma. The
        # bank is absent rather than fabricated if it has not been compiled.
        self.motifs = load_bank(MotifBankConfig(d_query=cfg.d_pair,
                                                d_value=cfg.d_pair,
                                                dropout=cfg.dropout))
        self.motif_mix = (nn.Linear(cfg.d_pair, cfg.d_pair)
                          if self.motifs is not None else None)
        # §10: once a stiffness field exists the equilibrium ensemble is nearly
        # free. Equilibrium breathing, not a folding pathway.
        self.dynamics = HarmonicEnsemble(DynamicsConfig(d_model=cfg.d_model,
                                                        dropout=cfg.dropout))
        # Stage 1 of §12.1: masked-token prediction. The output projection is
        # TIED to the input embedding -- the same 13 x d matrix read both ways.
        # Untied it would add 13d parameters that have to learn the identity of
        # a symbol the embedding already knows, and tying is what makes the
        # pretraining and structural vocabularies (D6) one model rather than two.
        self.mlm_norm = nn.LayerNorm(cfg.d_model)
        self.mlm_bias = nn.Parameter(torch.zeros(cfg.n_symbols))

    def forward(self, tokens: torch.Tensor, mod_ids: torch.Tensor,
                chem: torch.Tensor, mask: torch.Tensor,
                pair_index: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                feats: Optional[RouterFeatures] = None,
                n_loops: Optional[int] = None,
                deep_supervision: bool = False,
                mlm: bool = False,
                dynamics: bool = False) -> Dict:
        x = self.embed(tokens, mod_ids, chem)
        iterates = []

        def supervise(h: torch.Tensor, it: int) -> None:
            if deep_supervision:
                iterates.append((it, h))

        h, aux = self.trunk(x, mask, feats, n_loops=n_loops, supervise=supervise)

        pair = None
        if pair_index is not None:
            ii, jj = pair_index
            pair = self.pair_proj(torch.cat([h[0, ii], h[0, jj]], dim=-1))
            if self.motifs is not None:
                retrieved, minfo = self.motifs(pair)
                pair = pair + self.motif_mix(retrieved)
                aux["motif_gate"] = minfo["gate_mean"]
        out = self.heads(h, mask, pair)
        if mlm:
            out["mlm_logits"] = (self.mlm_norm(h) @ self.embed.tok.weight.T
                                 + self.mlm_bias)
        # §10's ensemble is opt-in, and the default is OFF. Its selected
        # inversion is O(L) sequential 6x6 solves -- 1,241 ms at L=981 -- plus
        # an eigendecomposition per step, and nothing supervises it unless the
        # batch carries X-ray B-factors (D12).
        #
        # It defaulted to ON, which is not what opt-in means: MLM pretraining
        # never passes the flag, so stage 1 computed the whole ensemble on
        # every step for an output nothing reads, and **ran out of memory doing
        # it** -- 8.57 GiB requested inside `eigvalsh` with 4.19 GiB free. A
        # default that every caller must remember to switch off is a default
        # that will be paid for by the caller who forgets.
        if dynamics:
            dyn = self.dynamics(h, mask)
            # the structure head already emits K-state weights from the token
            # track; the ensemble's are the physics-derived ones, so they are
            # kept under their own name rather than silently overwriting
            out["ensemble_state_logits"] = dyn.pop("state_logits")
            out.update(dyn)
        out["hidden"] = h
        out["aux"] = aux
        if deep_supervision:
            out["iterates"] = iterates
        return out

    # -- §5.4, countable ---------------------------------------------------
    def param_counts(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        moe_all = sum(p.numel() for b in self.trunk.blocks for p in b.ff.parameters())
        moe_active = sum(b.ff.n_active_params for b in self.trunk.blocks)
        return {
            "total": total,
            "active": total - moe_all + moe_active,
            "embedding": sum(p.numel() for p in self.embed.parameters()),
            "trunk": sum(p.numel() for p in self.trunk.parameters()),
            "heads": sum(p.numel() for p in self.heads.parameters()),
            "experts": sum(p.numel() for b in self.trunk.blocks
                           for p in b.ff.experts.parameters()),
            "effective_layers": self.cfg.n_loops * self.cfg.n_blocks,
        }


def summarise(cfg: PharosConfig, name: str = "") -> Dict:
    m = Pharos(cfg)
    pc = m.param_counts()
    return {"name": name, "d_model": cfg.d_model, "blocks": cfg.n_blocks,
            "loops": cfg.n_loops, **pc}


if __name__ == "__main__":
    rows = [summarise(PharosConfig.small(), "PHAROS-Small"),
            summarise(PharosConfig.mini(), "PHAROS-Mini"),
            summarise(PharosConfig.base_v2(), "Base-v2")]
    print(f"{'model':14s} {'d':>4s} {'blk':>4s} {'loop':>5s} {'eff':>4s} "
          f"{'total':>12s} {'active':>12s}")
    for r in rows:
        print(f"{r['name']:14s} {r['d_model']:4d} {r['blocks']:4d} {r['loops']:5d} "
              f"{r['effective_layers']:4d} {r['total']:12,d} {r['active']:12,d}")
    print("\n§5.4 states: Small 149M/61M, Mini 67M/30M, Base-v2 1,401M/269M")
