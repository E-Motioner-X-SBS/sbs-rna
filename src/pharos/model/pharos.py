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
from .diffusion import DiffusionPairFeatures
from .heads import HeadConfig, PharosHeads
from .shared_moe import MEASURED_WIDTH_SHARED400, SharedMoEConfig
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
    #: Mean nucleus-routing width, for the ACTIVE parameter count only. See
    #: `SharedMoEConfig.typical_width`; `None` falls back to `max_k // 4`.
    typical_width: Optional[float] = None
    #: Finding 103: make the twelve causal `gdn` blocks bidirectional.
    #: Default OFF; zero-gated, so turning it on does not change a loaded
    #: checkpoint's forward pass.
    bidirectional_gdn: bool = False
    #: Finding 109: rounds of triangle multiplicative update on the decoder's
    #: pair representation. 0 disables it. Zero-gated, so switching it on
    #: does not change a loaded checkpoint's forward pass.
    triangle_layers: int = 0
    triangle_c: int = 32

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
        # n_loops 3, not 8.
        #
        # 8 was aspirational: the model is trained at whatever depth the
        # trainer passes, and measured at step 8,750 it reads 1.6632 bits at
        # the 2 loops it saw, 1.9711 at 4 and 2.0064 at 8 -- it keeps 2.9% of
        # its advantage over the unigram baseline at the depth this config
        # advertised. 144 effective layers was a number no checkpoint could
        # deliver.
        #
        # 3 with `--sample-loops` is free: uniform over 1..3 has mean 2.0, and
        # cost is `6 + 2(mean - 1)` FLOPs per active parameter per token, so it
        # is 8.0 -- exactly what a fixed 2 costs. The model becomes usable at
        # one, two and three loops for nothing, and 54 effective layers is a
        # number the run can actually stand behind.
        c = cls(d_model=768, n_blocks=18, n_loops=3, n_heads=12, window=128,
                d_pair=160, n_experts=512, d_expert=2304, top_k=6, n_shared=1)
        c.shared_experts = True
        # max_k = n_experts: a token may genuinely fire all 512. Merging
        # makes that the same matmul as firing one, so the only reason left to
        # cap the width would be to force specialisation -- and the nucleus
        # threshold already does that, adaptively, per token.
        c.max_k, c.threshold_rel = 512, 1.0
        # MEASURED, not max_k // 4. The nucleus router fires 15.28 experts on
        # average at step 7,000; the old estimate said 128 and overstated the
        # active count by 20.3M. See `SharedMoEConfig.typical_width`.
        c.typical_width = MEASURED_WIDTH_SHARED400
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
            bidirectional_gdn=self.bidirectional_gdn,
            moe=(SharedMoEConfig(
                    d_model=self.d_model, d_expert=self.d_expert,
                    n_experts=self.n_experts, n_shared=self.n_shared,
                    rank=self.rank, max_k=self.max_k,
                    threshold_rel=self.threshold_rel, dropout=self.dropout,
                    typical_width=self.typical_width)
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


#: Mechanisms that are BUILT and not WIRED, with the reason, in the shape
#: `test_pharos.py` property 7c uses for the heads. An output nothing reads
#: and a module nothing calls are the same defect; the heads got a pinned
#: declaration during the 09-26 audit and the modules did not.
#:
#: `elec` was the only entry and is now wired -- see `VirtualDistance` and
#: the `pair_bias_fn` in `forward`. Property 7e asserts this dict is TRUE
#: rather than merely present: an entry here must really have no caller, and
#: a module with a caller must not be listed.
UNWIRED: Dict[str, str] = {}


class ElectrostaticBias(nn.Module):
    """§6.2 — the screened-Coulomb pair bias, from `physics.manning`.

    `B_elec(r) = -lam * q_eff^2 * l_B * exp(-r/lambda_D) / r`, with `q_eff`
    and the Debye length from Manning condensation (§6.1) at the given ionic
    condition. The behaviour it reproduces, which `test_manning.py` pins: as
    salt rises the screening length shortens 9.61 -> 6.88 A and the bias
    weakens -0.0097 -> -0.0064.

    **This calls `manning.b_elec` rather than restating it.** It used to
    carry its own copy of the formula, and the copy was a different
    function: it hardcoded `q_eff = -0.196` instead of deriving it, took a
    `kappa` where the canonical form takes a Debye length, and **omitted the
    Bjerrum length l_B = 7.158 A entirely**, making it 7.16x too small. Its
    own docstring said the bias "is computed from `physics.manning`, not
    learned" -- the intent, not the code. Two definitions of one equation is
    how they drift; the same defect as `PharosHeads.loss` in finding 36, in
    the physics instead of the losses.

    `log_scale` is the one learned quantity and starts at 0, so the bias
    enters at exactly its physical magnitude.

    SITE-SPECIFIC CONDENSATION
    --------------------------
    `q_eff` above is Manning's result for an infinite uniformly charged rod,
    so it is one number for every phosphate in the molecule. Real RNA is
    neither infinite nor uniform, and checking §6.2 against `md_rnaions`
    (Hayes et al., the reference code for generalized Manning condensation)
    made the consequence precise: raising Mg2+ from 0 to 15 mM shortens the
    screening length 9.61 -> 7.98 A, and leaves `theta` and `q_eff`
    BIT-IDENTICAL, because `IonicCondition.screening()` evaluates them at
    `z = 1` whatever is in the solution. The whole response to divalent ion
    therefore runs through the Debye length -- and `B_elec` goes as
    `q_eff^2`, so the channel left out is the larger one: 4x against 1.2x.

    That model gives every phosphate its own dynamical `theta_i` and finds
    it by minimising a free energy; there is no closed form for a K+/Mg2+
    mixture. `theta_head` predicts it from the representation instead, and
    is squashed into `[theta(z=1), theta(z=2)] = [0.8044, 0.9022]` so the
    head cannot leave the interval on which "condensed fraction" means
    anything.

    `site_scale` is zero-initialised, in the same discipline as
    `bias_scale`: at init `theta_i` is exactly the rod value for every
    residue and the bias is bit-identical to what it was, so no checkpoint
    changes numerically when this is added.
    """

    def __init__(self, learn_scale: bool = True, d_model: int | None = None):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1), requires_grad=learn_scale)
        self.theta_head = nn.Linear(d_model, 1) if d_model else None
        self.site_scale = nn.Parameter(torch.zeros(1)) if d_model else None

    def site_charges(self, h: torch.Tensor, cond=None) -> torch.Tensor:
        """Per-residue effective charge, `(B, L)`, inside Manning's bounds.

        Written as `q_rod + deviation` rather than as `-(1 - theta)` so the
        thing the head actually supplies -- a departure from the rod value --
        is the thing in the expression. At `site_scale = 0` the deviation is
        identically zero and every residue carries Manning's `q_eff`.

        The gate opens before the head can learn: at `site_scale = 0` the
        derivative with respect to `theta_head`'s parameters is zero, so the
        head is inert until the gate moves. That is finding 64's shape and it
        is deliberate here rather than accidental -- the alternative is a
        non-zero initial deviation, which would change every loaded
        checkpoint's forward pass. `site_scale` itself DOES take gradient at
        init (its derivative is the deviation the head happens to propose),
        and the measured behaviour of the sibling gate `bias_scale` is that
        it reaches 1e-2 within 200 steps of stage 2/3. The probe asserts both
        halves rather than trusting them.
        """
        from ..physics.manning import condensation_bounds, effective_charge
        lo, hi = condensation_bounds()
        u = torch.sigmoid(self.theta_head(h).squeeze(-1))
        return effective_charge() + (hi - lo) * self.site_scale * u

    def forward(self, dist: torch.Tensor, cond=None,
                h: torch.Tensor | None = None) -> torch.Tensor:
        from ..physics.manning import (IonicCondition as _IC, b_elec,
                                       b_elec_sitewise)
        # `lam` stays a TENSOR. Casting it with float() -- as this did until
        # the gradient probe caught it -- severs the graph, and `log_scale`
        # then never receives a gradient however the bias is used.
        lam = torch.exp(self.log_scale)
        d = dist.clamp(min=1.0)
        if self.theta_head is None or h is None:
            return b_elec(d, cond or _IC(), lam=lam)
        q = self.site_charges(h, cond)                       # (B, L)
        return b_elec_sitewise(d, q.unsqueeze(-1), q.unsqueeze(-2),
                               cond or _IC(), lam=lam)


class VirtualDistance(nn.Module):
    """A pairwise distance estimate in angstrom, from a hidden state.

    §6.2 needs "the previous loop's distance estimate", and the architecture
    stopped providing one when head 3 became a denoiser: a diffusion decoder
    emits no coordinates in a forward pass, which is why the bias sat
    unwired (finding 51). The pair track could supply distances, but densely
    that is `L^2 * d_pair` -- 21M pairs at L=4,608, every loop -- which is
    why wiring it was recorded as a costed change.

    This is the cheap form: project each residue to three numbers and take
    the pairwise Euclidean distance. `L^2 * 3` rather than `L^2 * 128`, so
    at L=1,024 it is a 4 MB matrix instead of 537 MB -- and one scalar per
    pair is all an attention bias can use.

    It is a LEARNED estimate, labelled as such. The initial weight scale
    puts the median pairwise separation near 20 A so the screened Coulomb
    starts in the regime it was derived for. The physics on top is NOT
    learned: `b_elec` is closed form from the ionic condition. The only
    gains are `ElectrostaticBias.log_scale` and `FullAttention.bias_scale`,
    and the latter is **zero-initialised**, so switching this on leaves
    every existing checkpoint numerically unchanged until the bias earns its
    way in.
    """

    def __init__(self, d_model: int, d_coord: int = 3, scale_a: float = 20.0):
        super().__init__()
        self.proj = nn.Linear(d_model, d_coord, bias=False)
        # |c_i - c_j| ~ sqrt(2 * d_coord) * std(c) and std(c) = w_std *
        # sqrt(d_model) for a unit-variance h, which the trunk's final
        # LayerNorm guarantees. Solve for the target median separation.
        nn.init.normal_(self.proj.weight,
                        std=scale_a / ((2 * d_coord) ** 0.5 * d_model ** 0.5))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        c = self.proj(h).float()
        return torch.cdist(c, c)


class Pharos(nn.Module):
    """The whole model, minus training-only machinery."""

    def __init__(self, cfg: PharosConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = InputEmbedding(cfg)
        self.trunk = TokenTrunk(cfg.trunk())
        self.heads = PharosHeads(cfg.head_cfg())
        self.pair_proj = nn.Linear(2 * cfg.d_model, cfg.d_pair)
        # Coevolution enters here and nowhere else. A pair carries one scalar --
        # the APC-corrected mutual information of the two alignment columns --
        # and it is projected in through its OWN zero-initialised layer rather
        # than concatenated into `pair_proj`. Two reasons. Concatenating would
        # change that layer's input width, so every checkpoint trained without
        # coevolution would stop loading. And zero-init means the feature starts
        # contributing exactly nothing, so a model that already works does not
        # regress the moment the feature is switched on -- it has to earn its
        # way in. 12% of the corpus has no Rfam family and passes zeros here,
        # which is the same thing as "no information" only because the layer is
        # zero-initialised and the scalar is non-negative.
        self.coev_proj = nn.Linear(1, cfg.d_pair, bias=False)
        nn.init.zeros_(self.coev_proj.weight)
        # The pair representation head 3 reads as an attention bias. Built here
        # rather than inside the head because it is a function of the TRUNK's
        # output, which is what the rest of the pair track is built from too.
        self.diff_pair = DiffusionPairFeatures(
            cfg.d_model, cfg.d_pair,
            triangle_layers=cfg.triangle_layers,
            triangle_c=cfg.triangle_c)
        self.elec = ElectrostaticBias(d_model=cfg.d_model)
        # §6.2's missing half: the distance estimate the bias is a function
        # of. See `VirtualDistance` -- three numbers per residue, so the
        # pair matrix is a scalar field and not a `d_pair` tensor.
        self.vdist = VirtualDistance(cfg.d_model)
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
                dynamics: bool = False,
                ionic=None,
                electrostatics: bool = True) -> Dict:
        x = self.embed(tokens, mod_ids, chem)
        iterates = []

        def supervise(h: torch.Tensor, it: int) -> None:
            if deep_supervision:
                iterates.append((it, h))

        # §6.2, wired. `pair_bias_fn(h, loop)` is called AFTER each loop and
        # its result feeds the NEXT one, so loop 0 runs with `pair_bias=None`
        # exactly as the trunk's contract says -- the circularity §5.3
        # records, represented rather than fabricated.
        #
        # `electrostatics=False` turns it off for the ablation that gives the
        # claim its meaning: the ionic condition is supposed to change what
        # attends to what, and that is only demonstrable against a run where
        # it does not.
        pair_bias_fn = None
        if electrostatics:
            def pair_bias_fn(hh: torch.Tensor, loop: int) -> torch.Tensor:
                # `hh` goes in as well as the distance: the effective charge
                # is per-residue now, not one rod value for the molecule.
                return self.elec(self.vdist(hh), ionic, h=hh)

        h, aux = self.trunk(x, mask, feats, n_loops=n_loops,
                            pair_bias_fn=pair_bias_fn, supervise=supervise)

        pair = None
        if pair_index is not None:
            # `(ii, jj)` indexes ONE sequence and `(bb, ii, jj)` says which.
            # This read `h[0, ii]` unconditionally, so on a batch of more than
            # one it built every pair from sequence 0's hidden states and
            # returned a plausibly-shaped answer about the wrong chain.
            # Nothing in training hit it -- `train_pharos.py` builds its own
            # pairs with a real per-chain `bidx`, because it needs one index
            # set per chain and the coevolution injection -- so the only
            # caller of this path was the test that validates the interface,
            # and it asserts shapes. Two implementations of one thing, and
            # the public one was the broken one.
            if len(pair_index) == 3:
                bb, ii, jj = pair_index
            else:
                ii, jj = pair_index
                if tokens.shape[0] != 1:
                    raise ValueError(
                        f"pair_index=(ii, jj) is ambiguous for a batch of "
                        f"{tokens.shape[0]}: pass (bb, ii, jj) to say which "
                        f"sequence each pair belongs to")
                bb = torch.zeros_like(ii)
            pair = self.pair_proj(torch.cat([h[bb, ii], h[bb, jj]], dim=-1))
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
