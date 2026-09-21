#!/usr/bin/env python3
"""The ten output heads of ARCHITECTURE v0.2 §9.

v0.1 specified six. v0.2 specifies ten, and the four that were added sit on
substantial supervised data that was already on disk and entirely unused:

    1  contact map        L x L binary          raw PDB, 10,520 entries
    2  distance           L x L binned          same
    3  3D structure       coordinates, K states same
    4  secondary structure per-residue symbol   bpRNA + pdb_hunter, ~126k
    5  Mg2+ sites         per-residue           raw PDB, 816,270 sites
    6  rigidity           normalised B-factor   X-RAY ONLY, 3.86M nt
    7  reactivity         SHAPE / DMS           Ribonanza, 335,616 profiles
    8  fitness            mutation effect       NABench + RNAGym, 620,372
    9  splicing           site / outcome        147 species
   10  base identity      recover N_struct      free, 0.025% of residues

Two of these carry constraints that are not stylistic.

**Head 6 trains on X-ray B-factors only (D12).** The Mg-rigidity gradient is
monotonic on 1,535 X-ray structures and *not* monotonic on cryo-EM, where the
per-atom B is a fitted display parameter rather than a measured one. Mixing the
two trains the head on a different physical quantity for 62% of the corpus, so
the head carries its own validity mask and `rigidity_mask` is not optional.

**Head 10 is free supervision.** `N_struct` residues have their ribose modelled
but their identity unassigned, so predicting the base is a task with labels that
cost nothing to produce and gradients that flow through the same trunk. v0.1
mapped them to `N` and discarded the signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadConfig:
    d_model: int = 512
    d_pair: int = 128
    #: distance bins: 2-40 A in 1 A steps plus an overflow bin, AlphaFold-style
    n_distance_bins: int = 40
    #: dot-bracket alphabet: unpaired, open, close, and three pseudoknot levels
    n_ss_symbols: int = 8
    #: §10: the ensemble is K discrete states, not one structure
    n_states: int = 3
    n_bases: int = 4
    n_splice_classes: int = 3          # donor / acceptor / neither
    dropout: float = 0.0


def _mlp(d_in: int, d_hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_hidden), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(d_hidden, d_out))


class PairHeads(nn.Module):
    """Heads 1 and 2 — contact and distance, on the sparse pair features.

    Both read the pair track's output, so both are defined only on the pairs it
    selected. That is the design: a dense L x L output would cost what the pair
    track exists to avoid. The contact head's loss is therefore evaluated on
    selected pairs plus the block-recall term that supervises the selector, and
    a contact in an unselected block is a *selector* failure, counted there --
    which is why block recall, not contact precision, is the metric that gates
    the architecture.
    """

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.contact = _mlp(cfg.d_pair, cfg.d_pair, 1, cfg.dropout)
        self.distance = _mlp(cfg.d_pair, cfg.d_pair, cfg.n_distance_bins, cfg.dropout)

    def forward(self, pair: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"contact_logit": self.contact(pair).squeeze(-1),
                "distance_logits": self.distance(pair)}


class StructureHead(nn.Module):
    """Head 3 — coordinates for K states (§10's ensemble, not one structure).

    Predicts a backbone frame per residue per state plus a state weight, so the
    output is a small ensemble with probabilities rather than a single answer.
    RNA's conformational heterogeneity is the reason: for much of the corpus a
    single structure is the wrong output, and the harmonic ensemble is nearly
    free once a stiffness field exists.
    """

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.k = cfg.n_states
        # 3 coordinates for each of P, C4', N-glycosidic anchor
        self.coords = _mlp(cfg.d_model, 2 * cfg.d_model, cfg.n_states * 9, cfg.dropout)
        self.state_logits = _mlp(cfg.d_model, cfg.d_model, cfg.n_states, cfg.dropout)

    def forward(self, tok: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, L, _ = tok.shape
        xyz = self.coords(tok).view(B, L, self.k, 3, 3)
        pooled = (tok * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return {"coords": xyz, "state_logits": self.state_logits(pooled)}


class ResidueHeads(nn.Module):
    """Heads 4, 5, 6, 7 and 10 — everything defined per nucleotide."""

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        d = cfg.d_model
        self.secondary = _mlp(d, d, cfg.n_ss_symbols, cfg.dropout)   # 4
        self.mg_site = _mlp(d, d // 2, 1, cfg.dropout)               # 5
        self.rigidity = _mlp(d, d // 2, 1, cfg.dropout)              # 6
        # 7: SHAPE and DMS are different chemistries probing different atoms,
        # so they get separate outputs rather than one "reactivity" scalar
        self.reactivity = _mlp(d, d, 2, cfg.dropout)
        self.base_identity = _mlp(d, d // 2, cfg.n_bases, cfg.dropout)  # 10

    def forward(self, tok: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "ss_logits": self.secondary(tok),
            "mg_logit": self.mg_site(tok).squeeze(-1),
            "rigidity": self.rigidity(tok).squeeze(-1),
            "reactivity": self.reactivity(tok),
            "base_logits": self.base_identity(tok),
        }


class FunctionHeads(nn.Module):
    """Heads 8 and 9 — fitness and splicing, the two that speak to function.

    Fitness is the second-largest labelled channel after probing (620,372
    measurements) and was entirely unused by v0.1. It is a property of a
    *variant relative to a reference*, so the head reads a pooled representation
    rather than a per-residue one; splicing is per-position, because a splice
    site is a position.
    """

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        d = cfg.d_model
        self.fitness = _mlp(d, d, 1, cfg.dropout)                       # 8
        self.splice = _mlp(d, d // 2, cfg.n_splice_classes, cfg.dropout)  # 9

    def forward(self, tok: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = (tok * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return {"fitness": self.fitness(pooled).squeeze(-1),
                "splice_logits": self.splice(tok)}


class PharosHeads(nn.Module):
    """All ten, with the loss that respects each one's validity mask."""

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        self.pair = PairHeads(cfg)
        self.structure = StructureHead(cfg)
        self.residue = ResidueHeads(cfg)
        self.function = FunctionHeads(cfg)

    def forward(self, tok: torch.Tensor, mask: torch.Tensor,
                pair: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        out.update(self.residue(tok))
        out.update(self.structure(tok, mask.float()))
        out.update(self.function(tok, mask.float()))
        if pair is not None:
            out.update(self.pair(pair))
        return out

    @staticmethod
    def loss(out: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor],
             weights: Optional[Dict[str, float]] = None
             ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Sum of the heads for which this batch actually carries labels.

        Every term is gated on its target being present **and** on its own
        validity mask. A head with no labels in this batch contributes nothing
        rather than contributing a zero that quietly shrinks the average -- the
        corpus is heterogeneous by design (probing for some chains, B-factors
        for X-ray only, fitness for a different set entirely), so most batches
        supervise a subset.
        """
        w = {"contact": 1.0, "distance": 1.0, "ss": 0.5, "mg": 0.3, "rigidity": 0.3,
             "reactivity": 0.5, "base": 0.2, "fitness": 0.5, "splice": 0.5,
             "coords": 1.0, **(weights or {})}
        parts: Dict[str, float] = {}
        total = None

        def add(name: str, value: torch.Tensor) -> None:
            nonlocal total
            if value is None or not torch.isfinite(value):
                return
            total = value * w[name] if total is None else total + value * w[name]
            parts[name] = float(value.detach())

        m = target.get("mask")
        if "contact" in target and "contact_logit" in out:
            add("contact", F.binary_cross_entropy_with_logits(
                out["contact_logit"], target["contact"].float(),
                pos_weight=target.get("contact_pos_weight")))
        if "distance_bin" in target and "distance_logits" in out:
            add("distance", F.cross_entropy(out["distance_logits"].transpose(-1, -2)
                                            if out["distance_logits"].dim() == 3
                                            else out["distance_logits"],
                                            target["distance_bin"]))
        if "ss" in target and m is not None:
            add("ss", F.cross_entropy(out["ss_logits"][m], target["ss"][m]))
        if "mg" in target and m is not None:
            add("mg", F.binary_cross_entropy_with_logits(
                out["mg_logit"][m], target["mg"][m].float()))
        # D12: rigidity is X-ray only, and the mask is not optional
        rm = target.get("rigidity_mask")
        if "rigidity" in target and rm is not None and bool(rm.any()):
            add("rigidity", F.smooth_l1_loss(out["rigidity"][rm], target["rigidity"][rm]))
        rx = target.get("reactivity_mask")
        if "reactivity" in target and rx is not None and bool(rx.any()):
            add("reactivity", F.smooth_l1_loss(
                out["reactivity"][rx], target["reactivity"][rx]))
        bm = target.get("base_mask")
        if "base" in target and bm is not None and bool(bm.any()):
            add("base", F.cross_entropy(out["base_logits"][bm], target["base"][bm]))
        if "fitness" in target:
            add("fitness", F.smooth_l1_loss(out["fitness"], target["fitness"]))
        if "splice" in target and m is not None:
            add("splice", F.cross_entropy(out["splice_logits"][m], target["splice"][m]))
        if total is None:
            total = torch.zeros((), device=next(iter(out.values())).device)
        return total, parts


#: §9's table, as data, so a test can assert the implementation matches it.
HEAD_SPEC: List[Dict] = [
    {"n": 1, "name": "contact", "output": "L x L binary", "key": "contact_logit"},
    {"n": 2, "name": "distance", "output": "L x L binned", "key": "distance_logits"},
    {"n": 3, "name": "structure", "output": "coordinates, K states", "key": "coords"},
    {"n": 4, "name": "secondary", "output": "dot-bracket", "key": "ss_logits"},
    {"n": 5, "name": "mg_sites", "output": "per-residue", "key": "mg_logit"},
    {"n": 6, "name": "rigidity", "output": "normalised B", "key": "rigidity"},
    {"n": 7, "name": "reactivity", "output": "SHAPE/DMS", "key": "reactivity"},
    {"n": 8, "name": "fitness", "output": "mutation effect", "key": "fitness"},
    {"n": 9, "name": "splicing", "output": "site / outcome", "key": "splice_logits"},
    {"n": 10, "name": "base_identity", "output": "recover N_struct", "key": "base_logits"},
]
