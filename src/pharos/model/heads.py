#!/usr/bin/env python3
"""The eleven output heads of ARCHITECTURE v0.2 §9.

v0.1 specified six. v0.2 specified ten, and the four that were added sat on
substantial supervised data that was already on disk and entirely unused.
Head 11 was added later, for the opposite reason: not unused data, but a
quantity nothing was predicting at all.

    1  contact            L x L binary          raw PDB, 10,520 entries
    2  distance           L x L binned          same
    3  structure          coordinates, K states same
    4  secondary          dot-bracket           bpRNA + pdb_hunter, ~126k
    5  reactivity         SHAPE / DMS           Ribonanza, 335,616 profiles
    6  mg_sites           per-residue           raw PDB, 816,270 sites
    7  rigidity           normalised B-factor   X-RAY ONLY, 3.86M nt
    8  disorder           per-residue           unobserved residues, 46,448
    9  geometry           Leontis-Westhof class 103,965 annotated pairs
   10  motif              motif class           RNA 3D Motif Atlas, 667
   11  torsion            eta/theta/chi_tilde   derived from P/C4'/N

**This table is `HEAD_SPEC` below, and `test_pharos.py` parses it out of this
docstring and compares the two.** It is written twice because a docstring
cannot be generated, and it had drifted: §9 renumbered when the diffusion
decoder and the two base-pair heads arrived, and this table still read
"5 Mg2+, 6 rigidity, 7 reactivity, 8 fitness, 9 splicing, 10 base identity"
-- the scheme §9 abandoned -- thirty lines above the corrected `HEAD_SPEC`.
Finding 37 of the 09-26 audit found that drift, fixed it in the TRAINERS, and
recorded that "all references now agree with HEAD_SPEC". This file was not
checked, so the claim was false in the one place the numbering is defined.
The parse test exists so the next such claim is enforced rather than asserted.

`fitness`, `splice_logits` and `base_logits` are produced and §9 does not
number them -- see `EXTRA_HEAD_KEYS`.

Two heads carry constraints that are not stylistic.

**Rigidity (head 7) trains on X-ray B-factors only (D12).** The Mg-rigidity
gradient is monotonic on 1,535 X-ray structures and *not* monotonic on
cryo-EM, where the per-atom B is a fitted display parameter rather than a
measured one. Mixing the two trains the head on a different physical quantity
for 62% of the corpus, so the head carries its own validity mask and
`rigidity_mask` is not optional.

**Torsions (head 11) are derived, not stored.** The shards carry P, C4' and
the glycosidic N -- not enough for the classical alpha..zeta, exactly enough
for the eta/theta pseudotorsions of Duarte & Pyle, which were defined on
these atoms for this reason. They are computed from the batch's own
coordinates at each step rather than written into the corpus: a torsion is a
deterministic function of coordinates already present, so a stored copy
would be a second source of truth with nothing to gain. See
`pharos.data.torsions`, whose torch path is checked against a numpy
reference to 1e-9 and whose sign convention is pinned by planar cis/trans
cases -- the first draft had every angle 180 degrees from IUPAC, which is
self-consistent and therefore invisible to everything except that test.

**Base identity is free supervision.** `N_struct` residues have their ribose
modelled but their identity unassigned, so predicting the base is a task with
labels that cost nothing to produce and gradients that flow through the same
trunk. v0.1 mapped them to `N` and discarded the signal. §9 does not number
this head; it is `base_logits` in `EXTRA_HEAD_KEYS`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import DiffusionConfig, DiffusionStructureHead


#: Head 2's bin edges in angstrom, left-closed, with everything beyond the
#: last edge in bin 39.
#:
#: THE UNIFORM SCHEME COLLAPSED THE HEAD, and the histogram says why. Head 2
#: is scored on the pair population the CONTACT head samples: every true
#: contact plus uniform random negatives at |i-j| >= 4. Measured over 891,804
#: such pairs from 394 training chains, the old `floor(d - 2)` clamped to
#: 2-41 A put **41.88% of all mass in the single catch-all bin**, all of it
#: negatives, while no other bin held more than 3.74%. A head predicting that
#: one bin scores the majority rate, which is exactly what finding 75
#: measured: `dist_acc` 0.478 against `dist_major` 0.478, macro 0.026 against
#: a 1/39 = 0.0256 chance.
#:
#: Narrowing the range makes it WORSE, which was the obvious fix and the
#: wrong one: 2-22 A at 0.5 A pushes more of the population into the
#: catch-all and takes the majority to 68.83%. The scheme scored best on the
#: measured distribution is equal-frequency, which reaches the theoretical
#: maximum entropy ln(40) with a 2.50% majority and by construction has no
#: class to collapse onto.
#:
#: These edges are the hybrid: eight fixed 0.875 A bins over 3-10 A, then 32
#: equal-frequency bins above. That costs 3.5% of the maximum entropy and
#: buys 6.7x the resolution in the base-pairing range, where a Watson-Crick
#: pair has a characteristic C4'-C4' distance and a 5.88 A first bin cannot
#: see it. Majority 3.01%, entropy 3.5583 of a possible 3.6889 nats.
#:
#: FIXED CONSTANTS, derived once from the TRAINING split. Recomputing them
#: per batch would make the target non-stationary -- the same label meaning a
#: different distance from one step to the next -- and deriving them from
#: anything but train would leak.
DISTANCE_BIN_EDGES: Tuple[float, ...] = (
    3.0000, 3.8750, 4.7500, 5.6250, 6.5000,
    7.3750, 8.2500, 9.1250, 10.0001, 11.3619,
    12.2427, 13.0706, 14.3230, 15.1573, 16.3933,
    17.5547, 18.7989, 20.6606, 22.9835, 25.0816,
    26.8192, 28.8208, 31.0584, 33.4061, 35.8351,
    38.2468, 40.7319, 43.2079, 45.7963, 48.7638,
    51.8173, 54.8410, 58.1599, 62.1919, 66.3752,
    70.6863, 76.2620, 81.5226, 88.6063, 100.7812,
)


def distance_bin(d):
    """Angstrom -> bin index, for a tensor or an array. The ONE definition.

    The trainer used to inline `floor(d - 2).clamp(0, nb - 1)`. One equation
    written in one place, because two copies of a binning rule drift and the
    label silently changes meaning -- the same defect as `ElectrostaticBias`
    carrying its own copy of `b_elec`.
    """
    import torch
    if torch.is_tensor(d):
        e = torch.as_tensor(DISTANCE_BIN_EDGES, device=d.device, dtype=d.dtype)
        return torch.clamp(torch.searchsorted(e.contiguous(), d.contiguous(),
                                              right=True) - 1,
                           0, len(DISTANCE_BIN_EDGES) - 1)
    import numpy as _np
    return _np.clip(_np.searchsorted(DISTANCE_BIN_EDGES, d, side="right") - 1,
                    0, len(DISTANCE_BIN_EDGES) - 1)


@dataclass
class HeadConfig:
    d_model: int = 512
    d_pair: int = 128
    #: Distance bins. NOT uniform -- see `DISTANCE_BIN_EDGES` above for the
    #: measured reason. Must equal `len(DISTANCE_BIN_EDGES)`, which the test
    #: module asserts.
    n_distance_bins: int = 40
    #: Head 9. Leontis-Westhof pair families, `_ndb_struct_na_base_pair.
    #: hbond_type_12` in every RNA mmCIF: 12 families (the three edges --
    #: Watson-Crick, Hoogsteen, Sugar -- crossed with cis/trans orientation)
    #: plus class 0 for "annotated but unclassified", which the archive writes
    #: as `?` and which is 6.9% of the 103,965 annotated pairs on the sample.
    #: A pair that is not base-paired at all is handled by the mask, not by a
    #: class: absence of a pair is not a kind of pair.
    n_lw_classes: int = 13
    #: Head 11. Backbone pseudotorsions, two per residue plus a glycosidic
    #: pseudo-angle: eta, theta, chi_tilde. See `pharos.data.torsions` for
    #: why these three and not the classical alpha..zeta -- the corpus stores
    #: P, C4' and N, and eta/theta is the representation defined on exactly
    #: those atoms. Each is emitted as an unnormalised (sin, cos) pair.
    n_torsions: int = 3
    #: Head 10. Motif kind from the RNA 3D Motif Atlas bank on disk -- 413
    #: internal loops and 254 hairpin loops over 667 entries -- plus a "neither"
    #: class, because most pairs are in neither and a posterior that cannot say
    #: so is forced to guess between two wrong answers.
    n_motif_classes: int = 3
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
    """contact, distance and Leontis-Westhof class, on the sparse pairs.

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
        # lw_logits -- Leontis-Westhof geometry class.
        #
        # A contact says two residues touch and a distance says how far apart.
        # Neither says HOW they are paired, and for RNA that is most of the
        # information: a cis Watson-Crick/Watson-Crick pair builds a helix, a
        # trans Hoogsteen/Sugar-edge pair builds a tertiary contact, and the
        # two have the same C4'-C4' distance. Predicting the family is what
        # separates a model that knows RNA is double-stranded from one that
        # knows RNA folds.
        self.geometry = _mlp(cfg.d_pair, cfg.d_pair, cfg.n_lw_classes, cfg.dropout)
    def forward(self, pair: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"contact_logit": self.contact(pair).squeeze(-1),
                "distance_logits": self.distance(pair),
                "lw_logits": self.geometry(pair)}


class StructureHead(nn.Module):
    """The RETAINED coordinate regressor, kept for the §5 ablation.

    Not head 3. `HEAD_SPEC` entry 3 is `DiffusionStructureHead`, which
    replaced this; `PharosHeads` does not instantiate this class and the
    ablation the architecture doc cites is the only caller.

    Coordinates for K states (§10's ensemble, not one structure).

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
    """Everything defined per nucleotide. Keys, not numbers: `HEAD_SPEC`
    below is the numbering, and writing it a second time here is how the
    module docstring came to disagree with it."""

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        d = cfg.d_model
        self.secondary = _mlp(d, d, cfg.n_ss_symbols, cfg.dropout)   # ss_logits
        self.mg_site = _mlp(d, d // 2, 1, cfg.dropout)               # mg_logit
        self.rigidity = _mlp(d, d // 2, 1, cfg.dropout)              # rigidity
        # reactivity: SHAPE and DMS are different chemistries probing
        # different atoms, so they get separate outputs rather than one scalar
        self.reactivity = _mlp(d, d, 2, cfg.dropout)
        self.base_identity = _mlp(d, d // 2, cfg.n_bases, cfg.dropout)
        # motif_logits -- motif-class posterior, per RESIDUE.
        #
        # The motif bank has always RETRIEVED a motif and mixed its descriptor
        # into the representation without ever committing to an answer that
        # could be scored. This head commits: hairpin loop, internal loop, or
        # neither. That makes the bank's contribution falsifiable rather than
        # merely present.
        #
        # Per residue and not per pair, because loop membership is a property
        # of a nucleotide. Putting it on the pair track would have forced a
        # self-pair -- `pair_proj(cat(h, h))` -- which is a shape that
        # typechecks and means nothing.
        self.motif = _mlp(d, d, cfg.n_motif_classes, cfg.dropout)
        # torsion_sincos -- head 11, the LOCAL geometry.
        #
        # Every other structural head here is non-local: a contact, a
        # distance, a pair family. None of them says what shape the backbone
        # takes between two residues, and that is most of what distinguishes
        # a fold from a contact map -- many geometries satisfy the same
        # contacts, which is the ambiguity the diffusion head was added to
        # resolve and the one it has the least supervision for.
        #
        # Emitted as an UNNORMALISED (sin, cos) per angle, AlphaFold-style:
        # regressing the angle itself puts a discontinuity at +-pi, where
        # 179 degrees and -179 are two degrees apart and a squared error
        # calls them 358. The loss normalises and adds a term pulling the
        # norm to 1, so the magnitude stays free to express confidence
        # during training without the direction being distorted by it.
        self.torsion = _mlp(d, d, cfg.n_torsions * 2, cfg.dropout)
        self.n_torsions = cfg.n_torsions

    def forward(self, tok: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "ss_logits": self.secondary(tok),
            "mg_logit": self.mg_site(tok).squeeze(-1),
            "rigidity": self.rigidity(tok).squeeze(-1),
            "reactivity": self.reactivity(tok),
            "base_logits": self.base_identity(tok),
            "motif_logits": self.motif(tok),
            "torsion_sincos": self.torsion(tok).view(
                *tok.shape[:-1], self.n_torsions, 2),
        }


class FunctionHeads(nn.Module):
    """fitness and splicing, the two that speak to function.

    Neither is numbered by §9 -- they are in `EXTRA_HEAD_KEYS`. This
    docstring called them "heads 8 and 9", which are disorder and
    Leontis-Westhof geometry.

    Fitness is the second-largest labelled channel after probing (620,372
    measurements) and was entirely unused by v0.1. It is a property of a
    *variant relative to a reference*, so the head reads a pooled representation
    rather than a per-residue one; splicing is per-position, because a splice
    site is a position.
    """

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        d = cfg.d_model
        self.fitness = _mlp(d, d, 1, cfg.dropout)              # fitness
        self.splice = _mlp(d, d // 2, cfg.n_splice_classes,
                           cfg.dropout)                       # splice_logits

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
        # Head 3 is a diffusion decoder, not a coordinate regression. The old
        # StructureHead is kept below for the ablation the architecture doc
        # cites, and is not wired in.
        self.structure = DiffusionStructureHead(DiffusionConfig(
            d_model=cfg.d_model, d_pair=cfg.d_pair,
            n_layers=getattr(cfg, "n_diff_layers", 6),
            n_heads=getattr(cfg, "n_heads", 8),
            dropout=cfg.dropout))
        self.residue = ResidueHeads(cfg)
        self.function = FunctionHeads(cfg)

    def forward(self, tok: torch.Tensor, mask: torch.Tensor,
                pair: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        out.update(self.residue(tok))
        # The diffusion head does not emit coordinates in a forward pass: it
        # denoises, so it needs either a target (training) or a sampling loop
        # (inference). Both are driven by the trainer, which owns the noise
        # schedule; a head that silently sampled 50 denoising steps inside
        # every forward would make an MLM step 50x slower for nothing.
        out.update(self.function(tok, mask.float()))
        if pair is not None:
            out.update(self.pair(pair))
        return out

    #: `PharosHeads.loss` used to live here: a static method that summed every
    #: head under a documented weight table -- contact 1.0, distance 1.0,
    #: ss 0.5, mg 0.3, rigidity 0.3, reactivity 0.5, base 0.2, fitness 0.5,
    #: splice 0.5, coords 1.0 -- and silently skipped any term that came out
    #: non-finite.
    #:
    #: Nothing ever called it. Not `train_pharos.py`, not
    #: `train_sequence_stages.py`, not a test. The real weights are assembled
    #: inline in each trainer and they are NOT those: stage 5 uses mg 0.3,
    #: rigidity 0.3, fluctuation 0.1, base 0.2, motif 0.3, lw 0.5, contact 1.0,
    #: distance 0.5, structure `--structure-weight`, balance 1.0. The one that
    #: disagreed outright was distance, 1.0 here against 0.5 in the trainer,
    #: and this table was the one a reader would find first.
    #:
    #: It also carried the project's only non-finite guard, on a path no
    #: gradient ever took, while the four trainers that do take gradients had
    #: none. That is now `pharos.train.guard.check_loss`, called by all four.
    #:
    #: Deleted rather than wired up, because two loss definitions is how they
    #: drift: this one had already drifted, and nothing could tell.

#: §9's table, as data, so a test can assert the implementation matches it.
#:
#: It had stopped being §9's table. §9 renumbered when the diffusion decoder and
#: the two base-pair heads arrived -- it now reads 5 reactivity, 6 Mg, 7
#: rigidity, 8 disorder, 9 base-pair geometry, 10 motif-class posterior -- and
#: this list still carried the old 5 Mg, 6 rigidity, 7 reactivity, 8 fitness, 9
#: splicing, 10 base_identity. So the one test whose job is to assert that the
#: implementation matches the specification was comparing it against a stale
#: private copy, and neither of the two newest heads was in it to be checked at
#: all. The same failure as the rest of the audit: the check existed, it ran,
#: and it could not have caught the thing it was for.
#:
#: `forward` is False for head 3, whose output is coordinates: the diffusion
#: decoder produces them from `sample()` and scores them in `loss()`, and a
#: plain forward pass never emits a `coords` key. Asserting one does is how
#: this entry failed for as long as the diffusion head has existed.
HEAD_SPEC: List[Dict] = [
    {"n": 1, "name": "contact", "output": "L x L binary", "key": "contact_logit"},
    {"n": 2, "name": "distance", "output": "L x L binned", "key": "distance_logits"},
    {"n": 3, "name": "structure", "output": "coordinates, K states",
     "key": "coords", "forward": False},
    {"n": 4, "name": "secondary", "output": "dot-bracket", "key": "ss_logits"},
    {"n": 5, "name": "reactivity", "output": "SHAPE/DMS", "key": "reactivity"},
    {"n": 6, "name": "mg_sites", "output": "per-residue", "key": "mg_logit"},
    {"n": 7, "name": "rigidity", "output": "normalised B", "key": "rigidity"},
    {"n": 8, "name": "disorder", "output": "per-residue", "key": "disorder_logit"},
    {"n": 9, "name": "geometry", "output": "Leontis-Westhof class, per pair",
     "key": "lw_logits"},
    {"n": 10, "name": "motif", "output": "motif class, per residue",
     "key": "motif_logits"},
    {"n": 11, "name": "torsion",
     "output": "eta/theta/chi_tilde as (sin, cos), per residue",
     "key": "torsion_sincos"},
]

#: Heads the model produces that §9 no longer numbers. Listed rather than left
#: out: an output nothing names is an output nothing checks, and these three
#: were exactly that once §9 renumbered past them.
EXTRA_HEAD_KEYS: List[str] = ["fitness", "splice_logits", "base_logits"]
