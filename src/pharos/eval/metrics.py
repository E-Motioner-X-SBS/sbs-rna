#!/usr/bin/env python3
"""The metrics CASP and RNA-Puzzles actually score on.

Four of them, because they fail in different directions and a model that is
good only by one of them is not good.

**RMSD** after optimal superposition is the number everybody quotes and the
one that lies most. It is dominated by the worst-placed residue and it grows
with length, so 8 Angstrom on a 30-mer is bad and 8 on a 700-mer is a result.
It is here because it is the common currency, not because it is informative.

**TM-score** normalises that away. `d0` grows with length, so the score is
comparable across targets, and the 0.45 threshold is the field's rough line
between "the fold is right" and "the fold is not right". It needs a search
over superpositions -- the optimal one for TM-score is not the one that
minimises RMSD, because TM-score deliberately down-weights outliers and RMSD
is driven by them.

**lDDT** needs no superposition at all. It asks, for every pair of atoms close
in the reference, whether the prediction preserves that distance. A model that
gets every local contact right but hinges one domain scores terribly on RMSD
and well on lDDT, and for RNA -- which is modular, and whose junctions hinge --
that distinction is the whole game.

**INF** is the RNA-specific one, and the only one that knows RNA is made of
base pairs. A prediction can have a respectable TM-score with the wrong
secondary structure; INF catches that, because it is the Matthews correlation
over the base-pair set rather than over coordinates.

Every function takes a mask and honours it. Unresolved reference atoms are
excluded from every sum, never zero-filled -- see `structure.py`.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

#: lDDT's inclusion radius and its four tolerance thresholds, in Angstrom.
#: The thresholds are the protein-standard 0.5/1/2/4; the radius is 15, also
#: the standard. RNA is more extended than protein, so a larger radius pulls in
#: pairs whose distance says little about local quality.
LDDT_R0 = 15.0
LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
#: GDT-TS cutoffs. The protein convention is 1/2/4/8; RNA is scored on
#: 1/2/4/8 as well by CASP's RNA assessors, so they are kept.
GDT_CUTOFFS = (1.0, 2.0, 4.0, 8.0)


# ---------------------------------------------------------------------------
# superposition
# ---------------------------------------------------------------------------

def kabsch(mobile: np.ndarray, target: np.ndarray,
           weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation and translation carrying `mobile` onto `target`.

    Reflections are excluded. A reflection minimises the same least-squares
    objective and preserves every distance, so an RMSD check cannot detect one
    -- but it mirrors chirality, and a mirrored RNA has the wrong sugar pucker
    and a left-handed helix. The determinant correction is not optional.
    """
    w = np.ones(len(mobile)) if weights is None else np.asarray(weights, float)
    w = w / max(w.sum(), 1e-9)
    mc = (mobile * w[:, None]).sum(0)
    tc = (target * w[:, None]).sum(0)
    h = ((mobile - mc) * w[:, None]).T @ (target - tc)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, tc - r @ mc


def superpose(mobile: np.ndarray, target: np.ndarray,
              weights: Optional[np.ndarray] = None) -> np.ndarray:
    r, t = kabsch(mobile, target, weights)
    return mobile @ r.T + t


def rmsd(pred: np.ndarray, ref: np.ndarray, *, superimpose: bool = True) -> float:
    """Root-mean-square deviation over `(N, 3)` point sets, in Angstrom."""
    if len(pred) == 0:
        return float("nan")
    p = superpose(pred, ref) if superimpose else pred
    return float(np.sqrt(((p - ref) ** 2).sum(-1).mean()))


# ---------------------------------------------------------------------------
# TM-score
# ---------------------------------------------------------------------------

def d0_rna(length: int) -> float:
    """RNA-align's length normalisation, `0.6*sqrt(L-0.5) - 2.5`.

    The protein formula (`1.24*(L-15)^(1/3) - 1.8`) is calibrated on protein
    compactness and is too generous for RNA, which is more extended per
    residue; using it inflates RNA TM-scores by roughly 0.1 at typical lengths.
    Floored at 0.3 so short targets stay scorable.
    """
    if length <= 1:
        return 0.3
    return max(0.6 * np.sqrt(max(length - 0.5, 0.0)) - 2.5, 0.3)


def tm_score(pred: np.ndarray, ref: np.ndarray, *,
             d0: Optional[float] = None, n_norm: Optional[int] = None) -> float:
    """TM-score with the standard iterative fragment search.

    The superposition maximising TM-score is not the one minimising RMSD: the
    `1/(1+(d/d0)^2)` kernel saturates, so moving an already-hopeless residue
    closer buys nothing while RMSD would chase it. Taking the Kabsch fit over
    all residues therefore *understates* TM-score, sometimes badly. The search
    below is Zhang and Skolnick's: seed on fragments of decreasing length at
    every offset, and from each seed iterate "superimpose, keep what is close,
    superimpose again" to a fixed point.
    """
    n = len(ref)
    if n == 0 or len(pred) != n:
        return float("nan")
    norm = n if n_norm is None else n_norm
    dd0 = d0_rna(norm) if d0 is None else d0
    d0sq = dd0 * dd0

    best = 0.0
    frag = n
    while frag >= 4:
        for start in range(0, n - frag + 1, max(1, frag // 2)):
            sel = np.arange(start, start + frag)
            cut = dd0 + 1.0
            for _ in range(20):
                if len(sel) < 3:
                    break
                r, t = kabsch(pred[sel], ref[sel])
                dist2 = ((pred @ r.T + t - ref) ** 2).sum(-1)
                score = float((1.0 / (1.0 + dist2 / d0sq)).sum() / norm)
                best = max(best, score)
                nxt = np.where(dist2 < cut * cut)[0]
                if len(nxt) < 3:
                    cut += 0.5
                    continue
                if len(nxt) == len(sel) and np.array_equal(nxt, sel):
                    break
                sel = nxt
        frag = frag // 2 if frag > 4 else 0
    return best


def gdt_ts(pred: np.ndarray, ref: np.ndarray) -> float:
    """GDT-TS: mean fraction of residues under 1/2/4/8 A, best superposition.

    Uses the TM-score search to pick the superposition, for the same reason:
    the RMSD-optimal fit is dragged around by residues no cutoff will ever
    include.
    """
    n = len(ref)
    if n == 0:
        return float("nan")
    best = 0.0
    frag = n
    while frag >= 4:
        for start in range(0, n - frag + 1, max(1, frag // 2)):
            sel = np.arange(start, start + frag)
            for _ in range(20):
                if len(sel) < 3:
                    break
                r, t = kabsch(pred[sel], ref[sel])
                dist = np.sqrt(((pred @ r.T + t - ref) ** 2).sum(-1))
                best = max(best, float(np.mean([(dist < c).mean()
                                                for c in GDT_CUTOFFS])))
                nxt = np.where(dist < GDT_CUTOFFS[-1])[0]
                if len(nxt) < 3 or np.array_equal(nxt, sel):
                    break
                sel = nxt
        frag = frag // 2 if frag > 4 else 0
    return best


# ---------------------------------------------------------------------------
# lDDT
# ---------------------------------------------------------------------------

def lddt(pred: np.ndarray, ref: np.ndarray, *,
         mask: Optional[np.ndarray] = None, r0: float = LDDT_R0,
         exclude_neighbours: int = 0,
         per_residue: bool = False) -> float | np.ndarray:
    """Local Distance Difference Test over `(N, 3)` point sets.

    No superposition: it compares the two distance matrices directly, so a
    correct local geometry scores well even if a hinge has swung. That is the
    right behaviour for RNA, where a three-way junction can be locally perfect
    and globally 20 Angstrom out.

    `exclude_neighbours` drops `|i-j| <= k` pairs. At k=0 the trivially-correct
    bonded distances are included, which is the lDDT convention for atoms;
    raise it to score only non-local contacts.
    """
    n = len(ref)
    if n < 2:
        return np.full(n, np.nan) if per_residue else float("nan")
    m = np.ones(n, bool) if mask is None else np.asarray(mask, bool)

    dr = np.linalg.norm(ref[:, None] - ref[None, :], axis=-1)
    dp = np.linalg.norm(pred[:, None] - pred[None, :], axis=-1)
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    valid = (dr < r0) & (sep > exclude_neighbours) & m[:, None] & m[None, :]

    diff = np.abs(dr - dp)
    hit = sum((diff < t) for t in LDDT_THRESHOLDS) / len(LDDT_THRESHOLDS)
    if per_residue:
        den = valid.sum(1)
        out = np.divide((hit * valid).sum(1), den,
                        out=np.full(n, np.nan), where=den > 0)
        return out
    den = valid.sum()
    return float((hit * valid).sum() / den) if den else float("nan")


# ---------------------------------------------------------------------------
# base pairs and INF
# ---------------------------------------------------------------------------

def inf(pred_pairs: set, ref_pairs: set) -> float:
    """Interaction Network Fidelity: the MCC over a base-pair set.

    RNA-Puzzles' own metric. MCC rather than F1 because the negative class --
    the O(L^2) pairs that are correctly *not* predicted -- carries real
    information for RNA: predicting every pair would give perfect recall and a
    useless structure, and MCC punishes that where F1 is lenient.

    Returns NaN when the reference has no pairs, which is honest: INF is
    undefined for an unpaired target rather than 0 or 1.
    """
    tp = len(pred_pairs & ref_pairs)
    fp = len(pred_pairs - ref_pairs)
    fn = len(ref_pairs - pred_pairs)
    if not ref_pairs:
        return float("nan")
    den = np.sqrt(float(tp + fp) * (tp + fn)) if (tp + fp) and (tp + fn) else 0.0
    return float(tp / den) if den > 0 else 0.0


def clash_score(coords: np.ndarray, *, min_dist: float = 2.0,
                exclude_neighbours: int = 1) -> float:
    """Fraction of non-bonded atom pairs closer than `min_dist`.

    Diffusion samplers produce self-intersecting chains when the denoiser is
    undertrained, and no superposition metric notices: TM-score and lDDT are
    both happy with a structure that passes through itself. This is the cheap
    check that it does not.
    """
    flat = coords.reshape(-1, 3)
    n = len(flat)
    if n < 2:
        return 0.0
    d = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1)
    per = coords.shape[1] if coords.ndim == 3 else 1
    res = np.arange(n) // per
    sep = np.abs(res[:, None] - res[None, :])
    off = ~np.eye(n, dtype=bool) & (sep > exclude_neighbours)
    return float(((d < min_dist) & off).sum() / max(off.sum(), 1))
