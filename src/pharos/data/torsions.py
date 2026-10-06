#!/usr/bin/env python3
"""Backbone pseudotorsions from the three atoms the corpus actually stores.

The shards carry `BACKBONE_ATOMS = ("P", "C4'", "N")` -- one phosphate, one
sugar carbon, one glycosidic nitrogen per residue. That is not enough for the
six classical backbone torsions (alpha..zeta need O5', C5', C3', O3', none of
which are stored), and writing a head for angles the data cannot supply would
be a head that trains on nothing.

It is exactly enough for the **eta/theta pseudotorsion** representation of
Duarte & Pyle (PNAS 95:11212, 1998), which was designed for this: two angles
per residue, defined only on P and C4', that separate A-form helical from
non-helical backbone as cleanly as the full six-angle description.

    eta_i   = torsion(C4'_{i-1}, P_i,     C4'_i,   P_{i+1})
    theta_i = torsion(P_i,       C4'_i,   P_{i+1}, C4'_{i+1})

Both are undefined at the chain ends -- eta needs residue i-1, theta needs
i+1 -- and both are undefined across a break in the chain, which is why the
validity mask below is computed from the coordinate mask rather than assumed.
A residue whose neighbour is unresolved has no pseudotorsion, and filling one
in with zero would be a fabricated angle in the middle of real ones.

A third angle uses the nitrogen, and is NOT chi:

    chi_tilde_i = torsion(P_i, C4'_i, N_i, P_{i+1})

Real chi is O4'-C1'-N9-C4 and needs four atoms of which we store one. What
this measures is the orientation of the glycosidic bond relative to the
backbone, which is the information chi carries that the backbone alone does
not. It is named `chi_tilde` everywhere rather than `chi` so that nobody
reads a literature chi distribution against it.

Angles are returned in RADIANS on (-pi, pi].
"""
from __future__ import annotations

import numpy as np

#: Index of each atom in the stored `coords` array's atom axis.
P, C4, N = 0, 1, 2

#: The angles this module produces, in the order the head's channels use.
TORSION_NAMES = ("eta", "theta", "chi_tilde")


def dihedral(p0, p1, p2, p3):
    """Torsion angle about the p1-p2 bond, radians on (-pi, pi].

    The `atan2` form rather than `arccos(dot)`: arccos loses the sign, so it
    cannot tell a conformation from its mirror image, and RNA is chiral. The
    sign is the whole point.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    # The bond vectors run ALONG the chain: p0->p1, p1->p2, p2->p3. Writing
    # the first one backwards (p0 - p1, which reads naturally as "from the
    # middle atom") negates n1 and puts every angle exactly 180 degrees from
    # the IUPAC convention. It is self-consistent, so nothing downstream
    # would have complained -- a head would have fitted the offset angles
    # perfectly -- and every comparison against a published eta/theta plot
    # would have been wrong. The planar cis/trans cases in the test module
    # are there to pin the convention, and they are what caught it.
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2n = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-9, None)
    m = np.cross(n1, b2n)
    x = (n1 * n2).sum(-1)
    y = (m * n2).sum(-1)
    return np.arctan2(y, x)


def pseudotorsions(coords, residue_mask=None):
    """`(angles, valid)` for one chain.

    `coords` is `(L, 3, 3)` -- residue, atom (P/C4'/N), xyz.
    `residue_mask` is `(L,)` and marks residues whose coordinates are real.

    Returns `angles (L, 3)` in radians and `valid (L, 3)` saying which of
    them are defined. Undefined entries are `nan`, not zero: zero is a
    perfectly good torsion angle and would be indistinguishable from a real
    one, which is the failure mode this project keeps finding.
    """
    coords = np.asarray(coords, dtype=np.float64)
    L = coords.shape[0]
    if residue_mask is None:
        residue_mask = np.ones(L, dtype=bool)
    residue_mask = np.asarray(residue_mask, dtype=bool)

    ang = np.full((L, 3), np.nan)
    if L < 2:
        return ang, np.zeros((L, 3), dtype=bool)

    prev = np.zeros(L, dtype=bool)
    prev[1:] = residue_mask[:-1] & residue_mask[1:]
    nxt = np.zeros(L, dtype=bool)
    nxt[:-1] = residue_mask[:-1] & residue_mask[1:]

    # eta_i needs i-1 and i+1
    e = np.zeros(L, dtype=bool)
    e[1:-1] = prev[1:-1] & nxt[1:-1]
    if e.any():
        i = np.flatnonzero(e)
        ang[i, 0] = dihedral(coords[i - 1, C4], coords[i, P],
                             coords[i, C4], coords[i + 1, P])
    # theta_i needs i+1 only
    t = np.zeros(L, dtype=bool)
    t[:-1] = nxt[:-1]
    if t.any():
        i = np.flatnonzero(t)
        ang[i, 1] = dihedral(coords[i, P], coords[i, C4],
                             coords[i + 1, P], coords[i + 1, C4])
    # chi_tilde_i needs i+1 only
    if t.any():
        i = np.flatnonzero(t)
        ang[i, 2] = dihedral(coords[i, P], coords[i, C4],
                             coords[i, N], coords[i + 1, P])

    valid = np.isfinite(ang)
    return ang, valid


def sin_cos(angles, valid=None):
    """`(L, 3, 2)` unit vectors, the form a head predicts.

    An angle is a point on a circle and a network that regresses the number
    is penalised for the discontinuity at +-pi -- 179 degrees and -179 are
    two degrees apart and the squared error calls them 358. Predicting
    `(sin, cos)` and comparing on the circle removes that entirely, which is
    the same reason AlphaFold's torsion head emits a two-vector per angle.
    """
    a = np.asarray(angles, dtype=np.float64)
    out = np.stack([np.sin(a), np.cos(a)], axis=-1)
    if valid is not None:
        out = np.where(np.asarray(valid)[..., None], out, np.nan)
    return out


# --------------------------------------------------------------------------
# The batched torch path the trainer uses.
#
# Torsions are a deterministic function of coordinates the batch already
# carries, so they are computed on the fly rather than stored in the shards:
# no rebuild, no second copy to go stale, and one source of truth. The numpy
# functions above stay as the REFERENCE -- the test module asserts the two
# agree to 1e-9 on random input, because a fast path nobody checks against a
# reference is how a silently different angle gets trained on.
# --------------------------------------------------------------------------

def _dihedral_t(p0, p1, p2, p3):
    import torch
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    b2n = b2 / b2.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    m = torch.cross(n1, b2n, dim=-1)
    return torch.atan2((m * n2).sum(-1), (n1 * n2).sum(-1))


def pseudotorsions_torch(coords, residue_mask):
    """`(angles, valid)` for a padded batch.

    `coords` is `(B, L, 3, 3)`, `residue_mask` is `(B, L)` and true where the
    residue's coordinates are real. Returns `(B, L, 3)` angles in radians and
    the matching validity mask. Invalid entries are ZERO here rather than nan
    -- a nan would poison the loss even after masking, because `0 * nan` is
    nan and the gradient follows. The mask is what carries "undefined"; the
    value must simply be finite.
    """
    import torch
    B, L = coords.shape[0], coords.shape[1]
    dev = coords.device
    ang = coords.new_zeros((B, L, 3))
    valid = torch.zeros((B, L, 3), dtype=torch.bool, device=dev)
    if L < 3:
        return ang, valid

    m = residue_mask.bool()
    link = m[:, :-1] & m[:, 1:]                 # residue i and i+1 both real

    # eta_i = C4'(i-1), P(i), C4'(i), P(i+1)   -- needs links (i-1,i) and (i,i+1)
    e = link[:, :-1] & link[:, 1:]              # (B, L-2), for i = 1..L-2
    a = _dihedral_t(coords[:, :-2, C4], coords[:, 1:-1, P],
                    coords[:, 1:-1, C4], coords[:, 2:, P])
    ang[:, 1:-1, 0] = torch.where(e, a, torch.zeros_like(a))
    valid[:, 1:-1, 0] = e

    # theta_i = P(i), C4'(i), P(i+1), C4'(i+1) -- needs link (i, i+1)
    b = _dihedral_t(coords[:, :-1, P], coords[:, :-1, C4],
                    coords[:, 1:, P], coords[:, 1:, C4])
    ang[:, :-1, 1] = torch.where(link, b, torch.zeros_like(b))
    valid[:, :-1, 1] = link

    # chi_tilde_i = P(i), C4'(i), N(i), P(i+1)
    c = _dihedral_t(coords[:, :-1, P], coords[:, :-1, C4],
                    coords[:, :-1, N], coords[:, 1:, P])
    ang[:, :-1, 2] = torch.where(link, c, torch.zeros_like(c))
    valid[:, :-1, 2] = link
    return ang, valid
