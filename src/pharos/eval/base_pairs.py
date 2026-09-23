#!/usr/bin/env python3
"""Watson-Crick base pairs, from full atoms and from the model's three.

INF needs a base-pair set for both the reference and the prediction. The
reference is full-atom, so its pairs come from the hydrogen bond itself:
N1 of a purine to N3 of a pyrimidine, under 3.3 Angstrom, with the two base
planes roughly parallel. A prediction from this model is three atoms per
residue -- P, C4' and the glycosidic N -- and has no base planes at all, so it
needs a different criterion.

**The criterion was measured, not assumed.** Running the full-atom detector
over the RNA-Puzzles reference structures and recording what those same pairs
look like in the 3-atom projection gives:

    glycosidic N-N   8.63 +- 0.72 A   (1st-99th percentile 6.30-9.25)
    C4'-C4'         14.67 +- 0.98 A   (1st-99th percentile 11.72-15.53)

so the windows below are those percentiles, widened slightly. Two constraints
rather than one because either alone admits stacking: residues one apart in an
A-form helix sit at N-N 4.5 A, and the first attempt at a full-atom detector
(N1-N3 distance only, no separation floor) called 41 adjacent stacked pairs
base pairs and dragged the measured mean from 8.63 down to 8.26.

`WC_AGREEMENT` records what this costs: measured against the full-atom
criterion over the 17 RNA-Puzzles reference structures, the 3-atom detector
runs at precision 0.768 and recall 0.812 (F1 0.790, 377 true / 114 false / 87
missed). So an INF gap under roughly 0.1 between two predictions is inside the
detector's own error and should not be read as a difference between models.
`test_metrics.py` re-derives these numbers and fails if they drift, so the
docstring cannot quietly go stale.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

Pair = Tuple[int, int]

#: Base combinations that can form a Watson-Crick or wobble pair.
WC_COMPATIBLE = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"),
                 ("G", "U"), ("U", "G")}
#: Minimum sequence separation. A hairpin needs at least three unpaired
#: residues in its loop, so |i-j| < 4 across one chain is not a pair; the floor
#: is 3 to stay permissive about unusual loops while still excluding stacking.
MIN_SEPARATION = 3
#: Measured windows -- see the module docstring.
NN_RANGE = (6.0, 9.8)
C4_RANGE = (11.0, 16.8)
#: |dot(n_i, n_j)| for two base planes that are stacked-parallel enough to pair.
COPLANAR_MIN = 0.75
#: N1(purine)-N3(pyrimidine) hydrogen bond, full atom.
HBOND_MAX = 3.3

#: Measured agreement of `geometric_pairs` with `full_atom_pairs`, over the
#: RNA-Puzzles reference set. Re-derived by `test_metrics.py`.
WC_AGREEMENT = {"precision": 0.768, "recall": 0.812, "f1": 0.790,
                "n_structures": 17, "tp": 377, "fp": 114, "fn": 87}

_RING = {"A": ("N9", "C8", "N7", "C5", "C4"), "G": ("N9", "C8", "N7", "C5", "C4"),
         "C": ("N1", "C2", "N3", "C4", "C5", "C6"),
         "U": ("N1", "C2", "N3", "C4", "C5", "C6")}


def full_atom_pairs(path) -> Set[Pair]:
    """WC pairs from a full-atom deposition, indexed into the RNA residue list.

    Indices match `structure.read_structure`'s residue order, so the two are
    directly comparable.
    """
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    res = [r for ch in st[0] for r in ch if r.name.strip().upper() in _RING]
    normals = [_plane_normal(r) for r in res]
    wc = [r.find_atom("N1" if r.name.strip().upper() in ("A", "G") else "N3", "*")
          for r in res]

    out: Set[Pair] = set()
    for i in range(len(res)):
        if wc[i] is None or normals[i] is None:
            continue
        for j in range(i + MIN_SEPARATION + 1, len(res)):
            if wc[j] is None or normals[j] is None:
                continue
            if (res[i].name.strip(), res[j].name.strip()) not in WC_COMPATIBLE:
                continue
            if wc[i].pos.dist(wc[j].pos) > HBOND_MAX:
                continue
            if abs(float(np.dot(normals[i], normals[j]))) < COPLANAR_MIN:
                continue
            out.add((i, j))
    return out


def _plane_normal(res) -> Optional[np.ndarray]:
    names = _RING.get(res.name.strip().upper())
    if not names:
        return None
    pts = [res.find_atom(a, "*") for a in names]
    if any(p is None for p in pts):
        return None
    m = np.array([[p.pos.x, p.pos.y, p.pos.z] for p in pts])
    return np.linalg.svd(m - m.mean(0))[2][-1]


def geometric_pairs(coords: np.ndarray, seq: str, *,
                    mask: Optional[np.ndarray] = None,
                    chain_ids: Optional[Sequence] = None) -> Set[Pair]:
    """WC pairs from `(L, 3, 3)` P/C4'/N coordinates, the model's own output.

    Greedy one-to-one: a residue pairs with at most one partner, best candidate
    first. RNA base pairing *is* one-to-one for the WC face, and without the
    constraint a residue in a crowded helix picks up two or three partners and
    INF's false-positive count explodes for a structure that is actually right.
    """
    L = len(seq)
    if L == 0:
        return set()
    m = np.ones(L, bool) if mask is None else np.asarray(mask, bool)
    n = coords[:, 2]                       # glycosidic nitrogen
    c4 = coords[:, 1]
    dn = np.linalg.norm(n[:, None] - n[None, :], axis=-1)
    dc = np.linalg.norm(c4[:, None] - c4[None, :], axis=-1)

    cand: List[Tuple[float, int, int]] = []
    for i in range(L):
        if not m[i]:
            continue
        for j in range(i + 1, L):
            if not m[j]:
                continue
            same = chain_ids is None or chain_ids[i] == chain_ids[j]
            if same and (j - i) <= MIN_SEPARATION:
                continue
            if (seq[i], seq[j]) not in WC_COMPATIBLE:
                continue
            if not (NN_RANGE[0] <= dn[i, j] <= NN_RANGE[1]):
                continue
            if not (C4_RANGE[0] <= dc[i, j] <= C4_RANGE[1]):
                continue
            # rank by how central the N-N distance is in its window
            cand.append((abs(dn[i, j] - 8.63), i, j))

    cand.sort()
    used: Set[int] = set()
    out: Set[Pair] = set()
    for _d, i, j in cand:
        if i in used or j in used:
            continue
        used.add(i); used.add(j); out.add((i, j))
    return out


def agreement(pred: Set[Pair], ref: Set[Pair]) -> Dict[str, float]:
    """Precision, recall and F1 of one pair set against another."""
    tp = len(pred & ref)
    p = tp / len(pred) if pred else float("nan")
    r = tp / len(ref) if ref else float("nan")
    f = 2 * p * r / (p + r) if p and r and (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f,
            "n_pred": len(pred), "n_ref": len(ref)}
