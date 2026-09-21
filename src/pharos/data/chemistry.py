#!/usr/bin/env python3
"""The 24-dimensional per-nucleotide chemistry vector of ARCHITECTURE v0.2 §4.

Specified in `CHEMISTRY.md`; this is the implementation. v0.1 fed the trunk
nucleotide identity and little else and then asked a physics module to reason
about hydrogen bonding, protonation and stacking. The physics had no chemical
substrate. These 24 numbers are that substrate.

Layout (see CHEMISTRY.md for the derivation of every value)::

     0-4  one-hot A / C / G / U / MOD
     5    purine flag
     6-11 H-bond donors and acceptors, per edge (WC, Hoogsteen/CH, sugar)
    12    pKa of the ring nitrogen that protonates
    13    shifted-pKa flag          <- STRUCTURAL; see the warning below
    14-15 C3'-endo / C2'-endo pucker propensity
    16-17 relative polarisability, stacking-energy prior
    18    phosphate charge state
    19    2'-OH present
    20-22 modification class: methylation / pseudouridylation / other
    23    local GC fraction, +/-16 window

**The separation that must not be violated.** Everything here is available at
inference from sequence alone, with one exception: dim 13 is set from context,
and the contexts that shift a pKa are structural. `residue_chemistry` therefore
defaults it to 0 and it can only be raised on recycles >= 1, the same
circularity that affects router features at recycle 0 (v0.2 §5.3). Setting it
from a known structure during training and from nothing at inference would be a
train/test mismatch. Structural observables -- Saenger and Leontis-Westhof
class, B-factor, Mg distance, SHAPE/DMS reactivity -- are supervision targets
and live in `attributes.py`, in a physically separate array, so that feeding one
in as a feature is not something a caller can do by accident.

Modified residues are resolved through the **PDB Chemical Component
Dictionary**, not a hand-written list of component codes. `scripts/sampling/
resolve_ccd_parents.py` emits `ccd_parents.json`, which maps every modification
the archive contains to its parent base and class; this module loads it if it is
present and degrades honestly if it is not (`MOD` set, class `other`, parent
unknown). A curated list of residue names is exactly what defect #22 was about.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

N_DIMS = 24
BASES = ("A", "C", "G", "U")
#: dim 23's window half-width
GC_WINDOW = 16

#: Dims 5-19 for the four standard ribonucleotides. Literature values compiled
#: into a static table: they are properties of the chemistry, not of our corpus,
#: so they are cited rather than re-derived (CHEMISTRY.md, "Provenance").
#:                purine wcD wcA hgD hgA sgD sgA  pKa  shift  c3'  c2'  pol  stk  chg  2'OH
_STATIC: Dict[str, Sequence[float]] = {
    "A": (1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 3.5, 0.0, 0.80, 0.20, 0.82, 0.85, -1.0, 1.0),
    "C": (0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 4.2, 0.0, 0.78, 0.22, 0.61, 0.63, -1.0, 1.0),
    "G": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.2, 0.0, 0.82, 0.18, 1.00, 1.00, -1.0, 1.0),
    "U": (0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 9.2, 0.0, 0.76, 0.24, 0.58, 0.55, -1.0, 1.0),
}
#: Pseudouridine's N1-H is an extra donor on a face uridine does not use. It is
#: the commonest modification in the archive and it changes base-pairing
#: capacity, which is why CHEMISTRY.md gives it its own dim rather than pooling
#: it into "other".
_PSU_EXTRA_HOOGSTEEN_DONOR = 1.0

_CLASS_DIM = {"methylation": 20, "pseudouridine": 21, "other": 22}


@lru_cache(maxsize=1)
def _ccd_table() -> Dict[str, Dict]:
    """`comp_id -> {parent, class, ...}` from the CCD resolution, if available."""
    p = (Path(__file__).resolve().parents[3]
         / "data/samples/analysis/ccd_parents.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("components", {})
    except (OSError, ValueError):
        return {}


def resolve(comp_id: str) -> tuple[str, str, bool]:
    """`(parent_base, modification_class, is_modified)` for an mmCIF comp id.

    A standard base resolves to itself, unmodified. Anything else is looked up
    in the CCD table; an unknown component is reported as modified, parent `N`,
    class `other` -- which is the honest answer, not a guess.
    """
    c = comp_id.strip().upper()
    if c in BASES:
        return c, "", False
    rec = _ccd_table().get(c)
    if not rec:
        return "N", "other", True
    parent = rec.get("parent", "?")
    return (parent if parent in BASES else "N",
            rec.get("class", "other") or "other", True)


def residue_chemistry(comp_id: str, *, five_prime_terminus: bool = False,
                      deoxy: bool = False, shifted_pka: bool = False) -> np.ndarray:
    """The 24-dim vector for one residue.

    `shifted_pka` sets dim 13 and must stay False at recycle 0 -- see the module
    docstring. `deoxy` clears dim 19 for the DNA residues that appear inside the
    hybrid duplexes raw PDB entries genuinely contain.
    """
    v = np.zeros(N_DIMS, dtype=np.float32)
    parent, mclass, is_mod = resolve(comp_id)

    # dims 0-4: one-hot, with MOD firing for anything outside A/C/G/U
    if is_mod:
        v[4] = 1.0
    if parent in BASES:
        v[BASES.index(parent)] = 1.0

    # dims 5-19 come from the parent's chemistry; an unresolvable residue gets
    # the element-wise mean, which is a stated fallback rather than a silent one
    if parent in _STATIC:
        v[5:20] = _STATIC[parent]
    else:
        v[5:20] = np.mean([_STATIC[b] for b in BASES], axis=0)

    if mclass == "pseudouridine":
        v[8] += _PSU_EXTRA_HOOGSTEEN_DONOR
    v[13] = 1.0 if shifted_pka else 0.0
    if five_prime_terminus:
        v[18] = 0.0          # free 5'-OH carries no phosphate charge
    if deoxy:
        v[19] = 0.0
        v[14], v[15] = 0.30, 0.70     # deoxyribose sits south
    if is_mod:
        v[_CLASS_DIM.get(mclass, 22)] = 1.0
    return v


def chain_chemistry(comp_ids: Iterable[str], *, deoxy_mask: Sequence[bool] | None = None,
                    gc_window: int = GC_WINDOW) -> np.ndarray:
    """`(L, 24)` for a whole chain, including dim 23's local GC fraction.

    Dim 23 is a windowed mean over the chain's own resolved parents, so a
    modified G still counts as a G -- which is the point of resolving parents
    rather than mapping every modification to `N`.
    """
    ids = list(comp_ids)
    L = len(ids)
    out = np.zeros((L, N_DIMS), dtype=np.float32)
    if L == 0:
        return out
    deoxy = list(deoxy_mask) if deoxy_mask is not None else [False] * L
    if len(deoxy) != L:
        raise ValueError(f"deoxy_mask has {len(deoxy)} entries for {L} residues")

    parents = []
    for i, c in enumerate(ids):
        out[i] = residue_chemistry(c, five_prime_terminus=(i == 0), deoxy=deoxy[i])
        parents.append(resolve(c)[0])

    # local GC over a +/-gc_window box, via a cumulative sum so it is O(L)
    gc = np.fromiter((1.0 if p in ("G", "C") else 0.0 for p in parents),
                     dtype=np.float64, count=L)
    known = np.fromiter((1.0 if p in BASES else 0.0 for p in parents),
                        dtype=np.float64, count=L)
    cg = np.concatenate(([0.0], np.cumsum(gc)))
    ck = np.concatenate(([0.0], np.cumsum(known)))
    lo = np.maximum(np.arange(L) - gc_window, 0)
    hi = np.minimum(np.arange(L) + gc_window + 1, L)
    denom = ck[hi] - ck[lo]
    out[:, 23] = np.where(denom > 0, (cg[hi] - cg[lo]) / np.maximum(denom, 1.0), 0.0)
    return out


# ---------------------------------------------------------------------------
# Vectorised form: the per-residue vector is a lookup, so build the table
# ---------------------------------------------------------------------------
#
# `chain_chemistry` is a Python loop calling `residue_chemistry` once per
# residue, and in the training loop it ran on CPU for every sequence of every
# batch -- twice, because `encode_batch` computed it from the original sequence
# and `masked_chemistry` then recomputed it from the masked one and threw the
# first away. At 26 sequences a step that is tens of thousands of Python calls
# between two GPU kernels.
#
# Nothing about it needs to be a loop. Every dim except 23 is a pure function
# of the symbol (and of two booleans), so the whole thing is a table indexed by
# token id; dim 23's window is a cumulative sum. The tables below are built by
# calling `residue_chemistry` itself rather than by restating its rules, so the
# fast path cannot drift from the definition -- if the chemistry changes, the
# table changes with it.


def chemistry_tables(symbols: Sequence[str], *, deoxy: bool = False
                     ) -> tuple[np.ndarray, np.ndarray]:
    """`(S, 24)` interior and 5'-terminal rows, one per symbol.

    Dim 23 is left at zero: it is the only dim that depends on neighbours, and
    it is filled by the caller's windowed sum.
    """
    interior = np.stack([residue_chemistry(s, deoxy=deoxy) for s in symbols])
    terminal = np.stack([residue_chemistry(s, five_prime_terminus=True, deoxy=deoxy)
                         for s in symbols])
    return interior, terminal


def gc_lookups(symbols: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-symbol `(is G or C, parent is a standard base)` for dim 23.

    The second is the denominator: a residue whose parent does not resolve to
    A/C/G/U is excluded from the window rather than counted as not-GC, which is
    what `chain_chemistry` does and is the reason a modified G still counts.
    """
    gc = np.array([1.0 if resolve(s)[0] in ("G", "C") else 0.0 for s in symbols],
                  dtype=np.float32)
    known = np.array([1.0 if resolve(s)[0] in BASES else 0.0 for s in symbols],
                     dtype=np.float32)
    return gc, known


if __name__ == "__main__":
    tbl = _ccd_table()
    print(f"CCD table: {len(tbl):,} components"
          if tbl else "CCD table absent -- modifications degrade to class 'other'")
    demo = ["G", "PSU", "A", "OMG", "U", "C", "XYZ"]
    v = chain_chemistry(demo)
    print(f"\n{'residue':8s} {'parent':7s} {'class':14s} onehot        GC")
    for c, row in zip(demo, v):
        p, k, _ = resolve(c)
        print(f"{c:8s} {p:7s} {k or '-':14s} {row[:5].astype(int)}  {row[23]:.3f}")
