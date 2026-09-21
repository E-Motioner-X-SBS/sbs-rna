#!/usr/bin/env python3
"""The structural tokenizer of VOCAB.md — parent base plus a modification id.

VOCAB.md's finding was that a 5-symbol vocabulary is right for *pretraining*
(elDORS is pre-normalised to ACGTN, so widening adds dead tokens — the
NucleicBERT vocab-25 mistake) and wrong for *structures*, which are not
normalised. Its decision was "5 + DNA{A,C,G,T,U} + I + UNK + a learned `MOD`
embedding keyed by the mmCIF `comp_id`, falling back to the parent base".

This implements that as **two parallel fields** rather than one widened symbol
set:

    token   the parent base -- what the residue *is* chemically
    mod_id  which modification it carries, or 0 for none

Keeping them separate is what makes the fallback automatic. A 2'-O-methylguanosine
is `token=G, mod_id=<OMG>`: every parameter that keys off G sees a G, and the
`MOD` embedding adds the difference. Collapsing the two into one symbol would
force the model to relearn guanosine's chemistry once per modification, and
there are **370 modification species** in the archive — a vocabulary of 370
one-shot symbols over 75,434 residues, most of them seen a handful of times.

The `mod_id` table is built from `ccd_parents.json`, ordered by frequency, so
low ids are the common modifications and a model may truncate the tail to a
single `MOD_RARE` bucket without re-indexing anything.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

#: Base symbols. Order is fixed: dims 0-3 line up with `chemistry.BASES`.
SYMBOLS: Tuple[str, ...] = (
    "A", "C", "G", "U",          # 0-3  standard ribonucleotides
    "N",                         # 4    N_struct: identity unassigned, ribose modelled
    "DA", "DC", "DG", "DT", "DU",  # 5-9  DNA, from hybrid duplexes
    "I",                         # 10   inosine -- common enough to earn a symbol
    "UNK",                       # 11   nothing modelled
    "PAD",                       # 12
)
SYM2ID: Dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}
PAD_ID = SYM2ID["PAD"]
N_SYMBOLS = len(SYMBOLS)

#: `mod_id` 0 means "not modified". Ids 1..N index MOD_SPECIES.
NO_MOD = 0

_DNA = {"DA": "DA", "DC": "DC", "DG": "DG", "DT": "DT", "DU": "DU"}
_ANALYSIS = Path(__file__).resolve().parents[3] / "data/samples/analysis"


@lru_cache(maxsize=1)
def _mod_table() -> Tuple[List[str], Dict[str, int], Dict[str, Dict]]:
    """`(species_by_frequency, comp_id -> mod_id, comp_id -> ccd record)`."""
    p = _ANALYSIS / "ccd_parents.json"
    if not p.exists():
        return [], {}, {}
    try:
        comps = json.loads(p.read_text())["components"]
    except (OSError, ValueError, KeyError):
        return [], {}, {}
    order = sorted(comps, key=lambda c: (-comps[c].get("count", 0), c))
    return order, {c: i + 1 for i, c in enumerate(order)}, comps


def mod_vocab_size() -> int:
    """Number of distinct modification species, excluding `NO_MOD`."""
    return len(_mod_table()[0])


def encode_residue(comp_id: str) -> Tuple[int, int]:
    """`(token, mod_id)` for one mmCIF component id.

    Resolution order, and the reason for it:

    1. a standard ribonucleotide is itself;
    2. a DNA residue keeps its own symbol -- raw entries genuinely contain
       hybrid duplexes, and calling a DT a U would be a chemistry error, not a
       normalisation;
    3. `UNK`/`N` are the two distinct unknowns VOCAB.md separates: `N_struct`
       has its ribose modelled and can train the geometry heads, `UNK` cannot;
    4. anything else is a modification: `token` is its CCD parent base and
       `mod_id` names the species. A modification with no declared parent --
       inosine, L-nucleotides, locked analogues, 11.4% of modified residues --
       gets token `I` if it is inosine and `N` otherwise, because `N_struct` is
       exactly "identity unassigned, full ribose modelled".
    """
    c = comp_id.strip().upper()
    if c in ("A", "C", "G", "U"):
        return SYM2ID[c], NO_MOD
    if c in _DNA:
        return SYM2ID[c], NO_MOD
    if c == "UNK":
        return SYM2ID["UNK"], NO_MOD
    if c == "N":
        return SYM2ID["N"], NO_MOD

    _, ids, comps = _mod_table()
    mid = ids.get(c, NO_MOD)
    rec = comps.get(c, {})
    parent = rec.get("parent", "?")
    if parent in ("A", "C", "G", "U"):
        return SYM2ID[parent], mid
    if c == "I" or "INOSIN" in (rec.get("name") or "").upper():
        return SYM2ID["I"], mid
    # a modification we hold no dictionary entry for: it is still a residue with
    # modelled geometry, so it trains the geometry heads as N_struct
    return SYM2ID["N"], mid


def encode_chain(comp_ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """`(tokens int8 (L,), mod_ids int16 (L,))` for a chain."""
    L = len(comp_ids)
    tok = np.empty(L, dtype=np.int8)
    mod = np.zeros(L, dtype=np.int16)
    for i, c in enumerate(comp_ids):
        tok[i], mod[i] = encode_residue(c)
    return tok, mod


def is_deoxy(tokens: np.ndarray) -> np.ndarray:
    """Boolean mask of DNA positions, for `chemistry.chain_chemistry`."""
    lo, hi = SYM2ID["DA"], SYM2ID["DU"]
    return (tokens >= lo) & (tokens <= hi)


if __name__ == "__main__":
    print(f"symbols: {N_SYMBOLS}  ({', '.join(SYMBOLS)})")
    print(f"modification species: {mod_vocab_size():,}")
    demo = ["G", "PSU", "OMG", "DT", "I", "UNK", "N", "ZZZ", "0G"]
    tok, mod = encode_chain(demo)
    order, _, comps = _mod_table()
    print(f"\n{'comp':6s} {'token':6s} {'mod_id':>7s}  parent/name")
    for c, t, m in zip(demo, tok, mod):
        rec = comps.get(c, {})
        print(f"{c:6s} {SYMBOLS[t]:6s} {m:7d}  "
              f"{rec.get('parent','-')}/{(rec.get('name') or '-')[:44]}")
