#!/usr/bin/env python3
"""Reference structures for the blind tests, reduced to the model's 3 atoms.

`diffusion.N_ATOM` is 3: phosphate, C4' and the glycosidic nitrogen (N9 on a
purine, N1 on a pyrimidine). Everything in this package therefore speaks that
representation, and the parser's job is to project CASP and RNA-Puzzles
depositions -- which are full-atom, multi-model, multi-chain, and frequently
missing residues -- onto it without inventing coordinates.

**Missing atoms are reported, not filled.** A residue whose phosphate was not
resolved is marked in `mask` rather than given a guessed position. Scoring
against invented coordinates is how a metric flatters a model: the reference
gets a plausible atom, the prediction matches it, and a residue nobody observed
contributes to the score. Every metric here consumes the mask.

**Chains are kept separate and then concatenated in deposition order**, because
RNA-Puzzles targets are routinely multimers (rp02 is an eight-chain square) and
a prediction is scored against the whole assembly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

#: Atom order, matching `diffusion.N_ATOM`.
ATOMS = ("P", "C4'", "N")
#: The glycosidic nitrogen is N9 on a purine and N1 on a pyrimidine.
_PURINE = {"A", "G", "DA", "DG", "I"}

#: Residue names that count as RNA, before CCD resolution. Modified residues
#: are resolved through `data.chemistry`, which reads the PDB Chemical
#: Component Dictionary -- a hand-written list here would be defect #22 again.
_STANDARD = {"A", "C", "G", "U"}


@dataclass
class Structure:
    """One chain set reduced to `(L, 3, 3)` coordinates plus a validity mask."""

    coords: np.ndarray        # (L, 3, 3) float32, Angstrom
    mask: np.ndarray          # (L, 3) bool -- was this atom actually resolved?
    seq: str                  # (L,) in ACGUN
    chain_ids: np.ndarray     # (L,) object -- which chain each residue came from
    resnum: np.ndarray        # (L,) int -- author numbering, for alignment
    name: str = ""

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    @property
    def residue_mask(self) -> np.ndarray:
        """`(L,)` -- a residue counts when all three of its atoms resolved."""
        return self.mask.all(axis=1)

    def subset(self, idx: Sequence[int] | np.ndarray) -> "Structure":
        i = np.asarray(idx, dtype=int)
        return Structure(self.coords[i], self.mask[i],
                         "".join(self.seq[int(k)] for k in i),
                         self.chain_ids[i], self.resnum[i], self.name)


def _glycosidic(resname: str) -> str:
    return "N9" if resname.strip().upper() in _PURINE else "N1"


def read_structure(path: str | Path, *, model: int = 0,
                   chains: Optional[Sequence[str]] = None) -> Structure:
    """Parse a `.pdb` or `.cif` deposition into the 3-atom representation.

    Only the first model is read by default: RNA-Puzzles submissions bundle up
    to five models per group in separate files, but NMR references carry many
    in one, and scoring a model-averaged structure is not a thing.
    """
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    if len(st) == 0:
        raise ValueError(f"{path}: no models")
    md = st[model]

    xyz: List[np.ndarray] = []
    msk: List[np.ndarray] = []
    seq: List[str] = []
    cid: List[str] = []
    num: List[int] = []

    for chain in md:
        if chains is not None and chain.name not in chains:
            continue
        for res in chain:
            parent = _parent_base(res.name)
            if parent is None:
                continue
            want = ("P", "C4'", _glycosidic(res.name if res.name.strip().upper()
                                            in _PURINE else parent))
            # a modified purine keeps N9 even though its comp id is not "A"/"G"
            if parent in ("A", "G"):
                want = ("P", "C4'", "N9")
            c = np.zeros((3, 3), dtype=np.float32)
            m = np.zeros(3, dtype=bool)
            for k, aname in enumerate(want):
                at = res.find_atom(aname, "*")
                if at is not None:
                    c[k] = (at.pos.x, at.pos.y, at.pos.z)
                    m[k] = True
            if not m.any():
                continue                      # residue present in name only
            xyz.append(c); msk.append(m); seq.append(parent)
            cid.append(chain.name); num.append(int(res.seqid.num))

    if not xyz:
        raise ValueError(f"{path}: no RNA residues found")
    return Structure(np.stack(xyz), np.stack(msk), "".join(seq),
                     np.array(cid, dtype=object), np.array(num, dtype=int),
                     Path(path).stem)


def _parent_base(resname: str) -> Optional[str]:
    """`A`/`C`/`G`/`U` for an RNA residue, `N` for a modified one, else None."""
    r = resname.strip().upper()
    if r in _STANDARD:
        return r
    if r in ("DA", "DC", "DG", "DT", "DU"):
        return None                           # DNA: not our target
    try:
        from ..data.chemistry import resolve
    except ImportError:
        return None
    parent, _cls, is_mod = resolve(r)
    if not is_mod:
        return parent if parent in _STANDARD else None
    # a modification the CCD resolves to a standard base is still that base;
    # one it cannot resolve is reported as N and kept, because it occupies a
    # position in the chain and dropping it would shift every index after it
    return parent if parent in _STANDARD else ("N" if _known_rna_mod(r) else None)


def _known_rna_mod(comp_id: str) -> bool:
    try:
        from ..data.chemistry import _ccd_table
    except ImportError:
        return False
    return comp_id in _ccd_table()


def align_by_resnum(ref: Structure, pred: Structure) -> tuple[Structure, Structure]:
    """Pair residues by (chain, author number), the RNA-Puzzles convention.

    Predictions are submitted with the target's own numbering, so this is an
    exact join rather than a sequence alignment -- and when a submission has
    renumbered, the join comes up short and the caller sees a small coverage
    rather than a silently wrong superposition.
    """
    index: Dict[tuple, int] = {}
    for i, (c, n) in enumerate(zip(pred.chain_ids, pred.resnum)):
        index.setdefault((c, int(n)), i)
    # single-chain targets are routinely submitted under a different chain
    # letter; fall back to number alone when that is unambiguous
    by_num: Dict[int, int] = {}
    if len(set(pred.chain_ids)) == 1 and len(set(ref.chain_ids)) == 1:
        for i, n in enumerate(pred.resnum):
            by_num.setdefault(int(n), i)

    ri, pi = [], []
    for j, (c, n) in enumerate(zip(ref.chain_ids, ref.resnum)):
        k = index.get((c, int(n)))
        if k is None:
            k = by_num.get(int(n))
        if k is not None:
            ri.append(j); pi.append(k)
    return ref.subset(ri), pred.subset(pi)
