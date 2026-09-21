#!/usr/bin/env python3
"""Training examples from raw PDB entries — the whole archive, not a derivative.

Everything upstream of this file established what the data *is*. This turns it
into what a model consumes: for each RNA chain, the token sequence, the 24-dim
chemistry, the contact set, the block-occupancy targets the Hierarchical Pair
Track's selectors are supervised on, and the metadata the curriculum needs.

Three decisions are baked in, each measured rather than assumed.

**It reads raw entries, not RNA3DB or gRNASolo.** §11.4: those two cover 7,943
of 10,520 RNA-bearing entries, and the 2,581 they omit hold the entire top 30 of
the `effective_c` distribution. A pipeline built on them inherits their snapshot
date, which is how `target_c` came to be fitted to 21.14 when the archive
contains 23.30.

**Modified residues keep their identity.** `vocab.encode_residue` emits the CCD
parent base plus a modification id, so a 2'-O-methylguanosine trains as a G that
carries OMG rather than as an `N`. Raw entries are 0.565% modified against the
derivatives' 0.025%; mapping them to `N` would discard the thing raw entries
were acquired for.

**Quality is carried, not filtered on.** D16 weights 3D examples by structure
quality instead of applying a resolution cutoff, because a cutoff throws away
the cryo-EM majority. The weight travels with the example; the sampler decides
what to do with it.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.chemistry import N_DIMS, chain_chemistry           # noqa: E402
from data.mmcif_entities import rna_chain_coords             # noqa: E402
from data.vocab import PAD_ID, encode_chain, is_deoxy        # noqa: E402

#: Contact definition, identical to every measurement in the project so the
#: labels and the published statistics describe the same object.
CONTACT_CUTOFF = 8.0
MIN_SEPARATION = 4
#: Hierarchical Pair Track levels (ARCHITECTURE v0.2 §7): b1 coarse, b2 fine.
BLOCK_SIZES = (16, 4)
#: Training window. The lower bound is where a contact map carries structure;
#: the upper is the pair-track measurement window, NOT the context window --
#: chains up to 4,450 nt exist and the model's context is 4,608 (D20).
MIN_LENGTH, MAX_LENGTH = 32, 4608
#: Below this a chain has no tertiary structure worth supervising.
MIN_CONTACTS = 8


@dataclass
class ChainExample:
    """One RNA chain, ready to train on."""
    pdb: str
    chain: str
    length: int
    tokens: np.ndarray            # int8   (L,)
    mod_ids: np.ndarray           # int16  (L,)
    chem: np.ndarray              # float16 (L, 24)
    contacts: np.ndarray          # int32  (n, 2), i < j, j - i >= MIN_SEPARATION
    resolution: Optional[float] = None
    method: str = "?"
    clashscore: Optional[float] = None
    train_weight: float = 1.0
    rfam: Optional[str] = None
    has_protein: bool = False
    ribosome_like: bool = False
    n_modified: int = 0

    @property
    def contacts_per_nt(self) -> float:
        return len(self.contacts) / max(self.length, 1)

    def block_labels(self, b: int) -> np.ndarray:
        """`(nb, nb)` 0/1: does this block contain at least one true contact?

        These are the targets for `block_occupancy_loss`. Producing them here
        rather than in the training loop means the expensive part -- the
        contact set -- is computed once, at build time, over the whole archive.
        """
        nb = (self.length + b - 1) // b
        lab = np.zeros((nb, nb), dtype=np.uint8)
        if len(self.contacts):
            lab[self.contacts[:, 0] // b, self.contacts[:, 1] // b] = 1
        return lab

    def effective_c(self, b: int = 4) -> float:
        """Pairs per nucleotide implied by keeping every occupied block at `b`.

        The quantity `target_c` budgets for. Recomputing it from the stored
        example is how a shard proves it matches the published distribution.
        """
        if not len(self.contacts):
            return 0.0
        occ = len(np.unique((self.contacts[:, 0] // b).astype(np.int64)
                            * (self.length // b + 2)
                            + (self.contacts[:, 1] // b)))
        return occ * b * b / max(self.length, 1)

    def meta(self) -> Dict:
        d = asdict(self)
        for k in ("tokens", "mod_ids", "chem", "contacts"):
            d.pop(k)
        d["n_contacts"] = int(len(self.contacts))
        d["contacts_per_nt"] = round(self.contacts_per_nt, 4)
        d["effective_c_b4"] = round(self.effective_c(4), 2)
        return d


def contact_set(residue_atoms: Sequence[Sequence[Sequence[float]]],
                cutoff: float = CONTACT_CUTOFF,
                min_sep: int = MIN_SEPARATION) -> np.ndarray:
    """Residue pairs with any heavy atom within `cutoff`, `j - i >= min_sep`.

    Vectorised throughout. The interpreted version -- accumulate `(i, j)` tuples
    in a set -- reached 28 GB on a 37,455-atom chain and stopped progressing;
    see the performance note in DECISIONS.md.
    """
    counts = [len(a) for a in residue_atoms]
    n_at = sum(counts)
    if n_at == 0:
        return np.empty((0, 2), dtype=np.int32)
    flat = np.empty((n_at, 3), dtype=np.float64)
    owner = np.empty(n_at, dtype=np.int64)
    k = 0
    for i, at in enumerate(residue_atoms):
        if not at:
            continue
        flat[k:k + len(at)] = at
        owner[k:k + len(at)] = i
        k += len(at)

    pr = cKDTree(flat).query_pairs(cutoff, output_type="ndarray")
    if pr.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    a, b = owner[pr[:, 0]], owner[pr[:, 1]]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keep = (hi - lo) >= min_sep
    if not keep.any():
        return np.empty((0, 2), dtype=np.int32)
    lo, hi = lo[keep], hi[keep]
    n = len(residue_atoms)
    uniq = np.unique(lo * n + hi)
    return np.stack((uniq // n, uniq % n), axis=1).astype(np.int32)


def build_entry(path: Path, entry_meta: Optional[Dict] = None,
                min_length: int = MIN_LENGTH, max_length: int = MAX_LENGTH,
                min_contacts: int = MIN_CONTACTS) -> List[ChainExample]:
    """Every usable RNA chain in one mmCIF entry."""
    em = entry_meta or {}
    try:
        chains = rna_chain_coords(path)
    except Exception:                                        # noqa: BLE001
        return []
    out: List[ChainExample] = []
    for ch, residues in chains.items():
        L = len(residues)
        if not (min_length <= L <= max_length):
            continue
        comps = [c for _, c, _ in residues]
        atoms = [a for _, _, a in residues]
        contacts = contact_set(atoms)
        if len(contacts) < min_contacts:
            continue
        tokens, mods = encode_chain(comps)
        chem = chain_chemistry(comps, deoxy_mask=is_deoxy(tokens))
        out.append(ChainExample(
            pdb=path.stem.replace(".cif", ""), chain=ch, length=L,
            tokens=tokens, mod_ids=mods, chem=chem.astype(np.float16),
            contacts=contacts,
            resolution=em.get("resolution"), method=em.get("method", "?"),
            clashscore=em.get("clashscore"),
            train_weight=float(em.get("train_weight", 1.0)),
            rfam=em.get("rfam_family"),
            has_protein=bool(em.get("has_protein", False)),
            ribosome_like=bool(em.get("ribosome_like", False)),
            n_modified=int((mods != 0).sum()),
        ))
    return out


# ---------------------------------------------------------------------------
# Sharding. One .npz per shard with every chain's arrays concatenated and an
# offset table, which keeps the file count low and the load a single mmap.
# ---------------------------------------------------------------------------

def write_shard(path: Path, examples: Sequence[ChainExample]) -> Dict:
    tok = np.concatenate([e.tokens for e in examples]) if examples else np.empty(0, np.int8)
    mod = np.concatenate([e.mod_ids for e in examples]) if examples else np.empty(0, np.int16)
    chem = (np.concatenate([e.chem for e in examples]) if examples
            else np.empty((0, N_DIMS), np.float16))
    con = (np.concatenate([e.contacts for e in examples]) if examples
           else np.empty((0, 2), np.int32))
    res_off = np.cumsum([0] + [e.length for e in examples]).astype(np.int64)
    con_off = np.cumsum([0] + [len(e.contacts) for e in examples]).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, tokens=tok, mod_ids=mod, chem=chem, contacts=con,
                        res_off=res_off, con_off=con_off)
    return {"file": path.name, "n_chains": len(examples),
            "n_residues": int(res_off[-1]), "n_contacts": int(con_off[-1]),
            "chains": [e.meta() for e in examples]}


class ShardReader:
    """Random access to a shard without unpacking it.

    `np.load` on an `.npz` is lazy per array, so a shard costs one decompress
    per field on first touch and nothing thereafter.
    """

    def __init__(self, path: Path, meta: Optional[Sequence[Dict]] = None):
        self.path = Path(path)
        self._z = np.load(self.path)
        self.res_off = self._z["res_off"]
        self.con_off = self._z["con_off"]
        self.meta = list(meta) if meta is not None else []

    def __len__(self) -> int:
        return len(self.res_off) - 1

    def __getitem__(self, i: int) -> Dict:
        a, b = int(self.res_off[i]), int(self.res_off[i + 1])
        c, d = int(self.con_off[i]), int(self.con_off[i + 1])
        out = {"tokens": self._z["tokens"][a:b], "mod_ids": self._z["mod_ids"][a:b],
               "chem": self._z["chem"][a:b], "contacts": self._z["contacts"][c:d],
               "length": b - a}
        if i < len(self.meta):
            out["meta"] = self.meta[i]
        return out


def pad_batch(items: Sequence[Dict], pad_id: int = PAD_ID) -> Dict[str, np.ndarray]:
    """Right-pad a list of examples into rectangular arrays plus a mask."""
    B = len(items)
    L = max((int(x["length"]) for x in items), default=0)
    tok = np.full((B, L), pad_id, dtype=np.int16)
    mod = np.zeros((B, L), dtype=np.int16)
    chem = np.zeros((B, L, N_DIMS), dtype=np.float32)
    mask = np.zeros((B, L), dtype=bool)
    for i, x in enumerate(items):
        n = int(x["length"])
        tok[i, :n] = x["tokens"]
        mod[i, :n] = x["mod_ids"]
        chem[i, :n] = x["chem"]
        mask[i, :n] = True
    return {"tokens": tok, "mod_ids": mod, "chem": chem, "mask": mask,
            "contacts": [x["contacts"] for x in items],
            "lengths": np.array([int(x["length"]) for x in items], dtype=np.int32)}
