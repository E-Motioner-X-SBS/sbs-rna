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
from data.mmcif_entities import (residue_labels,             # noqa: E402
                                 rna_chain_coords, rna_chain_backbone)
from data.vocab import PAD_ID, encode_chain, is_deoxy        # noqa: E402

#: Contact definition, identical to every measurement in the project so the
#: labels and the published statistics describe the same object.
CONTACT_CUTOFF = 8.0
MIN_SEPARATION = 4
#: Hierarchical Pair Track levels (ARCHITECTURE v0.2 §7): b1 coarse, b2 fine.
BLOCK_SIZES = (16, 4)
#: Training window. The upper bound is the context window (D20); chains reach
#: 4,450 nt and nothing is lost at 4,608.
#:
#: The lower bound is measured, not assumed. Dropping it from 32 to 8 over a
#: 400-entry stride sample adds **+10.6% chains for +0.3% residues and +0.1%
#: contacts**: the excluded chains are numerous and nearly empty. `MIN_CONTACTS`
#: is the filter that matters -- it is about whether a chain has tertiary
#: structure to learn from, which is the actual question, and it is not a
#: proxy for length. 32 is kept because moving it would churn every pinned
#: figure in the corpus for a tenth of a percent of the signal.
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
    #: The diffusion head's supervision: P / C4' / glycosidic-N per residue, in
    #: `rna_chain_coords` order so index i is the same residue as `tokens[i]`.
    #: Stored as float16 -- 0.002 A of quantisation against coordinates whose
    #: own uncertainty is tenths of an Angstrom, for half the corpus size.
    #: `coord_mask` is false where an atom was not resolved (every 5' terminus
    #: has no phosphate); the loss skips those rather than fitting a guess.
    coords: Optional[np.ndarray] = None        # float16 (L, 3, 3)
    coord_mask: Optional[np.ndarray] = None    # bool    (L, 3)
    resolution: Optional[float] = None
    method: str = "?"
    clashscore: Optional[float] = None
    train_weight: float = 1.0
    rfam: Optional[str] = None
    has_protein: bool = False
    ribosome_like: bool = False
    n_modified: int = 0
    # ---- the supervision the depositor already produced (heads 5, 6, 10, 11)
    mg_site: Optional[np.ndarray] = None       # uint8 (L,)   head 5
    b_factor_z: Optional[np.ndarray] = None    # float16 (L,) head 6
    unknown_base: Optional[np.ndarray] = None  # uint8 (L,)   head 10
    #: the disorder label (§10) is about the POLYMER, not the coordinates: an unobserved
    #: residue has no atoms, so it cannot be a mask over the modelled residues
    unobserved_seq_id: Optional[np.ndarray] = None
    n_polymer: int = 0
    #: D12 -- the rigidity head trains on X-ray B-factors ONLY. cryo-EM per-atom
    #: B is a fitted display parameter, on a different scale entirely (7PAS
    #: reads a mean of 987 against 1VY7's 72), and the Mg-rigidity gradient is
    #: monotonic on X-ray and not on cryo-EM. The flag travels with the example
    #: so a training loop cannot mix them by accident.
    rigidity_valid: bool = False

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
        for k in ("tokens", "mod_ids", "chem", "contacts", "mg_site",
                  "b_factor_z", "unknown_base", "unobserved_seq_id",
                  "coords", "coord_mask"):
            d.pop(k, None)
        d["n_mg_sites"] = int(self.mg_site.sum()) if self.mg_site is not None else 0
        d["n_unobserved"] = (len(self.unobserved_seq_id)
                             if self.unobserved_seq_id is not None else 0)
        d["n_unknown_base"] = (int(self.unknown_base.sum())
                               if self.unknown_base is not None else 0)
        d["n_contacts"] = int(len(self.contacts))
        d["frac_backbone_resolved"] = (
            round(float(self.coord_mask.all(1).mean()), 4)
            if self.coord_mask is not None else None)
        d["contacts_per_nt"] = round(self.contacts_per_nt, 4)
        d["effective_c_b4"] = round(self.effective_c(4), 2)
        return d


#: A chain is a usable disorder observation only if most of its declared
#: polymer was actually resolved. 7ANE chain 2 declares an 18,998-nt polymer and
#: models 604 of it: the other 18,394 positions are not "disordered", they are
#: the rest of a viral genome that was never in the crystal. 250 chains of
#: 16,604 (1.5%) are fragments like this and they hold **472,825 of the
#: 1,379,892** unobserved positions -- 34% of the signal, and all of it the
#: wrong kind. Training a disorder head on them teaches it that RNA is mostly
#: disordered.
DISORDER_MAX_POLYMER_RATIO = 3.0
DISORDER_MIN_POLYMER = 200


def disorder_is_meaningful(length: int, n_polymer: int) -> bool:
    """Whether this chain's unobserved positions are flexibility or truncation."""
    if not n_polymer or n_polymer <= DISORDER_MIN_POLYMER:
        return True
    return n_polymer <= DISORDER_MAX_POLYMER_RATIO * max(length, 1)


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
        # Same parser, same residue order, so frame i is residue i. Parsed in
        # the same call site as the contacts so the two cannot drift apart.
        backbones = rna_chain_backbone(path)
    except Exception:                                        # noqa: BLE001
        return []
    try:
        labels = residue_labels(path)
    except Exception:                                        # noqa: BLE001
        labels = {}
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
        bb = backbones.get(ch)
        if bb is None or len(bb[2]) != L:
            # the two parses disagree about this chain; drop it rather than
            # pair a frame with the wrong residue
            continue
        bb_xyz, bb_msk = bb[0].astype(np.float16), bb[1]
        tokens, mods = encode_chain(comps)
        chem = chain_chemistry(comps, deoxy_mask=is_deoxy(tokens))
        lab = labels.get(ch, {})
        def _arr(key, dtype, n=L):
            a = lab.get(key)
            if a is None or len(a) != n:
                return np.zeros(n, dtype=dtype)
            return np.asarray(a, dtype=dtype)
        out.append(ChainExample(
            pdb=path.stem.replace(".cif", ""), chain=ch, length=L,
            tokens=tokens, mod_ids=mods, chem=chem.astype(np.float16),
            contacts=contacts,
            coords=bb_xyz, coord_mask=bb_msk,
            resolution=em.get("resolution"), method=em.get("method", "?"),
            clashscore=em.get("clashscore"),
            train_weight=float(em.get("train_weight", 1.0)),
            rfam=em.get("rfam_family"),
            has_protein=bool(em.get("has_protein", False)),
            ribosome_like=bool(em.get("ribosome_like", False)),
            n_modified=int((mods != 0).sum()),
            mg_site=_arr("mg_site", np.uint8),
            b_factor_z=_arr("b_factor_z", np.float16),
            unknown_base=_arr("unknown_base", np.uint8),
            unobserved_seq_id=np.asarray(lab.get("unobserved_seq_id", []),
                                         dtype=np.int32),
            n_polymer=int(lab.get("n_polymer", L)),
            rigidity_valid=bool(lab.get("rigidity_valid", False)),
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

    def cat(attr, dtype, width=None):
        parts = [getattr(e, attr) for e in examples]
        parts = [p if p is not None else np.zeros(e.length, dtype=dtype)
                 for p, e in zip(parts, examples)]
        return (np.concatenate(parts).astype(dtype) if parts
                else np.empty(0, dtype))

    mg = cat("mg_site", np.uint8)
    bz = cat("b_factor_z", np.float16)
    ub = cat("unknown_base", np.uint8)
    unob = (np.concatenate([e.unobserved_seq_id if e.unobserved_seq_id is not None
                            else np.empty(0, np.int32) for e in examples])
            if examples else np.empty(0, np.int32))
    unob_off = np.cumsum([0] + [0 if e.unobserved_seq_id is None else len(e.unobserved_seq_id)
                                for e in examples]
                         ).astype(np.int64)
    # (sum_L, 3, 3) and (sum_L, 3), concatenated on the residue axis so the
    # same res_off slices them as slices every other per-residue field
    xyz = (np.concatenate([e.coords if e.coords is not None
                           else np.zeros((e.length, 3, 3), np.float16)
                           for e in examples]).astype(np.float16)
           if examples else np.empty((0, 3, 3), np.float16))
    xyz_m = (np.concatenate([e.coord_mask if e.coord_mask is not None
                             else np.zeros((e.length, 3), bool)
                             for e in examples])
             if examples else np.empty((0, 3), bool))
    res_off = np.cumsum([0] + [e.length for e in examples]).astype(np.int64)
    con_off = np.cumsum([0] + [len(e.contacts) for e in examples]).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, tokens=tok, mod_ids=mod, chem=chem, contacts=con,
                        res_off=res_off, con_off=con_off,
                        mg_site=mg, b_factor_z=bz, unknown_base=ub,
                        unobserved_seq_id=unob, unob_off=unob_off,
                        coords=xyz, coord_mask=xyz_m)
    return {"file": path.name, "n_chains": len(examples),
            "n_residues": int(res_off[-1]), "n_contacts": int(con_off[-1]),
            "chains": [e.meta() for e in examples]}


class ShardReader:
    """Random access to a shard, with the arrays decompressed exactly once.

    **`np.load` on an `.npz` is not lazily indexable.** `NpzFile.__getitem__`
    decompresses the *whole* named array on every access, so reading one chain's
    slice costs a full-shard decompression -- and reading 512 chains costs 512 of
    them. An earlier version of this class indexed `self._z[key][a:b]` directly
    and its docstring claimed "one decompress per field on first touch and
    nothing thereafter", which is exactly what NpzFile does not do. Measured
    consequence: block-scorer training ran at **4 s/step with the GPU at 0%
    utilisation** -- about 13 hours for 8 epochs, all of it in zlib.

    So every field is materialised once, in `_load`, and sliced from memory
    thereafter. A shard is ~25 MB expanded and the whole 33-shard set is under a
    gigabyte, so the arrays are simply held.
    """

    _FIELDS = ("tokens", "mod_ids", "chem", "contacts", "mg_site", "b_factor_z",
               "unknown_base", "unobserved_seq_id", "unob_off",
               "coords", "coord_mask")

    def __init__(self, path: Path, meta: Optional[Sequence[Dict]] = None):
        self.path = Path(path)
        with np.load(self.path) as z:
            self._a = {k: z[k] for k in self._FIELDS if k in z.files}
            self.res_off = z["res_off"]
            self.con_off = z["con_off"]
        self.meta = list(meta) if meta is not None else []

    def __len__(self) -> int:
        return len(self.res_off) - 1

    def __getitem__(self, i: int) -> Dict:
        a, b = int(self.res_off[i]), int(self.res_off[i + 1])
        c, d = int(self.con_off[i]), int(self.con_off[i + 1])
        A = self._a
        out = {"tokens": A["tokens"][a:b], "mod_ids": A["mod_ids"][a:b],
               "chem": A["chem"][a:b], "contacts": A["contacts"][c:d],
               "length": b - a}
        for key in ("mg_site", "b_factor_z", "unknown_base",
                    "coords", "coord_mask"):
            if key in A:
                out[key] = A[key][a:b]
        if "unob_off" in A:
            uo = A["unob_off"]
            out["unobserved_seq_id"] = A["unobserved_seq_id"][
                int(uo[i]):int(uo[i + 1])]
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
    # per-residue supervision, padded alongside. Each carries its own validity
    # mask: a head must never be trained on a zero that means "absent" rather
    # than "measured zero", and B-factors in particular are only valid for
    # X-ray chains (D12).
    mg = np.zeros((B, L), dtype=np.uint8)
    bz = np.zeros((B, L), dtype=np.float32)
    ub = np.zeros((B, L), dtype=np.uint8)
    # The diffusion target. `coord_mask` is false for padding AND for atoms the
    # depositor never resolved, and the two are indistinguishable downstream on
    # purpose: both mean "there is no observation here", which is exactly the
    # condition under which the loss must not contribute.
    xyz = np.zeros((B, L, 3, 3), dtype=np.float32)
    xyz_m = np.zeros((B, L, 3), dtype=bool)
    for i, x in enumerate(items):
        n = int(x["length"])
        tok[i, :n] = x["tokens"]
        mod[i, :n] = x["mod_ids"]
        chem[i, :n] = x["chem"]
        mask[i, :n] = True
        if "mg_site" in x:
            mg[i, :n] = x["mg_site"]
        if "b_factor_z" in x:
            bz[i, :n] = x["b_factor_z"]
        if "unknown_base" in x:
            ub[i, :n] = x["unknown_base"]
        if x.get("coords") is not None and len(x["coords"]) == n:
            xyz[i, :n] = x["coords"]
            xyz_m[i, :n] = x["coord_mask"]
    meta = [x.get("meta") or {} for x in items]
    rigid_ok = np.array([bool(m.get("rigidity_valid")) for m in meta], dtype=bool)
    dis_ok = np.array([disorder_is_meaningful(int(m.get("length", 0)),
                                              int(m.get("n_polymer", 0) or 0))
                       for m in meta], dtype=bool)
    return {"tokens": tok, "mod_ids": mod, "chem": chem, "mask": mask,
            "contacts": [x["contacts"] for x in items],
            "lengths": np.array([int(x["length"]) for x in items], dtype=np.int32),
            "mg_site": mg, "b_factor_z": bz, "unknown_base": ub,
            "coords": xyz, "coord_mask": xyz_m,
            #: a residue is usable by the structure loss only when all three of
            #: its backbone atoms were resolved -- two points do not fix a frame
            "coord_residue_mask": mask & xyz_m.all(-1),
            # D12: the rigidity target is valid only where the structure is
            # X-ray, so the mask is per-chain AND per-residue
            "rigidity_mask": mask & rigid_ok[:, None],
            # head 10 recovers identity exactly where it was not assigned
            "base_mask": mask & (ub > 0),
            "unobserved_seq_id": [x.get("unobserved_seq_id") for x in items],
            # a chain whose declared polymer dwarfs the modelled part is a
            # truncation, not a disorder observation -- see the note above
            "disorder_valid": dis_ok,
            "n_polymer": np.array([int(m.get("n_polymer", 0) or 0) for m in meta],
                                  dtype=np.int32)}
