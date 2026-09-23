#!/usr/bin/env python3
"""Coevolution: Rfam alignments in, pair features out.

Until now the only coevolution signal reaching the model was `neff_over_l`, a
single scalar telling the MoE router how deep the family is. That is a hint
about how much to trust the sequence, not coevolution. Every competitive RNA
structure predictor consumes alignments, because covarying column pairs are the
strongest sequence-only evidence that two bases touch.

**Which alignments.** `data/families/rfam/full_alignments/` holds 4,077
Stockholm files and is the wrong set for this corpus: the families that
dominate it are absent. SSU rRNA (RF00177), LSU rRNA (RF02541/RF02543) and tRNA
(RF00005) have no full alignment, and together they are 84.4% of structural
residues -- only 3 of the top 10 families are covered.

`Rfam.seed.gz` covers **all 4,227 families**, including those: tRNA 954 seed
rows, SSU rRNA bacteria 99, LSU rRNA bacteria 102. Seeds are curated and
shallow where the full alignments are deep and partial, and shallow-but-present
beats deep-but-absent. `Rfam.full_region.gz` maps sequences to families and is
the route to deeper alignments built from our own 1.37B-sequence corpus; that
is a later, heavier path and this module is written so it can be swapped in.

**What comes out.** Two things, because they fail differently:

* `apc_mutual_information` -- an (L, L) matrix of APC-corrected mutual
  information. Classical, cheap, and strong on RNA: it needs no training, so it
  works on the first step of the first epoch. The APC correction subtracts the
  row/column background that otherwise makes conserved and high-entropy columns
  look coupled to everything.
* the sampled alignment itself, for the learned MSA track in
  `model/msa_track.py`, which can find structure that pairwise statistics miss.

Alignment columns are mapped onto the query by picking the seed row closest to
it and following that row's gaps, so an (L, L) feature always lines up with the
chain the model is folding.
"""
from __future__ import annotations

import gzip
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RFAM = ROOT / "data/families/rfam"
SEED = RFAM / "Rfam.seed.gz"
SEED_3D = RFAM / "Rfam.3d.seed.gz"
FULL_DIR = RFAM / "full_alignments"

#: RNA alphabet for coevolution counting: four bases plus gap. Anything else --
#: IUPAC ambiguity codes, which Rfam seeds contain -- folds into gap, because a
#: column position that is ambiguous carries no covariation signal and pretending
#: it is a fifth state adds noise to every pair it touches.
ALPHA = "ACGU-"
A_IDX = {c: i for i, c in enumerate(ALPHA)}
GAP = A_IDX["-"]


def _clean(seq: str) -> str:
    s = seq.upper().replace("T", "U").replace(".", "-").replace("~", "-")
    return "".join(c if c in A_IDX else "-" for c in s)


@lru_cache(maxsize=1)
def read_seed(path: Optional[str] = None) -> Dict[str, Dict]:
    """`accession -> {"id": name, "rows": [(name, aligned_seq), ...]}`.

    One pass over the concatenated Stockholm file. Rfam seeds wrap long
    alignments across several blocks per record, so rows accumulate by name.
    """
    p = Path(path) if path else SEED
    out: Dict[str, Dict] = {}
    ac: Optional[str] = None
    ident: Optional[str] = None
    rows: Dict[str, List[str]] = {}
    order: List[str] = []
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#=GF AC"):
                ac = line.split()[-1].split(".")[0]
            elif line.startswith("#=GF ID"):
                ident = line.split(None, 2)[2].strip() if len(line.split(None, 2)) > 2 else None
            elif line.startswith("//"):
                if ac and rows:
                    out[ac] = {"id": ident,
                               "rows": [(n, _clean("".join(rows[n]))) for n in order]}
                ac, ident, rows, order = None, None, {}, []
            elif line and not line.startswith("#"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    name, frag = parts
                    if name not in rows:
                        rows[name] = []
                        order.append(name)
                    rows[name].append(frag.strip())
    if ac and rows:
        out[ac] = {"id": ident,
                   "rows": [(n, _clean("".join(rows[n]))) for n in order]}
    return out


@lru_cache(maxsize=1)
def name_to_accession() -> Dict[str, str]:
    """Rfam family NAME (what the dataset carries) -> accession (what files use)."""
    return {v["id"]: k for k, v in read_seed().items() if v.get("id")}


def alignment_for(family: str) -> Optional[List[str]]:
    """Aligned rows for an Rfam family, by name or accession."""
    seeds = read_seed()
    ac = family if family in seeds else name_to_accession().get(family)
    if not ac or ac not in seeds:
        return None
    return [s for _, s in seeds[ac]["rows"]]


def encode_msa(rows: Sequence[str]) -> np.ndarray:
    """`(N, C)` uint8 over ALPHA."""
    if not rows:
        return np.zeros((0, 0), dtype=np.uint8)
    C = max(len(r) for r in rows)
    out = np.full((len(rows), C), GAP, dtype=np.uint8)
    for i, r in enumerate(rows):
        out[i, :len(r)] = np.frombuffer(
            r.ljust(C, "-").encode(), dtype=np.uint8)[:C]
    # translate ASCII -> alphabet index
    lut = np.full(256, GAP, dtype=np.uint8)
    for c, i in A_IDX.items():
        lut[ord(c)] = i
    return lut[out]


def sequence_weights(msa: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """Henikoff-style redundancy weights, capped for cost.

    Rfam seeds contain near-duplicate rows from related organisms. Counting them
    equally lets one over-sampled clade dominate every column statistic, which
    is the classic way a coevolution signal turns into a phylogeny signal.
    """
    N = msa.shape[0]
    if N <= 1:
        return np.ones(max(N, 1), dtype=np.float32)
    if N > 2000:                      # O(N^2) identity is the expensive part
        idx = np.random.default_rng(0).choice(N, 2000, replace=False)
        msa = msa[idx]
        N = 2000
    eq = (msa[:, None, :] == msa[None, :, :]).mean(-1)
    n_sim = (eq > threshold).sum(1).astype(np.float32)
    return (1.0 / np.maximum(n_sim, 1.0)).astype(np.float32)


def apc_mutual_information(msa: np.ndarray,
                           weights: Optional[np.ndarray] = None,
                           pseudocount: float = 0.5) -> np.ndarray:
    """`(C, C)` APC-corrected mutual information over alignment columns.

    Average Product Correction removes the background a column's own entropy
    contributes to every pair it appears in: without it the most conserved and
    the most variable columns both look coupled to the entire alignment, and the
    top-ranked "contacts" are an artefact of column composition.

        MI_apc(i,j) = MI(i,j) - mean_i(MI) * mean_j(MI) / mean(MI)
    """
    N, C = msa.shape
    if N == 0 or C == 0:
        return np.zeros((C, C), dtype=np.float32)
    w = np.ones(N, dtype=np.float32) if weights is None else weights[:N]
    W = float(w.sum()) or 1.0
    K = len(ALPHA)

    # single-column marginals, (C, K)
    oh = np.zeros((N, C, K), dtype=np.float32)
    np.put_along_axis(oh, msa[:, :, None].astype(np.int64), 1.0, axis=2)
    wi = (oh * w[:, None, None]).sum(0)
    fi = (wi + pseudocount) / (W + pseudocount * K)

    mi = np.zeros((C, C), dtype=np.float32)
    # pairwise joints, one column at a time to keep memory at O(C*K^2)
    for i in range(C):
        # (N, K) for column i, weighted against every other column
        oi = oh[:, i, :] * w[:, None]
        joint = np.einsum("nk,ncl->ckl", oi, oh)       # (C, K, K)
        pij = (joint + pseudocount / K) / (W + pseudocount)
        outer = fi[i][None, :, None] * fi[:, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = pij * np.log(np.maximum(pij, 1e-12) / np.maximum(outer, 1e-12))
        mi[i] = np.nansum(term, axis=(1, 2))
    np.fill_diagonal(mi, 0.0)

    m_row = mi.mean(1, keepdims=True)
    m_all = float(mi.mean()) or 1e-9
    apc = mi - (m_row @ m_row.T) / m_all
    np.fill_diagonal(apc, 0.0)
    return apc.astype(np.float32)


def _ungapped(row: str) -> str:
    return row.replace("-", "")


def map_to_query(rows: Sequence[str], query: str) -> Optional[np.ndarray]:
    """Alignment-column index for each query position, or None if no row fits.

    The seed row whose ungapped sequence is closest to the query supplies the
    gap pattern. Rfam seeds are curated so a family member is usually an
    excellent match; when the best match is poor the caller gets None and falls
    back to no coevolution rather than to a misaligned feature, which would be
    worse than none.
    """
    q = _clean(query).replace("-", "")
    if not rows or not q:
        return None
    best, best_score = None, -1.0
    for r in rows:
        u = _ungapped(r)
        if not u:
            continue
        n = min(len(u), len(q))
        if n < 0.5 * max(len(u), len(q)):
            continue
        score = sum(1 for a, b in zip(u[:n], q[:n]) if a == b) / max(n, 1)
        if score > best_score:
            best, best_score = r, score
    if best is None or best_score < 0.5:
        return None
    cols = np.array([c for c, ch in enumerate(best) if ch != "-"], dtype=np.int32)
    if cols.size == 0:
        return None
    L = len(q)
    if cols.size >= L:
        return cols[:L]
    out = np.full(L, -1, dtype=np.int32)
    out[:cols.size] = cols
    return out


def coevolution_matrix(family: str, query: str,
                       max_rows: int = 2000) -> Optional[np.ndarray]:
    """`(L, L)` APC-MI for `query`, or None when the family gives nothing usable."""
    rows = alignment_for(family)
    if not rows or len(rows) < 8:      # below this MI is noise, not signal
        return None
    if len(rows) > max_rows:
        idx = np.random.default_rng(0).choice(len(rows), max_rows, replace=False)
        rows = [rows[i] for i in idx]
    cols = map_to_query(rows, query)
    if cols is None:
        return None
    msa = encode_msa(rows)
    apc = apc_mutual_information(msa, sequence_weights(msa))
    L = cols.size
    out = np.zeros((L, L), dtype=np.float32)
    ok = cols >= 0
    ii = cols[ok]
    sub = apc[np.ix_(ii, ii)]
    oi = np.where(ok)[0]
    out[np.ix_(oi, oi)] = sub
    return out


# ---------------------------------------------------------------------------
# The cached form: per-family couplings, mapped into a chain's own indices
# ---------------------------------------------------------------------------
#
# `coevolution_matrix` recomputes the alignment's mutual information on every
# call, which costs 26 s for SSU_rRNA_bacteria and would be paid once per chain
# -- 14,593 times across the corpus for only 361 distinct families. The cache
# `scripts/precompute_coevolution.py` writes pays it 361 times instead, and
# stores the strongly-coupled pairs rather than the dense matrix, which for a
# 1,980-column rRNA is the difference between 8k pairs and 3.9M mostly-noise
# entries.

CACHE = ROOT / "data/derived/coevolution"


@lru_cache(maxsize=512)
def _cached_family(family: str):
    """`(pairs (n,2), score (n,))` in ALIGNMENT-column indices, or None."""
    f = CACHE / f"{family.replace('/', '_')}.npz"
    if not f.exists():
        return None
    try:
        z = np.load(f)
        return z["pairs"], z["score"]
    except (OSError, ValueError, KeyError):
        return None


@lru_cache(maxsize=4096)
def coevolution_pairs(family: str, query: str, top_k: Optional[int] = None
                      ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Cached couplings in THIS chain's residue indices: `(pairs, score)`.

    The cache is keyed by family and indexed by alignment column; a chain has
    its own gaps and its own length, so the columns have to be mapped back
    through the seed row that best matches the chain's sequence. A pair whose
    either end maps to a gap in this chain is dropped -- there is no residue to
    attach the coupling to, and inventing one is how a feature ends up pointing
    at the wrong nucleotide.

    Returns None when the family has no cache or no seed row fits, which the
    caller must treat as "no coevolution for this chain" rather than as zeros:
    12% of the corpus has no Rfam family at all and the model has to work
    without the feature anyway.

    Cached on `(family, query)`. `map_to_query` scans every seed row to find the
    best match, which at a few hundred chains a batch would dominate the data
    loader -- and the corpus is full of repeats, 444 tRNA chains among them,
    so the hit rate is high. The returned arrays are shared, so callers must
    not mutate them.
    """
    got = _cached_family(family)
    if got is None:
        return None
    rows = alignment_for(family)
    if not rows:
        return None
    cols = map_to_query(rows, query)          # (L,) alignment column per residue
    if cols is None:
        return None
    pairs, score = got

    # invert: alignment column -> residue index, -1 where this chain has a gap
    n_cols = int(max(pairs.max(initial=0), cols.max(initial=0))) + 1
    inv = np.full(n_cols, -1, dtype=np.int32)
    ok = cols >= 0
    inv[cols[ok]] = np.nonzero(ok)[0].astype(np.int32)

    a = np.where(pairs[:, 0] < n_cols, inv[np.clip(pairs[:, 0], 0, n_cols - 1)], -1)
    b = np.where(pairs[:, 1] < n_cols, inv[np.clip(pairs[:, 1], 0, n_cols - 1)], -1)
    keep = (a >= 0) & (b >= 0)
    if not keep.any():
        return None
    out = np.stack([a[keep], b[keep]], 1).astype(np.int32)
    sc = score[keep].astype(np.float32)
    if top_k is not None and len(sc) > top_k:
        sel = np.argsort(sc)[::-1][:top_k]
        out, sc = out[sel], sc[sel]
    return out, sc
