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


FULL_ALIGNMENTS = ROOT / "data/families/rfam/full_alignments"


def full_alignment_rows(family: str, max_rows: int = 2000,
                        seed: int = 0) -> Optional[List[str]]:
    """Up to `max_rows` sequences from a family's FULL alignment, streamed.

    The seed alignments this module reads by default are curated and tiny:
    chain-weighted over the 3D corpus, **67.2% of family-assigned chains**
    sit behind an effective depth (Henikoff Neff) **below 50**, and the
    median family is Neff **4.0**. APC-corrected mutual information from
    four effective sequences is noise, and the index has recorded that
    number since the cache was first built.

    For 213 of the corpus's 234 families a full alignment exists, covering
    38.5% of family-assigned chains -- including 5S_rRNA, the second
    largest at 2,235 chains, whose seed is 712 rows against **594,154** in
    the full set. `msa.py`'s standing argument against the full alignments
    is that the families that dominate the corpus (SSU/LSU rRNA, tRNA)
    have none, and that is true and is why this is a UNION and not a
    replacement: full where one exists, seed otherwise.

    Streamed and reservoir-sampled because 5S_rRNA's file is **701 MiB**
    and only `max_rows` of it is ever used. Rfam full alignments are one
    line per sequence, which this relies on and checks: a wrapped file
    would give a name twice, and it returns None rather than silently
    building an alignment out of half-sequences.
    """
    import random
    f = FULL_ALIGNMENTS / f"{family}.sto"
    if not f.exists():
        ac = name_to_accession().get(family)
        f = FULL_ALIGNMENTS / f"{ac}.sto" if ac else f
    if not f.exists():
        return None
    rng = random.Random(seed)
    res: List[str] = []
    names: List[str] = []
    n = 0
    try:
        with open(f, "r", errors="ignore") as fh:
            for ln in fh:
                if not ln.strip() or ln[0] == "#" or ln.startswith("//"):
                    continue
                parts = ln.split(None, 1)
                if len(parts) != 2:
                    continue
                nm, seq = parts[0], parts[1].strip()
                n += 1
                if len(res) < max_rows:
                    res.append(seq); names.append(nm)
                else:
                    j = rng.randrange(n)
                    if j < max_rows:
                        res[j] = seq; names[j] = nm
    except OSError:
        return None
    if len(set(names)) != len(names):      # wrapped: not one line per sequence
        return None
    if len(res) < 8 or len({len(r) for r in res}) != 1:
        return None                        # below the MI noise floor, or ragged

    # MATCH COLUMNS ONLY. Stockholm marks insert columns with lowercase
    # residues and `.` gaps, and a full alignment is mostly inserts: FMN
    # reads 816 columns of which 221 are match states, and the raw rows are
    # 84.6% `.`. `ALPHA` is "ACGU-", so `encode_msa` maps every lowercase
    # residue AND every dot to GAP -- which makes all rows look alike and
    # collapses the Henikoff weighting to **Neff 1.0**. Measured, on the
    # first version of this function, which otherwise looked like it
    # worked: FMN full "2000 rows -> Neff 1.0" against the seed's 21.2.
    #
    # A column is match-or-insert for the whole alignment, so the mask
    # comes from any one row: uppercase or `-` is a match state.
    keep = [i for i, ch in enumerate(res[0]) if ch.isupper() or ch == "-"]
    if len(keep) < 16:
        return None
    return ["".join(r[i] for i in keep).upper() for r in res]


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
        # `W + pseudocount * K`, not `W + pseudocount`. Laplace smoothing has
        # to be consistent between the joint and the marginals or the joint is
        # not a distribution: adding `pc/K` to each of K^2 cells adds `pc*K` of
        # mass, so the denominator must absorb `pc*K`. With `W + pc` the joint
        # summed to `(W + pc*K) / (W + pc)` -- measured 1.1905 at N=10, 1.0396
        # at N=50, 1.0199 at N=100, 1.0020 at N=1000 -- while `fi` above summed
        # to exactly 1.
        #
        # The bias is therefore DEPTH-DEPENDENT, which is the harmful part: a
        # shallow Rfam family got a systematically larger mutual information
        # than a deep one for the same actual coupling, and alignment depth is
        # a property of the family rather than of the structure.
        pij = (joint + pseudocount / K) / (W + pseudocount * K)
        outer = fi[i][None, :, None] * fi[:, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = pij * np.log(np.maximum(pij, 1e-12) / np.maximum(outer, 1e-12))
        mi[i] = np.nansum(term, axis=(1, 2))
    np.fill_diagonal(mi, 0.0)

    # The means include the zeroed diagonal, where Dunn et al. take them over
    # j != i. That scales the whole correction by (C-1)/C -- 1% at C=100 --
    # which is a uniform under-correction and not depth-dependent, so it is
    # left alone and recorded rather than changed under a model in flight.
    m_row = mi.mean(1, keepdims=True)
    m_all = float(mi.mean()) or 1e-9
    apc = mi - (m_row @ m_row.T) / m_all
    np.fill_diagonal(apc, 0.0)
    return apc.astype(np.float32)


def _ungapped(row: str) -> str:
    return row.replace("-", "")


def _kmer_profile(s: str, k: int = 5) -> set:
    return {s[i:i + k] for i in range(0, max(len(s) - k + 1, 0))}


def _best_row(rows: Sequence[str], q: str) -> Optional[tuple]:
    """The seed row most similar to `q`, by k-mer Jaccard.

    Positional identity was used here and is wrong for anything long. It walks
    two ungapped strings in lockstep, so a single indel decorrelates everything
    after it: real 1,500-nucleotide SSU rRNA chains scored 0.36-0.46 against
    their own family's seed rows and were rejected by the 0.5 threshold, which
    silently removed the two largest families in the corpus -- 3,450 chains --
    from coevolution entirely. K-mer overlap does not care where the indel is.
    """
    qk = _kmer_profile(q)
    if not qk:
        return None
    best, best_score = None, -1.0
    for r in rows:
        u = _ungapped(r)
        if not u:
            continue
        # length gate first: it is cheap and a 10x length mismatch is not the
        # same molecule however many k-mers happen to coincide
        lo, hi = min(len(u), len(q)), max(len(u), len(q))
        if lo < 0.5 * hi:
            continue
        uk = _kmer_profile(u)
        if not uk:
            continue
        j = len(qk & uk) / len(qk | uk)
        if j > best_score:
            best, best_score = r, j
    return (best, best_score) if best is not None else None


def map_to_query(rows: Sequence[str], query: str,
                 min_similarity: float = 0.10) -> Optional[np.ndarray]:
    """Alignment-column index for each query position, or None if no row fits.

    Two steps, because one was not enough. The seed row closest to the query is
    chosen by k-mer overlap, then the query is **actually aligned** to that
    row's ungapped sequence and the alignment carries query positions through
    the row's gap pattern into alignment columns.

    The previous version did neither: it scored rows by walking two ungapped
    strings position by position and then assumed query position i sat at the
    row's i-th non-gap column. Both assumptions hold for tRNA, which is 76
    nucleotides with almost no indels, and neither holds for rRNA. That is why
    coevolution appeared to work -- it was validated on tRNA.

    A query position with no counterpart in the seed row maps to -1 and the
    caller drops it, which is the honest answer for an insertion the family's
    alignment has no column for.
    """
    q = _clean(query).replace("-", "")
    if not rows or not q:
        return None
    got = _best_row(rows, q)
    if got is None:
        return None
    row, sim = got
    if sim < min_similarity:
        return None

    # column index of each non-gap character of the chosen row
    cols = np.array([c for c, ch in enumerate(row) if ch != "-"], dtype=np.int32)
    u = _ungapped(row)
    if cols.size == 0:
        return None

    out = np.full(len(q), -1, dtype=np.int32)
    try:
        from Bio import Align

        aligner = Align.PairwiseAligner(mode="global", match_score=1,
                                        mismatch_score=-1, open_gap_score=-5,
                                        extend_gap_score=-0.5)
        aln = aligner.align(u, q)[0]
        # aligned blocks are ((u_start, u_end), ...) paired with the same for q
        ub, qb = aln.aligned
        for (us, ue), (qs, qe) in zip(ub, qb):
            n = min(ue - us, qe - qs)
            if n <= 0:
                continue
            out[qs:qs + n] = cols[us:us + n]
    except Exception:                                        # noqa: BLE001
        # Biopython absent or the alignment failed: fall back to the positional
        # assumption, which is right for short indel-free families and is
        # better than returning nothing for them
        n = min(cols.size, len(q))
        out[:n] = cols[:n]
    if not (out >= 0).any():
        return None
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


@lru_cache(maxsize=512)
def family_depth(family: str) -> Optional[int]:
    """Rows in the family's alignment, or None if there is no cache.

    `precompute_coevolution.py` has always stored `n_rows` beside the
    couplings, and nothing ever read it. It is the numerator of `Neff/L`,
    the router conditioning feature that `RouterFeatures` declares,
    `moe.py` reads into `extra[:, 0]`, and no caller has ever set --
    finding 68, open since it was declared. The trainer's note that
    "Neff/L is not in the 3D set, so it is left absent rather than
    invented" was true when written; building this cache made it
    available and nobody went back.

    It is the **Rfam SEED row count**, and neither a redundancy-weighted
    Neff nor a full-alignment depth. Both distinctions matter and are
    measured, not assumed: over 60 cached families the median seed depth
    is **13** rows against a median **101** in the corresponding full
    alignment, and 5S_rRNA reads 712 against **594,154** -- an 834x gap.
    Calling this `Neff/L` without saying so would be inventing a feature
    that looks like alignment depth and is a curated sample size.

    It is still strictly more than the constant zero `extra[:, 0]` has
    carried in every stage, which is why it is offered at all -- behind
    `--router-neff`, off by default, as an A/B and not a fix.
    """
    f = CACHE / f"{family.replace('/', '_')}.npz"
    if not f.exists():
        return None
    try:
        return int(np.load(f)["n_rows"])
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
