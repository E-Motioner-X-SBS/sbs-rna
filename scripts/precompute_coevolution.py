#!/usr/bin/env python3
"""Cache APC-corrected coevolution couplings, once per Rfam family.

87.9% of the 3D corpus (14,593 of 16,604 chains) carries an Rfam family, and
those 14,593 chains draw on only **361 distinct families**. Coevolution is a
property of the family's alignment, not of the individual deposition, so
computing it per chain would do the same work forty times over. This computes
it 361 times and lets the chains map into it.

Dense storage is not an option either: SSU_rRNA_bacteria is 1,980 alignment
columns and LSU is longer, so an (L, L) float32 matrix per family runs to tens
of megabytes and most of it is noise. What a pair track can actually use is the
strongly-coupled pairs, so only the top `--per-column` couplings per column are
kept -- the same sparsity budget the hierarchical pair track already works to.

Cost, measured: 0.8 s for tRNA (954 rows x 118 columns), 25.8 s for
SSU_rRNA_bacteria (99 x 1,980). The whole set runs on CPU in well under an
hour, which is why this is a separate script rather than something the data
loader does on the fly.

    python scripts/precompute_coevolution.py            # every family in the corpus
    python scripts/precompute_coevolution.py --family tRNA
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharos.data.msa import (full_alignment_rows, alignment_for, apc_mutual_information,  # noqa: E402
                             encode_msa, sequence_weights)

MANIFEST = ROOT / "data/derived/pharos3d/manifest.json"
OUT = ROOT / "data/derived/coevolution"


def families_in_corpus() -> Dict[str, int]:
    """`{family: chain count}` for every family the 3D corpus actually uses."""
    if not MANIFEST.exists():
        return {}
    m = json.loads(MANIFEST.read_text())
    counts: Dict[str, int] = {}
    for shard in m.get("shards", []):
        for c in shard.get("chains", []):
            f = c.get("rfam")
            if f:
                counts[f] = counts.get(f, 0) + 1
    return counts


def top_couplings(mi: np.ndarray, per_column: int, min_sep: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """The `per_column * L` strongest pairs, as `(pairs (n,2) int32, score (n,))`.

    Near-diagonal pairs are dropped: adjacent alignment columns covary because
    they are adjacent, not because they are in contact, and admitting them
    would fill the budget with the one signal a structure predictor already
    has for free.
    """
    L = mi.shape[0]
    m = mi.copy()
    idx = np.arange(L)
    near = np.abs(idx[:, None] - idx[None, :]) < min_sep
    m[near] = -np.inf
    m = np.triu(m, 1)
    m[np.tril_indices(L)] = -np.inf

    k = min(L * L, max(per_column * L, 1))
    flat = m.ravel()
    if k >= flat.size:
        sel = np.argsort(flat)[::-1]
    else:
        sel = np.argpartition(flat, -k)[-k:]
        sel = sel[np.argsort(flat[sel])[::-1]]
    score = flat[sel]
    keep = np.isfinite(score)
    sel, score = sel[keep], score[keep]
    return np.stack([sel // L, sel % L], 1).astype(np.int32), score.astype(np.float32)


def run(families: List[str], per_column: int, min_sep: int,
        max_rows: int, force: bool, prefer_full: bool = False,
        out: Path = OUT) -> None:
    OUT_ = out
    globals()["OUT"] = out
    OUT_.mkdir(parents=True, exist_ok=True)
    index: Dict[str, Dict] = {}
    idx_path = OUT / "index.json"
    if idx_path.exists() and not force:
        index = json.loads(idx_path.read_text())

    t_start = time.time()
    for n, fam in enumerate(families, 1):
        safe = fam.replace("/", "_")
        dest = OUT / f"{safe}.npz"
        if dest.exists() and not force:
            continue
        # Finding 112: the FULL alignment where one exists, the seed
        # otherwise. Chain-weighted, 67.2% of family-assigned chains sit
        # behind an effective depth under Neff 50 and the median family
        # reads 4.0, which is noise; the full alignments lift the families
        # that have one by 16x to 35x. The families that dominate the
        # corpus (SSU/LSU rRNA, tRNA) have none, which is why this is a
        # union and not a replacement.
        source = "seed"
        rows = None
        if prefer_full:
            rows = full_alignment_rows(fam, max_rows)
            if rows:
                source = "full"
        if not rows:
            rows = alignment_for(fam)
        if not rows:
            index[fam] = {"status": "no_alignment"}
            print(f"[{n}/{len(families)}] {fam}: no alignment", flush=True)
            continue
        t0 = time.time()
        msa = encode_msa(rows)
        if msa.shape[0] > max_rows:
            # Henikoff weighting already caps the effective count; this caps the
            # arithmetic. Rows are taken evenly rather than from the top, so the
            # subsample is not the alignment's first N near-duplicates.
            msa = msa[np.linspace(0, msa.shape[0] - 1, max_rows).astype(int)]
        w = sequence_weights(msa)
        mi = apc_mutual_information(msa, w)
        pairs, score = top_couplings(mi, per_column, min_sep)
        # `source` is load-bearing: the pairs are indices into the
        # alignment they were computed from, and `coevolution_pairs` must
        # map them onto a chain through that same one.
        np.savez_compressed(dest, pairs=pairs, score=score,
                            n_cols=np.int32(msa.shape[1]),
                            n_rows=np.int32(msa.shape[0]),
                            source=np.str_(source))
        index[fam] = {"status": "ok", "file": dest.name, "source": source,
                      "n_rows": int(msa.shape[0]), "n_cols": int(msa.shape[1]),
                      "n_pairs": int(len(pairs)),
                      "neff": round(float(w.sum()), 1),
                      "seconds": round(time.time() - t0, 1)}
        print(f"[{n}/{len(families)}] {fam}: {msa.shape[0]} x {msa.shape[1]}, "
              f"{len(pairs):,} pairs, Neff {w.sum():.0f}, "
              f"{time.time()-t0:.1f}s", flush=True)
        idx_path.write_text(json.dumps(index, indent=1))

    idx_path.write_text(json.dumps(index, indent=1))
    ok = sum(1 for v in index.values() if v.get("status") == "ok")
    print(f"\n[coev] {ok} families cached, {len(index)-ok} without an alignment, "
          f"{time.time()-t_start:.0f}s total")
    # `--out` may be given relative to the CWD, which is not ROOT.
    try:
        _shown = OUT.resolve().relative_to(ROOT)
    except ValueError:
        _shown = OUT.resolve()
    print(f"[coev] -> {_shown}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", action="append", default=None,
                    help="one family (repeatable); default is every family the "
                         "3D corpus uses")
    ap.add_argument("--per-column", type=int, default=4,
                    help="couplings kept per alignment column")
    ap.add_argument("--min-sep", type=int, default=4,
                    help="drop pairs closer than this along the alignment")
    ap.add_argument("--max-rows", type=int, default=2000,
                    help="subsample alignments deeper than this")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prefer-full", action="store_true",
                    help="finding 112: use a family's FULL alignment when one "
                         "exists, seed otherwise. Lifts the families that "
                         "have one by 16x-35x in effective depth.")
    ap.add_argument("--out", type=lambda x: Path(x).resolve(), default=OUT,
                    help="cache directory. Write a --prefer-full build to a "
                         "SEPARATE directory and select it with "
                         "PHAROS_COEV_CACHE: swapping the feature under a "
                         "trained checkpoint is an A/B, not a fix.")
    args = ap.parse_args()

    if args.family:
        fams = args.family
    else:
        counts = families_in_corpus()
        if not counts:
            print("no corpus manifest; pass --family", file=sys.stderr)
            return 1
        # biggest first: the long rRNA families dominate the runtime, so a
        # partial run still covers the chains that matter most
        fams = [f for f, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
        print(f"[coev] {len(fams)} families covering {sum(counts.values()):,} chains")
    run(fams, args.per_column, args.min_sep, args.max_rows,
        args.force, prefer_full=args.prefer_full, out=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
