#!/usr/bin/env python3
"""Per-CHAIN Rfam assignment, from RNA3DB's Infernal cmscan.

The corpus assigned Rfam families **per entry**: `build_dataset` keys its
metadata by `pdb_id` alone, so every chain in a deposition inherits one family.
For a ribosome that is wrong for most of the chains in it -- a 76-nucleotide
tRNA sitting in a bacterial ribosome was labelled `SSU_rRNA_bacteria` along
with the 1,500-nucleotide small subunit it is bound to.

The damage shows up in the length distribution, which is the cheapest possible
check and settles it:

    label                     n      median length
    ours   SSU_rRNA_bacteria  3,606      122        <- SSU rRNA is ~1,500 nt
    ours   LSU_rRNA_bacteria  3,514      120        <- LSU rRNA is ~2,900 nt
    cmscan tRNA               3,246       76
    cmscan 5S_rRNA            2,233      120
    cmscan SSU_rRNA_bacteria  1,648    1,511
    cmscan LSU_rRNA_bacteria  1,730    2,871

Our buckets are mixtures spanning tRNA to LSU; cmscan's are tight around the
right size. The two disagree on 59.1% of the 13,496 chains both label.

This matters because the family selects the **alignment** coevolution is
computed from. A tRNA scored against the SSU rRNA alignment is not a weak
feature, it is a feature about a different molecule. `map_to_query` rejects
most such pairings (it requires a seed row within 50% length and 50% identity),
so the practical effect was silent loss of coverage rather than wrong
couplings -- but it means the headline "87.9% of chains have a family" was
counting families that mostly could not be used.

Writes `data/derived/chain_rfam.json`: `{"<pdb>_<chain>": {...}}` using each
chain's best-scoring hit above Infernal's own inclusion threshold.

    python scripts/build_chain_rfam.py
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "data/structures/databases/rna3db/rna3db_extracted/rna3db-cmscans"
OUT = ROOT / "data/derived/chain_rfam.json"


def parse(tbl: Path) -> dict:
    """`{chain_key: {family, accession, evalue, score}}`, best hit per chain.

    Only rows Infernal itself marks `!` are kept. The `?` rows are hits below
    the inclusion threshold and the table is full of them -- the first data row
    in the file is an HIV packaging signal matched to a ribosome chain at
    E-value 5.5, which is noise.
    """
    best: dict = {}
    n_rows = n_inc = 0
    for line in tbl.open():
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 18:
            continue
        n_rows += 1
        if p[16] != "!":
            continue
        n_inc += 1
        fam, acc, chain, score, ev = p[0], p[1], p[2], p[14], p[15]
        try:
            e, sc = float(ev), float(score)
        except ValueError:
            continue
        prev = best.get(chain)
        # rank by E-value, then by bit score: a ribosome chain matches several
        # related models and the tightest one is the right call
        if prev is None or (e, -sc) < (prev["evalue"], -prev["score"]):
            best[chain] = {"family": fam, "accession": acc,
                           "evalue": e, "score": sc}
    print(f"[rfam] {n_rows:,} scan rows, {n_inc:,} above the inclusion "
          f"threshold, {len(best):,} chains assigned")
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    tbls = sorted(TBL.glob("*.tbl")) if TBL.is_dir() else []
    tbls = [t for t in tbls if "nohits" not in t.name]
    if not tbls:
        print(f"no cmscan table under {TBL}; extract rna3db-cmscans.tar.gz",
              file=sys.stderr)
        return 1
    best = parse(tbls[-1])

    fam = collections.Counter(v["family"] for v in best.values())
    print(f"[rfam] {len(fam)} distinct families; "
          f"top: {', '.join(f'{k} {n}' for k, n in fam.most_common(5))}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(best))
    print(f"[rfam] -> {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
