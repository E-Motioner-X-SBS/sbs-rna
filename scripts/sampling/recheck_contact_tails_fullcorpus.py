#!/usr/bin/env python3
"""Re-derive the remaining TAIL claims of Facts 2 and 5.1 on the full corpus.

Two tail claims from the 180-structure sample survive unchecked after the
`target_c` and 4096-context failures:

  Fact 2 (Section 1)   "Median 4.40 contacts/nt, p95 5.42, **max 5.50** --
                        bounded by RNA's coordination geometry, independent of
                        length." The bound is load-bearing: it is the reason a
                        fixed per-nucleotide budget is safe at any L.

  Section 5.1          contact separation: "median 78 nt on long chains", and
                        a +/-512 band still misses 19% (0.812 captured at
                        L 1500-3000). This is what kills the banded fallback.

Both are maxima or extreme quantiles, and both come from n=180. The two tail
claims already tested on the full corpus (max effective c, longest chain) each
failed while their neighbouring means reproduced, so these are measured rather
than assumed.

Per-chain values are RETAINED here, unlike `analyze_block_sparsity.py` which
summarises into bins and discards them -- that is why the contacts/nt maximum
could not be recovered from the earlier run.

Usage
-----
    PY=/store/shuvam/.venv/bin/python
    $PY scripts/sampling/recheck_contact_tails_fullcorpus.py --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RNA3DB = ROOT / "data/structures/databases/rna3db/rna3db_extracted/rna3db-mmcifs"
GRNADE = ROOT / "data/structures/databases/grnade_rnasolo/extracted/raw"
OUT = ROOT / "data/samples/analysis"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recheck_block_sparsity_fullcorpus import (            # noqa: E402
    parse_mmcif, parse_pdb, contact_pairs, MIN_L, MAX_L, MIN_CONTACTS)

#: Bands from Section 5.1's table.
BANDS = [32, 64, 128, 256, 512]


def analyse(path_str: str) -> Optional[Dict]:
    p = Path(path_str)
    try:
        atoms = (parse_mmcif(p) if p.suffix == ".cif" else parse_pdb(p))["acgu"]
    except Exception:                                        # noqa: BLE001
        return None
    if not atoms:
        return None
    L = len(atoms)
    if not (MIN_L <= L <= MAX_L):
        return None
    pairs = contact_pairs(atoms)
    if len(pairs) < MIN_CONTACTS:
        return None
    sep = np.array([j - i for i, j in pairs], dtype=np.int64)
    return {
        "file": p.name, "L": L, "n_contacts": len(pairs),
        "contacts_per_nt": len(pairs) / L,
        "median_sep": float(np.median(sep)),
        "band": {str(b): float((sep <= b).mean()) for b in BANDS},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="contact_tails_fullcorpus.json")
    args = ap.parse_args()

    files = [str(p) for p in RNA3DB.rglob("*.cif")] + [str(p) for p in GRNADE.glob("*.pdb")]
    files.sort()
    if args.limit:
        files = files[::max(1, len(files) // args.limit)][:args.limit]
    print(f"[tails] {len(files):,} files, {args.workers} workers")

    rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, r in enumerate(ex.map(analyse, files, chunksize=16), 1):
            if r:
                rows.append(r)
            if n % 2000 == 0:
                print(f"[tails]   {n}/{len(files)} usable={len(rows)}", flush=True)

    cpn = np.array([r["contacts_per_nt"] for r in rows])
    res = {
        "n_chains": len(rows),
        "contacts_per_nt": {
            "published": {"median": 4.40, "p95": 5.42, "max": 5.50},
            "median": float(np.median(cpn)), "p95": float(np.percentile(cpn, 95)),
            "p99": float(np.percentile(cpn, 99)),
            "p999": float(np.percentile(cpn, 99.9)), "max": float(cpn.max()),
            "n_over_5_50": int((cpn > 5.50).sum()),
            "frac_over_5_50": float((cpn > 5.50).mean()),
            "worst": sorted(((r["contacts_per_nt"], r["file"], r["L"]) for r in rows),
                            reverse=True)[:5],
        },
        "separation_by_bin": [],
    }
    for lo, hi in [(500, 1500), (1500, 3000)]:
        sel = [r for r in rows if lo <= r["L"] < hi]
        if not sel:
            continue
        res["separation_by_bin"].append({
            "L_range": f"{lo}-{hi}", "n": len(sel),
            "median_sep": float(np.median([r["median_sep"] for r in sel])),
            "band_capture": {b: round(float(np.mean([r["band"][b] for r in sel])), 4)
                             for b in map(str, BANDS)},
        })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(res, indent=1))

    c = res["contacts_per_nt"]
    print(f"\ncontacts/nt   published median 4.40  p95 5.42  max 5.50")
    print(f"              full      median {c['median']:.2f}  p95 {c['p95']:.2f}  "
          f"p99 {c['p99']:.2f}  p99.9 {c['p999']:.2f}  MAX {c['max']:.2f}")
    print(f"              chains over the published max 5.50: {c['n_over_5_50']:,} "
          f"({100*c['frac_over_5_50']:.2f}%)")
    print(f"              worst: {[(round(a,2), b, l) for a, b, l in c['worst'][:3]]}")
    print(f"\nseparation    published: median 78 nt long chains; band 512 captures 0.812 (1500-3000)")
    for e in res["separation_by_bin"]:
        print(f"              {e['L_range']}: n={e['n']:,} median_sep={e['median_sep']:.0f}  "
              f"bands " + " ".join(f"{b}:{v}" for b, v in e["band_capture"].items()))
    print(f"\n[tails] -> {OUT / args.out}")


if __name__ == "__main__":
    main()
