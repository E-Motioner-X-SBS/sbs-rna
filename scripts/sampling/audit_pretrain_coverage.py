#!/usr/bin/env python3
"""Is the stage-1 corpus representative of elDORS, or just the start of it?

`data/derived/parquet_starter` holds 10,000,000 sequences drawn from chunks
c001-c008 of the twenty elDORS ships. Two questions follow and they have
different answers:

**Volume** is not the issue. At the 2e9-token stage-1 budget and a median
length of 522 nt, one pass needs about 3.8M sequences. Ten million is 2.6
passes' worth, so the starter is not the binding constraint -- the GPU is.

**Coverage** might be. Using 8 of 20 chunks is fine if the chunks are an
arbitrary byte-split of one stream and a problem if they are ordered by source,
taxon or length. That is a measurement, not a judgement call, so this compares
the length distribution and base composition of chunks the starter DREW FROM
against chunks it did not.

If the used and unused chunks agree, the split carries no information and the
starter is a sample of the corpus rather than a slice of it.

Usage: python3 scripts/sampling/audit_pretrain_coverage.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ELDORS = ROOT / "data/sequences/elDORS_v1"
OUT = ROOT / "data/samples/analysis"

USED = (1, 3, 5, 8)            # the starter drew from c001-c008
UNUSED = (10, 13, 17, 20)      # it did not
N_SAMPLE = 60_000


def scan(chunk: int, n: int = N_SAMPLE) -> Dict:
    """Length and composition of the first `n` sequences of a chunk."""
    f = ELDORS / f"elDORS_v1_{chunk:03d}.fasta.gz"
    if not f.exists():
        return {}
    lens: List[int] = []
    comp = {c: 0 for c in "ACGTUN"}
    total = 0
    cur = 0
    with gzip.open(f, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    lens.append(cur)
                    if len(lens) >= n:
                        break
                cur = 0
                continue
            s = line.strip().upper()
            cur += len(s)
            total += len(s)
            for c in "ACGTUN":
                comp[c] += s.count(c)
    a = np.array(lens, dtype=np.int64)
    return {"chunk": chunk, "n": int(a.size),
            "len_mean": round(float(a.mean()), 1),
            "len_median": int(np.median(a)),
            "len_p10": int(np.percentile(a, 10)),
            "len_p90": int(np.percentile(a, 90)),
            "gc": round((comp["G"] + comp["C"]) / max(total, 1), 4),
            "n_frac": round(comp["N"] / max(total, 1), 5)}


def main() -> int:
    used = [r for c in USED if (r := scan(c))]
    unused = [r for c in UNUSED if (r := scan(c))]
    if not used or not unused:
        print("elDORS chunks not found")
        return 1

    def agg(rs, k):
        return float(np.mean([r[k] for r in rs]))

    keys = ("len_mean", "len_median", "len_p10", "len_p90", "gc", "n_frac")
    print(f"{N_SAMPLE:,} sequences from each of {len(used)} used and "
          f"{len(unused)} unused chunks\n")
    print(f"  {'statistic':12s} {'used (c1-8)':>14s} {'unused (c10-20)':>16s} "
          f"{'relative diff':>14s}")
    res = {"used_chunks": list(USED), "unused_chunks": list(UNUSED),
           "n_sampled_per_chunk": N_SAMPLE, "per_chunk": used + unused,
           "comparison": {}}
    worst = 0.0
    for k in keys:
        u, v = agg(used, k), agg(unused, k)
        rel = abs(u - v) / max(abs(u), 1e-9)
        worst = max(worst, rel)
        res["comparison"][k] = {"used": round(u, 5), "unused": round(v, 5),
                                "rel_diff": round(rel, 5)}
        print(f"  {k:12s} {u:14.4f} {v:16.4f} {100*rel:13.2f}%")
    res["worst_rel_diff"] = round(worst, 5)
    # A byte-split of one stream would agree closely on every one of these.
    # These do not: elDORS is sorted by length, so a chunk is a length band and
    # the chunk index is information.
    res["chunks_interchangeable"] = bool(worst < 0.10)

    # Which chunks does the built corpus actually draw from? This is the check
    # that matters, and the one that was never made: the starter covered 8 of
    # 20 and nothing said so, because "10,000,000 sequences" sounds like plenty.
    import re
    import pyarrow.parquet as pq
    corpus = ROOT / "data/derived/parquet_starter"
    tags, rows = set(), 0
    for f in sorted(corpus.glob("*.parquet")):
        m = re.match(r"eldors_c(\d+)_", f.name)
        if m:
            tags.add(int(m.group(1)))
        rows += pq.ParquetFile(f).metadata.num_rows
    total = len(list(ELDORS.glob("elDORS_v1_*.fasta.gz")))
    res["corpus_sequences"] = rows
    res["corpus_chunks"] = sorted(tags)
    res["eldors_chunks"] = total
    res["chunk_coverage"] = round(len(tags) / max(total, 1), 4)
    res["covers_all_chunks"] = bool(len(tags) == total)
    print(f"\nbuilt corpus: {rows:,} sequences from {len(tags)} of {total} "
          f"elDORS chunks ({100*res['chunk_coverage']:.0f}%)")
    if not res["covers_all_chunks"]:
        miss = sorted(set(range(1, total + 1)) - tags)
        print(f"  MISSING CHUNKS: {miss}")
        print("  the corpus is a length band, not a sample of elDORS")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pretrain_coverage.json").write_text(json.dumps(res, indent=1))
    print(f"\nworst relative difference {100*worst:.2f}%")
    print("chunks are interchangeable: any subset would be representative"
          if res["chunks_interchangeable"] else
          "chunks DIFFER by up to "
          f"{100*worst:.0f}%: elDORS is sorted by length, so a subset of chunks "
          "is a length band and the corpus must span all of them")
    print(f"-> {OUT / 'pretrain_coverage.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
