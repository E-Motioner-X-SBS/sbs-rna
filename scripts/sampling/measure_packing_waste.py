#!/usr/bin/env python3
"""How much of a stage-1 step is padding, and what does fixing it cost?

Three packers over the same 400,000 elDORS sequences, at the shipped 24,576-
token budget:

  stream order   what shipped. Sequences are appended in corpus order and cut
                 when the batch would exceed the budget, so a 20-nt sequence
                 and a 1,024-nt one land in the same batch and everything pads
                 to 1,024.
  length-sorted  the pool is sorted before packing. This is the floor.
  quantised      length-sorted, then the batch width rounded up to
                 `LEN_QUANTUM` (64, half a GDN chunk). Costs padding back, and
                 buys 16 distinct tensor widths instead of 969 -- which is what
                 makes `torch.compile` usable, because the delta-rule chunk
                 loop is a Python loop and inductor specialises on the chunk
                 count.

Deterministic: same corpus, same budget, no RNG, no GPU. The numbers are pinned
in `verify_claims.py`.

Usage: python3 scripts/sampling/measure_packing_waste.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/derived/parquet_starter"
OUT = ROOT / "data/samples/analysis"

N_SEQ = 400_000
BUDGET, MAX_BATCH, MIN_LEN, MAX_LEN = 24576, 512, 20, 1024
QUANTUM = 64


def lengths(limit: int) -> List[int]:
    """Lengths from the corpus, striding across shards rather than reading the
    first few.

    elDORS is sorted by length and the shards are named after their chunk, so
    reading `sorted(glob)` in order samples only the longest chunk. Taking a
    fixed quota from every shard makes this a measurement of the corpus the
    trainer sees, which shuffles shard order for the same reason.
    """
    out: List[int] = []
    files = sorted(CORPUS.glob("*.parquet"))
    per = max(1, limit // max(len(files), 1))
    for f in files:
        got = 0
        for b in pq.ParquetFile(f).iter_batches(batch_size=8192,
                                                columns=["sequence"]):
            for s in b.column("sequence").to_pylist():
                if s is None:
                    continue
                n = len(s)
                if MIN_LEN <= n <= MAX_LEN:
                    out.append(n)
                    got += 1
                    if got >= per or len(out) >= limit:
                        break
            if got >= per or len(out) >= limit:
                break
        if len(out) >= limit:
            break
    return out


def _lengths_head(limit: int) -> List[int]:
    out: List[int] = []
    for f in sorted(CORPUS.glob("*.parquet")):
        for b in pq.ParquetFile(f).iter_batches(batch_size=8192,
                                                columns=["sequence"]):
            for s in b.column("sequence").to_pylist():
                if s is None:
                    continue
                n = len(s)
                if MIN_LEN <= n <= MAX_LEN:
                    out.append(n)
                    if len(out) >= limit:
                        return out
    return out


def pack(lens: List[int], *, sort: bool, quantum: int) -> dict:
    seq = sorted(lens) if sort else lens
    buf: List[int] = []
    real = pad = steps = 0
    widths, sizes = set(), []
    for n in seq:
        buf.append(n)
        w = max(buf)
        if quantum:
            w = ((w + quantum - 1) // quantum) * quantum
        if len(buf) * w >= BUDGET or len(buf) >= MAX_BATCH:
            real += sum(buf); pad += len(buf) * w; steps += 1
            widths.add(w); sizes.append(len(buf)); buf = []
    if buf:
        w = max(buf)
        if quantum:
            w = ((w + quantum - 1) // quantum) * quantum
        real += sum(buf); pad += len(buf) * w; steps += 1
        widths.add(w); sizes.append(len(buf))
    return {"steps": steps, "real_tokens": real, "padded_tokens": pad,
            "padding_fraction": round(1 - real / pad, 4),
            "distinct_widths": len(widths),
            "median_batch": int(np.median(sizes)), "max_batch": int(max(sizes))}


def main() -> int:
    lens = lengths(N_SEQ)
    res = {
        "n_sequences": len(lens), "quantum": QUANTUM,
        "token_budget": BUDGET, "max_batch": MAX_BATCH,
        "length_min": int(min(lens)), "length_median": int(np.median(lens)),
        "length_max": int(max(lens)),
        "stream_order": pack(lens, sort=False, quantum=0),
        "length_sorted": pack(lens, sort=True, quantum=0),
        "quantised": pack(lens, sort=True, quantum=QUANTUM),
    }
    a, b = res["stream_order"], res["quantised"]
    res["steps_saved_fraction"] = round(1 - b["steps"] / a["steps"], 4)
    res["useful_tokens_per_step_gain"] = round(
        (b["real_tokens"] / b["steps"]) / (a["real_tokens"] / a["steps"]), 4)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "packing_waste.json").write_text(json.dumps(res, indent=1))
    print(f"{len(lens):,} sequences, length {res['length_min']}/"
          f"{res['length_median']}/{res['length_max']}")
    for k in ("stream_order", "length_sorted", "quantised"):
        r = res[k]
        print(f"  {k:14s} steps {r['steps']:6,}  padding {100*r['padding_fraction']:5.1f}%"
              f"  widths {r['distinct_widths']:4d}  batch median {r['median_batch']:3d}")
    print(f"\nquantised vs shipped: {100*res['steps_saved_fraction']:.1f}% fewer steps, "
          f"{res['useful_tokens_per_step_gain']:.2f}x the useful tokens per step")
    print(f"-> {OUT / 'packing_waste.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
