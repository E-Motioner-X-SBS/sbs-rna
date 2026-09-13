#!/usr/bin/env python3
"""elDORS FASTA -> Parquet ETL for training-ready shards.

Converts multi-GB gzipped FASTA chunks into compressed Parquet shards with
columns (id: string, sequence: string). Optionally maps T->U for RNA models.

The converter is streaming (constant memory) and shard-aware: each ~N
sequences is flushed as one parquet file, which makes the corpus directly
loadable by HF datasets / PyTorch DataLoader without decompression storms.

Usage:
  # demo: first 1M sequences of chunk 001 into 4 shards of 250k
  python3 eldors_to_parquet.py --chunk 001 --limit 1000000 --shard-size 250000 \
      --out data/parquet/eldors_v1 --t2u

  # full chunk
  python3 eldors_to_parquet.py --chunk 001 --out data/parquet/eldors_v1 --t2u

  # all chunks (run in parallel externally, 8 chunks at a time)
  for i in $(seq -w 1 20); do python3 eldors_to_parquet.py --chunk 00$i ...; done
"""

from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Data root resolution order:
#   1. $SBS_RNA_DATA
#   2. <repo>/data                     (this checkout)
#   3. /store/shuvam/E-motioner-X-SBS/sbs-rna/data   (the original server path)
# The server path is a fallback, not a hard-coded assumption -- on any other
# machine it does not exist.
def _resolve_data_root() -> Path:
    import os
    env = os.environ.get("SBS_RNA_DATA")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "data"
        if (cand / "catalog").is_dir():
            return cand
    return Path("/store/shuvam/E-motioner-X-SBS/sbs-rna/data")

DATA = _resolve_data_root()
ELDORS = DATA / "sequences" / "elDORS_v1"


def stream_fasta(path: Path, limit: int | None):
    """Yield (id, sequence) pairs from gzipped FASTA."""
    n = 0
    header = None
    chunks: list[str] = []
    with gzip.open(path, "rt", errors="ignore") as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                    n += 1
                    if limit and n >= limit:
                        return
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True, help="e.g. 001")
    ap.add_argument("--out", required=True, help="output dir for parquet shards")
    ap.add_argument("--shard-size", type=int, default=1_000_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--t2u", action="store_true", help="map T->U (RNA alphabet)")
    ap.add_argument("--compression", default="zstd")
    args = ap.parse_args()

    src = ELDORS / f"elDORS_v1_{args.chunk}.fasta.gz"
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"c{args.chunk}"
    t0 = time.time()

    ids: list[str] = []
    seqs: list[str] = []
    shard = 0
    total = 0

    def flush() -> None:
        nonlocal shard, ids, seqs
        if not ids:
            return
        table = pa.table({"id": pa.array(ids), "sequence": pa.array(seqs)})
        dest = outdir / f"eldors_{tag}_shard{shard:04d}.parquet"
        pq.write_table(table, dest, compression=args.compression)
        print(
            f"  wrote {dest.name}: {len(ids):,} rows "
            f"({dest.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )
        ids, seqs = [], []
        shard += 1

    for rid, seq in stream_fasta(src, args.limit):
        if args.t2u:
            seq = seq.replace("T", "U").replace("t", "u")
        ids.append(rid)
        seqs.append(seq)
        total += 1
        if len(ids) >= args.shard_size:
            flush()
    flush()

    dt = time.time() - t0
    print(
        f"chunk {args.chunk}: {total:,} sequences -> {shard} shards in {dt / 60:.1f} min "
        f"({total / max(dt, 1):,.0f} seq/s)"
    )


if __name__ == "__main__":
    main()
