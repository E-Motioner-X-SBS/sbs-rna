#!/usr/bin/env python3
"""Pretraining corpus utilities: split convention + dataset sampling/stats.

1. split:  defines the reproducible train/val split convention for elDORS
           (hash of sequence header: md5(header) % 100 < 80 -> train).
2. sample: extracts N sequences per chunk into samples/ for dev/testing.
3. stats:  nucleotide composition + length distribution on a sample.

Usage:
  python3 corpus_tools.py split            # write splits/pretrain_split.json
  python3 corpus_tools.py sample 10000     # 10k seqs from chunk 001
  python3 corpus_tools.py stats            # stats on the sample
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/store/shuvam/E-motioner-X-SBS/sbs-rna/data")
ELDORS = ROOT / "sequences" / "elDORS_v1"
OUT = ROOT / "catalog"


def split() -> None:
    """Write the split convention document (no data movement)."""
    doc = {
        "corpus": "elDORS_v1",
        "total_sequences": 1320000000,
        "total_chunks": 20,
        "split_rule": {
            "description": "Per-sequence deterministic split by MD5 of the sequence header",
            "train": "md5(header).int % 100 < 80  (80%)",
            "validation": "md5(header).int % 100 in [80, 90)  (10%)",
            "test": "md5(header).int % 100 in [90, 100)  (10%)",
            "rationale": "Homology-independent, reproducible, no data movement; "
            "mirrors NucleicBERT's 80/20 with an explicit held-out test tier.",
        },
        "note": "Headers are the FASTA description lines; implement by streaming "
        "with Bio.SeqIO or `zcat | awk '/^>/{...}'`.",
    }
    (OUT / "splits").mkdir(parents=True, exist_ok=True)
    with open(OUT / "splits" / "pretrain_split.json", "w") as fh:
        json.dump(doc, fh, indent=2)
    print("wrote", OUT / "splits" / "pretrain_split.json")


def sample(n: int, chunk: str = "001") -> None:
    src = ELDORS / f"elDORS_v1_{chunk}.fasta.gz"
    dest_dir = OUT / "samples"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"eldors_{chunk}_first{n}.fasta"
    written = 0
    with gzip.open(src, "rt", errors="ignore") as fin, open(dest, "w") as fout:
        keep = False
        for line in fin:
            if line.startswith(">"):
                if written >= n:
                    break
                written += 1
                keep = True
            if keep:
                fout.write(line)
    print(f"wrote {dest} ({written} sequences)")


def stats() -> None:
    samp = sorted((OUT / "samples").glob("eldors_*_first*.fasta"))
    if not samp:
        print("no sample found, run `sample` first")
        return
    path = samp[0]
    nts: Counter = Counter()
    lens: list[int] = []
    n = 0
    cur = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if n > 0:
                    lens.append(cur)
                n += 1
                cur = 0
            else:
                seq = line.strip().upper()
                nts.update(seq)
                cur += len(seq)
        if n > 0:
            lens.append(cur)
    total = sum(nts.values())
    report = {
        "file": str(path),
        "sequences": n,
        "nucleotides": total,
        "composition": {k: round(v / total, 4) for k, v in nts.most_common(15)},
        "length": {
            "min": min(lens),
            "max": max(lens),
            "mean": round(sum(lens) / len(lens), 1),
            "median": sorted(lens)[len(lens) // 2],
        },
    }
    with open(OUT / "samples" / "sample_stats.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "split"
    if cmd == "split":
        split()
    elif cmd == "sample":
        sample(int(sys.argv[2]) if len(sys.argv) > 2 else 10000)
    elif cmd == "stats":
        stats()
    else:
        print(__doc__)
