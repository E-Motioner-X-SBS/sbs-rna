#!/usr/bin/env python3
"""Turn a MARS archive into RNA pretraining shards, filtering as it goes.

MARS is 1.73B sequences in 30 gzipped tar archives, 427 GB. Decision D18
excluded it because it yields **0.17% diverse structured ncRNA** for that size.
Measured here on 40,000 headers of part 01, which is the same story in more
detail:

    any RNA mention   43.80%        explicit DNA      12.30%
    mRNA              28.39%        gene/genomic      47.61%
    rRNA               8.65%        snRNA/snoRNA       0.35%
    tRNA               3.76%        ncRNA/lncRNA       0.01%
                                    miRNA              0.00%

So MARS is a general nucleotide archive. Loaded raw into an RNA model it would
shift the pretraining distribution toward mRNA and genomic DNA -- and the
vocabulary maps T to U at read time, so a panda satellite repeat arrives as
RNA-alphabet nonsense rather than as an obvious mistake. It is also 427 GB
against 599 GB of free disk, so the archives cannot simply be kept.

This script therefore does three things per archive:

1. **streams** the FASTA out of the `.tgz` without extracting it to disk;
2. **filters** to records whose annotation says RNA and does not say genomic
   DNA, and to the length window the trainer uses anyway;
3. **writes** parquet shards and, with `--delete-archive`, removes the `.tgz`.

Shards land in their own directory, not beside elDORS, so the two corpora stay
separable and MARS can be turned off by not passing its directory. Every
decision is counted and the counts are written beside the shards, because a
filter that discards 60% of an input should have to say so.

Usage:
    python3 scripts/mars_to_parquet.py --archive data/sequences/mars/OMIX003037-01.tgz
    python3 scripts/mars_to_parquet.py --all --delete-archive
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MARS_DIR = ROOT / "data/sequences/mars"
OUT_DIR = ROOT / "data/derived/parquet_mars"

#: Keep a record only if its annotation says RNA...
RNA_RE = re.compile(
    r"\b(?:rRNA|tRNA|mRNA|ncRNA|lncRNA|miRNA|microRNA|snRNA|snoRNA|siRNA|"
    r"piRNA|scRNA|tmRNA|SRP RNA|RNase [PM]RP|riboswitch|ribozyme|"
    r"ribosomal RNA|transfer RNA|messenger RNA|non-?coding RNA|"
    r"untranslated region|UTR|transcript)\b", re.I)
#: ...and does not say it is the genomic copy. "gene" alone is not
#: disqualifying -- "tRNA-Leu gene" is a tRNA -- but an explicit genomic,
#: satellite or chromosomal annotation is.
DNA_RE = re.compile(
    r"\b(?:genomic DNA|satellite|chromosome|complete genome|contig|scaffold|"
    r"whole genome shotgun|plasmid|mitochondrion, complete|clone)\b", re.I)

#: The structured non-coding RNA PHAROS is actually about. Measured on the
#: annotation-filtered set: mRNA/UTR is 68.2% of it, rRNA 22.1%, tRNA 9.2%,
#: everything else under 1%. So "RNA" and "the RNA this model folds" are very
#: different filters, and which one is wanted is a decision, not a detail.
STRUCTURED_RE = re.compile(
    r"\b(?:rRNA|tRNA|ncRNA|lncRNA|miRNA|microRNA|snRNA|snoRNA|siRNA|piRNA|"
    r"scRNA|tmRNA|SRP RNA|RNase [PM]RP|riboswitch|ribozyme|"
    r"ribosomal RNA|transfer RNA|non-?coding RNA)\b", re.I)

MIN_LEN, MAX_LEN = 20, 1024
SHARD_ROWS = 250_000


def iter_fasta(fh) -> Iterator[Tuple[str, str]]:
    """(header, sequence) from a FASTA stream, without loading it all."""
    hdr, buf = None, []
    for raw in fh:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        if line.startswith(">"):
            if hdr is not None:
                yield hdr, "".join(buf)
            hdr, buf = line[1:].strip(), []
        elif hdr is not None:
            buf.append(line.strip())
    if hdr is not None:
        yield hdr, "".join(buf)


def keep(header: str, seq: str, mode: str = "structured") -> Tuple[bool, str]:
    """Should this record train an RNA model? Returns (keep, reason if not)."""
    n = len(seq)
    if n < MIN_LEN or n > MAX_LEN:
        return False, "length"
    if DNA_RE.search(header):
        return False, "genomic"
    if mode == "structured":
        if not STRUCTURED_RE.search(header):
            return False, "not structured ncRNA"
    elif not RNA_RE.search(header):
        return False, "not annotated RNA"
    return True, ""


def corpus_hashes(dirs: List[Path]) -> set:
    """Sequence hashes already in the pretraining corpus.

    Deduplicating MARS against itself is not the interesting question. D18
    rejected MARS on **diverse** yield -- 0.17% -- and the way to measure that
    is against the corpus the model already trains on, not against MARS alone.
    """
    seen: set = set()
    for d in dirs:
        for f in sorted(Path(d).glob("*.parquet")):
            for b in pq.ParquetFile(f).iter_batches(batch_size=65536,
                                                    columns=["sequence"]):
                for s in b.column("sequence").to_pylist():
                    if s:
                        seen.add(hashlib.blake2b(
                            s.upper().replace("T", "U").encode(),
                            digest_size=8).digest())
    return seen


def convert(archive: Path, out_dir: Path, *, delete: bool = False,
            limit: int | None = None, mode: str = "structured",
            existing: set | None = None) -> Dict:
    part = archive.name.split("-")[-1].split(".")[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {"read": 0, "kept": 0, "length": 0, "genomic": 0,
                              "not annotated RNA": 0, "not structured ncRNA": 0,
                              "duplicate": 0, "already in corpus": 0}
    seen: set = set(existing or ())
    n_existing = len(seen)
    ids: List[str] = []
    seqs: List[str] = []
    shard = 0

    def flush() -> None:
        nonlocal shard, ids, seqs
        if not seqs:
            return
        path = out_dir / f"mars_p{part}_shard{shard:04d}.parquet"
        pq.write_table(pa.table({"id": ids, "sequence": seqs}), path,
                       compression="zstd")
        print(f"  wrote {path.name}: {len(seqs):,} rows "
              f"({path.stat().st_size/1e6:.1f} MB)", flush=True)
        shard += 1
        ids, seqs = [], []

    truncated = False
    try:
      with tarfile.open(archive, "r|gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            for hdr, seq in iter_fasta(fh):
                counts["read"] += 1
                ok, why = keep(hdr, seq, mode)
                if not ok:
                    counts[why] += 1
                    continue
                h = hashlib.blake2b(
                    seq.upper().replace("T", "U").encode(), digest_size=8).digest()
                if h in seen:
                    counts["already in corpus" if len(seen) == n_existing
                           else "duplicate"] += 1
                    continue
                seen.add(h)
                counts["kept"] += 1
                ids.append(hdr.split()[0])
                seqs.append(seq.upper())
                if len(seqs) >= SHARD_ROWS:
                    flush()
                if limit and counts["read"] >= limit:
                    break
            if limit and counts["read"] >= limit:
                break
    except (tarfile.ReadError, EOFError, gzip.BadGzipFile) as e:
        # A `.part` file being probed mid-download, or a download that was cut
        # short. Everything read before the break is still valid, so the count
        # is kept and the caller is told the archive was incomplete rather than
        # given a silent undercount.
        truncated = True
        print(f"  archive ends early ({type(e).__name__}); "
              f"counting what was readable", flush=True)
    flush()

    counts["archive"] = archive.name
    counts["truncated"] = truncated
    counts["mode"] = mode
    counts["corpus_hashes_loaded"] = n_existing
    counts["kept_fraction"] = round(counts["kept"] / max(counts["read"], 1), 4)
    counts["shards"] = shard
    (out_dir / f"mars_p{part}_counts.json").write_text(json.dumps(counts, indent=1))
    print(f"  {archive.name}: read {counts['read']:,}, kept {counts['kept']:,} "
          f"({100*counts['kept_fraction']:.1f}%), {shard} shards", flush=True)
    if delete and shard and not truncated:
        archive.unlink()
        print(f"  removed {archive.name} (converted)", flush=True)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, default=None)
    ap.add_argument("--all", action="store_true",
                    help="every completed .tgz in data/sequences/mars")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--delete-archive", action="store_true",
                    help="remove the .tgz once its shards are written; 427 GB "
                         "of archives does not fit beside their own output")
    ap.add_argument("--mode", choices=("structured", "rna"), default="structured",
                    help="structured: rRNA/tRNA/sn(o)/nc/lnc/mi/ribozyme only, "
                         "which is what this model folds. rna: anything "
                         "RNA-annotated, which measured 68.2%% mRNA")
    ap.add_argument("--dedup-against", type=Path, nargs="*", default=None,
                    help="parquet dirs whose sequences already train the model; "
                         "MARS records matching them add nothing")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many records, for measurement")
    args = ap.parse_args()

    if args.all:
        targets = sorted(MARS_DIR.glob("*.tgz"))
    elif args.archive:
        targets = [args.archive]
    else:
        print("pass --archive or --all")
        return 1
    if not targets:
        print("no completed archives yet (.part files are still downloading)")
        return 0
    existing = None
    if args.dedup_against:
        print(f"loading corpus hashes from {len(args.dedup_against)} dir(s)...",
              flush=True)
        existing = corpus_hashes(args.dedup_against)
        print(f"  {len(existing):,} sequences already in the corpus", flush=True)
    for t in targets:
        print(f"converting {t.name} ({t.stat().st_size/1e9:.1f} GB)", flush=True)
        convert(t, args.out, delete=args.delete_archive, limit=args.limit,
                mode=args.mode, existing=existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
