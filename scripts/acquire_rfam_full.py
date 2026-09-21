#!/usr/bin/env python3
"""Acquire every Rfam 15.1 full alignment (full_alignments/*.sto).

Rfam.tar.gz carries seed alignments only; the per-family full alignments
hold every matching sequence and are the substrate for the coevolution /
MSA track.  The FTP directory is 4,079 Stockholm files, ~4 GB total.

    uv run python scripts/acquire_rfam_full.py --workers 12
"""
from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "families" / "rfam" / "full_alignments"
BASE = "https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1/full_alignments/"
LOG = DEST.parent / "full_alignments.log"

MULT = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def list_files(session: requests.Session) -> list[tuple[str, int]]:
    html = session.get(BASE, timeout=120).text
    rows = re.findall(
        r'<a href="([^"]+\.sto)">[^<]+</a></td><td align="right">[^<]*'
        r'</td><td align="right">\s*([\d.]+)\s*([KMG]?)\s*</td>', html)
    out = []
    for name, size, unit in rows:
        out.append((name, int(float(size) * MULT[unit])))
    return out


def download(session: requests.Session, name: str, expected: int,
             max_attempts: int = 200) -> str:
    dest = DEST / name
    part = dest.with_name(dest.name + ".part")
    if dest.exists() and (not expected or dest.stat().st_size == expected):
        return "cached"
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(BASE + name, headers=headers, stream=True,
                             timeout=(30, 120)) as r:
                if have and r.status_code == 200:
                    have = 0
                    part.unlink(missing_ok=True)
                elif have and r.status_code == 416:
                    part.rename(dest)
                    return "ok"
                r.raise_for_status()
                with open(part, "ab" if have else "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            fh.write(chunk)
        except Exception:                                 # noqa: BLE001
            size = part.stat().st_size if part.exists() else 0
            if size > have:
                attempt = 0
            time.sleep(min(30, 3 * attempt))
            continue
        size = part.stat().st_size if part.exists() else 0
        if not expected or size == expected:
            part.rename(dest)
            return "ok"
    return "failed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "sbs-rna-acquisition/0.1 (academic)"
    files = list_files(session)
    if args.limit:
        files = files[:args.limit]
    log(f"{len(files)} full alignments, "
        f"{sum(s for _, s in files)/1e9:.2f} GB")
    counts = {"ok": 0, "cached": 0, "failed": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download, session, n, s): n for n, s in files}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                counts[fut.result()] += 1
            except Exception:                             # noqa: BLE001
                counts["failed"] += 1
                log(f"FAIL {futs[fut]}")
            if i % 250 == 0:
                log(f"  {i}/{len(files)} {counts} "
                    f"{(time.time()-t0)/60:.1f} min")
    log(f"done: {counts} in {(time.time()-t0)/60:.1f} min")
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
