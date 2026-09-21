#!/usr/bin/env python3
"""Acquire the complete RNA Mapping DataBase (RMDB).

RMDB is the archive of quantitative RNA structural-mapping experiments
(SHAPE, DMS, CMCT, 1M7, mutate-and-map, and — critically for PHAROS —
Mg2+ titration ladders: `*_MGTI_*`, `*_MGPH_*`, `*_MG50_*`).

The site's own detail pages link each construct to a GitHub release
asset under `DasLab/rmdb.github.io`; there is no single bulk tarball.
This script enumerates all five data releases, then downloads every
`.rdat` with byte-range resume.

Layout
------
    data/benchmarks/probing/rmdb/
        rmdb_manifest.json     site manifest (1,024 constructs)
        rmdb_assets.json       index of downloaded assets
        rdat/<RMDB_ID>.rdat    the measurements

Usage
-----
    uv run python scripts/acquire_rmdb.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "benchmarks" / "probing" / "rmdb"
RDAT = DEST / "rdat"
MANIFEST_URL = "https://rmdb.stanford.edu/manifest.json"
RELEASES_API = ("https://api.github.com/repos/DasLab/rmdb.github.io/"
                "releases?per_page=100")
ASSETS_API = ("https://api.github.com/repos/DasLab/rmdb.github.io/"
              "releases/{rid}/assets?per_page=100&page={page}")
LOG = DEST / "download.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    DEST.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def fetch_assets(session: requests.Session) -> list[dict]:
    rels = session.get(RELEASES_API, timeout=60).json()
    assets: list[dict] = []
    for rel in rels:
        rid = rel["id"]
        page = 1
        while True:
            r = session.get(ASSETS_API.format(rid=rid, page=page), timeout=60)
            r.raise_for_status()
            batch = r.json()
            assets.extend(a for a in batch if a["name"].endswith(".rdat"))
            if len(batch) < 100:
                break
            page += 1
    return assets


def download(session: requests.Session, url: str, dest: Path,
             expected: int, max_attempts: int = 300) -> tuple[str, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        have = part.stat().st_size if part.exists() else 0
        if expected and have >= expected:
            part.rename(dest)
            return "cached" if have == expected else "ok", ""
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(url, headers=headers, stream=True,
                             timeout=(30, 120)) as r:
                if have and r.status_code == 200:
                    have = 0
                    part.unlink(missing_ok=True)
                elif have and r.status_code == 416:
                    part.rename(dest)
                    return "ok", ""
                r.raise_for_status()
                with open(part, "ab" if have else "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            fh.write(chunk)
        except Exception as exc:                          # noqa: BLE001
            size = part.stat().st_size if part.exists() else 0
            if size > have:
                attempt = 0
            time.sleep(min(30, 3 * attempt))
            if attempt >= max_attempts:
                return "failed", f"{type(exc).__name__}: {str(exc)[:120]}"
            continue
        size = part.stat().st_size if part.exists() else 0
        if expected and size == expected:
            part.rename(dest)
            return "ok", ""
        if not expected and size > 0:
            part.rename(dest)
            return "ok", ""
    return "failed", "attempts exhausted"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    RDAT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "sbs-rna-acquisition/0.1 (academic)"

    mf = DEST / "rmdb_manifest.json"
    if not mf.exists():
        r = session.get(MANIFEST_URL, timeout=60)
        r.raise_for_status()
        mf.write_bytes(r.content)
        log(f"manifest: {len(r.json())} constructs")
    else:
        log(f"manifest cached ({len(json.loads(mf.read_text()))} constructs)")

    assets = fetch_assets(session)
    if args.limit:
        assets = assets[:args.limit]
    log(f"release assets: {len(assets)} .rdat files, "
        f"{sum(a['size'] for a in assets)/1e9:.2f} GB")

    done = {"ok": 0, "cached": 0, "failed": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for a in assets:
            dest = RDAT / a["name"]
            if dest.exists() and dest.stat().st_size == a["size"]:
                done["cached"] += 1
                continue
            futs[pool.submit(download, session, a["browser_download_url"],
                             dest, a["size"])] = a
        for n, fut in enumerate(as_completed(futs), 1):
            a = futs[fut]
            status, err = fut.result()
            done[status] = done.get(status, 0) + 1
            if status == "failed":
                log(f"FAIL {a['name']}: {err}")
            if n % 100 == 0:
                log(f"  {n}/{len(futs)} ({done}) "
                    f"{(time.time()-t0)/60:.1f} min")

    index = [{"name": a["name"], "size": a["size"],
              "url": a["browser_download_url"]} for a in assets]
    (DEST / "rmdb_assets.json").write_text(json.dumps(index, indent=1))
    log(f"done: {done}; {(time.time()-t0)/60:.1f} min")
    return 0 if done.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
