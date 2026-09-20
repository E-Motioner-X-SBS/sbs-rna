#!/usr/bin/env python3
"""Acquire the RAW PDB entries behind our 3D corpus — the ones that still carry ions.

Why this is needed
------------------
Both structural corpora on disk are derivatives and both are stripped:

  RNA3DB   per-chain mmCIF extracts, **zero HETATM**, B-factors written as 0.0
  RNASolo  per-chain PDB extracts,   **zero HETATM**, real B-factors retained

Four things in ARCHITECTURE.md therefore have no data to learn from or be
checked against:

  * Fact 3, the 1.76 sigma Mg-rigidity gradient          (needs Mg coordinates)
  * the ion inventory (Mg:K 9:1, 83% inner-sphere to OP1/OP2)
  * the Mg-site supervision head
  * the ionic-condition input, the design's headline novelty

and a fifth is measured on the wrong distribution: OQ-7's `target_c` breach is
driven by modified-residue density, which the derivatives carry at **0.025%**
against **1.05%** in raw PDB — a 42x under-representation of exactly the
quantity that pushes a chain over budget.

Raw entries fix all five at once. They also restore whole-entry context, which
is what G2 (98.95% of RNA residues are in complex) and G3 (93% of residues come
from ribosomes) are actually statements about — neither is measurable on
per-chain extracts.

Scope
-----
7,943 unique PDB entries, derived from the RNA3DB and RNASolo filenames
(`data/structures/raw_pdb_entrylist.txt`). Measured mean ~2.9 MB compressed,
so roughly 23 GB. Resumable: an entry already present and gzip-valid is skipped,
so re-running costs one stat per entry.

Usage
-----
    PY=/store/shuvam/.venv/bin/python
    $PY scripts/acquire_raw_pdb_entries.py --workers 8
    $PY scripts/acquire_raw_pdb_entries.py --limit 200      # trial slice
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
ENTRYLIST = ROOT / "data/structures/raw_pdb_entrylist.txt"
DEST = ROOT / "data/structures/raw_pdb_entries"
MANIFEST = ROOT / "data/structures/raw_pdb_manifest.json"
URL = "https://files.rcsb.org/download/{pdb}.cif.gz"
UA = {"User-Agent": "sbs-rna-architecture-research/0.1 (academic; contact via repo)"}

#: Minimum plausible gzip body. Anything smaller is an error page, not a structure.
MIN_BYTES = 400


def valid(path: Path) -> bool:
    """Present, non-trivial, decompressible, and containing coordinates.

    The size check alone is not enough: RCSB serves HTML error bodies with a
    200 for some obsoleted IDs, and those gzip-fail rather than being short.

    The coordinate check must stream rather than sample a fixed prefix. mmCIF
    puts metadata first and `_atom_site` can sit deep into the file -- 1C9S has
    it at byte 225,439 of 1.6 MB, so a 200 KB probe rejected three perfectly
    good entries as "invalid body" on the trial slice.
    """
    if not path.exists() or path.stat().st_size < MIN_BYTES:
        return False
    try:
        with gzip.open(path, "rb") as fh:
            tail = b""
            while True:
                buf = fh.read(1 << 20)
                if not buf:
                    return False
                if b"_atom_site" in tail + buf:
                    return True
                tail = buf[-16:]            # carry, in case the tag straddles
    except OSError:
        return False


def fetch(pdb: str, retries: int = 3) -> Dict:
    dest = DEST / f"{pdb.lower()}.cif.gz"
    if valid(dest):
        return {"pdb": pdb, "status": "cached", "bytes": dest.stat().st_size}
    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL.format(pdb=pdb.upper()), headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(body)
            if valid(tmp):
                tmp.rename(dest)
                return {"pdb": pdb, "status": "ok", "bytes": len(body)}
            tmp.unlink(missing_ok=True)
            last = f"invalid body ({len(body)} B)"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (404, 410):            # obsoleted / superseded: do not retry
                break
        except Exception as e:                                   # noqa: BLE001
            last = type(e).__name__
        time.sleep(1.5 * (attempt + 1))
    return {"pdb": pdb, "status": "failed", "error": last}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    entries: List[str] = [x.strip() for x in ENTRYLIST.read_text().split() if x.strip()]
    if args.limit:
        entries = entries[:args.limit]
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"[raw] {len(entries):,} entries -> {DEST}  ({args.workers} workers)")

    t0 = time.time()
    rows: List[Dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch, e): e for e in entries}
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 250 == 0:
                ok = sum(r["status"] in ("ok", "cached") for r in rows)
                gb = sum(r.get("bytes", 0) for r in rows) / 1e9
                rate = n / max(time.time() - t0, 1e-9)
                print(f"[raw]   {n}/{len(entries)}  ok={ok}  {gb:.1f} GB  "
                      f"{rate:.1f}/s  eta {(len(entries)-n)/max(rate,1e-9)/60:.0f} min",
                      flush=True)

    ok = [r for r in rows if r["status"] in ("ok", "cached")]
    failed = [r for r in rows if r["status"] == "failed"]
    manifest = {
        "n_requested": len(entries),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "total_bytes": sum(r.get("bytes", 0) for r in ok),
        "wall_seconds": round(time.time() - t0, 1),
        "source": URL,
        "failed": failed[:200],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"[raw] done: {len(ok):,} ok, {len(failed)} failed, "
          f"{manifest['total_bytes']/1e9:.1f} GB in "
          f"{manifest['wall_seconds']/60:.0f} min -> {MANIFEST}")
    if failed:
        print("[raw] first failures:", [(r['pdb'], r['error']) for r in failed[:8]])


if __name__ == "__main__":
    main()
