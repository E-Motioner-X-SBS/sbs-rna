#!/usr/bin/env python3
"""Verify every downloaded artifact: completeness, sizes, archive integrity.

    uv run python scripts/verify_downloads.py [--workers 12] [--group sequence]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire_all import DATA, HUNTER, build_jobs, complete  # noqa: E402

REPORT = DATA / "acquisition" / "verification_report.json"

ARCHIVE_EXT = (".tar.gz", ".tgz", ".tar.xz", ".zip", ".parquet", ".gz", ".xz")


def check_archive(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    if name.endswith((".tar.gz", ".tgz")):
        r = subprocess.run(["tar", "-tzf", str(path)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return "OK", ""
        # some mirrors serve an uncompressed tar under a .tar.gz name
        if "not in gzip format" in (r.stderr or ""):
            r2 = subprocess.run(["tar", "-tf", str(path)],
                                capture_output=True, text=True)
            if r2.returncode == 0:
                return "OK", "uncompressed tar (.tar.gz name)"
        return "FAIL", (r.stderr or r.stdout)[-200:]
    elif name.endswith(".tar.xz"):
        cmd = ["xz", "-t", str(path)]
    elif name.endswith(".xz"):
        cmd = ["xz", "-t", str(path)]
    elif name.endswith(".zip"):
        cmd = ["unzip", "-tq", str(path)]
    elif name.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
            pq.ParquetFile(str(path)).metadata
            return "OK", ""
        except Exception as e:                              # noqa: BLE001
            return "FAIL", str(e)[:200]
    elif name.endswith(".gz"):
        cmd = ["gzip", "-t", str(path)]
    else:
        return "SKIP", ""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        return "OK", ""
    return "FAIL", (r.stderr or r.stdout)[-200:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--group", default=None)
    ap.add_argument("--skip-eldors", action="store_true",
                    help="skip elDORS chunks (already sha256-verified)")
    args = ap.parse_args()

    jobs = build_jobs()
    if args.group:
        jobs = [j for j in jobs if j.group == args.group]

    report: dict = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "jobs_total": len(jobs), "jobs_incomplete": [],
                    "stray_parts": [], "archives": {}}

    for j in jobs:
        if not complete(j):
            report["jobs_incomplete"].append(j.name)
    print(f"jobs incomplete: {report['jobs_incomplete'] or 'none'}")

    for root in (DATA, HUNTER / "RNA_Database", HUNTER / ".harvest_bundles"):
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and (p.name.endswith(".part")
                                    or p.name.endswith(".partial")):
                    report["stray_parts"].append(str(p))
    print(f"stray .part/.partial files: {len(report['stray_parts'])}")

    targets: list[Path] = []
    for p in sorted(DATA.rglob("*")):
        if not p.is_file() or not p.name.lower().endswith(ARCHIVE_EXT):
            continue
        if args.skip_eldors and "elDORS_v1" in str(p):
            continue
        targets.append(p)
    print(f"archives to verify: {len(targets)} ({args.workers} workers)")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(check_archive, p): p for p in targets}
        for n, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            status, err = fut.result()
            report["archives"][str(p)] = {"status": status, "error": err}
            if status != "OK":
                print(f"  {status} {p} {err}")
            if n % 20 == 0:
                print(f"  {n}/{len(targets)} checked "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
    bad = [k for k, v in report["archives"].items() if v["status"] == "FAIL"]
    report["archives_total"] = len(targets)
    report["archives_failed"] = bad
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"\nfailed archives: {len(bad)}")
    for b in bad:
        print("  " + b)
    ok = (not report["jobs_incomplete"] and not report["stray_parts"]
          and not bad)
    print("VERDICT:", "ALL VERIFIED" if ok else "ISSUES FOUND")
    print(f"report: {REPORT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
