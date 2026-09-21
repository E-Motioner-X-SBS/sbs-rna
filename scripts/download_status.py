#!/usr/bin/env python3
"""Progress report for the acquisition run in scripts/acquire_all.py.

    uv run python scripts/download_status.py
"""
from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire_all import DATA, Job, build_jobs, complete  # noqa: E402


def bytes_on_disk(job: Job) -> int:
    if job.kind == "cmd":
        if job.dest and job.dest.exists() and job.dest.is_dir():
            total = 0
            for p in job.dest.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
            return total
        return 0
    if job.dest is None:
        return 0
    total = 0
    part = job.dest.with_name(job.dest.name + ".part")
    if part.exists():
        total += part.stat().st_size
    if job.dest.exists() and job.dest.is_file():
        total += job.dest.stat().st_size
    return total


def main() -> int:
    jobs = build_jobs()
    running = subprocess.run(["pgrep", "-af", "[a]cquire_all.py"],
                             capture_output=True, text=True).stdout.splitlines()
    groups = sorted({j.group for j in jobs})
    for g in groups:
        gj = [j for j in jobs if j.group == g]
        done = sum(complete(j) for j in gj)
        have = sum(bytes_on_disk(j) for j in gj)
        want = sum(j.expected or 0 for j in gj)
        print(f"\n== {g}: {done}/{len(gj)} complete, {have/1e9:.1f} GB on disk"
              + (f" of ~{want/1e9:.1f} GB expected" if want else ""))
        for j in gj:
            got = bytes_on_disk(j)
            mark = "DONE" if complete(j) else ("PART" if got else "    ")
            pct = f"{100*got/j.expected:5.1f}%" if j.expected else "     "
            print(f"  {mark} {j.name:<28} {got/1e9:8.3f} GB  {pct}")
    print("\n== processes ==")
    for line in running:
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
