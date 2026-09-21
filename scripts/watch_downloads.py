#!/usr/bin/env python3
"""Watchdog: keep every acquisition group running until all jobs are done.

    setsid nohup uv run python scripts/watch_downloads.py > data/acquisition/logs/watchdog.log 2>&1 &

Every cycle it
  1. checks each group's process is alive,
  2. restarts it when jobs are pending and it is not,
  3. stops (and writes state/ALL_DONE) when every job is complete.

Group workers can be tuned with --workers-sequence etc.  Completed
directory jobs (harvest, raw PDB) are recorded with markers by
acquire_all.py, so a restart only resumes what is unfinished.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire_all import LOGS, ROOT, STATE, UV, build_jobs, complete  # noqa: E402

GROUPS = ["sequence", "catalog", "long"]


def running(group: str) -> bool:
    r = subprocess.run(["pgrep", "-f", f"[a]cquire_all.py --group {group}"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def fmt_done(jobs) -> str:
    done = sum(complete(j) for j in jobs)
    return f"{done}/{len(jobs)}"


def spawn(group: str, workers: int) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    out = open(LOGS / f"{group}.out", "ab")
    subprocess.Popen(
        [UV, "run", "python", "scripts/acquire_all.py",
         "--group", group, "--workers", str(workers)],
        cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"[watchdog {time.strftime('%H:%M:%S')}] spawned {group} "
          f"({workers} workers)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--workers-sequence", type=int, default=10)
    ap.add_argument("--workers-catalog", type=int, default=12)
    ap.add_argument("--workers-long", type=int, default=2)
    ap.add_argument("--max-cycles", type=int, default=0,
                    help="0 = run until all done")
    args = ap.parse_args()
    workers = {"sequence": args.workers_sequence,
               "catalog": args.workers_catalog,
               "long": args.workers_long}

    jobs = build_jobs()
    cyc = 0
    while True:
        cyc += 1
        pending = [j for j in jobs if not complete(j)]
        line = "  ".join(
            f"{g}:{fmt_done([j for j in jobs if j.group == g])}"
            for g in GROUPS)
        print(f"[watchdog {time.strftime('%H:%M:%S')}] cycle {cyc}  {line}",
              flush=True)
        if not pending:
            STATE.mkdir(parents=True, exist_ok=True)
            (STATE / "ALL_DONE").write_text(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"[watchdog] ALL DOWNLOADS COMPLETE ({len(jobs)} jobs)",
                  flush=True)
            return 0
        for g in GROUPS:
            gp = [j for j in pending if j.group == g]
            if gp and not running(g):
                spawn(g, workers[g])
        if args.max_cycles and cyc >= args.max_cycles:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
