#!/usr/bin/env python3
"""Per-run training telemetry that survives the run being killed.

Every trainer here wrote its history into a Python list and serialised it to
JSON **once, at the end**. On a shared card that never runs: stage 1 reached
188.6M tokens across several fires and not one of them produced a history file,
because not one of them finished. The only record was the text log, which is
fine for reading and useless for plotting, diffing or asking when a metric
turned.

So metrics are appended to a CSV **as they are produced**, flushed on every
row, and a JSON manifest records what produced them. Kill the process at any
point and everything up to the last logged step is on disk.

Three files per stage, under `data/samples/analysis/runs/`:

    <stage>.csv              every row every run ever logged, append-only
    <stage>/<run_id>.json    the manifest: argv, config, git commit, device
    <stage>/<run_id>.csv     that run's rows alone, for convenience

`run_id` is a UTC timestamp fixed when the run starts, and it is a column in
the shared CSV, so rows from different runs never have to be told apart by
guessing -- which matters here, where a stage is resumed many times and the
step counter restarts are meaningful rather than noise.

`kind` distinguishes what a row is: `step` for a logging interval, `epoch` for
an epoch summary, `eval` for a validation pass, `event` for something that
happened (an OOM, a budget change, a resume). One file holds all of them and a
reader filters.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

#: Columns every stage writes, in this order, before its own.
CORE_FIELDS: List[str] = [
    "run_id", "stage", "kind", "timestamp", "elapsed_s", "epoch", "step",
]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _git_dirty(root: Path) -> Optional[bool]:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                             capture_output=True, text=True, timeout=10)
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class RunLog:
    """Append-only CSV plus a per-run manifest. Safe to kill at any moment."""

    def __init__(self, root: Path, stage: str, fields: Iterable[str], *,
                 run_id: Optional[str] = None,
                 manifest: Optional[Dict] = None) -> None:
        self.stage = stage
        self.run_id = run_id or _utc_stamp()
        self.t0 = time.time()
        self.fields = CORE_FIELDS + [f for f in fields if f not in CORE_FIELDS]

        base = Path(root) / "data/samples/analysis/runs"
        (base / stage).mkdir(parents=True, exist_ok=True)
        self.shared = base / f"{stage}.csv"
        self.per_run = base / stage / f"{self.run_id}.csv"
        self._fh: List = []
        self._wr: List = []
        for path in (self.shared, self.per_run):
            self._open(path)

        man = {
            "run_id": self.run_id, "stage": stage,
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "argv": sys.argv,
            "git_commit": _git_commit(Path(root)),
            "git_dirty": _git_dirty(Path(root)),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "pid": os.getpid(),
            "csv": str(self.per_run.relative_to(Path(root))),
            **(manifest or {}),
        }
        self.manifest_path = base / stage / f"{self.run_id}.json"
        self.manifest_path.write_text(json.dumps(man, indent=1, default=str))

    def _open(self, path: Path) -> None:
        """Append if the header matches; rotate the old file if it does not.

        A schema change must not corrupt a file that already holds a run's
        history, and it must not silently drop the new columns either.
        """
        if path.exists() and path.stat().st_size:
            with path.open(newline="") as fh:
                head = next(csv.reader(fh), [])
            if head != self.fields:
                n = 1
                while path.with_suffix(f".v{n}.csv").exists():
                    n += 1
                path.replace(path.with_suffix(f".v{n}.csv"))
        new = not (path.exists() and path.stat().st_size)
        fh = path.open("a", newline="")
        wr = csv.DictWriter(fh, fieldnames=self.fields, extrasaction="ignore",
                            restval="")
        if new:
            wr.writeheader()
            fh.flush()
        self._fh.append(fh)
        self._wr.append(wr)

    def log(self, kind: str, *, epoch=None, step=None, **row) -> None:
        """Write one row to both files and flush, so a kill loses nothing."""
        base = {
            "run_id": self.run_id, "stage": self.stage, "kind": kind,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_s": round(time.time() - self.t0, 1),
            "epoch": "" if epoch is None else epoch,
            "step": "" if step is None else step,
        }
        base.update({k: v for k, v in row.items() if v is not None})
        for fh, wr in zip(self._fh, self._wr):
            wr.writerow(base)
            fh.flush()

    def event(self, what: str, **row) -> None:
        """Something happened: an OOM, a resume, a budget change."""
        self.log("event", note=what, **row)

    def finish(self, **row) -> None:
        self.log("finish", **row)
        self.close()

    def close(self) -> None:
        for fh in self._fh:
            try:
                fh.close()
            except OSError:
                pass
        self._fh, self._wr = [], []
