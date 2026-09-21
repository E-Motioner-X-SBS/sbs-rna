#!/usr/bin/env python3
"""The telemetry must survive the thing it exists for: the process dying.

Run: python3 src/pharos/train/test_telemetry.py
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pharos.train.telemetry import RunLog                        # noqa: E402

fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} {detail}")
    if not ok:
        fails.append(name)


def rows(p: Path):
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    root = Path(tempfile.mkdtemp())
    try:
        F = ["tokens", "loss", "lr"]
        log = RunLog(root, "unit", F, run_id="RUN1")
        log.log("step", epoch=0, step=10, tokens=1000, loss=2.5, lr=1e-4)
        log.log("step", epoch=0, step=20, tokens=2000, loss=2.3, lr=2e-4)

        base = root / "data/samples/analysis/runs"
        shared, per = base / "unit.csv", base / "unit/RUN1.csv"

        print("== every row is on disk before the run ends ==")
        chk("shared csv has both rows without close()", len(rows(shared)) == 2,
            f"{len(rows(shared))} rows")
        chk("per-run csv has both rows", len(rows(per)) == 2)
        chk("manifest written at construction",
            (base / "unit/RUN1.json").exists())

        print("\n== a killed run leaves valid CSV, and the next one appends ==")
        del log                                   # no close(): simulate a kill
        log2 = RunLog(root, "unit", F, run_id="RUN2")
        log2.log("step", epoch=0, step=5, tokens=500, loss=2.9, lr=1e-4)
        r = rows(shared)
        chk("shared csv now holds both runs", len(r) == 3, f"{len(r)} rows")
        chk("rows carry their own run_id",
            [x["run_id"] for x in r] == ["RUN1", "RUN1", "RUN2"],
            str([x["run_id"] for x in r]))
        chk("a fresh per-run file per run", len(rows(base / "unit/RUN2.csv")) == 1)

        print("\n== kinds are distinguishable in one file ==")
        log2.event("OOM", tokens=500)
        log2.log("epoch", epoch=0, step=5, loss=2.7)
        kinds = [x["kind"] for x in rows(base / "unit/RUN2.csv")]
        chk("step / event / epoch all present",
            kinds == ["step", "event", "epoch"], str(kinds))

        print("\n== a schema change rotates rather than corrupts ==")
        log2.close()
        log3 = RunLog(root, "unit", F + ["new_column"], run_id="RUN3")
        log3.log("step", epoch=0, step=1, tokens=1, loss=1.0, lr=1.0,
                 new_column=42)
        chk("old file preserved under .v1.csv",
            (base / "unit.v1.csv").exists())
        chk("old rows still readable", len(rows(base / "unit.v1.csv")) == 5,
            f"{len(rows(base / 'unit.v1.csv'))} rows")
        chk("new file carries the new column",
            "new_column" in rows(shared)[0], str(list(rows(shared)[0])[-1]))
        log3.close()

        print("\n== manifest records what produced the numbers ==")
        man = json.loads((base / "unit/RUN1.json").read_text())
        for k in ("run_id", "stage", "started", "argv", "git_commit", "host",
                  "pid"):
            chk(f"manifest has {k}", k in man, str(man.get(k))[:40])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
