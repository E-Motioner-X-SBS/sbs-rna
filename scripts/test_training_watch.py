#!/usr/bin/env python3
"""Tests for the durable training watch.

This module exists because the watch's failure mode is *false positives*:
22 of its first 23 warnings were wrong, and a monitor that cries wolf is
worse than no monitor, because it trains its reader to skip the one real
alarm. Every check here asserts a severity, not just a message.

The OOM check is the worked example. Its comment said "only while it is
still happening", and it decided that by comparing the last two ROWS of a
csv -- a property of the file, not of the clock. Once a run ended on a
rising count the comparison stayed true forever, so a run that died at
16:03 was reported at 16:13 as "still taking them", in the same report
whose liveness line said no trainer was running.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import training_watch as tw                                  # noqa: E402

fails: list[str] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<54} {detail}")
    if not ok:
        fails.append(name)


class _FakeRun:
    """Stands in for the newest csv, with a controllable mtime."""

    def __init__(self, age_min: float) -> None:
        self._age = age_min

    def stat(self):
        return types.SimpleNamespace(st_mtime=tw.time.time() - self._age * 60)


def _oom(counts: list[int], age_min: float):
    """Run check_oom against a synthetic run of the given age."""
    saved = tw.newest_run
    tw.newest_run = lambda stage: _FakeRun(age_min)          # type: ignore
    try:
        F: list[tuple[str, str, str]] = []
        rows = [{"n_oom": str(c)} for c in counts]
        tw.check_oom(F, "stage5_3d/synthetic", rows)
        return F
    finally:
        tw.newest_run = saved                                # type: ignore


def main() -> int:
    print("== the OOM check must separate 'is happening' from 'happened' ==")

    F = _oom([40, 43, 46], age_min=2.0)
    chk("a LIVE run with a rising count is CRIT",
        len(F) == 1 and F[0][0] == "CRIT" and "still taking them" in F[0][2],
        F[0][2] if F else "nothing reported")

    F = _oom([40, 43, 46], age_min=247.0)
    chk("the SAME rows from a run that ended are not CRIT",
        len(F) == 1 and F[0][0] == "INFO",
        f"{F[0][0]}: {F[0][2]}" if F else "nothing reported")
    chk("and the message says so rather than claiming it continues",
        len(F) == 1 and "still taking them" not in F[0][2]
        and "ended" in F[0][2],
        F[0][2] if F else "")

    F = _oom([46, 46, 46], age_min=2.0)
    chk("a live run whose count has stopped rising is INFO",
        len(F) == 1 and F[0][0] == "INFO" and "not rising" in F[0][2],
        F[0][2] if F else "")

    F = _oom([0, 0, 0], age_min=2.0)
    chk("a run that has taken no OOMs says nothing at all",
        not F, f"{len(F)} finding(s); silence is the only right output here")

    F = _oom([1, 2, 3], age_min=2.0)
    chk("a live run with a SMALL rising count is WARN, not CRIT",
        len(F) == 1 and F[0][0] == "WARN",
        f"{F[0][0]}: {F[0][2]}" if F else "")

    print("\n== the liveness window is a named constant, not a literal ==")
    chk("LIVE_MIN exists and is a sane number of minutes",
        isinstance(getattr(tw, "LIVE_MIN", None), (int, float))
        and 5.0 <= tw.LIVE_MIN <= 120.0,
        f"LIVE_MIN = {getattr(tw, 'LIVE_MIN', None)}; stage 5 writes a row "
        f"per epoch at ~150 s, so this must be several missed rows")

    print("\n== an empty newest record must not blind the watch ==")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "stage5_3d"
        d.mkdir(parents=True)
        real = d / "20260101T000000Z.csv"
        real.write_text("epoch,n_oom\n0,0\n1,0\n")
        smoke = d / "20260102T000000Z.csv"
        smoke.write_text("epoch,n_oom\n")          # header only, as --smoke writes
        import os, time as _t
        os.utime(real, (_t.time() - 600, _t.time() - 600))
        os.utime(smoke, (_t.time(), _t.time()))

        saved = tw.RUNS
        tw.RUNS = Path(td)
        try:
            picked = tw.newest_run("stage5_3d")
            chk("newest_run skips a record with no data rows",
                picked is not None and picked.name == real.name,
                f"picked {picked.name if picked else None}; strictly-newest "
                f"gives {smoke.name}, whose 0 rows make the dispatch loop "
                f"skip EVERY check for that stage without a word")

            F: list = []
            tw.check_empty_newest(F)
            hit = [f for f in F if f[1] == "empty run"]
            chk("and the watch SAYS the newest record is empty",
                len(hit) == 1 and "stage5_3d" in hit[0][2],
                hit[0][2] if hit else "nothing reported -- silence here is "
                                      "what made this invisible: the report "
                                      "just had two fewer INFO lines")
            chk("reported as INFO, not as an alarm",
                bool(hit) and hit[0][0] == "INFO",
                "an old-but-real record is the right thing to read, so this "
                "is news and not a fault")
        finally:
            tw.RUNS = saved

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
