#!/usr/bin/env python3
"""A non-finite loss must stop the run, and the monitor must see one too.

Two halves of the same hole. A NaN loss steps the optimiser, writes NaN into
every weight, and the next checkpoint saves it -- so the run keeps going and
the file it leaves behind cannot be resumed from. Meanwhile the held-out
monitor parsed `nan` with `float()`, compared it against the unigram floor
(every comparison against NaN is False), and reported the diverged run OK.

Run: python3 src/pharos/train/test_guard.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from pharos.train.guard import NonFiniteLoss, check_loss        # noqa: E402

fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:58s} {detail}")
    if not ok:
        fails.append(name)


def raises(loss, parts=None):
    try:
        check_loss(torch.tensor(loss), 1, parts)
        return None
    except NonFiniteLoss as e:
        return str(e)


def main() -> int:
    print("== the guard stops a run that cannot compute a finite loss ==")
    chk("a finite loss passes untouched", raises(1.5, {"ce": 1.5}) is None)
    chk("a finite NEGATIVE loss passes too", raises(-3.0) is None,
        "some terms are legitimately negative; only non-finite is fatal")
    for bad, label in ((float("nan"), "NaN"), (float("inf"), "+Inf"),
                       (float("-inf"), "-Inf")):
        chk(f"{label} raises", raises(bad) is not None)
    msg = raises(float("nan"), {"contact": 0.5, "structure": float("nan")})
    chk("and names the component that went first",
        msg is not None and "structure" in msg and "contact" not in msg,
        "the heads are summed, so the sum only records that something broke")
    msg2 = raises(float("nan"), {"contact": 0.5})
    chk("and says so when no reported part explains it",
        msg2 is not None and "not reported" in msg2)

    print("\n== every trainer calls it, before the optimiser ==")
    for f in ("pretrain_mlm.py", "train_pharos.py",
              "train_sequence_stages.py", "train_block_scorer.py"):
        src = (ROOT / "scripts" / f).read_text()
        has = "check_loss(" in src and "from pharos.train.guard" in src
        # the call has to come BEFORE backward, or the weights are already gone
        ok_order = has and all(
            src.index("check_loss(", src.rindex("import check_loss"))
            < src.index(b) for b in (".backward()",))
        chk(f"{f} guards its loss", ok_order,
            "" if ok_order else "missing, or placed after backward()")

    print("\n== the monitor treats a NaN reading as unreadable, not as good ==")
    spec = importlib.util.spec_from_file_location(
        "tm", ROOT / "scripts" / "training_monitor.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    chk("nan -> None", tm.f("nan") is None,
        "float('nan') parses, and nan >= UNIGRAM_BITS is False, so it used "
        "to read as 'better than unigram'")
    chk("inf -> None", tm.f("inf") is None)
    chk("-inf -> None", tm.f("-inf") is None)
    chk("a real number still parses", tm.f("1.78488") == 1.78488)
    chk("junk still parses to the default", tm.f("", "x") == "x")
    chk("and NaN is not better than the unigram floor any more",
        tm.f("nan") is None and not (tm.f("nan", 9e9) < tm.UNIGRAM_BITS),
        f"UNIGRAM_BITS {tm.UNIGRAM_BITS}")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
