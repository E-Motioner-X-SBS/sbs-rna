#!/usr/bin/env python3
"""The property that makes WSD the right schedule here, stated as a test.

Run: python3 src/pharos/train/test_schedule.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pharos.train.schedule import (SHAPES, cosine,  # noqa: E402
                                   lr_for, warmup_stable_decay)

F: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:60s} {detail}")
    if not ok:
        F.append(name)


PEAK = 3e-4


def main() -> int:
    print("== the incumbent is reproduced exactly, so the switch is a no-op ==")
    # `train_sequence_stages.lr_at` is the rule in flight; this module must
    # return the same number for it until --lr-schedule says otherwise.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    try:
        from train_sequence_stages import lr_at as _incumbent
        worst = max(abs(cosine(s, 3626, PEAK) - _incumbent(s, 3626, PEAK))
                    for s in range(0, 3700, 7))
        chk("cosine() == train_sequence_stages.lr_at everywhere",
            worst < 1e-18, f"max |delta| {worst:.3e}")
    except Exception as e:                                   # noqa: BLE001
        chk("cosine() == train_sequence_stages.lr_at everywhere", False, str(e))

    print("\n== THE PROPERTY: the stable phase does not depend on the total ==")
    # This is the whole reason for the change. A cosine's value at step t is
    # a function of the total; in this setup the total is a CLI flag that
    # changed from 1 epoch to 2 between runs, and the resume silently
    # re-warmed an annealed model by 82x.
    step = 541
    c_short, c_long = cosine(step, 541, PEAK), cosine(step, 3626, PEAK)
    chk("cosine at one step, two totals: the rate JUMPS",
        c_long / c_short > 5.0,
        f"{c_short:.3e} -> {c_long:.3e} = {c_long/c_short:.0f}x")
    w_short = warmup_stable_decay(step, 2000, PEAK)
    w_long = warmup_stable_decay(step, 3626, PEAK)
    chk("wsd at one step, two totals: IDENTICAL",
        w_short == w_long == PEAK, f"{w_short:.3e} == {w_long:.3e} == peak")
    # and it holds across the whole stable phase, for any pair of totals
    bad = [(s, t1, t2) for s in range(250, 1500, 37)
           for t1, t2 in ((2000, 3626), (3626, 9000), (2000, 50000))
           if warmup_stable_decay(s, t1, PEAK) != warmup_stable_decay(s, t2, PEAK)]
    chk("...for every step in the stable phase and every pair of totals",
        not bad, f"{len(bad)} disagreements" if bad else "0 disagreements")

    print("\n== but it is still a schedule, not a constant ==")
    chk("it warms up from zero",
        warmup_stable_decay(0, 1000, PEAK) == 0.0
        and warmup_stable_decay(5, 1000, PEAK) < PEAK,
        f"step 0 {warmup_stable_decay(0,1000,PEAK):.2e}, "
        f"step 5 {warmup_stable_decay(5,1000,PEAK):.2e}")
    chk("warmup is an absolute step count, not a fraction of the total",
        warmup_stable_decay(50, 2000, PEAK) == warmup_stable_decay(50, 90000, PEAK),
        "a fractional warmup would make a long run warm up for longer, "
        "which is the cosine's bug in a different place")
    chk("it is exactly peak through the middle",
        warmup_stable_decay(500, 1000, PEAK) == PEAK)
    chk("and it anneals to ZERO at the end",
        warmup_stable_decay(1000, 1000, PEAK) == 0.0,
        "the cosine here floors at 0.1*peak and never anneals: "
        f"{cosine(1000, 1000, PEAK):.2e}")
    chk("the decay is monotone",
        all(warmup_stable_decay(s, 1000, PEAK)
            >= warmup_stable_decay(s + 1, 1000, PEAK)
            for s in range(800, 1000)))

    print("\n== every shape is available and every one is sane ==")
    for sh in SHAPES:
        v = [warmup_stable_decay(s, 1000, PEAK, shape=sh) for s in range(800, 1001)]
        chk(f"shape {sh!r}: peak -> 0, monotone, bounded",
            abs(v[0] - PEAK) < 1e-12 and abs(v[-1]) < 1e-12
            and all(a >= b - 1e-18 for a, b in zip(v, v[1:]))
            and all(0.0 <= x <= PEAK + 1e-18 for x in v),
            f"{v[0]:.2e} -> {v[-1]:.2e}")
    # 1-sqrt must spend LESS time near the peak than linear: that is the
    # whole claim -- get off the plateau fast, then crawl.
    a = sum(warmup_stable_decay(s, 1000, PEAK, shape="1-sqrt") for s in range(800, 1001))
    b = sum(warmup_stable_decay(s, 1000, PEAK, shape="linear") for s in range(800, 1001))
    chk("1-sqrt drops faster than linear, as the paper describes",
        a < b, f"area under decay {a:.5f} vs {b:.5f}")

    print("\n== clamping, because estimate_steps is an estimate ==")
    # OneCycleLR raises on the step past its total and that killed a finished
    # 8-epoch run at its very last step. Neither of these may ever raise.
    chk("past the total it clamps, never raises",
        warmup_stable_decay(5000, 1000, PEAK) == 0.0
        and abs(cosine(5000, 1000, PEAK) - PEAK * 0.1) < 1e-18,
        f"wsd {warmup_stable_decay(5000,1000,PEAK):.2e}, "
        f"cosine {cosine(5000,1000,PEAK):.2e}")
    chk("a zero or negative total does not divide by zero",
        warmup_stable_decay(0, 0, PEAK) >= 0.0 and cosine(0, 0, PEAK) >= 0.0)
    chk("a negative step does not go negative",
        warmup_stable_decay(-50, 1000, PEAK) == 0.0)

    print("\n== dispatch ==")
    chk("lr_for('wsd') and lr_for('cosine') agree with the functions",
        lr_for("wsd", 500, 1000, PEAK) == warmup_stable_decay(500, 1000, PEAK)
        and lr_for("cosine", 500, 1000, PEAK) == cosine(500, 1000, PEAK))
    try:
        lr_for("nope", 1, 2, 3)
        chk("an unknown name raises", False, "it did not")
    except ValueError:
        chk("an unknown name raises", True)
    try:
        warmup_stable_decay(1, 2, 3, shape="nope")
        chk("an unknown shape raises", False, "it did not")
    except ValueError:
        chk("an unknown shape raises", True)

    print()
    if F:
        print(f"FAILURES ({len(F)}): " + ", ".join(F))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
