#!/usr/bin/env python3
"""Compare each architecture A/B arm against the three-run baseline.

A mean over 17 targets hides the thing that decides whether a change is
real: whether it moved the SAME targets. The baseline's three runs span
0.0890 / 0.0913 / 0.0936 at identical settings, so a mean difference
under about 0.0046 is inside the spread the settings already produce,
and the init-RNG noise floor measured between two of them is 0.0023.

So this reports a paired comparison against each baseline run and a sign
test, and it states the verdict in those terms rather than in the
difference of two means.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "data/samples/analysis"
OUT = AN / "geom_ab/RESULT.md"

BASELINES = {
    "run 1 (blind_tests_v2)": AN / "blind_tests_v2.json",
    "run 2 (blind_ab_A)": AN / "blind_ab_A.json",
    "run 3 (blind_tests_v3)": AN / "blind_tests_v3.json",
}
ARMS = {
    "100  GeometricBias": AN / "blind_geom.json",
    "109  triangle update": AN / "blind_tri.json",
    "103  bidirectional gdn": AN / "blind_bigdn.json",
}
NOISE_FLOOR = 0.0023
SPREAD = 0.0046


def tm_by_target(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text()).get("targets", [])
    out = {}
    for r in rows:
        if r.get("source") != "rna_puzzles" or not r.get("n_competitors"):
            continue
        ph = r.get("pharos") or {}
        if ph.get("tm") is not None:
            out[r.get("target") or r.get("name")] = float(ph["tm"])
    return out


def sign_test_p(wins: int, n: int) -> float:
    """Two-sided exact binomial at p=0.5. `n` is ties-excluded.

    TWO-sided, deliberately, and it differs from the figure finding 93
    quotes. 93 records "9/17 wins, sign test p = 0.500" for its arm B;
    9 of 17 two-sided is **p = 1.000**, because 17 is odd and
    P(X >= 9) = P(X <= 8) = 0.5 exactly, so twice the tail is the whole
    mass. 0.500 is the ONE-sided value.

    It changes no conclusion there -- both say "indistinguishable from
    chance" -- but one-sided is the wrong default for a question asked as
    "did this help?", because the same test must be able to say "this
    hurt". Validated by replaying 93's arm B through this script: it
    reproduces +0.0025 and 9/17 against arm A.
    """
    if n == 0:
        return 1.0
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> int:
    bases = {n: tm_by_target(p) for n, p in BASELINES.items()}
    present = {n: tm_by_target(p) for n, p in ARMS.items() if p.exists()}
    present = {n: v for n, v in present.items() if v}
    if not present:
        print("[ab] no arm results yet; nothing to compare")
        return 1

    lines = ["# Architecture A/Bs against the three-run baseline", "",
             "Each arm: 40 epochs, 16,384-token budget, lr 2e-4, from the "
             "same stage-4 init, scored on 17 RNA-Puzzles targets.", "",
             "The baselines span **0.0046** at identical settings and the "
             "init-RNG noise floor measured between two of them is "
             "**0.0023**, so a mean difference under about 0.0046 is inside "
             "the spread the settings already produce. On 17 targets, "
             "two-sided p < 0.05 needs **13 wins**.", "",
             "| arm | baseline | n | arm TM | base TM | delta | wins | p |",
             "|---|---|---|---|---|---|---|---|"]

    summary = {}
    for aname, arm in present.items():
        deltas, ps = [], []
        for bname, base in bases.items():
            common = sorted(set(arm) & set(base))
            if not common:
                continue
            a = [arm[t] for t in common]
            b = [base[t] for t in common]
            am, bm = sum(a) / len(a), sum(b) / len(b)
            wins = sum(1 for x, y in zip(a, b) if x > y)
            ties = sum(1 for x, y in zip(a, b) if x == y)
            p = sign_test_p(wins, len(common) - ties)
            deltas.append(am - bm); ps.append(p)
            lines.append(f"| {aname} | {bname} | {len(common)} | {am:.4f} | "
                         f"{bm:.4f} | {am - bm:+.4f} | {wins}/{len(common)} | "
                         f"{p:.3f} |")
        if deltas:
            summary[aname] = (sum(deltas) / len(deltas), min(ps))

    lines += ["", "## Verdicts", "",
              "| arm | mean delta | best p | verdict |", "|---|---|---|---|"]
    for aname, (md, bp) in summary.items():
        if md > SPREAD and bp < 0.05:
            v = "**KEEP** — clears the spread and is significant"
        elif md > SPREAD:
            v = "**INCONCLUSIVE, leaning positive** — clears the spread, not significant"
        elif abs(md) <= SPREAD:
            v = "**NO EFFECT** — inside the spread three identical runs give"
        else:
            v = "**WORSE** — below the baselines by more than their spread"
        lines.append(f"| {aname} | {md:+.4f} | {bp:.3f} | {v} |")

    missing = [n for n in ARMS if n not in present]
    if missing:
        lines += ["", f"Not yet run: {', '.join(missing)}."]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[ab] written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
