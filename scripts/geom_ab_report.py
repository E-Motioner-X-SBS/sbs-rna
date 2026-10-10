#!/usr/bin/env python3
"""Compare the finding-100 arm against the three-run baseline, per target.

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
ARM = AN / "blind_geom.json"
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
    arm = tm_by_target(ARM)
    if not arm:
        print(f"[ab] no arm results at {ARM}; nothing to compare")
        return 1

    lines = ["# Finding 100 A/B — GeometricBias in the diffusion decoder", ""]
    lines.append(f"Arm: `{ARM.name}`, {len(arm)} RNA-Puzzles targets.")
    lines.append("")
    lines.append("| baseline | n paired | arm mean | base mean | delta | "
                 "arm wins | sign test p |")
    lines.append("|---|---|---|---|---|---|---|")

    verdicts = []
    for name, path in BASELINES.items():
        base = tm_by_target(path)
        common = sorted(set(arm) & set(base))
        if not common:
            continue
        a = [arm[t] for t in common]
        b = [base[t] for t in common]
        am, bm = sum(a) / len(a), sum(b) / len(b)
        wins = sum(1 for x, y in zip(a, b) if x > y)
        ties = sum(1 for x, y in zip(a, b) if x == y)
        p = sign_test_p(wins, len(common) - ties)
        verdicts.append((am - bm, p))
        lines.append(f"| {name} | {len(common)} | {am:.4f} | {bm:.4f} | "
                     f"{am - bm:+.4f} | {wins}/{len(common)} | {p:.3f} |")

    lines += ["", "## Verdict", ""]
    if verdicts:
        deltas = [d for d, _ in verdicts]
        mean_d = sum(deltas) / len(deltas)
        beats_all = all(d > SPREAD for d in deltas)
        any_sig = any(p < 0.05 for _, p in verdicts)
        lines.append(f"Mean delta over the three baselines: **{mean_d:+.4f}**.")
        lines.append(f"The baselines' own spread at identical settings is "
                     f"{SPREAD:.4f} and the init-RNG noise floor measured "
                     f"between two of them is {NOISE_FLOOR:.4f}.")
        lines.append("")
        if beats_all and any_sig:
            lines.append("**KEEP.** The arm beats every baseline run by more "
                         "than their own spread, and at least one paired sign "
                         "test is significant.")
        elif mean_d > SPREAD:
            lines.append("**INCONCLUSIVE, leaning positive.** The mean clears "
                         "the spread but no paired sign test does. More seeds "
                         "before this is called a win.")
        elif abs(mean_d) <= SPREAD:
            lines.append("**NO EFFECT.** The difference is inside the spread "
                         "three identical runs already produce. On the "
                         "project's own rule, that means do not keep it on "
                         "this evidence.")
        else:
            lines.append("**WORSE.** The arm is below the baselines by more "
                         "than their spread. Revert.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[ab] written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
