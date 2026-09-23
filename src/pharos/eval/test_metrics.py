#!/usr/bin/env python3
"""Behavioural tests for the blind-test metrics.

A metric that is wrong is worse than no metric, because it produces a number
that looks like evidence. These tests check properties that a correct
implementation must have and an incorrect one generally will not:

  * invariance to rigid motion, and NON-invariance to reflection -- the trap
    that let a mirrored structure pass an RMSD check earlier in this project;
  * the published RNA-Puzzles round-1 ranking, reproduced from the actual
    submissions, which no amount of self-consistent arithmetic gives you;
  * the 3-atom base-pair detector re-measured against the full-atom one, so
    `WC_AGREEMENT` in the docstring cannot silently drift from the code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pharos.eval.base_pairs import (agreement, full_atom_pairs,  # noqa: E402
                                    geometric_pairs)
from pharos.eval.blind_tests import BLIND, rna_puzzles  # noqa: E402
from pharos.eval.metrics import (d0_rna, gdt_ts, inf, kabsch,  # noqa: E402
                                 lddt, rmsd, tm_score, clash_score)
from pharos.eval.structure import align_by_resnum, read_structure  # noqa: E402

fails: list[str] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} {detail}")
    if not ok:
        fails.append(name)


def main() -> int:
    rng = np.random.default_rng(0)

    print("== a metric that moves with the structure is not a metric ==")
    x = rng.normal(size=(60, 3)) * 12.0
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    moved = x @ q.T + np.array([5.0, -2.0, 11.0])
    chk("RMSD is 0 after a rigid motion", rmsd(moved, x) < 1e-8,
        f"{rmsd(moved, x):.2e}")
    chk("TM-score is 1 after a rigid motion", abs(tm_score(moved, x) - 1) < 1e-6)
    chk("lDDT is 1 after a rigid motion", abs(lddt(moved, x) - 1) < 1e-9)
    chk("GDT-TS is 1 after a rigid motion", abs(gdt_ts(moved, x) - 1) < 1e-9)

    # A reflection preserves every pairwise distance, so lDDT -- which only
    # sees distances -- cannot tell, and must not be asked to. Superposition
    # CAN tell, and this is the check that it does.
    mirrored = x * np.array([1.0, 1.0, -1.0])
    chk("lDDT cannot see a reflection (by construction)",
        abs(lddt(mirrored, x) - 1) < 1e-9,
        "distances are preserved; this is why lDDT alone is not enough")
    chk("but superposition refuses to mirror", rmsd(mirrored, x) > 1.0,
        f"RMSD {rmsd(mirrored, x):.2f} -- Kabsch stays in SO(3)")
    r, _t = kabsch(mirrored, x)
    chk("and the rotation it returns is a rotation", np.linalg.det(r) > 0,
        f"det {np.linalg.det(r):+.3f}")

    print("\n== scrambling destroys the score ==")
    perm = rng.permutation(x)
    chk("TM-score collapses", tm_score(perm, x) < 0.25, f"{tm_score(perm, x):.3f}")
    chk("lDDT collapses", lddt(perm, x) < 0.25, f"{lddt(perm, x):.3f}")

    print("\n== d0 grows with length, so TM-score is comparable across targets ==")
    d30, d700 = d0_rna(30), d0_rna(700)
    chk("d0 increases with L", d30 < d700, f"L=30 -> {d30:.2f}, L=700 -> {d700:.2f}")
    # the same absolute error should score BETTER on a longer chain
    short = x[:30] + rng.normal(size=(30, 3)) * 2.0
    long_ = np.concatenate([x] * 12)[:700]
    long_p = long_ + rng.normal(size=(700, 3)) * 2.0
    chk("a 2 A error scores higher on a long target than a short one",
        tm_score(long_p, long_) > tm_score(short, x[:30]),
        f"{tm_score(long_p, long_):.3f} vs {tm_score(short, x[:30]):.3f}")

    print("\n== lDDT sees local quality that RMSD cannot ==")
    # a rigid hinge: the second half rotates about the join. Local geometry is
    # untouched, so lDDT stays high while RMSD blows up.
    hinged = x.copy()
    h = x[30:] - x[30]
    hinged[30:] = h @ q.T + x[30]
    chk("RMSD punishes a hinge", rmsd(hinged, x) > 5.0, f"{rmsd(hinged, x):.2f} A")
    chk("lDDT mostly does not", lddt(hinged, x) > 0.5, f"{lddt(hinged, x):.3f}")

    print("\n== INF is undefined, not zero, on an unpaired target ==")
    chk("no reference pairs -> NaN", np.isnan(inf({(1, 9)}, set())))
    chk("perfect agreement -> 1", abs(inf({(1, 9), (2, 8)}, {(1, 9), (2, 8)}) - 1) < 1e-9)
    chk("predicting everything is punished",
        inf({(i, j) for i in range(20) for j in range(i + 4, 20)},
            {(1, 9), (2, 8)}) < 0.2,
        "MCC, not recall")

    print("\n== a self-intersecting chain is caught ==")
    good = np.stack([np.arange(40) * 6.0, np.zeros(40), np.zeros(40)], 1)
    good = good[:, None, :] + np.zeros((1, 3, 3))
    bad = np.zeros_like(good)
    chk("a spread chain has no clashes", clash_score(good) == 0.0)
    chk("a collapsed one is all clash", clash_score(bad) > 0.9,
        f"{clash_score(bad):.2f}")

    have_data = (BLIND / "rna_puzzles_std").is_dir()
    if not have_data:
        print("\n(blind-test data absent -- skipping the real-structure checks)")
        return _report()

    print("\n== the published RNA-Puzzles round-1 ranking, reproduced ==")
    rp01 = next((t for t in rna_puzzles() if t.name == "rp01"), None)
    if rp01 is not None:
        ref = read_structure(rp01.reference)
        scored = []
        for f in rp01.competitors:
            try:
                a, b = align_by_resnum(ref, read_structure(f))
            except Exception:
                continue
            k = a.residue_mask & b.residue_mask
            if k.sum() < 10:
                continue
            scored.append((f.stem, rmsd(b.coords[k].reshape(-1, 3),
                                        a.coords[k].reshape(-1, 3))))
        scored.sort(key=lambda r: r[1])
        best, worst = scored[0], scored[-1]
        # Cruz et al. 2012 report the Das group best on Puzzle 1 at ~3.4 A and
        # Dokholyan last. If our RMSD were wrong, this would not come out.
        chk("Das places first", "das" in best[0].lower(), f"{best[0]} at {best[1]:.2f} A")
        chk("and at the published ~3.4 A", 2.5 < best[1] < 4.5, f"{best[1]:.2f} A")
        chk("Dokholyan places last", "dokholyan" in worst[0].lower(),
            f"{worst[0]} at {worst[1]:.2f} A")
        chk("every submission scores in a plausible range",
            all(1.0 < s < 25.0 for _n, s in scored),
            f"{len(scored)} models, {scored[0][1]:.1f}-{scored[-1][1]:.1f} A")

    print("\n== the 3-atom base-pair detector, re-measured ==")
    tp = fp = fn = 0
    n_struct = 0
    for t in rna_puzzles():
        try:
            ref_pairs = full_atom_pairs(t.reference)
            s = read_structure(t.reference)
        except Exception:
            continue
        if not ref_pairs:
            continue
        got = geometric_pairs(s.coords, s.seq, mask=s.residue_mask,
                              chain_ids=s.chain_ids)
        tp += len(got & ref_pairs)
        fp += len(got - ref_pairs)
        fn += len(ref_pairs - got)
        n_struct += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    chk("precision against the full-atom criterion", prec > 0.70,
        f"{prec:.3f} over {n_struct} structures")
    chk("recall against the full-atom criterion", rec > 0.70, f"{rec:.3f}")
    chk("F1 is high enough for INF to mean something", f1 > 0.70, f"{f1:.3f}")
    print(f"\n  WC_AGREEMENT = precision {prec:.3f}, recall {rec:.3f}, F1 {f1:.3f} "
          f"(tp {tp}, fp {fp}, fn {fn})")

    print("\n== INF ranks the real submissions sensibly ==")
    if rp01 is not None:
        ref_s = read_structure(rp01.reference)
        ref_bp = full_atom_pairs(rp01.reference)
        vals = []
        for f in rp01.competitors[:8]:
            try:
                p = read_structure(f)
            except Exception:
                continue
            vals.append(inf(geometric_pairs(p.coords, p.seq, mask=p.residue_mask,
                                            chain_ids=p.chain_ids), ref_bp))
        vals = [v for v in vals if not np.isnan(v)]
        chk("submissions get a real INF", len(vals) > 3 and max(vals) > 0.5,
            f"n={len(vals)} max {max(vals):.3f}")
        self_inf = inf(geometric_pairs(ref_s.coords, ref_s.seq,
                                       mask=ref_s.residue_mask,
                                       chain_ids=ref_s.chain_ids), ref_bp)
        chk("and the reference beats all of them against itself",
            self_inf >= max(vals) - 1e-9, f"reference {self_inf:.3f}")

    return _report()


def _report() -> int:
    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
