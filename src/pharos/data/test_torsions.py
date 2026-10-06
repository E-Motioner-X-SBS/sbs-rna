#!/usr/bin/env python3
"""Pseudotorsions: the invariants, then what the corpus actually contains.

Run: python3 src/pharos/data/test_torsions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pharos.data.torsions import (C4, N, P, TORSION_NAMES,  # noqa: E402
                                  dihedral, pseudotorsions, sin_cos)

fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:58s} {detail}")
    if not ok:
        fails.append(name)


def main() -> int:
    print("== the dihedral itself, against geometry with a known answer ==")
    # cis: all four atoms in a plane, 1 and 4 on the SAME side -> 0
    cis = dihedral([1., 1., 0.], [0., 0., 0.], [1., 0., 0.], [1., 1., 0.])
    chk("planar cis is 0", abs(cis) < 1e-12, f"{np.degrees(cis):+.6f} deg")
    # trans: opposite sides -> pi
    trans = dihedral([1., 1., 0.], [0., 0., 0.], [1., 0., 0.], [1., -1., 0.])
    chk("planar trans is 180", abs(abs(trans) - np.pi) < 1e-12,
        f"{np.degrees(trans):+.6f} deg")
    # a right angle out of plane -> +-90
    q = dihedral([1., 1., 0.], [0., 0., 0.], [1., 0., 0.], [1., 0., 1.])
    chk("perpendicular is 90", abs(abs(q) - np.pi / 2) < 1e-12,
        f"{np.degrees(q):+.6f} deg")

    print("\n== sign, which arccos(dot) would have thrown away ==")
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(4, 3))
    a = dihedral(*pts)
    mirror = pts * np.array([1., 1., -1.])      # reflect through z=0
    am = dihedral(*mirror)
    chk("mirroring flips the sign", abs(a + am) < 1e-12,
        f"{np.degrees(a):+.3f} -> {np.degrees(am):+.3f} deg")
    chk("and does not change the magnitude", abs(abs(a) - abs(am)) < 1e-12)

    print("\n== a torsion is invariant to rigid motion, by construction ==")
    # If it were not, a structure and the same structure rotated would train
    # the head towards two different answers.
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:                     # keep it a ROTATION
        Q[:, 0] *= -1
    moved = pts @ Q.T + np.array([7.3, -2.1, 0.5])
    chk("rotation + translation leaves it unchanged",
        abs(dihedral(*moved) - a) < 1e-10,
        f"delta {abs(dihedral(*moved) - a):.2e} rad")

    print("\n== ends and breaks are NaN, not zero ==")
    L = 12
    co = rng.normal(size=(L, 3, 3)) * 5.0
    ang, valid = pseudotorsions(co)
    chk("shape is (L, 3)", ang.shape == (L, 3), str(ang.shape))
    chk("eta is undefined at both ends",
        not valid[0, 0] and not valid[-1, 0])
    chk("theta and chi_tilde are undefined at the last residue",
        not valid[-1, 1] and not valid[-1, 2])
    chk("everything undefined is NaN, never 0.0",
        np.isnan(ang[~valid]).all() and np.isfinite(ang[valid]).all(),
        f"{(~valid).sum()} undefined of {valid.size}")
    chk("eta defined on L-2, theta/chi on L-1",
        valid[:, 0].sum() == L - 2 and valid[:, 1].sum() == L - 1
        and valid[:, 2].sum() == L - 1,
        f"{valid[:, 0].sum()}/{valid[:, 1].sum()}/{valid[:, 2].sum()}")

    print("\n== an unresolved residue removes exactly the angles it touches ==")
    m = np.ones(L, dtype=bool)
    m[5] = False
    ang2, valid2 = pseudotorsions(co, m)
    lost = np.flatnonzero(valid[:, 0] & ~valid2[:, 0])
    chk("dropping residue 5 kills eta at 4, 5 and 6",
        set(lost.tolist()) == {4, 5, 6}, f"lost eta at {lost.tolist()}")
    lost_t = np.flatnonzero(valid[:, 1] & ~valid2[:, 1])
    chk("and theta at 4 and 5", set(lost_t.tolist()) == {4, 5},
        f"lost theta at {lost_t.tolist()}")
    chk("angles away from the gap are UNCHANGED",
        np.allclose(ang[:3, 1], ang2[:3, 1], equal_nan=True))

    print("\n== sin/cos is the form the head predicts ==")
    sc = sin_cos(ang, valid)
    chk("shape is (L, 3, 2)", sc.shape == (L, 3, 2), str(sc.shape))
    norm = np.linalg.norm(sc[valid], axis=-1)
    chk("every defined pair is a UNIT vector",
        np.allclose(norm, 1.0, atol=1e-12),
        f"norm in [{norm.min():.12f}, {norm.max():.12f}]")
    chk("undefined pairs stay NaN", np.isnan(sc[~valid]).all())
    back = np.arctan2(sc[..., 0], sc[..., 1])
    chk("and it round-trips to the angle",
        np.allclose(back[valid], ang[valid], atol=1e-12))

    print("\n== ideal A-form RNA lands in the helical cluster ==")
    # Build an ideal A-form helix from its published parameters rather than
    # asserting a literature eta/theta: 32.7 deg twist, 2.81 A rise. If the
    # geometry is right the angles follow, and a single tight cluster is the
    # claim worth testing -- it is what makes eta/theta informative at all.
    twist, rise = np.radians(32.7), 2.81
    nres = 24
    co = np.zeros((nres, 3, 3))
    for i in range(nres):
        t = i * twist
        for k, (r, dz, dt) in enumerate([(8.8, 0.0, 0.0),      # P
                                         (9.5, 1.0, 0.33),     # C4'
                                         (4.5, 1.6, 0.75)]):   # N
            co[i, k] = [r * np.cos(t + dt), r * np.sin(t + dt), i * rise + dz]
    ang, valid = pseudotorsions(co)
    eta = np.degrees(ang[valid[:, 0], 0]) % 360
    theta = np.degrees(ang[valid[:, 1], 1]) % 360
    chk("eta is one tight cluster along an ideal helix",
        eta.std() < 1.0, f"mean {eta.mean():.2f} sd {eta.std():.4f} deg")
    chk("theta is one tight cluster too",
        theta.std() < 1.0, f"mean {theta.mean():.2f} sd {theta.std():.4f} deg")
    # and a helix of the opposite hand must NOT land in the same place
    left = co * np.array([1., -1., 1.])
    angL, validL = pseudotorsions(left)
    etaL = np.degrees(angL[validL[:, 0], 0]) % 360
    chk("a LEFT-handed helix is somewhere else entirely",
        abs(etaL.mean() - eta.mean()) > 30.0,
        f"{eta.mean():.1f} vs {etaL.mean():.1f} deg")

    print("\n== the torch fast path agrees with the numpy reference ==")
    # A fast path nobody checks against a reference is how a silently
    # different angle gets trained on for a week.
    import torch
    from pharos.data.torsions import pseudotorsions_torch
    rng2 = np.random.default_rng(7)
    B, L = 3, 40
    co = rng2.normal(size=(B, L, 3, 3)) * 6.0
    mk = rng2.random((B, L)) > 0.15                 # scattered unresolved
    mk[:, 0] = True
    at, vt = pseudotorsions_torch(torch.tensor(co), torch.tensor(mk))
    at, vt = at.numpy(), vt.numpy()
    worst, agree = 0.0, True
    for b in range(B):
        an, vn = pseudotorsions(co[b], mk[b])
        if not (vt[b] == vn).all():
            agree = False
        d = np.abs(at[b][vn] - an[vn])
        worst = max(worst, float(d.max()) if d.size else 0.0)
    chk("the validity masks are identical", agree)
    chk("the angles agree to 1e-9", worst < 1e-9, f"max delta {worst:.3e} rad")
    chk("invalid entries are 0.0 in torch, not nan",
        np.isfinite(at).all() and (at[~vt] == 0).all(),
        "a nan would poison the loss through the mask")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
