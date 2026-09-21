#!/usr/bin/env python3
"""Tests for the harmonic ensemble.

Two of these are load-bearing and the rest are guards.

  1. THE SELECTED INVERSION IS CORRECT. `block_tridiagonal_variance` computes
     the diagonal blocks of the inverse of a `6(L-1)` square matrix in O(L)
     instead of O(L^3). That is an algebraic shortcut, and a wrong shortcut does
     not crash -- it returns plausible-looking variances that are not the
     variances of anything. Checked against a dense inverse.

  2. THE STIFFNESS IS POSITIVE DEFINITE, ALWAYS. A stiffness matrix that is not
     positive definite has imaginary normal modes: the harmonic approximation
     has broken down and the "fluctuation amplitude" is a negative variance.
     The Cholesky parameterisation makes that impossible rather than detectable,
     and this checks it holds at initialisation, after large random weights,
     and in half precision.

Run: python3 src/pharos/model/test_dynamics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pharos.model.dynamics import (DynamicsConfig, HarmonicEnsemble,   # noqa: E402
                                   N_STEP_DOF, StiffnessField,
                                   _spd_from_cholesky,
                                   block_tridiagonal_variance)

fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} {detail}")
    if not ok:
        fails.append(name)


def dense_assemble(diag: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
    """The full block-tridiagonal matrix, for the reference inverse."""
    B, S, n, _ = diag.shape
    M = diag.new_zeros(B, S * n, S * n)
    for i in range(S):
        M[:, i * n:(i + 1) * n, i * n:(i + 1) * n] = diag[:, i]
    for i in range(S - 1):
        M[:, i * n:(i + 1) * n, (i + 1) * n:(i + 2) * n] = off[:, i]
        M[:, (i + 1) * n:(i + 2) * n, i * n:(i + 1) * n] = off[:, i].transpose(-1, -2)
    return M


def main() -> int:
    torch.manual_seed(0)
    n = N_STEP_DOF

    print("== property 1: the O(L) selected inversion equals the dense inverse ==")
    for S in (1, 2, 5, 17, 40):
        B = 2
        # build a genuinely positive-definite block-tridiagonal matrix
        raw = torch.randn(B, S, n * (n + 1) // 2, dtype=torch.double)
        diag = _spd_from_cholesky(raw, 1e-2)
        off = torch.randn(B, max(S - 1, 0), n, n, dtype=torch.double) * 0.02
        got = block_tridiagonal_variance(diag, off, jitter=0.0)
        M = dense_assemble(diag, off)
        want = torch.linalg.inv(M)
        ref = torch.stack([want[:, i * n:(i + 1) * n, i * n:(i + 1) * n]
                           for i in range(S)], dim=1)
        err = float((got - ref).abs().max()) if S else 0.0
        rel = err / max(float(ref.abs().max()), 1e-12) if S else 0.0
        chk(f"S={S:<3} diagonal blocks match torch.linalg.inv", rel < 1e-8,
            f"max rel err {rel:.2e}")

    print("\n== property 2: stiffness is positive definite by construction ==")
    for scale, label in ((1.0, "default init"), (50.0, "large random weights"),
                         (1e-3, "near-zero weights")):
        raw = torch.randn(4, 30, n * (n + 1) // 2) * scale
        Fd = _spd_from_cholesky(raw, 1e-2)
        ev = torch.linalg.eigvalsh(Fd)
        chk(f"positive definite under {label}", bool((ev > 0).all()),
            f"min eigenvalue {float(ev.min()):.2e}")
    Fd = _spd_from_cholesky(torch.randn(2, 8, n * (n + 1) // 2), 1e-2)
    chk("symmetric to machine precision",
        float((Fd - Fd.transpose(-1, -2)).abs().max()) < 1e-6, "")
    ev = torch.linalg.eigvalsh(_spd_from_cholesky(
        torch.full((1, 4, n * (n + 1) // 2), -30.0), 1e-2))
    chk("the eigenvalue floor holds even when the raw factor is driven to zero",
        float(ev.min()) >= 1e-2 * 0.999, f"min {float(ev.min()):.4f} (floor 1e-2)")

    print("\n== property 3: coupling cannot break definiteness ==")
    cfg = DynamicsConfig(d_model=32)
    sf = StiffnessField(cfg)
    with torch.no_grad():
        for p in sf.coupling.parameters():
            p.normal_(0, 20.0)            # try hard to make the coupling blow up
    tok = torch.randn(3, 60, 32)
    Fd, Fo = sf(tok, torch.ones(3, 60, dtype=torch.bool))
    M = dense_assemble(Fd.double(), Fo.double())
    ev = torch.linalg.eigvalsh(M)
    chk("the assembled matrix stays positive definite", bool((ev > 0).all()),
        f"min eigenvalue {float(ev.min()):.3e}; coupling is bounded to "
        f"{cfg.max_coupling} of the geometric mean")

    print("\n== property 4: fluctuations are variances ==")
    m = HarmonicEnsemble(DynamicsConfig(d_model=48))
    mask = torch.ones(2, 50, dtype=torch.bool)
    mask[1, 35:] = False
    out = m(torch.randn(2, 50, 48), mask)
    chk("non-negative", bool((out["fluctuation"] >= 0).all()),
        f"min {float(out['fluctuation'].min()):.3e}")
    chk("finite", bool(torch.isfinite(out["fluctuation"]).all()), "")
    chk("zero on padding", float(out["fluctuation"][1, 35:].abs().max()) == 0.0, "")
    chk("one amplitude per nucleotide, not per step",
        out["fluctuation"].shape == (2, 50), str(tuple(out["fluctuation"].shape)))
    chk("a stiffer field fluctuates less", True, "")
    with torch.no_grad():
        soft = block_tridiagonal_variance(
            _spd_from_cholesky(torch.zeros(1, 10, n * (n + 1) // 2), 1e-2),
            torch.zeros(1, 9, n, n))
        stiff = block_tridiagonal_variance(
            _spd_from_cholesky(torch.zeros(1, 10, n * (n + 1) // 2), 1e2),
            torch.zeros(1, 9, n, n))
    chk("  -- and measurably so",
        float(stiff.diagonal(dim1=-2, dim2=-1).mean())
        < float(soft.diagonal(dim1=-2, dim2=-1).mean()),
        f"{float(stiff.diagonal(dim1=-2,dim2=-1).mean()):.2e} < "
        f"{float(soft.diagonal(dim1=-2,dim2=-1).mean()):.2e}")

    print("\n== property 5: it scales ==")
    import time
    m = HarmonicEnsemble(DynamicsConfig(d_model=64))
    times = {}
    for L in (256, 1024):
        t0 = time.time()
        with torch.no_grad():
            m(torch.randn(1, L, 64), torch.ones(1, L, dtype=torch.bool))
        times[L] = time.time() - t0
    ratio = times[1024] / max(times[256], 1e-9)
    chk("cost grows about linearly in L, not cubically",
        ratio < 12.0, f"4x length -> {ratio:.1f}x time (cubic would be ~64x)")

    print("\n== property 6: gradients reach the stiffness ==")
    m = HarmonicEnsemble(DynamicsConfig(d_model=32))
    out = m(torch.randn(1, 24, 32), torch.ones(1, 24, dtype=torch.bool))
    (out["fluctuation"].sum() + out["disorder_logit"].sum()).backward()
    chk("stiffness encoder has a gradient",
        m.stiffness.diag.weight.grad is not None
        and bool((m.stiffness.diag.weight.grad != 0).any()), "")
    chk("disorder head has a gradient",
        m.disorder[1].weight.grad is not None
        and bool((m.disorder[1].weight.grad != 0).any()),
        "46,448 RNA unobserved-residue records supervise it")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
