#!/usr/bin/env python3
"""Tests for the three mixers of the hybrid stack.

The one that matters is the Gated DeltaNet. Its chunked form is an algebraic
rearrangement of a sequential recurrence, and a rearrangement that is subtly
wrong does not crash -- it trains to something slightly different from the
model that was designed, forever, silently. So it is checked against a naive
step-by-step implementation of the recurrence it claims to compute:

    S_t = a_t * S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T
    o_t = S_t^T q_t

The others are checked for the properties their names promise: sliding-window
attention must not see past its window, full attention must accept a physics
bias, and all three must ignore padding.

Run: python3 src/pharos/model/test_attention.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pharos.model.attention import (BLOCK_PATTERN, FullAttention,   # noqa: E402
                                    GatedDeltaNet, SlidingWindowAttention,
                                    _heads, make_mixer)

fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} {detail}")
    if not ok:
        fails.append(name)


def gdn_reference(mod: GatedDeltaNet, x: torch.Tensor) -> torch.Tensor:
    """The recurrence, one step at a time. Slow, obviously correct."""
    B, L, _ = x.shape
    h, dk = mod.h, mod.dk
    q = F.normalize(_heads(mod.q(x), h), dim=-1)
    k = F.normalize(_heads(mod.k(x), h), dim=-1)
    v = _heads(mod.v(x), h)
    a = torch.sigmoid(mod.a(x)).transpose(1, 2)          # B,h,L
    b = torch.sigmoid(mod.b(x)).transpose(1, 2)
    S = x.new_zeros(B, h, dk, v.shape[-1])
    outs = []
    for t in range(L):
        kt = k[:, :, t]                                   # B,h,dk
        vt = v[:, :, t]
        at = a[:, :, t].unsqueeze(-1).unsqueeze(-1)
        bt = b[:, :, t].unsqueeze(-1).unsqueeze(-1)
        # S (I - b k k^T) = S - b (S k) k^T ... acting on the key axis
        Sk = torch.einsum("bhde,bhd->bhe", S, kt)         # B,h,dv
        S = at * (S - bt * torch.einsum("bhd,bhe->bhde", kt, Sk)) \
            + bt * torch.einsum("bhd,bhe->bhde", kt, vt)
        outs.append(torch.einsum("bhd,bhde->bhe", q[:, :, t], S))
    o = torch.stack(outs, dim=2)
    o = o.transpose(1, 2).reshape(B, L, h * v.shape[-1])
    return mod.out(o * torch.sigmoid(mod.g(x)))


def main() -> int:
    torch.manual_seed(0)
    D, H = 64, 4

    print("== GDN: the chunked form computes the stated recurrence ==")
    for L, chunk in ((32, 8), (64, 16), (100, 16), (257, 64)):
        mod = GatedDeltaNet(D, H, chunk=chunk).double().eval()
        x = torch.randn(2, L, D, dtype=torch.double)
        with torch.no_grad():
            got, want = mod(x), gdn_reference(mod, x)
        err = float((got - want).abs().max())
        rel = err / max(float(want.abs().max()), 1e-12)
        chk(f"L={L:<4} chunk={chunk:<3} matches the sequential recurrence",
            rel < 1e-8, f"max rel err {rel:.2e}")

    print("\n== GDN: state behaviour ==")
    mod = GatedDeltaNet(D, H, chunk=16).double().eval()
    x = torch.randn(1, 200, D, dtype=torch.double)
    with torch.no_grad():
        y = mod(x)
    chk("output is finite over 200 steps", bool(torch.isfinite(y).all()),
        "the delta rule keeps the state contractive")
    # a repeated key must overwrite rather than accumulate: feeding the same
    # token 100 times must not make the output grow without bound
    xr = x[:, :1].repeat(1, 100, 1)
    with torch.no_grad():
        yr = mod(xr)
    growth = float(yr[0, -1].norm() / yr[0, 0].norm().clamp_min(1e-9))
    chk("a repeated token does not blow the state up", growth < 5.0,
        f"|o_100|/|o_1| = {growth:.2f} (accumulation would diverge)")

    print("\n== padding is ignored ==")
    for kind in ("gdn", "swa", "full"):
        m = make_mixer(kind, D, H, 16, 0.0).double().eval()
        x = torch.randn(1, 60, D, dtype=torch.double)
        mask = torch.ones(1, 60, dtype=torch.bool)
        mask[0, 40:] = False
        xp = x.clone()
        xp[0, 40:] = torch.randn(20, D, dtype=torch.double) * 100   # junk in the pad
        with torch.no_grad():
            a = m(x, mask=mask)[0, :40]
            b = m(xp, mask=mask)[0, :40]
        chk(f"{kind}: real positions unaffected by padding content",
            float((a - b).abs().max()) < 1e-8,
            f"max delta {float((a - b).abs().max()):.2e}")

    print("\n== sliding window really is a window ==")
    w = 8
    m = SlidingWindowAttention(D, H, window=w).double().eval()
    L = 64
    x = torch.randn(1, L, D, dtype=torch.double, requires_grad=True)
    y = m(x)
    y[0, 0].sum().backward()
    reach = (x.grad[0].abs().sum(-1) > 1e-12).nonzero().flatten()
    chk(f"position 0 depends only on positions <= {w}",
        int(reach.max()) <= w, f"furthest dependency {int(reach.max())}")

    print("\n== full attention accepts a physics bias ==")
    m = FullAttention(D, H).double().eval()
    x = torch.randn(1, 32, D, dtype=torch.double)
    bias = torch.zeros(1, 32, 32, dtype=torch.double)
    with torch.no_grad():
        base = m(x)
        same = m(x, pair_bias=bias)
    chk("a zero bias is a no-op", float((base - same).abs().max()) < 1e-9,
        "bias_scale starts at 0, so even a nonzero bias is initially inert")
    with torch.no_grad():
        m.bias_scale.fill_(1.0)
        strong = torch.full((1, 32, 32), -1e4, dtype=torch.double)
        strong[:, :, 0] = 0.0
        out = m(x, pair_bias=strong)
    chk("a bias that admits only position 0 collapses attention onto it",
        bool(torch.allclose(out, out[:, :1].expand_as(out), atol=1e-6)),
        "every query returns the same value vector")

    print("\n== the block pattern is §5.1's ==")
    chk("period-8 pattern", BLOCK_PATTERN ==
        ("gdn", "gdn", "swa", "gdn", "gdn", "swa", "gdn", "full"), str(BLOCK_PATTERN))
    two = BLOCK_PATTERN * 2
    chk("16 blocks: 10 GDN, 4 SWA, 2 FULL",
        (len(two), two.count("gdn"), two.count("swa"), two.count("full")) == (16, 10, 4, 2),
        f"{two.count('gdn')}/{two.count('swa')}/{two.count('full')}")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
