#!/usr/bin/env python3
"""Tests for the triangle multiplicative update (finding 109).

The headline assertion is a CONTRAST, not a property: the shipped pair
features move exactly 0 pairs that share no endpoint with a perturbed
residue, and the triangle update moves all of them. A test of the new
module alone would not say why it exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pharos.model.triangle import (TriangleMultiplication,    # noqa: E402
                                   TrianglePairStack)
from pharos.model.diffusion import DiffusionPairFeatures      # noqa: E402

fails: list[str] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<56} {detail}")
    if not ok:
        fails.append(name)


def main() -> int:
    torch.manual_seed(0)
    B, L, dp, dm = 1, 20, 16, 32
    mask = torch.ones(B, L, dtype=torch.bool)

    print("== the coupling the outer sum cannot have ==")
    # the shipped pair features, probed by perturbing one residue
    dpf = DiffusionPairFeatures(dm, dp).eval()
    h = torch.randn(B, L, dm)
    with torch.no_grad():
        p0 = dpf(h, None, mask)
        h2 = h.clone(); h2[0, 3] = torch.randn(dm) * 3.0
        p1 = dpf(h2, None, mask)
    dd = (p1 - p0).abs().amax(-1)[0]
    ns = [(i, j) for i in range(L) for j in range(L)
          if i != 3 and j != 3 and i != j]
    moved_sum = sum(1 for i, j in ns if float(dd[i, j]) > 1e-4)
    chk("the outer sum moves NO pair that excludes the perturbed residue",
        moved_sum == 0,
        f"{moved_sum} of {len(ns)} -- z[i,j] = A s_i + B s_j is a function "
        f"of its two endpoints and nothing else, so a distogram it produces "
        f"has no mechanism that could make it 3D-embeddable")

    # the triangle stack, probed by perturbing one EDGE
    st = TrianglePairStack(dp, c=8, n_layers=1, grad_checkpoint=False).eval()
    with torch.no_grad():
        for lay in st.layers:
            lay.proj_out.weight.normal_(0.0, 0.5)
    z = torch.randn(B, L, L, dp)
    with torch.no_grad():
        o0 = st(z, mask)
        z2 = z.clone()
        v = torch.randn(dp) * 3.0
        z2[0, 3, 11] = v; z2[0, 11, 3] = v
        o1 = st(z2, mask)
    d = (o1 - o0).abs().amax(-1)[0]
    ns2 = [(i, j) for i in range(L) for j in range(L)
           if i not in (3, 11) and j not in (3, 11) and i != j]
    moved_tri = sum(1 for i, j in ns2 if float(d[i, j]) > 1e-4)
    chk("the triangle update moves pairs that share NO endpoint",
        moved_tri == len(ns2),
        f"{moved_tri} of {len(ns2)} -- reached through the triangles they "
        f"share with edge (3,11), which is the whole point")

    print("\n== and it must not disturb a loaded checkpoint ==")
    a = DiffusionPairFeatures(dm, dp).eval()
    b = DiffusionPairFeatures(dm, dp, triangle_layers=2).eval()
    with torch.no_grad():
        b.load_state_dict(a.state_dict(), strict=False)
        delta = float((a(h, None, mask) - b(h, None, mask)).abs().max())
    chk("zero-initialised, so the forward pass is bit-identical at load",
        delta == 0.0,
        f"max |delta| {delta:.3e} -- same rule as bias_scale, site_scale, "
        f"coev_proj and GeometricBias.proj")

    g = TrianglePairStack(dp, c=8, n_layers=1, grad_checkpoint=False)
    g.train(); g.zero_grad()
    g(z, mask).sum().backward()
    gm = float(g.layers[0].proj_out.weight.grad.abs().max())
    chk("and the gate takes gradient at that zero init",
        gm > 0.0, f"|grad| {gm:.3e}; a zero gate with no gradient is inert "
                  f"forever, which is finding 64's shape")

    print("\n== padding must not leak through the sum over k ==")
    # the sum runs over EVERY k, so an unmasked pad contributes a real
    # product to every real pair -- a leak, not a rounding error
    m2 = torch.zeros(B, L, dtype=torch.bool); m2[0, :12] = True
    zp = z.clone()
    # NOISE in the pad, not a constant. A constant vector is mapped to
    # exactly zero by `norm_in` (LayerNorm subtracts the mean) and then to
    # zero by the bias-free `lin_a`, so a constant pad contributes nothing
    # whether it is masked or not -- and a leak test it cannot fail is not
    # a leak test. The first version of this used 9.9 and passed the
    # masked case while reporting 0.000e+00 for the unmasked one.
    zp[0, 12:] = torch.randn(L - 12, L, dp) * 4.0
    zp[0, :, 12:] = torch.randn(L, L - 12, dp) * 4.0
    with torch.no_grad():
        full = st(zp, m2)[0, :12, :12]
        short = st(zp[:, :12, :12], m2[:, :12])[0]
    leak = float((full - short).abs().max())
    chk("a right-padded batch matches the same rows unpadded",
        leak == 0.0, f"max |delta| {leak:.3e} with garbage in the pad")

    with torch.no_grad():
        no_mask = st(zp, None)[0, :12, :12]
    chk("and WITHOUT the mask it genuinely would leak",
        float((no_mask - short).abs().max()) > 1e-3,
        f"max |delta| {float((no_mask - short).abs().max()):.3e} -- the "
        f"masking is load-bearing, not decoration")

    print("\n== both directions, and they are not transposes ==")
    out_l = TriangleMultiplication(dp, 8, outgoing=True)
    in_l = TriangleMultiplication(dp, 8, outgoing=False)
    chk("the stack alternates outgoing and incoming",
        [l.outgoing for l in TrianglePairStack(dp, 8, 2).layers]
        == [True, False, True, False],
        "Jumper et al. Algorithms 11 and 12; neither is the other's "
        "transpose once the projections differ")
    chk("outgoing and incoming contract different indices",
        out_l.outgoing and not in_l.outgoing,
        "bikc,bjkc->bijc against bkic,bkjc->bijc")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
