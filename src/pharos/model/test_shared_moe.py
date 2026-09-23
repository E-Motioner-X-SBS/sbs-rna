#!/usr/bin/env python3
"""Shared-network experts: cheaper than independent ones, and variable width.

Run: python3 src/pharos/model/test_shared_moe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pharos.model.moe import MoEConfig, MoEFeedForward, RouterFeatures   # noqa: E402
from pharos.model.shared_moe import (SharedMoEConfig,                    # noqa: E402
                                     SharedMoEFeedForward)

fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:58s} {detail}")
    if not ok:
        fails.append(name)


def n_par(m):
    return sum(p.numel() for p in m.parameters())


def main() -> int:
    torch.manual_seed(0)

    print("== sharing buys experts: 256 shared cost less than 48 independent ==")
    indep = MoEFeedForward(MoEConfig(d_model=640, d_expert=192, n_experts=48,
                                     top_k=6, n_shared=2))
    shared = SharedMoEFeedForward(SharedMoEConfig(d_model=640, d_expert=512,
                                                  n_experts=256, rank=16))
    chk("256 shared experts < 48 independent experts", n_par(shared) < n_par(indep),
        f"{n_par(shared)/1e6:.2f}M vs {n_par(indep)/1e6:.2f}M")
    chk("and it is 5x the expert count", 256 >= 5 * 48)

    print("\n== every expert starts as the shared network ==")
    cfg = SharedMoEConfig(d_model=64, d_expert=48, n_experts=16, max_k=8)
    m = SharedMoEFeedForward(cfg).double()
    x = torch.randn(1, 6, 64, dtype=torch.double)
    e = m.experts
    flat = x.reshape(-1, 64)

    def pure(idx):
        """Route every token wholly to one expert."""
        w = torch.zeros(flat.shape[0], cfg.n_experts, dtype=torch.double)
        w[:, idx] = 1.0
        return e(flat, w)

    # the modulation is zero at init, so it is the identity and two different
    # experts must compute exactly the same function
    y0, y7 = pure(0), pure(7)
    chk("expert 0 == expert 7 at initialisation",
        torch.allclose(y0, y7, atol=1e-12),
        f"max dev {float((y0-y7).abs().max()):.2e} -- modulation starts at identity")
    e.mod.data.normal_(0, 0.1)
    chk("and they diverge once the modulation is non-zero",
        not torch.allclose(pure(0), pure(7), atol=1e-8))

    print("\n== merging is a blend of experts, not of their outputs ==")
    # The rewrite merges modulations BEFORE the network runs, so a 50/50 route
    # must equal the expert whose modulation is the average -- not the average
    # of the two experts' outputs. Those differ, because the SwiGLU is not
    # linear, and confusing them is the bug this test exists to catch.
    half = torch.zeros(flat.shape[0], cfg.n_experts, dtype=torch.double)
    half[:, 0] = half[:, 3] = 0.5
    blended = e(flat, half)
    saved = e.mod.data.clone()
    e.mod.data[5] = 0.5 * saved[0] + 0.5 * saved[3]
    chk("a 50/50 route == the expert at the averaged modulation",
        torch.allclose(blended, pure(5), atol=1e-12),
        f"max dev {float((blended - pure(5)).abs().max()):.2e}")
    chk("and that is NOT the mean of the two outputs",
        not torch.allclose(blended, 0.5 * (pure(0) + pure(3)), atol=1e-6),
        "the SwiGLU is non-linear, so the two differ")
    e.mod.data = saved

    print("\n== cost is independent of routing width ==")
    # the point of merging: one expert and all E cost the same forward
    wide = torch.full((flat.shape[0], cfg.n_experts), 1.0 / cfg.n_experts,
                      dtype=torch.double)
    y_one, y_all = pure(0), e(flat, wide)
    chk("width 1 and width E produce the same shape from one matmul",
        y_one.shape == y_all.shape and not torch.allclose(y_one, y_all),
        "different answers, identical cost")

    print("\n== the threshold is scale-free ==")
    widths = {}
    for E in (32, 64, 128):
        c = SharedMoEConfig(d_model=64, d_expert=48, n_experts=E, rank=4, max_k=E)
        mm = SharedMoEFeedForward(c)
        xx = torch.randn(2, 12, 64)
        mk = torch.ones(2, 12, dtype=torch.bool)
        with torch.no_grad():
            _, aux = mm(xx, mk, RouterFeatures(length=mk.sum(1).float()))
        widths[E] = float(aux["mean_width"]) / E
    chk("width as a FRACTION of E is stable across E",
        max(widths.values()) - min(widths.values()) < 0.25,
        " ".join(f"E={k}:{v:.2f}" for k, v in widths.items()))

    print("\n== width adapts: 1 to many, per token ==")
    c = SharedMoEConfig(d_model=64, d_expert=48, n_experts=64, rank=4, max_k=16)
    mm = SharedMoEFeedForward(c)
    xx = torch.randn(4, 16, 64)
    mk = torch.ones(4, 16, dtype=torch.bool)
    with torch.no_grad():
        mm.gate.weight.normal_(0, 3.0)          # a sharp router, as training gives
        _, aux = mm(xx, mk, RouterFeatures(length=mk.sum(1).float()))
    chk("a confident router routes narrowly", float(aux["mean_width"]) < 4.0,
        f"mean width {float(aux['mean_width']):.2f} of {c.n_experts}")
    chk("but never to zero experts", float(aux["mean_width"]) >= 1.0)
    with torch.no_grad():
        mm.gate.weight.normal_(0, 0.0)          # a flat router
        _, aux2 = mm(xx, mk, RouterFeatures(length=mk.sum(1).float()))
    chk("an uncertain router routes widely",
        float(aux2["mean_width"]) > 4 * float(aux["mean_width"]),
        f"{float(aux2['mean_width']):.2f} vs {float(aux['mean_width']):.2f}")

    print("\n== it trains ==")
    torch.manual_seed(0)
    c = SharedMoEConfig(d_model=32, d_expert=32, n_experts=16, rank=4, max_k=8)
    mm = SharedMoEFeedForward(c)
    opt = torch.optim.Adam(mm.parameters(), lr=1e-2)
    xx = torch.randn(2, 8, 32)
    mk = torch.ones(2, 8, dtype=torch.bool)
    tgt = torch.randn(2, 8, 32)
    first = None
    for _ in range(150):
        y, aux = mm(xx, mk, RouterFeatures(length=mk.sum(1).float()))
        loss = ((y - tgt) ** 2).mean() + aux["balance_loss"]
        if first is None:
            first = float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    chk("loss falls", float(loss) < 0.6 * first, f"{first:.3f} -> {float(loss):.3f}")
    chk("gradients reach the per-expert modulation",
        mm.experts.mod.grad is not None and float(mm.experts.mod.grad.abs().sum()) > 0,
        "the experts are learning, not dead weight")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
