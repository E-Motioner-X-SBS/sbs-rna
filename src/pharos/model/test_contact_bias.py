#!/usr/bin/env python3
"""Finding 92's mechanism: a pairing bias the trunk's attention can use.

The headline assertion is not a property of `ContactBias` but a
comparison of SHAPES: an outer sum `A h_i + B h_j` cannot represent
complementarity, and a bilinear form `<U h_i, V h_j>` can. Base pairing
is complementarity, so the shape the pair track uses is the wrong one
for it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pharos.model.pharos import ContactBias, PharosConfig, Pharos  # noqa: E402

fails: list[str] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<54} {detail}")
    if not ok:
        fails.append(name)


def _fit(model: nn.Module, h: torch.Tensor, target: torch.Tensor,
         steps: int = 1500) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(model(h)[0], target).backward()
        opt.step()
    with torch.no_grad():
        return float(((model(h)[0] > 0).float() == target).float().mean())


def main() -> int:
    torch.manual_seed(0)
    L, d = 24, 16

    print("== a sum cannot represent complementarity; a product can ==")
    # toy base pairing: i pairs with j iff type_i + type_j == 3, which is
    # exactly the A-U / C-G shape -- a statement about the two TOGETHER
    types = torch.randint(0, 4, (L,))
    target = ((types[:, None] + types[None, :]) == 3).float()
    h = torch.cat([torch.eye(4)[types], torch.zeros(L, d - 4)], 1)[None]
    base = 1.0 - float(target.mean())

    class Sum(nn.Module):               # DiffusionPairFeatures' shape
        def __init__(s):
            super().__init__()
            s.a, s.b = nn.Linear(d, 1), nn.Linear(d, 1)

        def forward(s, x):
            return s.a(x)[:, :, 0, None] + s.b(x)[:, None, :, 0]

    class Bil(nn.Module):               # ContactBias' shape
        def __init__(s, r=8):
            super().__init__()
            s.u = nn.Linear(d, r, bias=False)
            s.v = nn.Linear(d, r, bias=False)

        def forward(s, x):
            return torch.einsum("bir,bjr->bij", s.u(x), s.v(x))

    a_sum, a_bil = _fit(Sum(), h, target), _fit(Bil(), h, target)
    chk("the outer sum cannot even beat the always-zero baseline",
        a_sum <= base + 0.02,
        f"accuracy {a_sum:.3f} against a base rate of {base:.3f} -- it is "
        f"the shape `DiffusionPairFeatures` uses for the pair track")
    chk("the bilinear form fits it exactly",
        a_bil > 0.98,
        f"accuracy {a_bil:.3f}; 'i pairs with j' is about the two together, "
        f"and only a product can say that")

    print("\n== and the module keeps the register's discipline ==")
    m = ContactBias(32, rank=8).eval()
    hh = torch.randn(1, 12, 32)
    mk = torch.ones(1, 12, dtype=torch.bool)
    with torch.no_grad():
        z = m(hh, mk)
    chk("zero at init, so no checkpoint's forward pass moves",
        bool((z == 0).all()), f"shape {tuple(z.shape)}")

    m.train(); m.zero_grad()
    m(hh, mk).sum().backward()
    chk("and the gate takes gradient there",
        float(m.scale.grad.abs().max()) > 0,
        f"|grad scale| {float(m.scale.grad.abs().max()):.3e}")

    with torch.no_grad():
        m.eval(); m.scale.fill_(1.0)
        mk2 = torch.zeros(1, 12, dtype=torch.bool); mk2[0, :8] = True
        z2 = m(hh, mk2)
    chk("padding is zeroed on both axes",
        float(z2[0, 8:, :].abs().max()) == 0.0
        and float(z2[0, :, 8:].abs().max()) == 0.0,
        "a bias on a pad would steer attention toward nothing")

    print("\n== wired behind a config flag, off by default ==")
    chk("shared400 leaves it off", PharosConfig.shared400().contact_bias is False)
    with torch.device("meta"):
        c = PharosConfig.shared400(); c.contact_bias = True
        on = Pharos(c).param_counts()["total"]
        off = Pharos(PharosConfig.shared400()).param_counts()["total"]
    chk("and it costs a bilinear pair of projections",
        on - off == 50_689,
        f"+{on - off:,} parameters for rank {PharosConfig.shared400().contact_bias_rank}")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
