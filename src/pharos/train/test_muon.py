#!/usr/bin/env python3
"""Tests for the optimizer that trains the production model.

There were none. Muon owns 222 of shared400's tensors and 343M of its
parameters, it was chosen over AdamW on a measured A/B, and
`verify_claims.py` gated twenty test modules of which none touched it.

What these pin:

  ORTHOGONALITY   the Newton-Schulz iteration drives singular values into
                  [~0.7, ~1.3] in five steps. If it does not, the update is
                  not orthogonal and Muon is momentum SGD with extra matmuls.
  SCALE           the iteration normalises first, so NS(c*g) == NS(g). This
                  is what makes the non-standard momentum accumulation here
                  -- `buf = m*buf + g` rather than `buf = m*buf + (1-m)*g` --
                  immaterial rather than a 20x learning-rate error at
                  momentum 0.95.
  STACKS          a `(E, d, f)` expert tensor is E matrices, not one. It was
                  reshaped to `(E, d*f)` and orthogonalised ACROSS the expert
                  axis, which makes the experts mutually orthogonal instead
                  of making each expert's matrix orthogonal. Every
                  independent-expert config -- mini, small, base400 -- hit
                  it; shared400, which stage 1 runs, has only 2-D tensors in
                  Muon and did not.
  THE SPLIT       no embedding reaches Muon, because the MLM head is tied to
                  the token embedding and orthogonalising a vocabulary
                  interface is not what the spectral argument licenses.

Run: python3 src/pharos/train/test_muon.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pharos.train.muon import Muon, newton_schulz, muon_param_groups  # noqa: E402

fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} {detail}")
    if not ok:
        fails.append(name)


def main() -> int:
    torch.manual_seed(0)

    print("== the iteration produces an approximately orthogonal matrix ==")
    # ...for a WELL-CONDITIONED input. Five steps cannot rescue a square
    # random Gaussian, whose smallest singular value sits at the
    # Marchenko-Pastur edge near zero: measured, a 768x768 random matrix goes
    # from condition number 5,458 to 105, which is a 52x improvement and is
    # not [0.7, 1.3]. The module comment claimed the interval without
    # qualification, and 31% of shared400's Muon tensors are square.
    #
    # So the claim is split into the two things that are separately true: the
    # spectrum is always COMPRESSED and always bounded above, and it reaches
    # the stated interval when the input is not near-singular.
    for shape in ((768, 2304), (2304, 768), (64, 16), (16, 64)):
        g = torch.randn(*shape)
        sv = torch.linalg.svdvals(newton_schulz(g, 5).float())
        chk(f"{str(shape):12s} well-conditioned -> [0.6, 1.4]",
            0.6 <= float(sv.min()) and float(sv.max()) <= 1.4,
            f"[{float(sv.min()):.3f}, {float(sv.max()):.3f}]")
    for shape in ((512, 512), (768, 768)):
        g = torch.randn(*shape)
        si = torch.linalg.svdvals(g)
        so = torch.linalg.svdvals(newton_schulz(g, 5).float())
        ci, co = float(si.max() / si.min()), float(so.max() / so.min())
        chk(f"{str(shape):12s} square -> compressed, bounded above",
            co < ci / 10 and float(so.max()) <= 1.4,
            f"condition {ci:,.0f} -> {co:,.0f}, max {float(so.max()):.3f} "
            f"-- NOT [0.7, 1.3], and five steps cannot make it so")

    print("\n== and it is scale-invariant, which is why the momentum "
          "normalisation does not matter ==")
    g = torch.randn(128, 256)
    base = newton_schulz(g, 5).float()
    for c in (1e-3, 0.5, 20.0, 1e3):
        d = float((newton_schulz(g * c, 5).float() - base).abs().max())
        chk(f"NS({c:g} * g) == NS(g)", d < 2e-2, f"max delta {d:.2e}")

    print("\n== a stacked tensor is E matrices, not one ==")
    g3 = torch.randn(4, 64, 96)
    batched = newton_schulz(g3, 5)
    per = torch.stack([newton_schulz(g3[i], 5) for i in range(4)])
    chk("batched equals slice-by-slice",
        float((batched - per).abs().max()) == 0.0,
        f"max delta {float((batched - per).abs().max()):.2e}")
    svs = [torch.linalg.svdvals(batched[i].float()) for i in range(4)]
    chk("EVERY slice is orthogonalised, not the stack",
        all(0.6 <= float(s.min()) and float(s.max()) <= 1.4 for s in svs),
        "  ".join(f"[{float(s.min()):.2f},{float(s.max()):.2f}]" for s in svs))
    # the old behaviour, kept here as the thing that must NOT happen: flatten
    # to (E, d*f) and the SLICES are no longer individually orthogonal
    flat = newton_schulz(g3.reshape(4, -1), 5).view_as(g3)
    s0 = torch.linalg.svdvals(flat[0].float())
    chk("and flattening to (E, d*f) would not have been that",
        not (0.6 <= float(s0.min()) and float(s0.max()) <= 1.4),
        f"flattened slice 0 spans [{float(s0.min()):.3f}, {float(s0.max()):.3f}]")

    print("\n== a step moves downhill on a quadratic ==")
    torch.manual_seed(1)
    w = torch.nn.Parameter(torch.randn(32, 32))
    target = torch.randn(32, 32)
    opt = Muon([w], lr=0.05)
    first = float(((w - target) ** 2).sum())
    for _ in range(50):
        opt.zero_grad()
        ((w - target) ** 2).sum().backward()
        opt.step()
    last = float(((w - target) ** 2).sum())
    # 50 orthogonal steps at lr 0.05 move each direction by about 2.5, against
    # a starting distance of ~32 per direction, so a 0.5x threshold was a
    # statement about the step budget and not about the optimizer.
    chk("loss falls", last < first * 0.75, f"{first:.2f} -> {last:.2f}")

    print("\n== and it steps a stacked tensor too ==")
    torch.manual_seed(2)
    w3 = torch.nn.Parameter(torch.randn(8, 16, 16))
    tg3 = torch.randn(8, 16, 16)
    opt3 = Muon([w3], lr=0.05)
    f3 = float(((w3 - tg3) ** 2).sum())
    for _ in range(50):
        opt3.zero_grad()
        ((w3 - tg3) ** 2).sum().backward()
        opt3.step()
    l3 = float(((w3 - tg3) ** 2).sum())
    chk("loss falls on a (E, d, f) parameter", l3 < f3 * 0.5,
        f"{f3:.2f} -> {l3:.2f}")

    print("\n== the split sends the right tensors to the right optimiser ==")
    from pharos.model.pharos import Pharos, PharosConfig
    cfg = PharosConfig(d_model=64, n_blocks=4, n_loops=1, n_heads=4, d_pair=32,
                       n_experts=4, d_expert=32, top_k=2, n_shared=1,
                       max_length=128)
    m = Pharos(cfg)
    muon, groups = muon_param_groups(m)
    byid = {id(p): n for n, p in m.named_parameters()}
    names = [byid[id(p)] for p in muon]
    chk("every Muon tensor is at least 2-D",
        all(p.ndim >= 2 for p in muon), f"{len(muon)} tensors")
    chk("no embedding reaches Muon",
        not [n for n in names if "embed" in n or n.endswith("rel.weight")],
        "the MLM head is tied to the token embedding")
    chk("no parameter is in both halves",
        not ({id(p) for p in muon}
             & {id(p) for g in groups for p in g["params"]}), "")
    allp = {id(p) for p in m.parameters() if p.requires_grad}
    covered = {id(p) for p in muon} | {id(p) for g in groups for p in g["params"]}
    chk("and every trainable parameter is in one of them",
        allp == covered, f"{len(allp)} trainable, {len(covered)} assigned")
    chk("the no-decay group really has zero decay",
        any(g["weight_decay"] == 0.0 for g in groups),
        "norms, biases and the per-expert modulation")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
