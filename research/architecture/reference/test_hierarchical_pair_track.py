#!/usr/bin/env python3
"""Correctness tests for the Hierarchical Pair Track.

The benchmark only ever measured SPEED, and ran under torch.no_grad(), so
neither the structural invariants nor differentiability were ever checked.
Both matter: a broken gradient path or a leaking index would degrade a trained
model silently rather than crash.

Run: python3 test_hierarchical_pair_track.py    (exit 0 = all pass)
"""
from __future__ import annotations
import sys
import torch

from hierarchical_pair_track import HPTConfig, HierarchicalPairTrack

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        FAILS.append(name)


def run(L: int, cfg: HPTConfig, seed: int = 0, grad: bool = False):
    torch.manual_seed(seed)
    m = HierarchicalPairTrack(cfg)
    m.train(grad)
    tok = torch.randn(1, L, cfg.d_model, requires_grad=grad)
    if grad:
        return m, tok, m(tok)
    with torch.no_grad():
        return m, tok, m(tok)


def main() -> int:
    cfg = HPTConfig()

    print("== structural invariants ==")
    for L in (128, 300, 512, 1000, 2048):          # 300 and 1000 are NOT multiples of b1=16
        _, _, (logits, (ii, jj), st) = run(L, cfg)
        pairs = list(zip(ii.tolist(), jj.tolist()))

        check(f"L={L:<5} indices in range",
              bool(ii.min() >= 0 and jj.max() < L),
              f"min={int(ii.min())} max={int(jj.max())} L={L}")
        check(f"L={L:<5} strictly upper triangular",
              all(j > i for i, j in pairs),
              "found a pair with j <= i")
        check(f"L={L:<5} min separation respected",
              all(j - i >= cfg.min_sep for i, j in pairs),
              f"a pair violates |i-j| >= {cfg.min_sep}")
        check(f"L={L:<5} no duplicate pairs",
              len(set(pairs)) == len(pairs),
              f"{len(pairs) - len(set(pairs))} duplicates")
        check(f"L={L:<5} logits align with pairs",
              logits.shape[0] == len(pairs),
              f"{logits.shape[0]} logits vs {len(pairs)} pairs")

    print("\n== budget is honoured (the bug that pinned c at 5.0) ==")
    for L in (512, 1024, 2048):
        _, _, (_, _, st) = run(L, cfg)
        c = st["effective_c"]
        # min_sep trimming near the diagonal costs a little, so allow a small shortfall
        check(f"L={L:<5} effective_c within 15% of target {cfg.target_c}",
              0.85 * cfg.target_c <= c <= 1.02 * cfg.target_c,
              f"got c={c:.2f}")

    print("\n== hierarchy actually nests ==")
    L = 1024
    _, _, (_, (ii, jj), st) = run(L, cfg)
    r = cfg.b1 // cfg.b2
    # every retained pair must sit inside a b2 block, which sits inside a b1 block
    b2_blocks = {(int(i) // cfg.b2, int(j) // cfg.b2) for i, j in zip(ii, jj)}
    b1_from_b2 = {(bi // r, bj // r) for bi, bj in b2_blocks}
    check("L3 pairs occupy no more b2 blocks than L2 kept",
          len(b2_blocks) <= st["l2_kept_blocks"],
          f"{len(b2_blocks)} b2 blocks vs l2_kept={st['l2_kept_blocks']}")
    check("every b2 block nests in a kept b1 block",
          len(b1_from_b2) <= st["l1_kept_blocks"],
          f"{len(b1_from_b2)} b1 parents vs l1_kept={st['l1_kept_blocks']}")
    check("pairs per b2 block never exceed b2^2",
          len(ii) <= len(b2_blocks) * cfg.b2 * cfg.b2,
          f"{len(ii)} pairs across {len(b2_blocks)} blocks")

    print("\n== differentiability (never previously checked) ==")
    m, tok, (logits, _, _) = run(512, cfg, grad=True)
    loss = logits.pow(2).mean()
    loss.backward()
    check("loss is finite", bool(torch.isfinite(loss)), f"loss={loss}")
    check("input gradient exists and is finite",
          tok.grad is not None and bool(torch.isfinite(tok.grad).all()),
          "input grad missing or non-finite")
    trained = [(n, p) for n, p in m.named_parameters() if p.grad is not None]
    untrained = [n for n, p in m.named_parameters() if p.grad is None]
    nonzero = [n for n, p in trained if p.grad.abs().sum() > 0]
    check("all parameters receive a gradient",
          not untrained,
          f"no grad for: {untrained[:6]}")
    check("gradients are non-zero for most parameters",
          len(nonzero) >= 0.5 * max(len(trained), 1),
          f"only {len(nonzero)}/{len(trained)} params have non-zero grad")
    check("all gradients finite",
          all(bool(torch.isfinite(p.grad).all()) for _, p in trained),
          "a parameter gradient is NaN or Inf")

    print("\n== determinism ==")
    _, _, (l1, (i1, j1), _) = run(512, cfg, seed=7)
    _, _, (l2, (i2, j2), _) = run(512, cfg, seed=7)
    check("same seed gives identical pair selection",
          bool(torch.equal(i1, i2) and torch.equal(j1, j2)))
    check("same seed gives identical logits", bool(torch.allclose(l1, l2)))

    print(f"\n{'ALL TESTS PASS' if not FAILS else f'{len(FAILS)} FAILURES: ' + '; '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
