#!/usr/bin/env python3
"""The diffusion head must do the things the coordinate MLP could not.

Shape tests prove nothing here. What matters is: does it actually learn a
structure, is it insensitive to the global frame, and does it represent more
than one mode where the data has more than one?

Run: python3 src/pharos/model/test_diffusion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pharos.model.diffusion import (DiffusionConfig, DiffusionStructureHead,  # noqa: E402
                                    N_ATOM, random_rigid)

fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} {detail}")
    if not ok:
        fails.append(name)


def rmsd(a, b, mask):
    m = mask[..., None, None].to(a.dtype)
    return (((a - b) ** 2 * m).sum() / m.sum().clamp(min=1)).sqrt().item()


def main() -> int:
    torch.manual_seed(0)
    cfg = DiffusionConfig(d_model=64, d_pair=16, n_layers=2, n_heads=4,
                          sigma_data=4.0, n_steps=24)
    head = DiffusionStructureHead(cfg).double()
    B, L = 2, 12
    mask = torch.ones(B, L, dtype=torch.bool)
    single = torch.randn(B, L, cfg.d_model, dtype=torch.double)
    pair = torch.randn(B, L, L, cfg.d_pair, dtype=torch.double)
    coords = torch.randn(B, L, N_ATOM, 3, dtype=torch.double) * 4.0

    print("== EDM preconditioning is correct at both ends ==")
    # at sigma -> 0 the model must be the identity; at sigma -> inf, c_skip -> 0
    for s, want_skip in [(1e-3, 1.0), (1e4, 0.0)]:
        sk, co, ci, cn = head._c(torch.tensor([s], dtype=torch.double))
        chk(f"c_skip at sigma={s:g}", abs(sk.item() - want_skip) < 1e-3,
            f"{sk.item():.6f}")
    sk, co, ci, cn = head._c(torch.tensor([cfg.sigma_data], dtype=torch.double))
    chk("c_in normalises variance at sigma=sigma_data",
        abs(ci.item() - 1 / (2 ** 0.5 * cfg.sigma_data)) < 1e-9)

    print("\n== the denoiser starts as the identity (zero-init output) ==")
    x = torch.randn(B, L, N_ATOM, 3, dtype=torch.double)
    sig = torch.full((B,), 1.0, dtype=torch.double)
    d0 = head.denoise(x, sig, single, pair, mask)
    sk, _, _, _ = head._c(sig)
    chk("denoise == c_skip * x before training",
        torch.allclose(d0, sk.view(-1, 1, 1, 1) * x, atol=1e-10),
        f"max dev {(d0 - sk.view(-1,1,1,1)*x).abs().max():.2e}")

    print("\n== random_rigid removes the global frame ==")
    a = random_rigid(coords, mask)
    m = mask[..., None, None].double()
    chk("output is centred", float(((a * m).sum((1, 2)) / m.sum((1, 2))).abs().max()) < 1e-9)
    # A rotation preserves pairwise distances -- but so does a REFLECTION, so
    # distance alone is not enough. Chirality is the property that separates
    # them, and RNA is chiral, so the determinant is the test that matters.
    def pdist(x):
        f = x.reshape(x.shape[0], -1, 3)
        return ((f[:, :, None, :] - f[:, None, :, :]) ** 2).sum(-1).sqrt()
    ref = coords - coords.reshape(B, -1, 3).mean(1)[:, None, None, :]
    chk("pairwise distances are preserved",
        float((pdist(ref) - pdist(a)).abs().max()) < 1e-9,
        f"max dev {float((pdist(ref) - pdist(a)).abs().max()):.2e}")
    dets = []
    for _ in range(16):
        r = random_rigid(coords, mask)
        for bi in range(B):
            X = (coords - coords.reshape(B, -1, 3).mean(1)[:, None, None, :])[bi].reshape(-1, 3)
            R = torch.linalg.lstsq(X, r[bi].reshape(-1, 3)).solution
            dets.append(float(torch.det(R)))
    chk("every augmentation is a ROTATION, not a reflection",
        all(d > 0.99 for d in dets),
        f"min det {min(dets):.4f} over {len(dets)} draws -- reflections mirror chirality")
    b = random_rigid(coords, mask)
    chk("and it really is random (two draws differ)",
        not torch.allclose(a, b, atol=1e-6))

    print("\n== the loss is finite and sigma-weighted across the whole range ==")
    g = torch.Generator().manual_seed(1)
    out = head.loss(coords, single, pair, mask, generator=g)
    chk("loss is finite", torch.isfinite(out["loss"]).item(),
        f"{out['loss'].item():.4f}")
    chk("it reports the noise level it sampled", out["sigma"].item() > 0)

    print("\n== it can actually fit a structure ==")
    torch.manual_seed(0)
    head = DiffusionStructureHead(cfg).double()
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    target = torch.randn(1, L, N_ATOM, 3, dtype=torch.double) * 4.0
    s1 = torch.randn(1, L, cfg.d_model, dtype=torch.double)
    p1 = torch.randn(1, L, L, cfg.d_pair, dtype=torch.double)
    m1 = torch.ones(1, L, dtype=torch.bool)
    # Evaluate at a FIXED set of noise levels before and after. Comparing the
    # training MSE of step 0 against step 400 compares two different random
    # sigmas, and at a small sigma the MSE is tiny regardless of skill -- the
    # first version of this test "failed" on a model that was learning fine.
    # Per noise level, not averaged. At sigma=32 with sigma_data=4 the
    # preconditioner gives c_skip ~ 0.015: almost none of the signal survives
    # and the best possible prediction is the data mean, so that term is
    # irreducible. Averaging it in hides real improvement at the levels where
    # denoising is actually possible, which is what a single averaged number
    # did on the first attempt at this test.
    def fixed_eval(h):
        out = {}
        for s in (0.5, 2.0, 8.0, 32.0):
            gg = torch.Generator().manual_seed(11)
            x0 = target - target.reshape(1, -1, 3).mean(1)[:, None, None, :]
            sig = torch.full((1,), s, dtype=torch.double)
            noise = torch.randn(x0.shape, dtype=torch.double, generator=gg)
            with torch.no_grad():
                pred = h.denoise(x0 + s * noise, sig, s1, p1, m1)
            out[s] = float(((pred - x0) ** 2).mean())
        return out

    # Train WITHOUT the rotation augmentation. With one example and a fresh
    # random frame every step the target is a different structure each time, so
    # this would measure how fast a 2-layer net learns rotation equivariance,
    # not whether it can denoise. Equivariance is tested above, on its own.
    # Train across the whole sigma range with augment=False. Training at one
    # noise level and evaluating at four measures overfitting to that level:
    # the first version of this test did exactly that and the sigma=32 term
    # dominated the average.
    before = fixed_eval(head)
    g = torch.Generator().manual_seed(2)
    for _ in range(1200):
        out = head.loss(target, s1, p1, m1, generator=g, augment=False)
        opt.zero_grad(); out["loss"].backward(); opt.step()
    after = fixed_eval(head)
    # The achievable improvement is NOT uniform across sigma, and a flat
    # threshold pretends it is. c_skip = sigma_data^2/(sigma^2+sigma_data^2) is
    # 0.985 at sigma=0.5, so an untrained model already returns almost exactly
    # x0 and there is nearly nothing to win; at sigma=8 c_skip is 0.2 and the
    # network has to supply four fifths of the answer. The thresholds follow the
    # preconditioner rather than a round number.
    for lvl, factor, why in ((0.5, 1.0, "c_skip 0.985, almost no headroom"),
                             (2.0, 0.85, "c_skip 0.80"),
                             (8.0, 0.60, "c_skip 0.20, most of the work")):
        chk(f"denoising improves at sigma={lvl}", after[lvl] < factor * before[lvl],
            f"{before[lvl]:.3f} -> {after[lvl]:.3f}  ({why})")
    chk("and sigma=32 stays near the data variance, as it must",
        after[32.0] > after[8.0],
        f"sigma=32 {after[32.0]:.2f} (irreducible) vs sigma=8 {after[8.0]:.2f}")

    print("\n== EDM's weighting is calibrated, and a wrong divisor breaks it ==")
    # At initialisation `out` is zero-init, so D(x) = c_skip*x exactly, and the
    # sigma-weighted loss is ~1.0 BY CONSTRUCTION at every noise scale -- that
    # is the entire point of the weighting. Any constant offset is a
    # normalisation error. This test exists because the divisor counted
    # RESIDUES while the numerator summed residues x atoms x 3, so the loss
    # read 8.98 instead of 1.0: harmless to training, where a constant is
    # absorbed by the learning rate, but it would have given head 3 nine times
    # its intended weight against every other head in the stage-5 objective.
    # Save and restore the global RNG. Module init and `torch.randn` both draw
    # from it, so a block added here silently re-seeds every test after it --
    # which is what happened when this one was written: the sampling test below
    # went from 6.17 to 15.16 RMSD without a line of its own changing.
    _state = torch.get_rng_state()
    cfgw = DiffusionConfig(d_model=64, n_layers=2, n_heads=4)
    hw = DiffusionStructureHead(cfgw)
    gw = torch.Generator().manual_seed(4242)
    cw = torch.randn(4, 40, N_ATOM, 3, generator=gw) * cfgw.sigma_data
    mw = torch.ones(4, 40, dtype=torch.bool)
    sw = torch.zeros(4, 40, 64)
    vals = [float(hw.loss(cw, sw, None, mw,
                          generator=torch.Generator().manual_seed(k))["loss"])
            for k in range(24)]
    mean = sum(vals) / len(vals)
    torch.set_rng_state(_state)
    chk("the weighted loss sits at ~1.0 at initialisation", 0.5 < mean < 2.0,
        f"mean {mean:.3f} over 24 draws (9.0 would be the residue-divisor bug)")

    print("\n== sampling produces a structure, not noise ==")
    g = torch.Generator().manual_seed(3)
    s = head.sample(s1, p1, m1, n_steps=32, generator=g)
    chk("sample has the right shape", tuple(s.shape) == (1, L, N_ATOM, 3), str(tuple(s.shape)))
    chk("sample is finite", torch.isfinite(s).all().item())
    tgt_c = random_rigid(target, m1)
    chk("sample is nearer the target than pure noise",
        rmsd(s, tgt_c, m1) < rmsd(torch.randn_like(s) * cfg.sigma_data, tgt_c, m1),
        f"rmsd {rmsd(s, tgt_c, m1):.2f} vs noise {rmsd(torch.randn_like(s)*cfg.sigma_data, tgt_c, m1):.2f}")

    print("\n== two samples differ: it is a distribution, not a point estimate ==")
    s2 = head.sample(s1, p1, m1, n_steps=32, generator=torch.Generator().manual_seed(9))
    chk("different seeds give different structures",
        rmsd(s, s2, m1) > 1e-3, f"rmsd between samples {rmsd(s, s2, m1):.3f}")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
