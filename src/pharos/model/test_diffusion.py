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

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pharos.model.diffusion import (BOND_C4_N, BOND_P_P,           # noqa: E402
                                    DiffusionConfig, DiffusionStructureHead,
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
    # The EDM TERM, not `["loss"]`. The flat-bottomed geometry penalty was
    # added to `["loss"]` after this test was written, and on `cw` -- which is
    # Gaussian noise scaled by sigma_data, not a backbone -- it reports 62 A of
    # bond error and 86% of the total. The test then failed at 7.237 and looked
    # like a regression in the divisor it was written to guard, which was
    # intact at 0.994 the whole time. Subtracting the violation back out keeps
    # this assertion on the quantity its own comment describes.
    runs = [hw.loss(cw, sw, None, mw,
                    generator=torch.Generator().manual_seed(k))
            for k in range(24)]
    vals = [float(r["loss"]) - cfgw.violation_weight * float(r["violation"])
            for r in runs]
    mean = sum(vals) / len(vals)
    chk("the weighted EDM loss sits at ~1.0 at initialisation", 0.5 < mean < 2.0,
        f"mean {mean:.3f} over 24 draws (9.0 would be the residue-divisor bug)")

    # And the geometry term's own validity check, which it never had: on a
    # chain that IS bonded correctly it must be a minority of the objective.
    # A violation weight that dominates would spend head 3's gradient making
    # the backbone self-consistent instead of making it right, and nothing
    # downstream would say so -- `["loss"]` is one number and `TM-score` never
    # asks whether the chain is bonded.
    ch = torch.zeros(2, 32, N_ATOM, 3)
    step = torch.tensor([BOND_P_P[0], 0.0, 0.0])
    for i in range(32):
        ch[:, i, 0] = step * i                       # P, spaced at 6.01 A
        ch[:, i, 1] = step * i + torch.tensor([0.0, 4.0, 0.0])          # C4'
        ch[:, i, 2] = step * i + torch.tensor([0.0, 4.0 + BOND_C4_N[0], 0.0])  # N
    mc = torch.ones(2, 32, dtype=torch.bool)
    sc = torch.zeros(2, 32, 64)
    rc = [hw.loss(ch, sc, None, mc,
                  generator=torch.Generator().manual_seed(k)) for k in range(12)]
    share = (sum(cfgw.violation_weight * float(r["violation"]) for r in rc)
             / max(sum(float(r["loss"]) for r in rc), 1e-9))
    torch.set_rng_state(_state)
    chk("on a correctly bonded chain the geometry term is a minority term",
        share < 0.5,
        f"{share:.1%} of the objective at weight {cfgw.violation_weight}")

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

    print("\n== the training noise distribution is TIED to sigma_data ==")
    # The defect this exists to stop: `sigma_data` was correctly raised from
    # the image default 0.5 to 16.0 for angstrom coordinates, and `p_mean`
    # was left at Karras's -1.2, which is the value FOR sigma_data = 0.5.
    # The median training draw became sigma = 0.301 against data of scale 16,
    # so `c_skip = sd^2/(s^2 + sd^2)` was 0.99965 and the network was asked
    # for 0.035% of the task. It learned to polish and never learned to build.
    import math as _math
    from pharos.model.diffusion import DiffusionConfig as _DC

    def _cskip(sg, sd):
        return sd ** 2 / (sg ** 2 + sd ** 2)

    k = _DC(sigma_data=0.5)
    chk("at sigma_data = 0.5 the derivation reproduces Karras's -1.2",
        abs(k.p_mean - (-1.2)) < 0.01, f"p_mean {k.p_mean:+.4f}")
    for sd_ in (0.5, 4.0, 16.0, 64.0):
        c_ = _DC(sigma_data=sd_)
        ratio = _math.exp(c_.p_mean) / sd_
        chk(f"sigma_data {sd_:5.1f}: median sigma/sigma_data is scale-free",
            abs(ratio - 0.6016) < 1e-3,
            f"p_mean {c_.p_mean:+.4f}, median sigma {_math.exp(c_.p_mean):8.3f}, "
            f"ratio {ratio:.4f}")
    live = _DC()
    chk("the live config's median draw asks the network for real work",
        0.5 < _cskip(_math.exp(live.p_mean), live.sigma_data) < 0.85,
        f"c_skip at the median draw {_cskip(_math.exp(live.p_mean), live.sigma_data):.5f} "
        f"-- it was 0.99965 before p_mean was derived")
    # and the bad configuration must FAIL this, or the test proves nothing
    bad = _DC(sigma_data=16.0, p_mean=-1.2)
    chk("the configuration that shipped would FAIL that check",
        _cskip(_math.exp(bad.p_mean), bad.sigma_data) > 0.99,
        f"c_skip {_cskip(_math.exp(bad.p_mean), bad.sigma_data):.5f} at "
        f"p_mean = -1.2, sigma_data = 16")
    chk("an explicit p_mean is still honoured, for the ablation",
        bad.p_mean == -1.2)
    # the sampler must start where training actually has mass
    rng_ = np.random.default_rng(0)
    draws = np.exp(rng_.normal(live.p_mean, live.p_std, 200_000)).clip(
        live.sigma_min, live.sigma_max)
    above = float((draws > live.sigma_data).mean())
    chk("a third of training draws sit above sigma_data, where the fold is built",
        0.2 < above < 0.5, f"{above:.1%} above sigma_data = {live.sigma_data}")
    chk("and the sampler starts inside the trained range, not past it",
        live.sigma_max <= float(np.percentile(draws, 99.99)) * 3.0,
        f"sigma_max {live.sigma_max} against a 99.99th percentile draw of "
        f"{float(np.percentile(draws, 99.99)):.1f}")

    print("\n== the INFERENCE sampler must use the pair features ==")
    # `predict_structure._sample_with` is a second implementation of
    # `head.sample`, written to control the noise seed, and it hardcoded
    # `pair=None`. The decoder is trained with `diff_pair(hidden, coev)`,
    # whose reason for existing is the relative-position embedding, so
    # sampling without it asked the model to build a chain without being
    # told which residues are adjacent -- and all seventeen RNA-Puzzles
    # predictions came back with consecutive P at 16-19 A against 5.95.
    #
    # The direct test: with the SAME noise, a sampler that threads `pair`
    # gives a different answer with it than without it. One that ignores
    # the argument gives the same answer, which is the bug.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from predict_structure import _sample_with                # noqa: E402
    from pharos.model.diffusion import DiffusionPairFeatures   # noqa: E402

    _B, _L, _dm = 1, 14, 32
    _head = DiffusionStructureHead(DiffusionConfig(
        d_model=_dm, d_pair=16, n_layers=2, n_heads=2))
    _dp = DiffusionPairFeatures(_dm, 16)
    torch.manual_seed(0)
    # OPEN THE ZERO-INITIALISED GATES FIRST, the same way property 7f opens
    # `bias_scale`. `DenoiseBlock.out.weight` and `CoordDenoiser.out.weight`
    # are zero by design -- a deep denoiser starts as a shallow one -- so on
    # a freshly built head the attention output is exactly zero and `pair`
    # provably cannot matter. Testing an untrained head would have reported
    # "the sampler ignores pair" for a sampler that does not, which is the
    # false positive this comment exists to stop being rediscovered.
    with torch.no_grad():
        for _m in _head.modules():
            if isinstance(_m, torch.nn.Linear) and float(_m.weight.abs().max()) == 0:
                _m.weight.normal_(0.0, 0.3)
    _single = torch.randn(_B, _L, _dm)
    _mask = torch.ones(_B, _L, dtype=torch.bool)
    with torch.no_grad():
        _pair = _dp(_single, None)
    _noise = torch.randn(_B, _L, 3, 3)
    with torch.no_grad():
        _with = _sample_with(_head, _single, _pair, _mask, 6, _noise)
        _without = _sample_with(_head, _single, None, _mask, 6, _noise)
    _d = float((_with - _without).abs().max())
    chk("the same noise with and without pair gives DIFFERENT structures",
        _d > 1e-5, f"max |delta| {_d:.4e} -- zero would mean the sampler "
        f"drops the argument, which is how it shipped (gates opened first, "
        f"or a zero-init head makes this vacuous)")
    # INSIDE no_grad, like the two calls above. Outside it the autograd
    # path picks different kernels and the result differs in the last bits,
    # so the first draft of this check reported the sampler as
    # non-deterministic when the sampler is bit-exact -- verified
    # separately: two identical calls give `torch.equal` True.
    with torch.no_grad():
        _again = _sample_with(_head, _single, _pair, _mask, 6, _noise)
    chk("and it is deterministic given the noise",
        bool(torch.equal(_with, _again)),
        "otherwise the comparison above is meaningless")

    print("\n== and the pair features carry ADJACENCY, which is the point ==")
    # If the relative-position embedding did not distinguish |i-j| = 1 from
    # |i-j| = 7 there would be nothing for the backbone to read.
    with torch.no_grad():
        _pp = _dp(_single, None)
    _adj = _pp[0, torch.arange(_L - 1), torch.arange(1, _L)].mean(0)
    _far = _pp[0, torch.arange(_L - 7), torch.arange(7, _L)].mean(0)
    chk("adjacent pairs embed differently from |i-j| = 7",
        float((_adj - _far).abs().max()) > 1e-3,
        f"max |delta| {float((_adj - _far).abs().max()):.4e}")
    # and that it is the REL table doing it, not the single projections
    _dp2 = DiffusionPairFeatures(_dm, 16)
    with torch.no_grad():
        _dp2.rel.weight.zero_()
        _q = _dp2(_single, None)
    _adj2 = _q[0, torch.arange(_L - 1), torch.arange(1, _L)].mean(0)
    _far2 = _q[0, torch.arange(_L - 7), torch.arange(7, _L)].mean(0)
    chk("with the rel table zeroed the two collapse together",
        float((_adj2 - _far2).abs().max()) < float((_adj - _far).abs().max()),
        f"{float((_adj2 - _far2).abs().max()):.4e} against "
        f"{float((_adj - _far).abs().max()):.4e} -- so it IS the relative "
        f"position embedding that carries adjacency")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
