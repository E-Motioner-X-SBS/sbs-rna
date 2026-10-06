#!/usr/bin/env python3
"""Tests for the assembled model.

What these pin is not "does it run" but the handful of properties whose
violation would be invisible:

  1. SIZING          -- §5.4 quotes totals and active counts and calls the
                        arithmetic "verified exact". It never states the four
                        MoE numbers that determine them, so it could not be
                        checked. These tests pin what the built model actually
                        is, and record the gap.
  2. ONE-STEP GRAD   -- §5.2's memory claim is that loops cost compute and no
                        activation memory. That is true only if the early loops
                        run under no_grad; if they silently do not, the model
                        still trains and quietly uses N times the memory.
  3. THE LEAK        -- deep supervision must see DETACHED intermediate
                        iterates, or the recurrence is being backpropagated
                        through after all.
  4. RECYCLE 0       -- §5.3's circularity: at the first pass there is no
                        distance estimate and no pair-derived router feature,
                        and the implementation must represent that rather than
                        fabricate one.
  5. MASKING         -- padding must not change a real position's output, at
                        any length, through any of the three mixer types.

Run: python3 src/pharos/model/test_pharos.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Running this file as a script puts its own directory on sys.path, where
# `pharos.py` sits -- so a bare `import pharos` resolves to that MODULE and
# shadows the package, and the relative imports inside it then fail. Drop the
# script directory and put the source root first.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pharos.model.moe import RouterFeatures                       # noqa: E402
from pharos.model.pharos import Pharos, PharosConfig              # noqa: E402

fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} {detail}")
    if not ok:
        fails.append(name)


#: What the built family is, given the recipe in `PharosConfig`. Pinned so a
#: change to the MoE numbers cannot silently move the cost model.
#: Wiring §6.2 added `VirtualDistance`, exactly `d_model * 3` parameters --
#: 1,536 / 1,152 / 2,304 -- in BOTH columns, since it is dense and runs every
#: loop. That is 0.0006% of Small and buys the electrostatic bias a distance
#: estimate; the alternative, reading the pair track, was L^2 * d_pair.
EXPECTED = {
    "PHAROS-Small": (258_921_498, 82_760_730, 128),
    "PHAROS-Mini": (113_506_832, 39_189_008, 144),
    "Base-v2": (1_105_559_546, 312_836_090, 96),
}
#: What the DENSE diffusion decoder (head 3 plus its pair features) contributes
#: to the active column at each scale. It is the reason the numbers above moved
#: and the reason the active FRACTION is no longer flat: it is a fixed-shape
#: transformer stack, not part of the MoE recipe, so it is 28.6% of Mini's
#: active parameters and 14.2% of Base-v2's.
DIFFUSION_ACTIVE = {"PHAROS-Small": 19_846_793, "PHAROS-Mini": 11_202_441,
                    "Base-v2": 44_508_297}
#: What §5.4 prints. Kept beside the built numbers deliberately: the active
#: column is close (61 vs 62.7, 269 vs 267.9) and the total column is not.
SPEC_5_4 = {"PHAROS-Small": (149e6, 61e6), "PHAROS-Mini": (67e6, 30e6),
            "Base-v2": (1401e6, 269e6)}


def _raises(fn) -> bool:
    try:
        with torch.no_grad():
            fn()
    except ValueError:
        return True
    return False


def main() -> int:
    torch.manual_seed(0)

    print("== property 1: sizing is what the recipe says ==")
    cfgs = {"PHAROS-Small": PharosConfig.small(), "PHAROS-Mini": PharosConfig.mini(),
            "Base-v2": PharosConfig.base_v2()}
    built = {}
    for name, cfg in cfgs.items():
        pc = Pharos(cfg).param_counts()
        built[name] = pc
        t, a, eff = EXPECTED[name]
        chk(f"{name}: total / active / effective layers",
            (pc["total"], pc["active"], pc["effective_layers"]) == (t, a, eff),
            f"{pc['total']:,} / {pc['active']:,} / {pc['effective_layers']}")

    print("\n== property 1b: the family shares one MoE recipe ==")
    # The MoE recipe is still shared; the diffusion decoder is not part of it.
    # Head 3 is a dense transformer stack of a fixed shape, so it lands on the
    # active column as a flat addition that is 28.6% of Mini and 14.2% of
    # Base-v2 -- and the "active fraction is constant" assertion, which held
    # before head 3 became a denoiser, has been failing ever since at a 6.2
    # point spread. Net of the decoder the recipe is flat again, which is the
    # property that was actually being asserted, so that is what is asserted.
    fracs = [built[n]["active"] / built[n]["total"] for n in cfgs]
    net = [(built[n]["active"] - DIFFUSION_ACTIVE[n])
           / (built[n]["total"] - DIFFUSION_ACTIVE[n]) for n in cfgs]
    chk("active fraction is constant across the family, net of head 3",
        max(net) - min(net) < 0.03,
        f"net {', '.join(f'{100*f:.1f}%' for f in net)}  "
        f"(gross {', '.join(f'{100*f:.1f}%' for f in fracs)})")
    chk("head 3 is dense, and costs relatively more at small scale",
        (DIFFUSION_ACTIVE["PHAROS-Mini"] / built["PHAROS-Mini"]["active"]
         > DIFFUSION_ACTIVE["Base-v2"] / built["Base-v2"]["active"]),
        ", ".join(f"{n.split('-')[-1]} "
                  f"{100*DIFFUSION_ACTIVE[n]/built[n]['active']:.1f}%"
                  for n in cfgs))
    spec_fracs = [a / t for t, a in SPEC_5_4.values()]
    chk("§5.4's own active fractions are NOT constant (why it is unreproducible)",
        max(spec_fracs) - min(spec_fracs) > 0.2,
        f"{', '.join(f'{100*f:.1f}%' for f in spec_fracs)}")
    # §5.4's ACTIVE column predates the diffusion decoder and no longer matches
    # anything. Recorded as a measured gap rather than asserted away: the fix
    # is a documentation update, and until it lands this prints the size of it.
    gaps = {n: abs(built[n]["active"] - SPEC_5_4[n][1]) / SPEC_5_4[n][1]
            for n in cfgs}
    chk("§5.4's ACTIVE column is stale by a KNOWN amount, not an unknown one",
        all(g < 0.40 for g in gaps.values()),
        ", ".join(f"{n.split('-')[-1]} {100*g:.1f}%" for n, g in gaps.items())
        + "  -- head 3 became a denoiser and §5.4 was not rewritten")

    # a small config for the behavioural tests
    cfg = PharosConfig(d_model=64, n_blocks=4, n_loops=3, n_heads=4, d_pair=32,
                       n_experts=4, d_expert=32, top_k=2, n_shared=1, max_length=256)
    model = Pharos(cfg)
    B, L = 2, 48
    tok = torch.randint(0, 4, (B, L))
    mod = torch.zeros(B, L, dtype=torch.long)
    chem = torch.randn(B, L, 24)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[1, 30:] = False

    print("\n== property 2: the one-step gradient ==")
    seen: list[tuple] = []
    out = model(tok, mod, chem, mask, deep_supervision=True)
    for it, h in out["iterates"]:
        seen.append((it, bool(h.requires_grad)))
    chk("every loop but the last runs without grad",
        [g for _, g in seen] == [False] * (cfg.n_loops - 1) + [True], str(seen))
    chk("all loops are supervised", len(seen) == cfg.n_loops,
        f"{len(seen)} of {cfg.n_loops}")

    print("\n== property 3: gradients reach the weights ==")
    # The loss must touch the heads, or this checks nothing about them: a first
    # version summed only the hidden state and duly found the head had no
    # gradient, which was true and uninformative.
    loss = (out["hidden"].sum() + out["aux"]["balance_loss"]
            + out["ss_logits"].sum() + out["rigidity"].sum() + out["fitness"].sum())
    loss.backward()
    named = dict(model.named_parameters())
    for key in ("embed.tok.weight", "trunk.blocks.0.ff.gate.weight",
                "heads.residue.secondary.1.weight"):
        g = named[key].grad
        chk(f"{key} has a gradient", g is not None and bool((g != 0).any()),
            "" if g is None else f"|g| = {float(g.abs().sum()):.3e}")

    print("\n== property 4: recycle 0 has no pair-derived features ==")
    feats = RouterFeatures(length=torch.tensor([float(L)] * B), recycle=0)
    v0 = feats.vector(B, torch.device("cpu"), 6, 8)
    chk("the recycle flag is 0 on the first pass", float(v0[0, -1]) == 0.0, "")
    v1 = RouterFeatures(length=feats.length, recycle=1).vector(
        B, torch.device("cpu"), 6, 8)
    chk("and 1 thereafter", float(v1[0, -1]) == 1.0, "")
    chk("an absent feature is zero, not fabricated",
        float(v0[:, 6:8].abs().sum()) == 0.0,
        "Neff/L and in-complex default to 0 when unknown")

    print("\n== property 5: padding cannot change a real position ==")
    with torch.no_grad():
        a = model(tok, mod, chem, mask)["hidden"][1, :30]
        tok2, chem2 = tok.clone(), chem.clone()
        tok2[1, 30:] = 3
        chem2[1, 30:] = torch.randn(L - 30, 24) * 50
        b = model(tok2, mod, chem2, mask)["hidden"][1, :30]
    chk("junk in the padded region is invisible",
        float((a - b).abs().max()) < 1e-4, f"max delta {float((a-b).abs().max()):.2e}")

    # ...and that has to mean EVERY output, not just `hidden`. This property
    # was checked on the trunk alone, and the leak was downstream of it:
    # `StiffnessField` took a `mask` and ignored it, while
    # `block_tridiagonal_variance` is a sequential forward-BACKWARD recursion
    # whose backward sweep starts at the last step. So garbage in the padded
    # region propagated inwards and moved `fluctuation` at real positions by
    # 0.206 -- a stage-5 supervision target, under the one test written for
    # exactly this property.
    with torch.no_grad():
        fa = model(tok, mod, chem, mask, dynamics=True)
        fb = model(tok2, mod, chem2, mask, dynamics=True)
    for key in ("fluctuation", "disorder_logit", "stiffness_diag"):
        da = fa[key][1][:29] if key == "stiffness_diag" else fa[key][1][:30]
        db = fb[key][1][:29] if key == "stiffness_diag" else fb[key][1][:30]
        chk(f"  and in {key}", float((da - db).abs().max()) < 1e-4,
            f"max delta {float((da - db).abs().max()):.2e}")

    print("\n== property 6: the pair track is optional and consistent ==")
    ii = torch.tensor([0, 1, 2])
    jj = torch.tensor([10, 11, 12])
    bb = torch.tensor([0, 1, 1])
    with torch.no_grad():
        o = model(tok, mod, chem, mask, pair_index=(bb, ii, jj), dynamics=True)
    chk("pair heads fire only when pair indices are given",
        "contact_logit" in o and o["contact_logit"].shape == (3,),
        str(tuple(o["contact_logit"].shape)))
    # A pair must be built from the sequence it belongs to. `forward` read
    # `h[0, ii]` unconditionally, so on a batch of more than one it answered
    # about sequence 0 whatever `bb` said -- and this test asked only for the
    # SHAPE, which was right. The trainer never hit it because
    # `train_pharos.py` builds its own pairs with a real per-chain index; the
    # only caller of the broken path was the test that validates the model's
    # interface.
    chk("a two-tuple is refused on a batch of more than one",
        _raises(lambda: model(tok, mod, chem, mask, pair_index=(ii, jj))),
        "ambiguous, so it raises instead of picking sequence 0")
    with torch.no_grad():
        o_a = model(tok, mod, chem, mask,
                    pair_index=(torch.tensor([0]), ii[:1], jj[:1]))
        o_b = model(tok, mod, chem, mask,
                    pair_index=(torch.tensor([1]), ii[:1], jj[:1]))
    chk("and the same pair on two sequences gives two answers",
        float((o_a["contact_logit"] - o_b["contact_logit"]).abs().max()) > 1e-6,
        f"delta {float((o_a['contact_logit'] - o_b['contact_logit']).abs().max()):.3e} "
        f"-- zero here would mean `bb` is being ignored")
    with torch.no_grad():
        o2 = model(tok, mod, chem, mask)
    chk("and are absent otherwise", "contact_logit" not in o2, "")

    print("\n== property 7: all ten heads are produced ==")
    from pharos.model.heads import EXTRA_HEAD_KEYS, HEAD_SPEC
    missing = [h["name"] for h in HEAD_SPEC
               if h.get("forward", True) and h["key"] not in o]
    chk("§9 lists ten heads and every forward-pass head is produced",
        len(HEAD_SPEC) == 10 and not missing, f"missing: {missing or 'none'}")
    # Head 3 is the one entry whose output a forward pass cannot carry: the
    # diffusion decoder emits coordinates from `sample()`. Checked where it
    # actually lives rather than excused.
    chk("head 3's coordinates come from the diffusion decoder",
        hasattr(model.heads, "structure")
        and callable(getattr(model.heads.structure, "sample", None)), "")
    # And the heads §9 stopped numbering are still produced, so renumbering
    # cannot quietly drop an output the loss still reads.
    extra_missing = [k for k in EXTRA_HEAD_KEYS if k not in o]
    chk("the unnumbered heads are still produced", not extra_missing,
        f"missing: {extra_missing or 'none'}")

    print("\n== property 7b2: the docstring's table IS HEAD_SPEC ==")
    # §9's numbering is written twice -- once as `HEAD_SPEC`, once as a table
    # in `heads.py`'s module docstring, because a docstring cannot be
    # generated. Twice means drift, and it had: the docstring still read
    # "5 Mg2+, 6 rigidity, 7 reactivity, 8 fitness, 9 splicing, 10 base
    # identity" -- the scheme §9 ABANDONED -- thirty lines above the corrected
    # `HEAD_SPEC`, plus six inline `# <n>` annotations in the same file.
    #
    # Finding 37 of the 09-26 audit found that drift, fixed the TRAINERS, and
    # recorded "all references now agree with HEAD_SPEC". It did not check
    # this file, which is where the numbering is defined, so the claim was
    # false in the one place that matters. This parses the table rather than
    # asserting agreement.
    import re as _re
    from pharos.model import heads as _H
    _rows = _re.findall(r"^\s{3,4}(\d+)\s+(\w+)\s{2,}", _H.__doc__, _re.M)
    _doc = {int(n): nm for n, nm in _rows}
    _spec = {h["n"]: h["name"] for h in HEAD_SPEC}
    chk("the docstring table parses to ten rows", len(_doc) == 10, f"{len(_doc)}")
    chk("and every row matches HEAD_SPEC", _doc == _spec,
        "; ".join(f"{n}: doc {_doc.get(n)} vs spec {_spec.get(n)}"
                  for n in sorted(set(_doc) | set(_spec))
                  if _doc.get(n) != _spec.get(n)) or "all ten agree")
    # and no bare numeric head annotation survives anywhere in the file
    _src_h = Path(_H.__file__).read_text()
    _bare = _re.findall(r"#\s*(\d+)\s*$", _src_h, _re.M)
    chk("no bare `# <n>` head annotation is left to drift",
        not _bare, f"found {_bare}" if _bare else "keys are used instead")

    print("\n== property 7c: every head produced is either trained or declared ==")
    # every optional branch on at once, so the set below is the whole surface
    # the model can present and not whichever corner this test happened to run
    with torch.no_grad():
        o_full = model(tok, mod, chem, mask, pair_index=(bb, ii, jj),
                       dynamics=True, mlm=True)
    # A head that nothing supervises still emits a number, every forward pass,
    # for the rest of the project. `fitness` had a loss weight printed at
    # startup and written into two report files with no loss behind it;
    # `disorder_logit` was named in stage 5's own docstring as a head that
    # stage trains, with its label sitting in every shard, and no term ever
    # read it. Both looked exactly like the trained heads from outside.
    #
    # Pinned as three explicit sets rather than discovered by searching the
    # trainers for `out["key"]`. The first version of this test did search,
    # and reported `fitness` as trained -- because the comment that says
    # there is no `out["fitness"]` anywhere contains the string `out["fitness"]`.
    # A check that greps its own prose is the exact defect this file exists
    # to catch, committed inside the catcher.
    #
    # The lists are maintenance, deliberately: a new head fails this test
    # until someone classifies it, which is the one moment anybody will.
    TRAINED = {                      # a loss term reads this, somewhere
        "mlm_logits": "stage 1", "ss_logits": "stage 2",
        "reactivity": "stage 3", "contact_logit": "stage 5",
        "distance_logits": "stage 5", "lw_logits": "stage 5",
        "mg_logit": "stage 5", "motif_logits": "stage 5",
        "rigidity": "stage 5", "fluctuation": "stage 5",
        "base_logits": "stage 5",
        "fitness": "stage 6",
    }
    INDIRECT = {                     # no loss of its own; gradient arrives anyway
        "hidden": "the trunk output every head reads",
        "stiffness_diag": "fluctuation is computed from it, so the rigidity "
                          "loss reaches it",
        "stiffness_off": "same path as stiffness_diag",
    }
    UNTRAINED = {                    # emitted, no gradient, and why
        "disorder_logit":
            "the label names polymer positions that were never modelled, and "
            "the corpus tokens are the modelled residues only, so a target "
            "aligned to them is vacuously zero. Needs entity_poly_seq in the "
            "corpus: a build change, not a loss term.",
        "splice_logits":
            "no RNA splice-site corpus has been acquired.",
        "ensemble_state_logits":
            "the K-state mixture weights. \u00a710 defines the ensemble but no "
            "observable in the corpus distinguishes the states, so there is "
            "nothing to fit them to.",
    }
    _emitted = {k for k, v in o_full.items() if torch.is_tensor(v)}
    _declared = set(TRAINED) | set(INDIRECT) | set(UNTRAINED)
    chk("no output is emitted without being classified",
        not (_emitted - _declared),
        f"unclassified: {sorted(_emitted - _declared) or 'none'}")
    chk("nothing is classified that is no longer emitted",
        not (_declared - _emitted),
        f"stale: {sorted(_declared - _emitted) or 'none'}")
    chk("the three sets are disjoint",
        len(_declared) == len(TRAINED) + len(INDIRECT) + len(UNTRAINED),
        f"{len(TRAINED)} trained, {len(INDIRECT)} indirect, "
        f"{len(UNTRAINED)} untrained")
    chk("every untrained head says why, at length",
        all(len(v) > 40 for v in UNTRAINED.values()),
        ", ".join(sorted(UNTRAINED)))
    # `fitness` moved from UNTRAINED to TRAINED when stage 6 was wired. The
    # claim is checkable from here, so it is checked from here rather than
    # taken on the word of the table above.
    _st = (Path(__file__).resolve().parents[3]
           / "scripts/train_sequence_stages.py").read_text()
    chk("stage 6 reads out[\"fitness\"] and weights it",
        'out["fitness"]' in _st and '"fitness": 0.3' in _st,
        "train_sequence_stages.py")
    chk("every STAGE_WEIGHTS key has a loop that applies it",
        all(f'STAGE_WEIGHTS["{k}"]' in _st
            for k in ("ss", "reactivity", "fitness")),
        "ss, reactivity, fitness")

    print("\n== property 7e: §6.2's physics reaches the attention logits ==")
    # It did not. `TokenTrunk.forward` accepted a `pair_bias_fn`, nothing
    # passed one, and `ElectrostaticBias` was never called from anywhere in
    # the repository -- while three docstrings and ARCHITECTURE.md §6.2
    # asserted the coupling in the present tense (finding 51). The missing
    # half was a distance estimate, which head 3 stopped providing when it
    # became a denoiser.
    from pharos.model.pharos import UNWIRED
    from pharos.physics.manning import IonicCondition
    chk("nothing is declared unwired any more", not UNWIRED, f"{len(UNWIRED)}")

    ecfg = PharosConfig(d_model=64, n_blocks=8, n_loops=2, n_heads=4, d_pair=32,
                        n_experts=4, d_expert=32, top_k=2, n_shared=1,
                        max_length=128)
    em = Pharos(ecfg).eval()
    eB, eL = 2, 24
    et = torch.randint(0, 4, (eB, eL))
    em_mask = torch.ones(eB, eL, dtype=torch.bool)
    ech = torch.randn(eB, eL, ecfg.d_chem)
    # 1. it is INERT at the shipped initialisation, because `bias_scale` is
    #    zero -- so wiring it leaves every existing checkpoint untouched.
    with torch.no_grad():
        _off = em(et, torch.zeros_like(et), ech, em_mask, n_loops=2,
                  electrostatics=False)["hidden"]
        _on = em(et, torch.zeros_like(et), ech, em_mask, n_loops=2)["hidden"]
    chk("inert at init: bias_scale is zero, so nothing moves",
        float((_off - _on).abs().max()) == 0.0,
        f"max delta {float((_off - _on).abs().max()):.2e}")
    # 2. and once the gain opens, THE IONIC CONDITION CHANGES THE ANSWER,
    #    which is the whole of §6.2's claim.
    for _b in em.trunk.blocks:
        if _b.kind == "full":
            torch.nn.init.constant_(_b.mixer.bias_scale, 1.0)
    with torch.no_grad():
        _lo = em(et, torch.zeros_like(et), ech, em_mask, n_loops=2,
                 ionic=IonicCondition(mg_mM=0, k_mM=100))["hidden"]
        _hi = em(et, torch.zeros_like(et), ech, em_mask, n_loops=2,
                 ionic=IonicCondition(mg_mM=15, k_mM=150))["hidden"]
    chk("0 mM Mg and 15 mM Mg give different representations",
        float((_lo - _hi).abs().max()) > 1e-6,
        f"max delta {float((_lo - _hi).abs().max()):.2e} -- the screening "
        f"length moves 9.61 A to 6.88 A")
    # 3. the bias is the CANONICAL physics, not a second copy of it
    from pharos.physics.manning import b_elec
    _d = torch.full((1, 3, 3), 10.0)
    _c = IonicCondition()
    chk("the bias is manning.b_elec, not a restatement",
        # relative: the module runs the torch path in fp32 while the scalar
        # reference runs the math path in fp64, so they agree to fp32 eps,
        # not to 1e-9 absolute. A restatement that drops l_B is 7.16x out.
        abs(float(em.elec(_d, _c)[0, 0, 0]) - float(b_elec(10.0, _c)))
        <= 1e-6 * abs(float(b_elec(10.0, _c))),
        f"{float(em.elec(_d, _c)[0, 0, 0]):.6f} vs "
        f"{float(b_elec(10.0, _c)):.6f} -- the old copy omitted l_B and was "
        f"7.16x too small")
    # 4. loop 0 has no estimate, so it must run with no bias at all
    _seen = []
    _orig = em.elec.forward
    chk("the distance estimate is in angstrom, not arbitrary units",
        15.0 < float(em.vdist(torch.randn(1, 64, 64)).median()) < 30.0,
        f"median pairwise {float(em.vdist(torch.randn(1, 64, 64)).median()):.1f} A")

    print("\n== property 7f: every parameter gets a gradient, or is declared ==")
    # The mechanical version of 7e. A module nothing calls shows up as
    # parameters with no gradient, so exercise EVERY path at once -- pair
    # track, mlm, dynamics, deep supervision, and the diffusion head's own
    # loss -- and assert the dead set is exactly the declared one. A new
    # unwired module then fails this test instead of sitting for weeks.
    #
    # 8 blocks, not 4: the period-8 pattern puts `full` last, so a 4-block
    # model has no full-attention block and `bias_scale` -- one of the two
    # things finding 51 left without a gradient -- would not exist to check.
    cfg8 = PharosConfig(d_model=64, n_blocks=8, n_loops=2, n_heads=4, d_pair=32,
                        n_experts=4, d_expert=32, top_k=2, n_shared=1,
                        max_length=128)
    g = Pharos(cfg8)
    # Open the electrostatic gate. `bias_scale` starts at zero, and a zero
    # gate multiplies the gradient into `vdist`/`elec` by zero -- they would
    # be in the graph with an all-zero grad, which `grad is None` cannot see.
    # The bootstrap from the CLOSED gate is checked separately below.
    with torch.no_grad():
        for _b in g.trunk.blocks:
            if _b.kind == "full":
                _b.mixer.bias_scale.fill_(0.1)
    gB, gL = 2, 24
    gt = torch.randint(0, 4, (gB, gL))
    gm = torch.ones(gB, gL, dtype=torch.bool)
    go = g(gt, torch.zeros_like(gt), torch.randn(gB, gL, cfg8.d_chem), gm,
           pair_index=(torch.tensor([0, 1]), torch.tensor([1, 2]),
                       torch.tensor([10, 11])),
           mlm=True, dynamics=True, deep_supervision=True)
    gloss = go["aux"]["balance_loss"]
    for _k, _v in go.items():
        if torch.is_tensor(_v) and _v.is_floating_point():
            gloss = gloss + _v.float().sum()
    _coev = torch.rand(gB, gL, gL)          # exercise BOTH coevolution paths
    gloss = gloss + g.coev_proj(_coev[:, :2, 0].reshape(-1, 1)).sum()
    _pf = g.diff_pair(go["hidden"], _coev)
    gloss = gloss + g.heads.structure.loss(
        torch.randn(gB, gL, 3, 3), go["hidden"], _pf, gm)["loss"]
    gloss.backward()
    _dead = sorted(n for n, q in g.named_parameters() if q.grad is None)
    # Nothing is declared unwired any more, so the dead set must be EMPTY.
    chk("every parameter is in the graph",
        _dead == [], f"dead {_dead}" if _dead else
        f"all {sum(1 for _ in g.named_parameters())} reached")

    # `grad is None` is not enough for §6.2. Until the gradient probe caught
    # it, the trunk detached `pair_bias` before the differentiated pass and
    # recomputed nothing, so these two were absent from the graph entirely
    # while STILL perturbing the forward pass -- measurable in 7e, frozen at
    # init for the whole run. Assert the magnitude, not just the presence.
    _phys = {"elec.log_scale": g.elec.log_scale,
             "vdist.proj.weight": g.vdist.proj.weight}
    for _n, _q in _phys.items():
        _mag = 0.0 if _q.grad is None else float(_q.grad.abs().sum())
        chk(f"{_n} gets a gradient that is not merely zero",
            _mag > 0, f"|grad| {_mag:.3e}")

    # ...and the gate can open itself: at the zero init the physics is
    # correctly starved, but `bias_scale` must still move, or it never does.
    g2 = Pharos(cfg8)
    _o2 = g2(gt, torch.zeros_like(gt), torch.randn(gB, gL, cfg8.d_chem), gm,
             n_loops=2, mlm=True)
    _o2["mlm_logits"].sum().backward()
    _bs2 = [b.mixer.bias_scale for b in g2.trunk.blocks if b.kind == "full"][0]
    _bsg = 0.0 if _bs2.grad is None else float(_bs2.grad.abs().sum())
    chk("at the closed init gate, bias_scale still gets a gradient",
        _bsg > 0 and float(_bs2.abs().max()) == 0.0,
        f"gate {float(_bs2.abs().max()):.1f}, |grad| {_bsg:.3e}")

    print("\n== property 7d: the fitness head separates single-nt variants ==")
    # `fitness` is a MEAN-POOLED scalar. A one-nucleotide change in an 87 nt
    # construct moves the pooled representation by about 1/87 of one token's
    # delta, and if that lands under the head's own numerical noise then no
    # amount of training can give it a fitness landscape: the MSE would fall
    # (it would be predicting the mean of a zero-mean target) while every
    # Spearman came back NaN. The architecture has to be able to represent
    # the task before the trainer is asked to learn it.
    import numpy as _np
    _wt = "GGCCGGCAUGGUCCCAGCCUCCUCGCUGGCGCCGGCUGGGCAACAUUCCGAGGGGACCGUCCCC"
    _rng = _np.random.default_rng(0)
    _seqs = [_wt]
    for _ in range(11):
        _i = int(_rng.integers(0, len(_wt)))
        _alt = [c for c in "ACGU" if c != _wt[_i]][int(_rng.integers(0, 3))]
        _seqs.append(_wt[:_i] + _alt + _wt[_i + 1:])
    _ids = {"A": 0, "C": 1, "G": 2, "U": 3}
    _tk = torch.tensor([[_ids[c] for c in q] for q in _seqs])
    _mk = torch.ones_like(_tk, dtype=torch.bool)
    _ch = torch.zeros(_tk.shape[0], _tk.shape[1], cfg.d_chem)
    # dims 0-4 of the chemistry are the one-hot base identity, which is the
    # only part of it that differs between these variants; the rest is zero
    # so the probe measures the trunk's response to the substitution and not
    # to a random feature vector.
    _ch.scatter_(2, _tk.unsqueeze(-1), 1.0)
    with torch.no_grad():
        _f = model(_tk, torch.zeros_like(_tk), _ch, _mk,
                   n_loops=2)["fitness"].float().numpy()
    _d = _np.abs(_f[1:] - _f[0])
    _eps = float(_np.spacing(_np.abs(_f).max()))
    chk("one-nucleotide variants give distinct fitness values",
        len(_np.unique(_f)) == len(_f), f"{len(_np.unique(_f))}/{len(_f)} distinct")
    chk("the separation is far above fp32 spacing",
        float(_d.min()) > 1000 * _eps,
        f"smallest gap {_d.min():.3e}, fp32 eps {_eps:.3e}, "
        f"ratio {_d.min()/_eps:.3g}")

    print("\n== property 7b: the MLM head starts at chance, not off a cliff ==")
    import math as _m
    import torch.nn.functional as _F
    for d in (64, 256):
        c = PharosConfig(d_model=d, n_blocks=2, n_loops=1, n_heads=4, d_pair=32,
                         n_experts=4, d_expert=32, top_k=2, n_shared=1,
                         max_length=128)
        mm = Pharos(c)
        tk = torch.randint(0, 5, (4, 48))
        with torch.no_grad():
            lg = mm(tk, torch.zeros_like(tk), torch.randn(4, 48, 24),
                    torch.ones(4, 48, dtype=torch.bool), n_loops=1,
                    mlm=True)["mlm_logits"].float()
            l0 = float(_F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tk.reshape(-1)))
        # The head is TIED to the token embedding, so logits are h . W^T over d
        # dims. With nn.Embedding's default N(0,1) they inherit std ~sqrt(d) and
        # the initial loss was 39.19 against log(13) = 2.56 -- a run that opens
        # by unlearning its own initialisation. std 0.02 init is the fix.
        chk(f"d={d}: initial masked-token loss is near chance",
            abs(l0 - _m.log(c.n_symbols)) < 0.5,
            f"{l0:.3f} vs log({c.n_symbols}) = {_m.log(c.n_symbols):.3f}")
    chk("embeddings use std-0.02 init, not N(0,1)",
        float(Pharos(PharosConfig.small()).embed.tok.weight.std()) < 0.05,
        f"std {float(Pharos(PharosConfig.small()).embed.tok.weight.std()):.4f}")
    chk("the MLM head is tied, adding no output matrix",
        not any(n.startswith("mlm_head") for n, _ in model.named_parameters()),
        "only mlm_norm and mlm_bias are its own")

    print("\n== property 8: §10's ensemble outputs are present WHEN ASKED FOR ==")
    with torch.no_grad():
        off = model(tok, mod, chem, mask)
    chk("the ensemble is off by default", "fluctuation" not in off,
        "opt-in: MLM pretraining OOMed paying for it unasked")
    for key in ("fluctuation", "disorder_logit", "stiffness_diag",
                "ensemble_state_logits"):
        chk(f"{key} emitted", key in o, str(tuple(o[key].shape)) if key in o else "")
    # The collision this guards is real and the guard was checking for it
    # backwards. It demanded BOTH `state_logits` and `ensemble_state_logits` in
    # the output, but head 3 is a diffusion decoder now (§9.1) and emits
    # neither -- `StructureHead`, the regressor that owned `state_logits`, is
    # kept only for the ablation and is not wired in. So the assertion could
    # not pass, and what it was protecting -- that the physics ensemble's
    # token-derived weights never arrive under a name something else might read
    # as a structure output -- was never actually tested. It is now: the
    # ensemble's weights must be renamed, and a bare `state_logits` must not
    # appear in a forward pass at all.
    chk("the ensemble's state weights are renamed, not left to collide",
        "ensemble_state_logits" in o and "state_logits" not in o,
        "physics-derived and token-derived weights are different quantities")
    from pharos.model.heads import HeadConfig, StructureHead
    _hc = HeadConfig(d_model=32)
    _sh = StructureHead(_hc)
    with torch.no_grad():
        _so = _sh(torch.zeros(1, 5, 32), torch.ones(1, 5, dtype=torch.bool))
    chk("the retained regressor still runs, so the ablation stays runnable",
        set(_so) == {"coords", "state_logits"}, f"{sorted(_so)}")
    chk("fluctuations are non-negative variances",
        bool((o["fluctuation"] >= 0).all()), "")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
