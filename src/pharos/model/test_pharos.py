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
EXPECTED = {
    "PHAROS-Small": (238_897_870, 62_737_102, 128),
    "PHAROS-Mini": (102_191_684, 27_873_860, 144),
    "Base-v2": (1_060_697_006, 267_973_550, 96),
}
#: What §5.4 prints. Kept beside the built numbers deliberately: the active
#: column is close (61 vs 62.7, 269 vs 267.9) and the total column is not.
SPEC_5_4 = {"PHAROS-Small": (149e6, 61e6), "PHAROS-Mini": (67e6, 30e6),
            "Base-v2": (1401e6, 269e6)}


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
    fracs = [built[n]["active"] / built[n]["total"] for n in cfgs]
    chk("active fraction is constant across the family",
        max(fracs) - min(fracs) < 0.03,
        f"{', '.join(f'{100*f:.1f}%' for f in fracs)}")
    spec_fracs = [a / t for t, a in SPEC_5_4.values()]
    chk("§5.4's own active fractions are NOT constant (why it is unreproducible)",
        max(spec_fracs) - min(spec_fracs) > 0.2,
        f"{', '.join(f'{100*f:.1f}%' for f in spec_fracs)}")
    chk("the recipe does reproduce §5.4's ACTIVE column within 8%",
        all(abs(built[n]["active"] - SPEC_5_4[n][1]) / SPEC_5_4[n][1] < 0.08
            for n in cfgs),
        ", ".join(f"{n.split('-')[-1]} "
                  f"{100*abs(built[n]['active']-SPEC_5_4[n][1])/SPEC_5_4[n][1]:.1f}%"
                  for n in cfgs))

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

    print("\n== property 6: the pair track is optional and consistent ==")
    ii = torch.tensor([0, 1, 2])
    jj = torch.tensor([10, 11, 12])
    with torch.no_grad():
        o = model(tok, mod, chem, mask, pair_index=(ii, jj))
    chk("pair heads fire only when pair indices are given",
        "contact_logit" in o and o["contact_logit"].shape == (3,),
        str(tuple(o["contact_logit"].shape)))
    with torch.no_grad():
        o2 = model(tok, mod, chem, mask)
    chk("and are absent otherwise", "contact_logit" not in o2, "")

    print("\n== property 7: all ten heads are produced ==")
    from pharos.model.heads import HEAD_SPEC
    missing = [s["name"] for s in HEAD_SPEC if s["key"] not in o]
    chk("§9 lists ten heads and ten are produced",
        len(HEAD_SPEC) == 10 and not missing, f"missing: {missing or 'none'}")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
