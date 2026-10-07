#!/usr/bin/env python
"""Every information channel, checked for whether it carries information.

The recurring defect in this repository is a component that exists, is
measured, and whose output is never checked for basic validity: §6.2's
electrostatic bias ran for weeks with its parameters detached from the graph
(findings 64-65); `b_factor_z` handed 3,408 residues a fabricated extreme
because the statistics excluded unset B-factors and the normalisation did not
(finding 66). Both were invisible to every metric printed beside them.

This probe is the standing check against that class. It asks two questions of
every channel, and a channel that exists but is dead fails the first:

  A. does the CORPUS carry information in it -- non-constant, in range, with
     a plausible fill rate and, where there is physics to check against, the
     right physics?
  B. does that information REACH the model -- perturb the channel alone, and
     the output must move; and the parameters that consume it must receive a
     gradient that is not merely present but non-zero?

Run:  PYTHONPATH=src python scripts/probe_channels.py [--shards N]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

fails: list[str] = []
notes: list[str] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:58s} {detail}")
    if not ok:
        fails.append(name)


#: The chemistry dims that are constant for a STATED reason. Dims 6, 9 and 11
#: are `wcD`, `hgA` and `sgA`, and the literature table in `chemistry.py` gives
#: all four standard bases the same value for each; dim 13 is `shift`, which
#: nothing outside a unit test sets. They are exactly redundant with the bias
#: of `nn.Linear(24, d)`. Pinned so that a dim going constant for an UNSTATED
#: reason -- a loader slice off by one, a build that stopped populating a
#: field -- fails instead of passing unnoticed. See finding 67.
CONST_CHEM_DIMS = {6, 9, 11, 13}

#: Backbone geometry, measured over 2.2M residues of the v4 corpus and checked
#: against the chemistry rather than against itself. These are covalent-path
#: distances within a residue and the A-form rise between residues; they are
#: tight, and a mis-ordered `(P, C4', glycosidic N)` triple would scatter them.
BACKBONE_BANDS = {
    "P-C4' (intra-residue)": (3.6, 4.2, 0.40),
    "C4'-N (intra-residue)": (3.1, 3.7, 0.40),
    "P-P (consecutive)":     (5.3, 6.5, 1.20),
}


def section(t: str) -> None:
    print(f"\n== {t} ==")


# ---------------------------------------------------------------------------
# A. the corpus
# ---------------------------------------------------------------------------
def probe_corpus(n_shards: int) -> None:
    fs = sorted(glob.glob(str(ROOT / "data/derived/pharos3d/shard-*.npz")))[:n_shards]
    if not fs:
        chk("the corpus is present", False, "no shards under data/derived/pharos3d")
        return
    agg: dict[str, list] = {}
    per_chain_coords = []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        for k in ("tokens", "mod_ids", "mg_site", "b_factor_z", "unknown_base",
                  "loop_class", "lw_pairs", "contacts", "coord_mask"):
            agg.setdefault(k, []).append(z[k])
        agg.setdefault("chem", []).append(z["chem"].astype(np.float32))
        per_chain_coords.append((z["coords"].astype(np.float32),
                                 z["coord_mask"], z["res_off"]))
    cat = {k: np.concatenate(v) for k, v in agg.items()}
    n = cat["tokens"].size
    print(f"  ({len(fs)} shards, {n:,} residues)")

    section("A1. sequence channels")
    t = cat["tokens"]
    chk("tokens span the alphabet", len(np.unique(t)) >= 4,
        f"{len(np.unique(t))} distinct, range {t.min()}..{t.max()}")
    m = cat["mod_ids"]
    f_mod = float((m != 0).mean())
    chk("mod_ids fire, and are not the majority", 0 < f_mod < 0.10,
        f"{len(np.unique(m))} distinct, {100*f_mod:.2f}% non-zero")
    u = cat["unknown_base"]
    chk("unknown_base is a minority label", float((u != 0).mean()) < 0.05,
        f"{100*float((u != 0).mean()):.3f}% unassigned -- head 10's free supervision")

    section("A2. chemistry: all 24 dims, individually")
    ch = cat["chem"]
    const = {i for i in range(ch.shape[1]) if ch[:, i].std() == 0}
    chk("exactly the declared dims are constant", const == CONST_CHEM_DIMS,
        f"constant {sorted(const)}, declared {sorted(CONST_CHEM_DIMS)}"
        + ("" if const == CONST_CHEM_DIMS else "  <-- an UNDECLARED dead dim"))
    live = [i for i in range(ch.shape[1]) if i not in CONST_CHEM_DIMS]
    worst = min(live, key=lambda i: ch[:, i].std())
    chk("every non-declared dim varies", ch[:, worst].std() > 0,
        f"{len(live)} live dims, weakest is dim {worst} (sd {ch[:, worst].std():.4f})")
    fin = np.isfinite(ch).all()
    chk("no chemistry value is NaN or inf", bool(fin), "all finite" if fin else "NON-FINITE")
    notes.append(f"chemistry carries {len(live)} of 24 dims; "
                 f"{sorted(CONST_CHEM_DIMS)} are constant by construction (finding 67)")

    section("A3. the rigidity target")
    b = cat["b_factor_z"].astype(np.float32)
    fin = np.isfinite(b)
    # The exact invariant -- a finite z comes only from a POSITIVE B -- needs
    # the raw B-factor and so is tested at the parser, in
    # `test_mmcif_entities.py` property 10. What the corpus alone can show is
    # whether the surviving channel is actually standardised, which is what
    # finding 66 broke: the fabricated entries dragged sd from 0.94 to 3.33.
    chk("the finite target is a standardised z-score",
        abs(b[fin].mean()) < 0.05 and 0.95 < b[fin].std() < 1.05,
        f"mean {b[fin].mean():+.4f} sd {b[fin].std():.4f}")
    # unset B-factors must be ABSENT, not extreme and not zero: both of those
    # are a fabricated rigidity target, one of them merely quieter
    nan_f = float((~fin).mean())
    chk("unset B-factors are NaN rather than invented",
        0.0 < nan_f < 0.6, f"{100*nan_f:.2f}% of residues carry no target")
    ext = b[fin & (b <= -6.0)]
    chk("the negative tail is negligible and not a block",
        ext.size < 0.001 * b.size,
        f"{ext.size:,} residues below z = -6 ({100*ext.size/b.size:.4f}%), "
        f"{len(np.unique(ext))} distinct"
        if ext.size else "none")

    section("A4. structural channels")
    g = cat["mg_site"]
    fg = float((g != 0).mean())
    chk("mg_site is binary and sparse", set(np.unique(g)) <= {0, 1} and 0.001 < fg < 0.5,
        f"{100*fg:.3f}% of residues coordinate Mg")
    lw = cat["lw_pairs"]
    chk("lw_pairs cover the Leontis-Westhof classes", len(lw) > 0
        and len(np.unique(lw[:, 2])) >= 10,
        f"{len(lw):,} pairs, {len(np.unique(lw[:, 2]))} classes "
        f"({lw[:, 2].min()}..{lw[:, 2].max()})")
    lc = cat["loop_class"]
    chk("loop_class has more than one class", len(np.unique(lc)) >= 2,
        "  ".join(f"{int(v)}:{100*c/lc.size:.1f}%"
                  for v, c in zip(*np.unique(lc, return_counts=True))))
    chk("contacts are present", len(cat["contacts"]) > 0, f"{len(cat['contacts']):,}")
    cm = cat["coord_mask"]
    chk("backbone atoms are mostly resolved", cm.mean() > 0.90,
        f"P {100*cm[:, 0].mean():.1f}%  C4' {100*cm[:, 1].mean():.1f}%  "
        f"N {100*cm[:, 2].mean():.1f}%")

    section("A5. backbone geometry -- against the chemistry, not against itself")
    d: dict[str, list] = {k: [] for k in BACKBONE_BANDS}
    for xyz, mk, off in per_chain_coords:
        for a, e in zip(off[:-1], off[1:]):
            c, mm = xyz[a:e], mk[a:e]
            o = mm[:, 0] & mm[:, 1]
            if o.any():
                d["P-C4' (intra-residue)"].append(np.linalg.norm(c[o, 0]-c[o, 1], axis=-1))
            o = mm[:, 1] & mm[:, 2]
            if o.any():
                d["C4'-N (intra-residue)"].append(np.linalg.norm(c[o, 1]-c[o, 2], axis=-1))
            o = mm[:-1, 0] & mm[1:, 0]
            if o.any():
                d["P-P (consecutive)"].append(
                    np.linalg.norm(c[:-1][o, 0]-c[1:][o, 0], axis=-1))
    for name, (lo, hi, sd_max) in BACKBONE_BANDS.items():
        v = np.concatenate(d[name])
        med, sd = float(np.median(v)), float(v.std())
        chk(f"{name} is physical", lo < med < hi and sd < sd_max,
            f"median {med:.2f} A, sd {sd:.2f}  (band {lo}-{hi} A, sd < {sd_max})")


# ---------------------------------------------------------------------------
# B. the model
# ---------------------------------------------------------------------------
def probe_model() -> None:
    import torch
    from pharos.model.pharos import Pharos, PharosConfig
    from pharos.physics.manning import IonicCondition

    torch.manual_seed(0)
    cfg = PharosConfig(d_model=64, n_blocks=8, n_loops=2, n_heads=4, d_pair=32,
                       n_experts=4, d_expert=32, top_k=2, n_shared=1, max_length=128)
    mdl = Pharos(cfg).eval()
    B, L = 2, 32
    tok = torch.randint(0, 4, (B, L))
    # NOT zeros: `embed.mod` is `nn.Embedding(..., padding_idx=0)`, so row 0
    # never receives a gradient by construction and an all-zero probe would
    # report a dead channel that is merely an unexercised one.
    mod = torch.randint(1, 4, (B, L))
    chem = torch.randn(B, L, cfg.d_chem)
    mk = torch.ones(B, L, dtype=torch.bool)

    def run(**kw):
        with torch.no_grad():
            a = dict(tokens=tok, mod_ids=mod, chem=chem, mask=mk, n_loops=2)
            a.update(kw)
            return mdl(**a)["hidden"]

    base = run()

    section("B1. every input channel moves the output")
    t2 = tok.clone(); t2[0, 0] = (int(t2[0, 0]) + 1) % 4
    chk("tokens", float((run(tokens=t2) - base).abs().max()) > 1e-6,
        f"delta {float((run(tokens=t2)-base).abs().max()):.2e}")
    # ROTATE, like the token line above; do not assign a constant.
    #
    # This was `m2[0, 0] = 1` against a `mod` drawn from randint(1, 4), so
    # whenever the draw was already 1 the "perturbed" input was the original
    # and the delta was exactly 0.00e+00 -- a dead channel reported for a
    # channel that is fine. It survived because `torch.manual_seed(0)` is set
    # three lines before the MODEL is built, and the model's initialisation
    # consumes the stream: adding head 11 and `theta_head` shifted every
    # draw after it, `mod[0, 0]` became 1, and a probe that had passed all
    # night started failing the gate and blocking the pipeline.
    #
    # The seed made it deterministic, not correct. A perturbation has to be
    # different from what it perturbs by construction.
    m2 = mod.clone(); m2[0, 0] = 1 + (int(mod[0, 0]) % 3)   # {1,2,3}, != itself
    assert int(m2[0, 0]) != int(mod[0, 0]), "the perturbation must perturb"
    chk("mod_ids", float((run(mod_ids=m2) - base).abs().max()) > 1e-6,
        f"delta {float((run(mod_ids=m2)-base).abs().max()):.2e} "
        f"({int(mod[0, 0])} -> {int(m2[0, 0])})")
    # EVERY chemistry dim, one at a time: a dim the embedding never reads is
    # the same defect as a dim the corpus never fills.
    dead = []
    for i in range(cfg.d_chem):
        c2 = chem.clone(); c2[:, :, i] += 1.0
        if float((run(chem=c2) - base).abs().max()) <= 1e-7:
            dead.append(i)
    chk("all 24 chemistry dims reach the trunk", not dead,
        "none dead" if not dead else f"dims {dead} change nothing")
    # §6.2 is INERT at init on purpose -- `bias_scale` is zero so a loaded
    # checkpoint is numerically unchanged -- so test both states. Inert at
    # init is a property to keep; inert with the gate open is finding 64.
    lo0 = run(ionic=IonicCondition(mg_mM=0, k_mM=100))
    hi0 = run(ionic=IonicCondition(mg_mM=15, k_mM=150))
    chk("§6.2 is inert at init, so a checkpoint is unchanged",
        float((lo0 - hi0).abs().max()) == 0.0,
        f"delta {float((lo0-hi0).abs().max()):.2e} -- bias_scale starts at zero")
    with torch.no_grad():
        for blk in mdl.trunk.blocks:
            if blk.kind == "full":
                blk.mixer.bias_scale.fill_(0.1)
    lo = run(ionic=IonicCondition(mg_mM=0, k_mM=100))
    hi = run(ionic=IonicCondition(mg_mM=15, k_mM=150))
    chk("ionic condition moves the output once the gate opens",
        float((lo - hi).abs().max()) > 1e-6,
        f"delta {float((lo-hi).abs().max()):.2e} -- screening 9.61 A to 6.88 A")
    off = run(electrostatics=False, ionic=IonicCondition(mg_mM=15, k_mM=150))
    chk("and the ablation switch really disables it",
        float((off - run(electrostatics=False,
                         ionic=IonicCondition(mg_mM=0, k_mM=100))).abs().max()) == 0.0,
        "electrostatics=False is insensitive to salt, as an ablation must be")
    with torch.no_grad():
        for blk in mdl.trunk.blocks:
            if blk.kind == "full":
                blk.mixer.bias_scale.zero_()

    section("B2. the pair track, the motif bank and coevolution")
    pi = (torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([10, 11]))
    out = mdl(tok, mod, chem, mk, pair_index=pi, n_loops=2)
    # the pair track reaches the outputs as the three PER-PAIR heads; there
    # is no `pair` key, the tensor is consumed inside `forward`
    npair = pi[0].numel()
    for key in ("contact_logit", "distance_logits", "lw_logits"):
        v = out.get(key)
        chk(f"pair track -> {key}", v is not None and v.shape[0] == npair,
            f"{tuple(v.shape)} for {npair} pairs" if v is not None else "absent")
    if mdl.motifs is None:
        chk("the motif bank is compiled", False,
            "data/derived/motif_bank/bank.npz absent -- load_bank returned None")
    else:
        chk("the motif bank is compiled", True,
            f"{mdl.motifs.descriptor.shape[0]} motifs")
        chk("retrieval runs and reports a gate", "motif_gate" in out["aux"],
            f"gate_mean {float(out['aux']['motif_gate']):.4f}"
            if "motif_gate" in out["aux"] else "no motif_gate in aux")
        if "motif_gate" in out["aux"]:
            chk("and the gate is open enough to pass signal",
                float(out["aux"]["motif_gate"]) > 1e-3,
                f"{float(out['aux']['motif_gate']):.4f}")

    section("B3. gradients -- non-zero, not merely present")
    mdl.train()
    # open the electrostatic gate: it starts at zero, and a zero gate makes
    # the gradient into §6.2 zero without making it absent (finding 64)
    with torch.no_grad():
        for blk in mdl.trunk.blocks:
            if blk.kind == "full":
                blk.mixer.bias_scale.fill_(0.1)
    mdl.zero_grad(set_to_none=True)
    o = mdl(tok, mod, chem, mk, pair_index=pi, n_loops=2, mlm=True,
            dynamics=True, deep_supervision=True)
    loss = o["aux"]["balance_loss"]
    for v in o.values():
        if torch.is_tensor(v) and v.is_floating_point():
            loss = loss + v.float().sum()
    # head 3 is a DENOISER: it emits nothing in a plain forward pass, so it is
    # reachable only through its own loss. Both coevolution paths likewise.
    coev = torch.rand(B, L, L)
    loss = loss + mdl.coev_proj(coev[:, :2, 0].reshape(-1, 1)).sum()
    loss = loss + mdl.heads.structure.loss(
        torch.randn(B, L, 3, 3), o["hidden"], mdl.diff_pair(o["hidden"], coev),
        mk)["loss"]
    loss.backward()
    absent = [n for n, p in mdl.named_parameters() if p.grad is None]
    chk("no parameter is absent from the graph", not absent,
        f"all {sum(1 for _ in mdl.named_parameters())} reached"
        if not absent else f"{len(absent)} absent: {absent[:4]}")
    watch = {"elec.log_scale": mdl.elec.log_scale,
             "vdist.proj.weight": mdl.vdist.proj.weight,
             "embed.chem.weight": mdl.embed.chem.weight,
             "embed.mod.weight": mdl.embed.mod.weight,
             "coev_proj.weight": mdl.coev_proj.weight,
             "embed.tok.weight": mdl.embed.tok.weight}
    for name, p in watch.items():
        g = 0.0 if p.grad is None else float(p.grad.abs().sum())
        chk(f"{name} has a non-zero gradient", g > 0, f"|grad| {g:.3e}")


#: Router-conditioning fields each call site must supply, and why anything
#: missing is allowed to be missing. `RouterFeatures` reads an absent field as
#: ZERO, so a call site that forgets one does not raise -- it silently hands
#: the router a vector the model was never trained on. Inference passed
#: `length` alone while stage 5 trained with the pooled chemistry too, which
#: left 6 of 14 conditioning dims at zero (finding 69).
#: `length_bin_max` is listed because it is not decoration: it changes WHICH
#: bin a length maps to, so a caller that omits it routes 800, 1,200 and
#: 2,000 nt chains one bin away from where they trained. Inference must match
#: the stage whose weights it loads, which is stage 5.
ROUTER_CALLERS = {
    "scripts/train_pharos.py":           {"length", "in_complex", "chem_summary",
                                          "length_bin_max"},
    "scripts/train_sequence_stages.py":  {"length", "chem_summary"},
    "scripts/predict_structure.py":      {"length", "chem_summary",
                                          "length_bin_max"},
    "scripts/audit_router.py":           {"length", "chem_summary"},
}


def probe_router_conditioning() -> None:
    """Every call site builds the conditioning the model was trained on."""
    import ast
    for rel, expect in ROUTER_CALLERS.items():
        f = ROOT / rel
        if not f.exists():
            chk(f"{rel} exists", False, "missing")
            continue
        got = set()
        for node in ast.walk(ast.parse(f.read_text())):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "RouterFeatures"):
                got |= {k.arg for k in node.keywords if k.arg}
        got.discard("recycle")          # the trunk sets this per loop
        chk(f"{rel.split('/')[-1]} conditions on the trained fields",
            got == expect,
            f"{sorted(got)}" + ("" if got == expect
                                else f"  expected {sorted(expect)}"))
    # and the one field NOTHING supplies, declared so it cannot go stale
    import re
    src = (ROOT / "src/pharos/model/moe.py").read_text()
    consumed = "self.neff_over_l is not None" in src
    setters = [rel for rel in ROUTER_CALLERS
               if "neff_over_l" in (ROOT / rel).read_text()]
    chk("neff_over_l is consumed but set by nobody, as declared",
        consumed and not setters,
        "extra[:, 0] is permanently zero in every path (finding 68)"
        if not setters else f"now set by {setters} -- update the declaration")


def probe_device_contracts() -> None:
    """No production path builds a CPU generator for a CUDA tensor.

    The gate runs every suite with `CUDA_VISIBLE_DEVICES=""`, deliberately --
    a neighbour's OOM once turned a passing suite into DRIFT DETECTED. The
    cost is that a device-placement bug is structurally invisible to it, and
    one lived there: stage 5's `evaluate` passed `torch.Generator()`, which
    is a CPU generator whatever the tensors are, into `random_rigid`, so
    every stage-5 run trained to the end of the epoch and then died on the
    FIRST eval batch with "Expected a 'cuda' device type for generator but
    found 'cpu'". It never reported a validation number.

    A source check catches that class on CPU, which is where the gate runs.
    """
    import io
    import re
    import tokenize
    bad = []
    for f in sorted((ROOT / "scripts").rglob("*.py")) + \
             sorted((ROOT / "src").rglob("*.py")):
        if f.name.startswith("test_") or f.name == "probe_channels.py":
            continue   # tests pin their own device; this file states the rule
        text = f.read_text()
        # Strip comments and strings before matching. A scan that reads its
        # own pattern, or the comment explaining the pattern, reports itself
        # -- which is what the first version of this check did, twice.
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            continue
        for t in toks:
            if t.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            if t.type == tokenize.NAME and t.string == "Generator":
                ln = text.split("\n")[t.start[0] - 1]
                if re.search(r"torch\.Generator\(\s*\)", ln):
                    bad.append(f"{f.relative_to(ROOT)}:{t.start[0]}")
    chk("no production path builds a device-less torch.Generator",
        not bad, "none" if not bad else f"{bad}")


def probe_losses() -> None:
    """Does each head's loss actually respond to ITS target?

    A head can produce a correctly-shaped output, be logged every step, and
    have its loss wired to the wrong tensor. Perturbing one target at a time
    and watching which loss moves is the only check that catches that.
    """
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    from pharos.data.loader import Pharos3DDataset
    from pharos.model.pharos import Pharos, PharosConfig
    import train_pharos as T

    root = ROOT / "data/derived/pharos3d"
    if not (root / "manifest.json").exists():
        chk("the 3D corpus is present for the loss probe", False, str(root))
        return
    tr = Pharos3DDataset(root, split="train")
    t = T.to_device(tr.collate([0, 1, 2, 3, 4, 5]), torch.device("cpu"))
    if not bool(t["rigidity_mask"].any()):
        notes.append("no X-ray chain in the probe batch; heads 6/7 not exercised")
    cfg = PharosConfig(d_model=48, n_blocks=8, n_loops=2, n_heads=4, d_pair=24,
                       n_experts=4, d_expert=24, top_k=2, n_shared=1,
                       max_length=int(t["tokens"].shape[1]) + 8)
    torch.manual_seed(0)
    mdl = Pharos(cfg).eval()

    def run(tt):
        torch.manual_seed(1234)     # same sampled pairs and recycling count
        with torch.no_grad():
            return T.step_losses(mdl, tt, cfg, n_neg=8, n_loops=2)[1]

    base = run(t)
    cases = [
        ("mg_site",    ["mg"],                      lambda d: 1 - d),
        ("b_factor_z", ["rigidity", "fluctuation"], lambda d: d + 3.0),
        ("lw_val",     ["lw"],                      lambda d: (d + 1) % 13),
        ("loop_class", ["motif"],                   lambda d: (d + 1) % 3),
        # NOT a rigid motion: the structure loss is translation-invariant by
        # design, which is asserted separately below
        ("coords",     ["structure"],               lambda d: d * 1.5),
    ]
    for key, watch, fn in cases:
        tt = dict(t)
        tt[key] = fn(t[key].clone())
        got = run(tt)
        for w in watch:
            a, b = base.get(w), got.get(w)
            if a is None or b is None:
                chk(f"{w} is produced for this batch", False, f"{w} absent")
                continue
            chk(f"perturbing {key} moves {w}", abs(float(a) - float(b)) > 1e-9,
                f"{float(a):.5f} -> {float(b):.5f}")
    # An absent target must never reach a loss. 11.49% of residues carry NaN
    # for b_factor_z -- the entries that have no measured B-factor, which
    # before finding 66 carried a fabricated one -- so the mask is now load
    # bearing: one leak turns the whole rigidity loss into NaN.
    import numpy as _np
    leaked = tot_nan = 0
    rng = _np.random.default_rng(0)
    for _ in range(4):
        idxs = rng.choice(len(tr), 6, replace=False).tolist()
        tb = T.to_device(tr.collate(idxs), torch.device("cpu"))
        nan = ~torch.isfinite(tb["b_factor_z"].float())
        tot_nan += int(nan.sum())
        leaked += int((nan & tb["rigidity_mask"]).sum())
    chk("absent rigidity targets never reach the loss", leaked == 0,
        f"{tot_nan:,} NaN targets, {leaked} inside rigidity_mask")

    # and the one invariance that must HOLD: a rigid translation changes
    # nothing, or the structure head is learning the crystal frame
    tt = dict(t)
    tt["coords"] = t["coords"] + 7.0
    got = run(tt)
    chk("a rigid translation leaves the structure loss alone",
        abs(float(base["structure"]) - float(got["structure"])) < 1e-9,
        f"{float(base['structure']):.5f} -> {float(got['structure']):.5f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--skip-model", action="store_true")
    a = ap.parse_args()
    print("PHAROS channel probe -- does every channel carry information,")
    print("and does that information reach the model?")
    probe_corpus(a.shards)
    if not a.skip_model:
        probe_model()
        section("B4. every call site conditions the router the same way")
        probe_router_conditioning()
        section("B5. device contracts the CPU-only gate cannot see")
        probe_device_contracts()
        section("B6. every head's loss responds to its own target")
        probe_losses()
    if notes:
        print("\nRecorded, not failures:")
        for s in notes:
            print(f"  - {s}")
    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    # The gate greps stdout for this exact sentinel -- `ok = returncode == 0
    # and "ALL TESTS PASS" in stdout` -- so a probe that passes while printing
    # its own wording is reported as DRIFT with every check green. That is the
    # defect this file exists to catch, in the file itself.
    print("ALL TESTS PASS -- every channel carries information "
          "and reaches the model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
