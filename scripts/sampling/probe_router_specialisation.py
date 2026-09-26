"""Is the router specialising, or collapsed to uniform?

The Switch balance loss sits near its analytic floor -- `balance_weight *
n_blocks`, because `n_experts * sum(frac * pbar)` evaluates to 1.0 at perfect
uniformity and the trunk sums one term per block. On `shared400` that floor is
0.01 x 18 = 0.180 and the run reads 0.219..0.238. That says the LOAD is even.
It says nothing about whether any individual token is routed sharply, and the
two are opposite situations:

  specialising   each token concentrates on a few experts; the MEAN over tokens
                 is still uniform because different tokens pick different ones.
                 This is what a MoE is for.
  collapsed      every token spreads over every expert at ~1/E. The mean is
                 also uniform. This is a dense model paying MoE's memory bill.

`router_entropy` in the aux dict cannot tell them apart -- it is the entropy of
the mean, which is log(n_experts) in both cases by construction. The entropy of
each token's OWN distribution can, and so can the nucleus width the block
already reports; this measures both, on a checkpoint, without disturbing a run.

Three things this script got wrong for three days, all of the same shape -- it
ran, it was scheduled every six hours, and nothing checked that what came back
meant anything:

  * it read `pretrain_small.pt`, which stopped existing on 2026-09-24 when the
    run moved to `shared400`. Every scheduled invocation since then died on
    `torch.load` and the only trace was the word FAILED in a cron log.
  * it hooked `MoEFeedForward`, and `shared400` builds `SharedMoEFeedForward`,
    which is not a subclass. Had the path been right it would have hooked ZERO
    blocks, averaged an empty list to nan, and written nan into the json.
  * the history mixed runs with nothing recording which model produced a row,
    so a 32-expert entry and a 512-expert one sorted into one trajectory.

So: the target is explicit, the hook covers both block types, the run refuses
to write anything that fails a range check, and every row carries the
checkpoint and the shape of the model it came from.

Usage:
    python3 scripts/sampling/probe_router_specialisation.py
    python3 scripts/sampling/probe_router_specialisation.py --device cpu --batches 2
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/store/shuvam/E-motioner-X-SBS/sbs-rna")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import pretrain_mlm as M                                              # noqa: E402
from pharos.model.pharos import Pharos, PharosConfig                  # noqa: E402
from pharos.model.moe import RouterFeatures, MoEFeedForward           # noqa: E402
from pharos.model.shared_moe import SharedMoEFeedForward              # noqa: E402
from pharos.data.chemistry_torch import BatchChemistry                # noqa: E402
from pharos.data.vocab import SYMBOLS                                 # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default="data/derived/checkpoints/pretrain_shared400.pt",
                help="the checkpoint the LIVE run writes; the default tracked "
                     "pretrain_small.pt long after that file was renamed")
ap.add_argument("--device", default="cuda")
ap.add_argument("--batches", type=int, default=6)
ap.add_argument("--token-budget", type=int, default=12288)
ap.add_argument("--n-loops", type=int, default=0,
                help="0 uses the checkpoint's own cfg.n_loops; the probe used "
                     "to hardcode 2, which measures a depth the config no "
                     "longer trains at")
ap.add_argument("--out", default="data/samples/analysis/router_specialisation.json")
args = ap.parse_args()

ck = ROOT / args.ckpt
if not ck.exists():
    have = sorted(p.name for p in (ROOT / "data/derived/checkpoints").glob("*.pt"))
    sys.exit(f"no checkpoint at {ck}\n"
             f"  the checkpoints that DO exist:\n"
             + "".join(f"    {h}\n" for h in have)
             + "  pass --ckpt, and fix the default if the run has moved.")

dev = torch.device(args.device)
sd = torch.load(ck, map_location="cpu", weights_only=False)
cfg = PharosConfig(**sd["cfg"])
model = Pharos(cfg).to(dev)
model.load_state_dict({k.replace("._orig_mod.", "."): v
                       for k, v in sd["model"].items()})
model.eval()
n_loops = args.n_loops or cfg.n_loops
print(f"checkpoint: {ck.name}, {sd['tokens']/1e6:.1f}M tokens, step {sd['step']:,}")
print(f"model: d_model {cfg.d_model}, {cfg.n_blocks} blocks, "
      f"{cfg.n_experts} experts, n_loops {n_loops}")

ent, top1, usage, widths, wmaxes = [], [], [], [], []


def hook(mod, hargs, kwargs, out):
    x = hargs[0]
    mask = hargs[1] if len(hargs) > 1 else kwargs.get("mask")
    h = mod.norm(x)
    B, L, D = x.shape
    cond = (kwargs.get("feats") or (hargs[2] if len(hargs) > 2 else None)
            or RouterFeatures()).vector(B, x.device, mod.cfg.n_length_bins,
                                        mod.cfg.d_router_extra)
    gin = torch.cat([h, cond.unsqueeze(1).expand(B, L, -1).to(h.dtype)], dim=-1)
    p = torch.softmax((mod.gate(gin) + mod.expert_bias).float(), -1)
    m = mask.unsqueeze(-1).float()
    n = m.sum().clamp(min=1)
    ent.append(float((-(p.clamp_min(1e-9).log() * p).sum(-1, keepdim=True) * m).sum() / n))
    top1.append(float((p.max(-1, keepdim=True).values * m).sum() / n))
    usage.append(((p * m).sum((0, 1)) / n).cpu().numpy())
    # The nucleus width is what `shared400` actually claims, and the block
    # already computes it. Recomputing it from `p` here would measure a
    # different thing -- the pre-nucleus softmax -- so take the block's own.
    aux = out[1] if isinstance(out, tuple) and len(out) > 1 else None
    if isinstance(aux, dict) and "mean_width" in aux:
        widths.append(float(aux["mean_width"]))
        wmaxes.append(float(aux["max_width"]))


BLOCKS = (MoEFeedForward, SharedMoEFeedForward)
hs = [m_.register_forward_hook(hook, with_kwargs=True)
      for m_ in model.modules() if isinstance(m_, BLOCKS)]
print(f"{len(hs)} MoE blocks hooked "
      f"({', '.join(sorted({type(m_).__name__ for m_ in model.modules() if isinstance(m_, BLOCKS)}))})")
if not hs:
    sys.exit("hooked ZERO blocks -- the isinstance filter does not match this "
             "model's feed-forward class, so every number below would be a "
             "mean over an empty list. Refusing to write.")

bc = BatchChemistry(SYMBOLS, dev)
rng = np.random.default_rng(0)
with torch.no_grad():
    for i, g in enumerate(M.iter_batches(M.CORPUS, 20, 1024, args.token_budget,
                                         256, rng, shards=1)):
        tok, mask, lengths = M.encode_batch(g)
        tk = torch.as_tensor(tok, device=dev)
        mk = torch.as_tensor(mask, device=dev)
        chem = bc(tk, mk)
        n = mk.sum(1)
        f = RouterFeatures(length=n.float(),
                           chem_summary=(chem.sum(1) / n.unsqueeze(1).clamp(min=1))[:, :5])
        if dev.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(tk, torch.zeros_like(tk), chem, mk, feats=f,
                      n_loops=n_loops, mlm=True)
        else:
            model(tk, torch.zeros_like(tk), chem, mk, feats=f,
                  n_loops=n_loops, mlm=True)
        if i + 1 >= args.batches:
            break
for h in hs:
    h.remove()

E = cfg.n_experts
mx = math.log(E)
u = np.stack(usage)
mean_ent = float(np.mean(ent))
mean_w = float(np.mean(widths)) if widths else None
max_w = float(np.max(wmaxes)) if wmaxes else None
# the trainer's own definition of a dead expert, so the two agree
dead = float((u.mean(0) < 0.1 / E).mean())

print(f"\nn_experts {E}, uniform entropy log({E}) = {mx:.3f}")
print(f"  per-token routing entropy   {mean_ent:.3f}  ({100*mean_ent/mx:.1f}% of uniform)")
print(f"  mean top-1 probability      {np.mean(top1):.4f}  (uniform = {1/E:.4f})")
print(f"  mean expert load, min/max   {u.mean(0).min():.5f} / {u.mean(0).max():.5f}"
      f"  (uniform = {1/E:.5f})")
print(f"  dead experts (<10% of uniform share)  {100*dead:.1f}%")
print(f"  per-block entropy spread    {np.min(ent):.3f} .. {np.max(ent):.3f}")
if mean_w is not None:
    print(f"  nucleus width, mean/max     {mean_w:.2f} / {max_w:.0f} of {E}")

# ---- basic validity, before anything is written ---------------------------
#
# Every number above is a mean over a list this script filled itself, and the
# failure mode that motivated this rewrite is a list that was empty or a model
# that was the wrong one. A quantity outside the only range it can legally take
# is not a result; it is a bug with a decimal point.
bad = []
if not ent:
    bad.append("no router readings collected")
if not math.isfinite(mean_ent) or not (0.0 < mean_ent <= mx + 1e-6):
    bad.append(f"per-token entropy {mean_ent} outside (0, log({E})={mx:.3f}]")
if mean_w is not None and not (1.0 <= mean_w <= E + 1e-6):
    bad.append(f"mean nucleus width {mean_w} outside [1, {E}]")
if max_w is not None and not (1.0 <= max_w <= E + 1e-6):
    bad.append(f"max nucleus width {max_w} outside [1, {E}]")
if not math.isfinite(float(u.sum())):
    bad.append("expert usage contains non-finite values")
if bad:
    sys.exit("REFUSING TO WRITE -- " + "; ".join(bad))

OUT = ROOT / args.out
# The history can hold entries from DIFFERENT runs and different MODELS:
# `--restart` resets both the token count and the step, so two runs collide on
# either, and a size change makes the entropies incomparable outright. The
# checkpoint name plus its mtime distinguishes them, and `n_experts` is
# recorded so a reader can see at a glance that two rows are not one curve.
rec = {"ckpt": ck.name, "tokens": int(sd["tokens"]), "step": int(sd["step"]),
       "checkpoint_mtime": int(ck.stat().st_mtime),
       # fp32 on cpu against bf16 autocast on cuda: the softmax that these
       # entropies come from is not the same arithmetic, so a series must say
       # which one it is -- the same column the held-out csv had to grow
       "device": dev.type,
       "d_model": cfg.d_model, "n_blocks": cfg.n_blocks, "n_loops": n_loops,
       "n_experts": E, "uniform_entropy": round(mx, 4),
       "token_router_entropy": round(mean_ent, 4),
       "frac_of_uniform": round(mean_ent / mx, 4),
       "mean_top1_prob": round(float(np.mean(top1)), 4),
       "uniform_top1_prob": round(1.0 / E, 4),
       "expert_load_min": round(float(u.mean(0).min()), 6),
       "expert_load_max": round(float(u.mean(0).max()), 6),
       "dead_expert_frac": round(dead, 4),
       "mean_width": None if mean_w is None else round(mean_w, 3),
       "max_width": None if max_w is None else int(max_w),
       "per_block_entropy": [round(float(e), 4) for e in ent]}
hist = []
if OUT.exists():
    try:
        hist = json.loads(OUT.read_text()).get("history", [])
    except (OSError, ValueError):
        hist = []
# rows written before this script knew which checkpoint it had read all came
# from the small run, because that is the only path it could ever open
for h in hist:
    h.setdefault("ckpt", "pretrain_small.pt")
hist = [h for h in hist
        if (h.get("ckpt"), h.get("tokens"), h.get("checkpoint_mtime"))
        != (rec["ckpt"], rec["tokens"], rec["checkpoint_mtime"])] + [rec]
hist.sort(key=lambda h: (h.get("ckpt", ""), h.get("tokens", 0)))
OUT.write_text(json.dumps({"history": hist}, indent=1, allow_nan=False))
print(f"\n-> {OUT} ({len(hist)} probe(s) recorded)")

if rec["frac_of_uniform"] > 0.95 and (mean_w is None or mean_w > 0.5 * E):
    print("   ROUTER IS EFFECTIVELY UNIFORM: the MoE is not specialising here.")
elif mean_w is not None:
    print(f"   routing is sharp: {mean_w:.1f} of {E} experts per token "
          f"({100*mean_w/E:.1f}%), {100*dead:.1f}% dead.")
