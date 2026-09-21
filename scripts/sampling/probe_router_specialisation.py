"""Is the router specialising, or collapsed to uniform?

The Switch balance loss sits at its analytic floor -- 0.161 nats, which is
exactly `balance_weight` 0.01 x 16 blocks x 1.0, and 1.0 is what
`n_experts * sum(frac * pbar)` evaluates to at perfect uniformity. That says
the LOAD is even. It says nothing about whether any individual token is routed
sharply, and the two are opposite situations:

  specialising   each token concentrates on a few experts; the MEAN over tokens
                 is still uniform because different tokens pick different ones.
                 This is what a MoE is for.
  collapsed      every token spreads over every expert at ~1/E. The mean is
                 also uniform. This is a dense model paying MoE's memory bill.

`router_entropy` in the aux dict cannot tell them apart -- it is the entropy of
the mean, which is log(n_experts) in both cases by construction. The entropy of
each token's OWN distribution can, and this measures it, on a checkpoint,
without disturbing a run.

Reads whatever `pretrain_small.pt` currently holds, so it can be re-run against
a training job as it progresses to get a trajectory rather than a snapshot.

Usage: python3 scripts/sampling/probe_router_specialisation.py
"""
import sys, math, numpy as np, torch
from pathlib import Path
ROOT = Path("/store/shuvam/E-motioner-X-SBS/sbs-rna")
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"scripts"))
import pretrain_mlm as M
from pharos.model.pharos import Pharos, PharosConfig
from pharos.model.moe import RouterFeatures, MoEFeedForward
from pharos.data.chemistry_torch import BatchChemistry
from pharos.data.vocab import SYMBOLS

dev = torch.device("cuda")
ck = ROOT/"data/derived/checkpoints/pretrain_small.pt"
sd = torch.load(ck, map_location="cpu", weights_only=False)
cfg = PharosConfig(**sd["cfg"])
model = Pharos(cfg).to(dev)
model.load_state_dict({k.replace("._orig_mod.", "."): v for k, v in sd["model"].items()})
model.eval()
print(f"checkpoint: {sd['tokens']/1e6:.1f}M tokens, step {sd['step']:,}")

ent, top1, usage = [], [], []
def hook(mod, args, kwargs, out):
    x = args[0]; mask = args[1] if len(args) > 1 else kwargs.get("mask")
    h = mod.norm(x)
    B, L, D = x.shape
    cond = (kwargs.get("feats") or (args[2] if len(args) > 2 else None)
            or RouterFeatures()).vector(B, x.device, mod.cfg.n_length_bins,
                                        mod.cfg.d_router_extra)
    gin = torch.cat([h, cond.unsqueeze(1).expand(B, L, -1).to(h.dtype)], dim=-1)
    p = torch.softmax((mod.gate(gin) + mod.expert_bias).float(), -1)
    m = mask.unsqueeze(-1).float()
    n = m.sum().clamp(min=1)
    ent.append(float((-(p.clamp_min(1e-9).log()*p).sum(-1, keepdim=True)*m).sum()/n))
    top1.append(float((p.max(-1, keepdim=True).values*m).sum()/n))
    usage.append(((p*m).sum((0,1))/n).cpu().numpy())

hs = [m_.register_forward_hook(hook, with_kwargs=True)
      for m_ in model.modules() if isinstance(m_, MoEFeedForward)]
print(f"{len(hs)} MoE blocks hooked")

bc = BatchChemistry(SYMBOLS, dev)
rng = np.random.default_rng(0)
with torch.no_grad():
    for i, g in enumerate(M.iter_batches(M.CORPUS, 20, 1024, 12288, 256, rng, shards=1)):
        tok, mask, lengths = M.encode_batch(g)
        tk = torch.as_tensor(tok, device=dev); mk = torch.as_tensor(mask, device=dev)
        chem = bc(tk, mk); n = mk.sum(1)
        f = RouterFeatures(length=n.float(),
                           chem_summary=(chem.sum(1)/n.unsqueeze(1).clamp(min=1))[:, :5])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(tk, torch.zeros_like(tk), chem, mk, feats=f, n_loops=2, mlm=True)
        if i >= 5: break
for h in hs: h.remove()

E = cfg.n_experts
mx = math.log(E)
u = np.stack(usage)
print(f"\nn_experts {E}, uniform entropy log({E}) = {mx:.3f}")
print(f"  per-token routing entropy   {np.mean(ent):.3f}  "
      f"({100*np.mean(ent)/mx:.1f}% of uniform)")
print(f"  mean top-1 probability      {np.mean(top1):.3f}  (uniform = {1/E:.3f})")
print(f"  mean expert load, min/max   {u.mean(0).min():.4f} / {u.mean(0).max():.4f}"
      f"  (uniform = {1/E:.4f})")
print(f"  per-block entropy spread    {np.min(ent):.3f} .. {np.max(ent):.3f}")

import json
OUT = ROOT / "data/samples/analysis"
# The history can hold entries from DIFFERENT runs: `--restart` resets both the
# token count and the step, so two runs collide on either. The checkpoint's
# mtime does not, so it is what distinguishes them.
rec = {"tokens": int(sd["tokens"]), "step": int(sd["step"]),
       "checkpoint_mtime": int(ck.stat().st_mtime),
       "n_experts": E, "uniform_entropy": round(mx, 4),
       "token_router_entropy": round(float(np.mean(ent)), 4),
       "frac_of_uniform": round(float(np.mean(ent)) / mx, 4),
       "mean_top1_prob": round(float(np.mean(top1)), 4),
       "uniform_top1_prob": round(1.0 / E, 4),
       "expert_load_min": round(float(u.mean(0).min()), 5),
       "expert_load_max": round(float(u.mean(0).max()), 5),
       "per_block_entropy": [round(float(e), 4) for e in ent]}
f = OUT / "router_specialisation.json"
hist = []
if f.exists():
    try:
        hist = json.loads(f.read_text()).get("history", [])
    except (OSError, ValueError):
        hist = []
hist = [h for h in hist
        if (h.get("tokens"), h.get("checkpoint_mtime"))
        != (rec["tokens"], rec["checkpoint_mtime"])] + [rec]
hist.sort(key=lambda h: h["tokens"])
f.write_text(json.dumps({"history": hist}, indent=1))
print(f"\n-> {f} ({len(hist)} probe(s) recorded)")
if rec["frac_of_uniform"] > 0.95:
    print("   ROUTER IS EFFECTIVELY UNIFORM: the MoE is not specialising here.")
