#!/usr/bin/env python3
"""What does PHAROS actually cost to train, and where does the cost go?

The architecture spec gives a parameter budget but never a training cost. This
builds the cost model from first principles so optimisations can be ranked by
measured impact rather than intuition.

Conventions (stated, not assumed):
  - Transformer training FLOPs ~ 6 * N_active * D_tokens  (fwd 2N, bwd 4N)
  - Attention adds ~ 6 * L * d per token per full-attention layer (the term
    6ND omits), which matters at our sequence lengths
  - MoE: FLOPs scale with ACTIVE params; memory scales with TOTAL params
  - Optimizer state: AdamW 2 fp32 moments; Muon 1 momentum buffer
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data" / "samples" / "analysis"

# ---- architecture as specified ----
d, n_blocks = 768, 32
dff_moe, dff_dense = 512, 3072
n_routed, n_shared, topk = 32, 2, 4
n_moe_blocks = n_blocks // 2
n_full, n_swa, n_gdn = 4, 8, 20
w_swa = 128

def params():
    attn = n_blocks * 4 * d * d
    per_expert = 3 * d * dff_moe
    moe_total = n_moe_blocks * (n_routed + n_shared) * per_expert
    moe_active = n_moe_blocks * (topk + n_shared) * per_expert
    dense = (n_blocks - n_moe_blocks) * 3 * d * dff_dense
    other = 45e6 + 35e6          # pair track + heads/decoder
    return {"total": attn + moe_total + dense + other,
            "active": attn + moe_active + dense + other}

P = params()
print(f"PHAROS-Base: total {P['total']/1e6:.0f}M, active {P['active']/1e6:.0f}M\n")

def attn_flops_per_token(L: int) -> float:
    """Extra attention FLOPs per token beyond the 6N term (QK^T and AV)."""
    full = n_full * 2 * 2 * L * d          # 2 matmuls, 2 FLOPs each
    swa  = n_swa  * 2 * 2 * min(w_swa, L) * d
    gdn  = n_gdn  * 2 * 2 * 16 * d         # linear attention: O(1) state per token
    return full + swa + gdn

def train_flops(tokens: float, L: int, n_active: float, loops: int = 1) -> float:
    """Training FLOPs including refinement loops, as the trunk actually runs them.

    Defect #17 added the loop count, which had been omitted entirely, but
    charged every loop a full forward+backward: `loops * 6N`. The implementation
    does not do that. `trunk.py` runs loops 1..N-1 under `torch.no_grad()` and
    differentiates only the last -- that is what "one-step gradient" means -- so
    an intermediate loop costs a forward, 2N, not 6N.

        6N + 2N(loops - 1),  not  6N * loops

    At the specified 3 loops that is 10N against 18N: the old model overcharged
    training by 1.8x. Deep supervision does add backward passes, but through the
    HEADS, on detached inputs, not through the trunk -- a much smaller term and
    not one that scales with N_active.

    `scripts/pretrain_mlm.py` uses the same convention to report MFU, so the
    cost model and the trainer cannot disagree about what a step costs.
    """
    fwd = 2 * n_active * tokens + attn_flops_per_token(L) * tokens
    fwd_bwd = 6 * n_active * tokens + 3 * attn_flops_per_token(L) * tokens
    return fwd_bwd + max(loops - 1, 0) * fwd

# ---- realistic hardware ----
HW = {"A100-80G bf16": 312e12, "H100-80G bf16": 989e12, "H100-80G fp8": 1979e12}

# MODEL FLOPS UTILISATION -- MEASURED, not assumed.
#
# 0.35 stood here as "a realistic MoE figure" and every A100-hour in the
# architecture descended from it. Nothing had ever measured it. Running
# PHAROS-Small on the A100 in this box says otherwise: a profile of a stage-1
# step puts only ~18% of GPU time in tensor-core GEMMs, the rest in elementwise
# work, copies and the delta-rule scan, and the achieved MFU is read off
# `mlm_pretrain.json` below.
#
# The measurement is on PHAROS-Small (63M active, d_model 512). PHAROS-Base has
# d_model 768 and wider matmuls and would do better, so the measured figure is
# a floor for Base rather than an estimate of it -- which is why both are
# printed. What is NOT defensible is continuing to quote 0.35 as though it had
# been observed.
MFU_ASSUMED = 0.35


def measured_mfu() -> float | None:
    """Achieved MFU from the last stage-1 run, or None if it has not run.

    The best interval, not the mean: early intervals include `torch.compile`
    warm-up and the first batch of each of the 8 quantised widths, which are a
    one-off cost of the run rather than a property of the steady state.
    """
    f = OUT / "pretrain_small_results.json"
    if not f.exists():
        return None
    try:
        hist = json.loads(f.read_text()).get("history", [])
    except (OSError, ValueError):
        return None
    vals = [h["mfu"] for h in hist if h.get("mfu")]
    return max(vals) if vals else None


MFU_MEASURED = measured_mfu()
MFU = MFU_MEASURED or MFU_ASSUMED
print(f"MFU: assumed {MFU_ASSUMED:.0%}"
      + (f", MEASURED {MFU_MEASURED:.1%} on PHAROS-Small/A100 -- using the measurement"
         if MFU_MEASURED else ", no measurement on disk -- using the assumption"))

print("=== Stage 1 (MLM pretraining) cost, as specified ===")
print("  corpus: elDORS 1.32B seqs; median ~245 nt cross-chunk => ~336B tokens for 1 epoch")
TOK_FULL = 1.32e9 * 245
for L in (2048,):
    LOOPS = 3   # PHAROS-Base as first specified
    F = train_flops(TOK_FULL, L, P["active"], loops=LOOPS)
    print(f"  tokens {TOK_FULL/1e9:.0f}B, L={L}: {F/1e21:.2f} ZFLOPs")
    for name, peak in HW.items():
        hours = F / (peak * MFU) / 3600
        print(f"    {name:16s} {hours:10,.0f} GPU-hours  = {hours/24/8:7,.0f} days on 8 GPUs")

print("\n=== memory per GPU (optimizer + params + grads) ===")
for opt, mult in [("AdamW fp32 moments", 2), ("Muon momentum only", 1)]:
    for prec, pb in [("bf16 params/grads", 2)]:
        mem = P["total"] * (pb + pb + mult * 4) / 1e9
        print(f"  {opt:22s} {prec:20s} {mem:6.2f} GB (before activations)")
print("  note: MoE memory scales with TOTAL params even though FLOPs scale with ACTIVE")

print("\n=== where does the FLOP budget actually go? ===")
L = 2048
base = 6 * P["active"]
extra = 3 * attn_flops_per_token(L)
print(f"  per token at L={L}:")
print(f"    6*N_active (FFN+proj)   {base/1e9:8.2f} GFLOPs   {100*base/(base+extra):5.1f}%")
print(f"    attention terms          {extra/1e9:8.2f} GFLOPs   {100*extra/(base+extra):5.1f}%")
print(f"  => the model is DOMINATED by the 6N term, not attention.")
print(f"     Cutting context length alone saves little; cutting ACTIVE PARAMS is the lever.")
json.dump({"params": P, "tokens_1epoch": TOK_FULL}, open(OUT/"training_cost_base.json","w"), indent=2)
