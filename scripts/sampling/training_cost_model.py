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
    """Training FLOPs INCLUDING refinement loops.

    Defect #17: this previously omitted `loops` entirely. Section 10b.3 specifies
    deep supervision at EVERY segment with a one-step (detached) gradient, so each
    of the `loops` segments is its own forward+backward. Cost therefore scales
    linearly with the loop count. Activation MEMORY does not -- that is what the
    detach buys -- but compute does.
    """
    per_pass = 6 * n_active * tokens + 3 * attn_flops_per_token(L) * tokens
    return loops * per_pass

# ---- realistic hardware ----
HW = {"A100-80G bf16": 312e12, "H100-80G bf16": 989e12, "H100-80G fp8": 1979e12}
MFU = 0.35      # model FLOPs utilisation; 35% is a realistic MoE figure

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
