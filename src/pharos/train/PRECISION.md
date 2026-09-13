# Numeric precision — measured, not conventional

## What the data supports

| Measurement | Value | Implication |
|---|---|---|
| elDORS composition entropy | **2.0167 bits/nt** | a 512-dim bf16 token is **4,062x** the identity content |
| Full attribute set per nucleotide | **58.9 bits** | still **139x** over-provisioned |
| Catalogue median resolution | **3.10 A** | fp16 coordinates are **3,100x** finer than the noise floor |

**Nothing in the RNA token path is stored in the mantissa.** Low precision is
therefore safe on information grounds. The binding constraint is FLOPs
(`6 * N_active * T`), not bits.

## Formats: what actually applies

| Format | Trainable? | Requires | Verdict for PHAROS |
|---|---|---|---|
| **NF4** (QLoRA NormalFloat4) | **No** | — | **Wrong tool.** W4A16 only, and the 4-bit base is *frozen* — gradients flow only to LoRA adapters. It is a finetuning/inference format for an existing backbone. PHAROS trains from scratch. |
| **MXFP4** | yes | Blackwell | needs **~36% more tokens** than NVFP4 to reach the same loss at 8B/1T |
| **NVFP4** | **yes** | **Blackwell** | E2M1, 16-element blocks, FP8 (E4M3) scale. Pretraining recipe reports **no measurable accuracy loss vs an FP8 baseline**. |
| **FP8** | yes | Hopper+ | `<0.25%` loss error vs BF16 (validated at DeepSeek V2/V2-Lite scale, ~1T tokens — *not* at 671B) |
| **BF16** | yes | any | baseline |

## Hardware-explicit cost — PHAROS-Small, 323B tokens

Defect #15: the earlier model quoted **A100-hours** while applying a **1.60x FP8**
lever. **A100 (Ampere) has no FP8 tensor cores.** That figure existed on no real
machine.

Levers split by what they depend on:
- **hardware-independent** (any generation): Muon 2.00x, read down-weighting 1.19x -> **2.38x**
- **hardware-dependent**: FP8 needs Hopper+, NVFP4 needs Blackwell

| Hardware + format | raw GPU-h | with levers |
|---|---|---|
| A100 bf16 | 299 | **126** |
| H100 bf16 | 94 | 40 |
| **H100 fp8** | 47 | **20** |
| B200 bf16 | 41 | 17 |
| **B200 NVFP4** | 10 | **4** |

**Honest headline**: 126 h on A100 bf16 · 20 h on H100 fp8 · 4 h on B200 NVFP4.

## The decisive constraint: model size x token count

**"Low-Bit Quantization Favors Undertrained LLMs"** (ACL 2025, arXiv 2411.17691,
1500+ checkpoints): quantization-induced degradation is **worst for small models
trained on many tokens**. Directly: *"smaller models require higher precision at
high training token counts."*

| Model | active | tok/param @323B | x Chinchilla | 4-bit |
|---|---|---|---|---|
| Micro | 12M | 26,917 | 1,346x | worst case |
| Mini | 30M | 10,767 | 538x | worst case |
| **Small** | **61M** | **5,295** | **265x** | **worst case** |
| Base-v2 | 269M | 1,201 | 60x | risky |

The NVFP4 pretraining validation was at **8B / 1T tokens = 125 tok/param**.
PHAROS-Small is **5,295**. The "no measurable accuracy loss" result is from a
different regime and does not transfer.

RNA makes it worse: measured entropy **2.0167 bits/nt** vs ~11 for an English
token, so the corpus is more redundant than the token count implies.

**This is why the token budget was cut rather than the precision** (D2/REV-3).
25B tokens gives 410 tok/param and saves **12.9x** — against FP8's 1.6x — with
no accuracy cost, because the dropped tokens are redundant.

## QAT

A *deployment* technique here, not a training one. QAT beats PTQ at INT4 and
LLM-QAT reaches within 1-2% of FP — but on **large** models. At 61M active and
265x Chinchilla the QiD law predicts the largest degradation, so recovery is
hardest exactly here. **No QAT in stage 1.** If 4-bit inference is later
required, use QAT rather than PTQ and **measure QiD at our own (size, tokens)
point**. Recorded caveat: post-QAT gradients tend toward zero, biasing the
straight-through estimator and reducing convergence stability.

## Recommendation

1. Train in **BF16 first** — the model is small enough that precision is not the
   constraint, and a correct baseline matters more than speed.
2. **FP8 on Hopper** once the baseline is reproduced; the <0.25% figure is
   well-attested.
3. **NVFP4: do NOT adopt at this size.** It is a genuine pretraining format
   (unlike NF4), but its validation point is 125 tok/param and we are at 5,295.
4. **Never NF4** for this model. It cannot train a from-scratch backbone.
5. Keep **coordinates and distances in a binned/int representation**: at 3.10 A
   median resolution, storing them finer than ~0.1 A encodes noise.
6. Keep embeddings, norms, the router and attention softmax in higher precision
   — DeepSeek's own FP8 recipe does exactly this.
