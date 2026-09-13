# PHAROS — Decided Parameters and Formats (cycle 3)

> Every decision here traces to a measurement in `data/samples/analysis/` or to
> a source verified by search this session. Where a prior decision is overturned
> it says so.

## The conflict that forced these decisions

The design simultaneously wanted three things that **cannot all hold**:

| | Want | From |
|---|---|---|
| (a) | the **smallest viable model** (61M active) | cycle-2 right-sizing |
| (b) | the **full 323B-token corpus** (one epoch of elDORS) | data availability |
| (c) | **4-bit** training/deployment | cycle-3 precision work |

**"Low-Bit Quantization Favors Undertrained LLMs"** (ACL 2025, arXiv 2411.17691,
1500+ checkpoints) finds quantization-induced degradation (QiD) is **worst for
small models trained on many tokens**, and states directly that *"smaller models
require higher precision at high training token counts."*

PHAROS-Small at 61M active × 323B tokens is **5,295 tokens/parameter = 265×
Chinchilla-optimal**. That is the worst-case corner of exactly that law.

| Model | active | tok/param @323B | × Chinchilla | 4-bit verdict |
|---|---|---|---|---|
| PHAROS-Micro | 12M | 26,917 | 1,346× | **worst case — avoid** |
| PHAROS-Mini | 30M | 10,767 | 538× | **worst case — avoid** |
| **PHAROS-Small** | **61M** | **5,295** | **265×** | **worst case — avoid** |
| Base-v2 | 269M | 1,201 | 60× | risky — validate QiD first |

RNA makes this **worse, not better**. Measured sequence entropy is **2.0167
bits/nt** against ~11 bits for an English token, so 323B RNA tokens carry
~0.1 TB of information where 323B LM tokens carry ~0.4 TB. The effective
over-training is larger than the raw ratio suggests.

---

## D1 — Model size: **KEEP PHAROS-Small (61M active, 149M total)**

Unchanged. The right-sizing argument survives: the 3D task carries ~2.8 MB of
information and is data-limited, not capacity-limited. Ladder retained
(Micro/Mini/Small/Base-v2) with Small as default.

## D2 — Token budget: **CUT 323B → staged 25B, stop when metrics flatten** [REVERSED]

**Overturns** the earlier "323B tokens, deliberate over-training is standard".
That defence was argued for **269M** active (1,201 tok/param). At **61M** it
becomes 5,295 tok/param and puts the model in the QiD worst case for no
information gain — the corpus is redundant, not rich.

- 25B tokens → **410 tok/param ≈ 20× Chinchilla**. Still deliberately
  over-trained for inference efficiency, without entering the degradation corner.
- The staged schedule (5B/25B/100B/323B) already exists; the change is that
  **25B is now the expected stopping point, not a waypoint.**
- The 151-nt read down-weighting is promoted from a cost lever to a
  **redundancy-removal requirement**: 9 of 20 chunks have median length exactly
  151 nt and carry 16.2% of nucleotide mass with little long-range signal.

## D3 — Training format: **BF16 first, FP8 on Hopper. NOT 4-bit.** [REVERSED]

| Format | Decision | Reason |
|---|---|---|
| **BF16** | **train here** | precision is not the constraint (139× over-provisioned); a correct baseline matters more than speed |
| **FP8** (Hopper+) | **adopt after baseline** | `<0.25%` loss error vs BF16, validated at DeepSeek V2/V2-Lite scale (~1T tokens), *not* at 671B |
| **NVFP4** (Blackwell) | **do NOT adopt for this configuration** | the pretraining recipe was validated at **8B/1T ≈ 125 tok/param**. PHAROS-Small is **5,295** tok/param — a different regime entirely. The "no measurable loss" result does not transfer. |
| **MXFP4** | no | needs ~36% more tokens than NVFP4 for equal loss |
| **NF4** | **never** | W4A16 with a **frozen** 4-bit base; gradients reach only LoRA adapters. It is a finetuning/inference format. PHAROS trains from scratch. |

Keep in higher precision regardless: **embeddings, norms, router, attention
softmax** — this is what DeepSeek's own FP8 recipe does.

## D4 — Quantization-aware training: **defer, and only for deployment**

QAT is a *deployment* technique here, not a training one. It is worth it only if
4-bit inference is actually required.

- QAT beats PTQ substantially at INT4; LLM-QAT reaches within 1–2% of FP.
- But those results are on **large** models. At 61M active and 265× Chinchilla
  the QiD law predicts the largest degradation, so the recovery task is hardest
  exactly here.
- **Decision**: no QAT in stage 1. If 4-bit inference is later required, run QAT
  (not PTQ) **and measure QiD at the actual (size, token) point** rather than
  assuming a published number transfers.
- Straight-through estimator caveat recorded: gradients after QAT tend toward
  zero, biasing STE and reducing convergence stability.

## D5 — Where the freed budget goes: **attributes, not mantissa** [user's insight, corrected]

Measured: a fully attributed nucleotide carries **58.9 bits**; a 512-dim bf16
token holds **8,192 bits** — **139× over-provisioned**. So `d_model` is *compute*
space, not storage.

**The correct form of "more attributes instead of more precision"**:
attributes enter as **input features through one projection**. They do **not**
justify a wider `d_model`, because training FLOPs go as `6·N_active` and
widening `d` is quadratic in the FFN. Attributes are nearly free; width is not.

At inference, only **26.0 bits of 58.9** are available for an arbitrary sequence
(identity, position, local GC, recycled pairing state, ionic condition,
in-complex flag). The rest are **supervision targets, not inputs** — feeding
B-factor or Mg-distance in would leak the answer. `data/attributes.py` keeps the
two sets physically separate.

## D6 — Vocabulary: **two vocabularies, one model** [changed]

Finding G7: **8.90%** of structural residues fall outside `{A,C,G,U}`.

- **Pretraining**: 5 symbols. elDORS is pre-normalised; widening adds dead
  tokens — NucleicBERT's vocab-25 mistake.
- **Structural**: 5 + DNA + inosine + UNK + a learned `MOD` embedding keyed by
  mmCIF `comp_id`. Until built, the honest statement is that modified residues
  map to `N` and their geometry is not predicted.

## D7 — Cost, hardware-explicit [defect #15 corrected]

A100 (Ampere) has **no FP8 tensor cores**, so the published "~79 A100-hours"
mixed an Ampere baseline with a Hopper-only lever and existed on no real machine.

At the **revised 25B-token budget**, PHAROS-Small:

| Hardware + format | GPU-hours (with Muon + read down-weighting, 2.38×) |
|---|---|
| A100 bf16 | **~10** |
| H100 fp8 | **~1.6** |

(323B-token figures, for reference: 126 h A100 bf16, 20 h H100 fp8.)

**The token-budget cut is a far larger saving than any precision lever** — 12.9×
against FP8's 1.6× — and it costs no accuracy, because the tokens being dropped
are redundant rather than informative.

---

## Summary of reversals

| # | Was | Now | Why |
|---|---|---|---|
| REV-3 | 323B tokens, over-training is standard | **25B staged, stop on plateau** | argued at 269M; at 61M it is 265× Chinchilla and the QiD worst case |
| REV-4 | FP8 1.60× lever on "A100-hours" | **hardware-explicit; A100 has no FP8** | defect #15 — not a real unit |
| REV-5 | 4-bit as a goal (NVFP4/NF4) | **BF16 → FP8; no 4-bit at this size** | NF4 cannot train from scratch; NVFP4 validated at 125 tok/param, we are at 5,295 |
