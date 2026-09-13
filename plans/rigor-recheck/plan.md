# Plan — Cycle 3: core architecture development

## What the user asked for

1. Develop the **core architecture** and the **dataset** side, rigorously.
2. Analyse how *every* RNA statistic can be used to improve the model.
3. **Update the whole repo** — where implementation code will actually live.
4. **Precision**: FP8 vs NVIDIA **NF4** / 4-bit. The user's insight is the key
   one: *RNA information is extremely constrained* — 4 symbols, positions, 3D
   coordinates. So where is precision actually a bottleneck? If it isn't, spend
   the saved bits on **more attributes per token** instead of more mantissa.
5. Audit the architecture again; find improvements.

## The precision hypothesis, stated so it can be falsified

**H1**: An RNA nucleotide token carries ~2 bits of identity. Representing it in
a 512-dim bf16 vector spends 8,192 bits. The bottleneck is therefore *not*
numeric precision anywhere in the token path, and precision can be traded for
**feature breadth** (more attributes per nucleotide) at equal memory.

**H2**: 3D coordinates do not need fp16. Our own catalogue records median
resolution **3.10 A**; coordinate precision finer than ~0.1 A is below the
experimental noise floor. If true, coordinates/distances can be int8/fp8-binned
without information loss.

Both are testable against data already in the repo. **Neither may be asserted
until measured** (rule 5).

## Step-locked order

| Step | Target |
|---|---|
| P1 | Verify NF4 / NVFP4 / FP8 facts by search (rules 1, 10) — no claim from memory |
| P2 | Measure actual information content: sequence entropy, coordinate precision floor, B-factor and stiffness precision |
| P3 | Derive the precision budget per tensor class from P2, not from convention |
| P4 | Dataset statistics -> concrete per-token attribute list ("more attributes, fewer bits") |
| P5 | Repo layout for implementation: where code, configs, checkpoints, data loaders go |
| P6 | Architecture improvements implied by P1-P5; update all three deliverables |

## Success criterion

Every precision recommendation traces to a measured number or a cited source,
never to "fp16 is standard". Every new attribute traces to a statistic measured
on the catalogue or the sampled structures.
