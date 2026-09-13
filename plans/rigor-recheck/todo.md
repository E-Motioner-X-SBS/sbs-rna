# Master To-Do — Cycle 3

Legend: [ ] open · [x] done+tested · [BLOCKED: r] · [DEFERRED: r]

## Priority 0 — precision (user's specific ask)
- [ ] P1a Verify NF4 (QLoRA NormalFloat4): what it is, what it costs, where valid
- [ ] P1b Verify NVFP4 / Blackwell FP4 and current FP8 training practice
- [ ] P1c Does 4-bit apply to TRAINING or only inference/adapters? (decisive)
- [ ] P2a Measure sequence information content (entropy/nt) from the catalogue
- [ ] P2b Measure coordinate precision floor vs experimental resolution
- [ ] P2c Measure precision needed for B-factors, stiffness, distances
- [ ] P3  Derive a per-tensor-class precision budget from P2

## Priority 1 — datasets -> model attributes
- [ ] P4a Inventory every per-nucleotide / per-pair statistic available
- [ ] P4b Rank by information gain per bit stored
- [ ] P4c Specify the token attribute vector that replaces spent precision

## Priority 2 — repo
- [ ] P5  Implementation layout: src/, configs/, data loaders, checkpoints, tests

## Priority 3 — architecture
- [ ] P6  Improvements implied by P1-P5; propagate to all three deliverables
