# Master To-Do — Cycle 3

Legend: [ ] open · [x] done+tested · [BLOCKED: r] · [DEFERRED: r]

## Priority 0 — precision (user's specific ask)
- [x] P1a Verify NF4 (QLoRA NormalFloat4): what it is, what it costs, where valid
- [x] P1b Verify NVFP4 / Blackwell FP4 and current FP8 training practice
- [x] P1c Does 4-bit apply to TRAINING or only inference/adapters? (decisive)
- [x] P2a Measure sequence information content (entropy/nt) from the catalogue
- [x] P2b Measure coordinate precision floor vs experimental resolution
- [DEFERRED: subsumed] P2c B-factor/stiffness precision — subsumed by P2b:
          at 3.10 A median resolution nothing downstream needs fp16
- [x] P3  Derive a per-tensor-class precision budget from P2

## Priority 1 — datasets -> model attributes
- [x] P4a Inventory every per-nucleotide / per-pair statistic available
- [x] P4b Rank by information gain per bit stored
- [x] P4c Specify the token attribute vector that replaces spent precision

## Priority 2 — repo
- [x] P5  Implementation layout: src/, configs/, data loaders, checkpoints, tests

## Priority 3 — architecture
- [x] P6  Improvements implied by P1-P5; propagate to all three deliverables

## Cycle-3 additions
- [x] QAT researched and decided (D4: deployment technique, deferred)
- [x] QiD scaling law applied to our size ladder -> the conflict, 3 reversals
- [x] Repo layout created: src/pharos/, configs/, with a tested physics module
- [DEFERRED: needs hardware] Benchmark Muon 2.00x on our actual shapes
- [DEFERRED: needs training] Validate the 25B stopping point empirically

## Cycle 4 — internal coherence
- [x] C1 recycle x3 vs 8 loops -> DEFECT #16, ladder now separates train/serve depth
- [x] C2 loops in the FLOP budget -> DEFECT #17, all costs corrected, guard extended
- [x] C3 -> DEFECT #18: never specified. Fixed at 8 heads x 64 dims.
- [x] C4 -> DEFECT #19: the 10M line was a placeholder and the DECODER was never
          specified, so it could not be checked. Itemised: heads 1.65M (cycle-3
          additions only 0.80M), motif bank 0.17M, 4-block frame decoder 12.59M.
          The decoder alone exceeds the line. Totals 149M/61M -> ~153M/~65M.
- [x] C5 [ADDED-CYCLE-4] DEFECT #18b: my own cycle-2 global replace corrupted
          5 parentheticals into "46,447 RNA all-polymer". Repaired; diagrams
          re-rendered. Same unguarded-replace class cycle 1 warned about.
