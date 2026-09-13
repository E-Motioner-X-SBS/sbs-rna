# Master To-Do — Cycle 2

Legend: [ ] open · [x] done+tested · [BLOCKED: r] · [DEFERRED: r]

## Priority 0 — suspected defects
- [ ] S1  Is the 143,871 unobserved-residue count RNA-only, or does it include
          protein chains from ribosome structures? Now a novelty claim.
- [ ] S2  A100-hours: table 299 vs text ~79 for PHAROS-Small. Reconcile.

## Priority 1 — re-derive the right-sizing
- [ ] S3a PHAROS-Small 149M total / 61M active from component arithmetic
- [ ] S3b 128 vs 96 effective layers; 9.4x total-parameter ratio
- [ ] S3c Mini/Micro rows (67M/30M, 23M/12M) internally consistent

## Priority 2 — re-derive the new measurements
- [ ] S4  Stiffness headroom: NLL values and the three gains
- [ ] S5  Aux block-occupancy loss implemented + tested, not just specified

## Priority 3 — literature (anti-hallucination rules 1, 10)
- [ ] S6a gRNAde multi-state "3-5%, best at 3"
- [ ] S6b RNAnneal "16 experimentally-resolved conformations", 10 states
- [ ] S6c "pentameric scale = minimum range of elastic couplings"
- [ ] S6d Muon ~2x compute efficiency; FP8 <0.25% loss error; HRM +13pp loop

## Priority 0b — architecture audit (user request, cycle 2)
- [x] G1  Length coverage -> single chains fit 4096 (max 3,679); whole entries
          reach 11,478. Scope statement added.
- [x] G2  **98.96% of RNA residues are in protein complexes.** Largest
          generalization hazard; was undocumented. Risk added.
- [x] G3  **93.07% of residues are ribosomal.** Headline stats relabelled.
- [x] G7  **8.90% of residues outside vocab-5** (DNA/UNK/inosine). Risk added.
- [x] G6  Router at recycle 0 undefined -> design gap logged, fix specified.
- [ ] G5  Capacity: 61M active vs ERNIE-RNA's 86M dense, while doing strictly
          more tasks. §12 already flags "Small could underperform" -> DEFERRED
          to training, it is an experiment not an argument.

## Priority 4 — consistency
- [ ] S7  Cross-document scan after 16 commits; guard still catches regressions
