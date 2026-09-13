# Master To-Do — Cycle 2

Legend: [ ] open · [x] done+tested · [BLOCKED: r] · [DEFERRED: r]

## Priority 0 — suspected defects
- [x] S1  Is the 143,871 unobserved-residue count RNA-only, or does it include
          protein chains from ribosome structures? Now a novelty claim.
- [x] S2  A100-hours: table 299 vs text ~79 for PHAROS-Small. Reconcile.

## Priority 1 — re-derive the right-sizing
- [x] S3a PHAROS-Small 149M total / 61M active from component arithmetic
- [x] S3b 128 vs 96 effective layers; 9.4x total-parameter ratio
- [x] S3c Mini/Micro rows (67M/30M, 23M/12M) internally consistent

## Priority 2 — re-derive the new measurements
- [x] S4  Stiffness headroom: NLL values and the three gains
- [x] S5  Aux block-occupancy loss implemented + tested, not just specified

## Priority 3 — literature (anti-hallucination rules 1, 10)
- [x] S6a gRNAde -> VERIFIED this session
- [x] S6b RNAnneal -> VERIFIED this session
- [x] S6c pentameric couplings -> VERIFIED this session
- [x] S6d Muon VERIFIED exact; HRM VERIFIED; FP8 VERIFIED but **misattributed to
          671B** (defect #14, corrected) and the HRM **memorization caveat did not
          propagate** to where the argument is made (corrected)

## Priority 0b — architecture audit (user request, cycle 2)
- [x] G1  Length coverage -> single chains fit 4096 (max 3,679); whole entries
          reach 11,478. Scope statement added.
- [x] G2  **98.96% of RNA residues are in protein complexes.** Largest
          generalization hazard; was undocumented. Risk added.
- [x] G3  **93.07% of residues are ribosomal.** Headline stats relabelled.
- [x] G7  **8.90% of residues outside vocab-5** (DNA/UNK/inosine). Risk added.
- [x] G6  Router at recycle 0 undefined -> design gap logged, fix specified.
- [DEFERRED: needs training, not analysis] G5 Capacity: 61M active vs
          ERNIE-RNA's 86M dense while doing strictly more tasks. Already flagged
          in §12 ("Small could simply underperform"), and the staged token budget
          makes the first informative checkpoint ~1 GPU-day. This is an
          experiment, not an argument, and cannot be closed on paper.

## Priority 4 — consistency
- [x] S7  Cross-document scan -> **PASS**. 19/19 shared tokens consistent across
          all three documents; all 7 retracted phrasings confirmed absent. The 4
          apparent gaps were formatting only (LaTeX `{,}` separators, `8.9` vs
          `8.90`), not drift. Guard re-verified to fail on a tampered document.
