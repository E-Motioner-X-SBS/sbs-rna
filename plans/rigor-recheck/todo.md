# Master To-Do — Cycle 5

Legend: [ ] open · [x] done+tested · [BLOCKED: r] · [DEFERRED: r]

## Priority 0 — buildability
- [x] B1  Component inventory: SPECIFIED / PARTIAL / MISSING
- [x] B2  Circularity sweep beyond the router (motif bank is the prime suspect)
- [x] B3  MoE expert width: d_ff=128 at d=512 is 0.25x — justified or arbitrary?
- [x] B4  Loss weights lambda_1..lambda_5 — specified anywhere?
- [x] B5  Close or explicitly defer every MISSING item

Outcome: 35/37 specified; 2 deliberately PROVISIONAL (loss weights, motif-bank
key construction) with documented procedures. Defects #20 and #21 found.
- [DEFERRED: 5B checkpoint] top_k 4 -> 8 decision (defect #21)
- [DEFERRED: 5B checkpoint] LR, warmup, loss-weight sweep
