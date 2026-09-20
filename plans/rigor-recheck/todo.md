# Master To-Do — Cycles 8-11

Legend: [ ] open · [x] done+tested · [BLOCKED: r] · [DEFERRED: r]

## Priority 0 — audit the verification apparatus
- [x] C25 Measure both mmCIF serialisations across every category the scripts read
- [x] C26 Fix `loops()`; re-run geometry + both stiffness-headroom analyses
- [x] C27 Propagate the deltas (56 exact-match replacements across 8 files)
- [x] C28 Compare every guard tolerance to the precision of the value it guards
- [x] C29 Derive the default tolerance from how the value is written
- [x] C30 Adversarially verify the guard rejects the two values that slipped through
- [x] C31 Rebuild main.pdf (40pp, 0 boxes); republish blueprint (v26)

## Closed in cycle 11 — the oldest outstanding items
- [x] C13 Shared canonical geometry loader (`rna_chain_coords`,
      `longest_rna_chain`) added to `mmcif_entities.py`; 7 new test properties
- [x] C14 Definition sensitivity measured chain-by-chain over all 179 structures:
      34.1% of chains change length, worst +27.9%, but max effective c moves
      19.03 -> 19.04 and **nothing breaches the c=20 budget under either rule**
- [x] C16 Vectorise the comparison with a cKDTree — the first implementation did
      not terminate, which is indistinguishable from never having been run
- [x] C17 Pin 8 new claims and 6 new tokens; rebuild (41pp, 0 boxes); publish v29
- [x] C32 Audit the cross-document token check (OQ-6) — **defect #28**
- [x] C33 Adversarially verify it by deleting every occurrence of the claims
- [x] C34 Fix the two latent truncated tokens exposed by the tightening
- [x] C35 Boundary matching + context patterns; print which mode each check used

## Deferred
- [x] **OQ-7 CLOSED — NEGATIVE. `target_c = 20` is breached.** Measured on
  **20,266 chains** (RNA3DB + RNASolo, the server corpus) instead of 180:
  worst chain **effective c = 21.14** (`7PAS_1_3`), 5.7% over budget; 1 chain
  in 20,266; p99 19.13, p99.9 19.52. Bin means reproduce (17.14 vs published
  17.2) — only the tail was wrong, which n=180 could not have estimated.
  Caveat: this corpus carries **0.025%** non-ACGU residues vs 1.05% in raw-PDB
  BGSU, a 42x under-representation of the driver, so the true tail is likely
  worse. `target_c` must be raised (24 clears the observed max by 12%) or the
  track must handle overflow. `scripts/sampling/recheck_block_sparsity_fullcorpus.py`,
  `data/samples/analysis/block_sparsity_fullcorpus.json`.
- [DEFERRED: 5B checkpoint] top_k 4 -> 8 decision (defect #21)
- [DEFERRED: 5B checkpoint] LR, warmup, loss-weight sweep
- [DEFERRED: needs external data] RMDB Mg2+ titration acquisition
