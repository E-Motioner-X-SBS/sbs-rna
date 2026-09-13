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
- [DEFERRED: needs a larger sample] **OQ-7** — `target_c = 20` has only **4.8%
  headroom** at the worst chain (19.04). Adequate on these 180 structures; the
  quantity that pushes a chain up is modified-residue density, which this corpus
  under-represents. Widen c or measure the distribution on more data before
  fixing it. Recorded in ARCHITECTURE.md §5 rather than silently resized.
- [DEFERRED: 5B checkpoint] top_k 4 -> 8 decision (defect #21)
- [DEFERRED: 5B checkpoint] LR, warmup, loss-weight sweep
- [DEFERRED: needs external data] RMDB Mg2+ titration acquisition
