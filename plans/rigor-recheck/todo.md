# Master To-Do — Cycles 8-10

Legend: [ ] open · [x] done+tested · [BLOCKED: r] · [DEFERRED: r]

## Priority 0 — audit the verification apparatus
- [x] C25 Measure both mmCIF serialisations across every category the scripts read
- [x] C26 Fix `loops()`; re-run geometry + both stiffness-headroom analyses
- [x] C27 Propagate the deltas (56 exact-match replacements across 8 files)
- [x] C28 Compare every guard tolerance to the precision of the value it guards
- [x] C29 Derive the default tolerance from how the value is written
- [x] C30 Adversarially verify the guard rejects the two values that slipped through
- [x] C31 Rebuild main.pdf (40pp, 0 boxes); republish blueprint (v26)

## Open — the oldest outstanding items, now three cycles old
- [ ] C13 Migrate the remaining analysis scripts onto `mmcif_entities.py`.
      Raised cycle 6, still open. 13 scripts carry their own RNA definitions and
      the resolver has changed twice since — any copied definition is drifting
      with nothing to detect it.
- [ ] C14 Re-derive the residue-weighted measurements (stiffness, ion
      coordination, block occupancy, contact scaling) on the canonical basis.
      Only the G-findings have been done.
- [x] C32 Audit the cross-document token check (OQ-6) — **defect #28**
- [x] C33 Adversarially verify it by deleting every occurrence of the claims
- [x] C34 Fix the two latent truncated tokens exposed by the tightening
- [x] C35 Boundary matching + context patterns; print which mode each check used

## Deferred
- [DEFERRED: 5B checkpoint] top_k 4 -> 8 decision (defect #21)
- [DEFERRED: 5B checkpoint] LR, warmup, loss-weight sweep
- [DEFERRED: needs external data] RMDB Mg2+ titration acquisition
