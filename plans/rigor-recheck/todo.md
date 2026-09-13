# Master To-Do — Cycle 7

Legend: [ ] open · [x] done+tested · [BLOCKED: r] · [DEFERRED: r]

## Priority 0 — audit the instrument
- [x] C15 Inspect the 18 "no RNA" structures in the raw mmCIF, not via the parser
- [x] C16 Characterise the failure class (mmCIF key-value serialisation)
- [x] C17 Establish whether the class is random w.r.t. any published claim
- [x] C18 Handle both mmCIF serialisations in `entity_poly_types()`
- [x] C19 Resolve hybrid DNA/RNA chains per residue via the `O2'` test
- [x] C20 Re-derive every G-finding; retract the cycle-6 "G2 strengthens" claim
- [x] C21 Re-propagate to ARCHITECTURE.md, main.tex, blueprint.html
- [x] C22 Extend the resolver test suite (20 properties, both forms, hybrids)
- [x] C23 Update `verify_claims.py` to cycle-7 values; add the cycle-6-vs-7 pair
      to the correction-table guard so the retraction cannot be silently erased
- [x] C24 Rebuild main.pdf (39pp, 0 over/underfull); republish blueprint (v24)

## Carried forward — still open
- [ ] C13 Migrate the remaining analysis scripts onto `mmcif_entities.py`.
      Raised in cycle 6 and **not done**: 13 scripts still carry their own
      divergent RNA definitions. Cycle 7 raised its priority — the resolver has
      now changed twice, so any script that copied an older definition is drifting
      from the canonical one in a way nothing detects.
- [ ] C14 Re-derive the residue-weighted measurements (stiffness, ion
      coordination, block occupancy, contact scaling) on the canonical basis.
      Cycles 6-7 re-derived only the G-findings.

## Deferred
- [DEFERRED: 5B checkpoint] top_k 4 -> 8 decision (defect #21)
- [DEFERRED: 5B checkpoint] LR, warmup, loss-weight sweep
- [DEFERRED: needs external data] RMDB Mg2+ titration acquisition
