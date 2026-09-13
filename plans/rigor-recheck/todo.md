# Master To-Do — Cycle 6

Legend: [ ] open · [x] done+tested · [BLOCKED: r] · [DEFERRED: r]

## Priority 0 — the unit of measurement
- [x] C1  Diff every RNA-residue parser over the same 180 files, both directions
- [x] C2  Derive the definition from `_entity_poly.type`, not a curated list
- [x] C3  Explain the 24.87% non-ACGU on the first canonical run
- [x] C4  Add the `label_seq_id` polymer test; confirm solvent leakage is zero
- [x] C5  Re-derive G1/G2/G3/G7 canonically (`audit_generalization_canonical.py`)
- [x] C6  Propagate to ARCHITECTURE.md, main.tex, blueprint.html
- [x] C7  **Audit the propagation** — 11 affected numbers, not 4
- [x] C8  Test the resolver (`test_mmcif_entities.py`, 15 properties)
- [x] C9  Wire all three suites into `verify_claims.py`
- [x] C10 Add canonical tokens + two anti-corruption guards to `verify_claims.py`
- [x] C11 Measure the hybrid-chain exclusion rather than assuming it immaterial
- [x] C12 Rebuild main.pdf (39pp, 0 over/underfull); republish blueprint (v23)

## Carried forward
- [DEFERRED: 5B checkpoint] top_k 4 -> 8 decision (defect #21)
- [DEFERRED: 5B checkpoint] LR, warmup, loss-weight sweep
- [DEFERRED: needs external data] RMDB Mg2+ titration acquisition for
  ion-conditioning supervision (PDB is survivorship-biased to 5-15 mM)

## Opened by cycle 6, for cycle 7
- [ ] C13 Migrate the remaining analysis scripts onto `mmcif_entities.py`.
      They still carry their own divergent definitions, so defect #22 can
      recur the moment one of them is re-run. `analyze_ions_motifs.py` and
      `audit_generalization.py` are the two known-divergent ones; the other
      13 have not been checked.
- [ ] C14 Re-derive the residue-weighted measurements (stiffness, ion
      coordination, block occupancy, contact scaling) on the canonical basis.
      Cycle 6 re-derived only the G-findings. The others were never checked
      against the canonical count and inherit whichever definition their
      script invented.
