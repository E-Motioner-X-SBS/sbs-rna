# Rigor Recheck — PHAROS empirical claims

**Note on location**: the repo's `plans/` already holds RNA data-acquisition
documents (12-*, 13-*, CONTINUATION_STATE). Rigor-mode state lives in this
subfolder to avoid collision.

| Field | Value |
|---|---|
| Task | Recheck every empirical claim in the PHAROS architecture work |
| Cycle | **6** — one canonical definition of "an RNA residue" |
| Phase | 6 AUDIT (cycle 6) |
| Started | 2026-09-13 (cycle 1) |
| Open TODO | 0 open, 3 deferred to the 5B checkpoint |
| Defects to date | **24** |

## Why cycle 6 exists

Cycle 5 closed with the observation that the newest defects are **absences
rather than errors** — components never specified, checks never run. Cycle 6
started from a two-line discrepancy that had been visible for four cycles and
never chased:

```
analyze_ions_motifs.py   ->  307,965 RNA residues, 179 structures
audit_generalization.py  ->  308,370 RNA residues, 180 structures
```

Two scripts reading the same 180 files disagreed. Neither was wrong about its
own definition, because **there was no definition** — every script had invented
one. That is defect #22.

## Scope — cycle 6

- The RNA-residue definition used by every analysis script
- `src/pharos/data/mmcif_entities.py` (new shared resolver) + its tests
- `scripts/sampling/audit_generalization_canonical.py` (re-derivation)
- Every G-finding (G1, G2, G3, G7) in ARCHITECTURE.md §10e, main.tex §Generalization,
  blueprint §16
- The propagation of the correction itself (this is where #24 lives)

## Defect base rate

| Cycle | Defects | Character |
|---|---|---|
| 0 (build) | 5 | implementation bugs producing plausible numbers |
| 1 (audit) | 4 | overclaims + a physics constant |
| 2 | 3 | wrong population, unfair in-sample comparison, unimplemented loss |
| 3 | 2 | misattributed citation, impossible hardware |
| 4 | 4 | mixed units, an omitted term in every cost, two unspecified components |
| 5 | 2 | circularity, copied template without its active capacity |
| **6** | **3** | **no shared definition; DNA/solvent contamination; partial propagation of the fix** |

**Twenty-four defects. Every one produced a plausible number rather than an
error.** Defect #24 is the first to be caused by *a previous cycle's repair*.
