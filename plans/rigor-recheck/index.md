# Rigor Recheck — PHAROS empirical claims

**Note on location**: the repo's `plans/` already holds RNA data-acquisition
documents (12-*, 13-*, CONTINUATION_STATE). Rigor-mode state lives in this
subfolder to avoid collision.

| Field | Value |
|---|---|
| Task | Recheck every empirical claim in the PHAROS architecture work |
| Cycle | **7** — audit the cycle-6 audit |
| Phase | 6 AUDIT (cycle 7) |
| Started | 2026-09-13 (cycle 1) |
| Open TODO | 2 open (C13, C14), 3 deferred |
| Defects to date | **25** |

## Why cycle 7 exists

Cycle 6 closed with an unexplained residual, logged as OQ-3 rather than waved
away: **18 of 180 structures in a non-redundant _RNA_ list appeared to contain no
RNA.** Cycle 6 reasoned that it changed no published number — they contribute
zero residues either way — and deferred it.

That reasoning was wrong, and the error was consequential.

16 of the 18 were a **parser failure**, not an absence. mmCIF serialises a
category in two forms and the resolver handled only one; the key-value form is
what the PDB writes when a category has exactly one row — i.e. **a structure
with a single polymer entity, which means a small isolated RNA**. Their absence
inflated G2's "RNA residues in complexes" from 98.95% to a spurious 99.70%, and
**cycle 6 published that inflation as a finding: "G2 strengthens".**

A defect introduced while auditing a definition produced a false result which the
audit then published, one commit before this cycle caught it.

## Scope — cycle 7

- OQ-3: the 18 structures (-> **defect #25**)
- `entity_poly_types()`: both mmCIF serialisations
- Hybrid DNA/RNA chains: resolve per residue instead of excluding wholesale
- Re-derive every G-finding; re-propagate to all three deliverables
- Extend the resolver test suite to cover both forms and the hybrid rule

## Defect base rate

| Cycle | Defects | Character |
|---|---|---|
| 0 (build) | 5 | implementation bugs producing plausible numbers |
| 1 (audit) | 4 | overclaims + a physics constant |
| 2 | 3 | wrong population, unfair comparison, unimplemented loss |
| 3 | 2 | misattributed citation, impossible hardware |
| 4 | 4 | mixed units, an omitted term in every cost, two unspecified components |
| 5 | 2 | circularity, copied template without its active capacity |
| 6 | 3 | no shared definition; DNA/solvent contamination; partial propagation |
| **7** | **1** | **the audit's own tooling parsed half the format, and published the artifact as a result** |

**Twenty-five defects.** Defect #24 was caused by a repair; **#25 was caused by
the audit tool itself and produced a published false finding.** The corrections
are now a defect source in their own right, which is the whole argument for
continuing the loop rather than declaring convergence.
