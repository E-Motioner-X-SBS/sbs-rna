# Plan — Cycle 6: one canonical definition of "an RNA residue"

## Why this framing

Cycles 3-5 established that **derived** quantities fail more often than
measurements, and cycle 5 sharpened it: the newest failures are *absences* —
a component never specified, a check never run on anything but its first
subject. Cycle 6 tests the most basic absence available:

> **Does the project have a definition of its own unit of measurement?**

Every residue-weighted number in the work — stiffness, ion coordination, block
occupancy, contact scaling, the G-findings — is a count of RNA residues. If the
scripts disagree about what one is, every such number inherits the disagreement.

## Method

1. Quantify the disagreement: run every parser over the same 180 files and diff
   per structure, in **both** directions.
2. Derive the authoritative definition from the format itself, not from a
   curated residue list. mmCIF *declares* polymer type.
3. Implement it once, in `src/pharos/data/`, and **test it** — a shared
   definition that is untested only relocates the problem.
4. Re-derive every affected finding under it.
5. Propagate to all deliverables. **Audit the propagation**, because cycle 4
   already showed a repair can introduce its own defect (#18b).

## Step-locked order

| Step | Target | Outcome |
|---|---|---|
| C1 | Diff the parsers per structure | **defect #22** — disagree on 72/180 |
| C2 | Build `mmcif_entities.py` from `_entity_poly.type` | first run: 404,205 |
| C3 | Why is it 24.87% non-ACGU? | **defect #23** — HOH/MG in RNA auth chains |
| C4 | Add the `label_seq_id` polymer test | 306,857, 1.04%, solvent leakage 0 |
| C5 | Re-derive G1/G2/G3/G7 canonically | G2 strengthens, G7 was 8.6x too high |
| C6 | Propagate to ARCHITECTURE / main.tex / blueprint | partial — see C7 |
| C7 | **Audit the propagation itself** | **defect #24** — 4 of 11 numbers moved |
| C8 | Test the resolver; wire every suite into the guard | 15 properties, all pass |
