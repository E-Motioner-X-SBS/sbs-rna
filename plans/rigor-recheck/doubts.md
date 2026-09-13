# Doubts log

| ID | Doubt | Raised | Resolution |
|----|-------|--------|------------|
| D1 | Rigidity reports 31 X-ray; ionic audit reports 63 from same files | C1 P0 | **RESOLVED — not a data bug.** Independent count confirms 63 X-ray of 180. `parse()` recovers the method for all 63. `analyze()` drops 32 because they have <30 RNA residues (verified: all 32 have 2-28 residues), hitting the `len(res) < 30` guard. The measurement is sound. **But the documentation is defective**: ARCHITECTURE.md, main.tex and blueprint.html all say the exclusion was cryo-EM comparability and cite "31 X-ray structures" without stating the >=30-residue minimum. DOC FIX REQUIRED (-> T20). |
