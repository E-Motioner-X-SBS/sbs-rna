# Plan — Cycles 8-9: audit the verification apparatus

## Why this framing

Cycle 7 found that the RNA resolver — an instrument built *by this audit, for
this audit* — had parsed only half of mmCIF's serialisation grammar, and that the
resulting artifact was published as a finding. That raises a question no earlier
cycle had asked:

> **Who checks the checkers?**

Cycles 0-6 treated the tooling as a given and audited its outputs. If the tooling
is defective, its outputs agree with each other perfectly and are all wrong
together — which reads exactly like confirmation.

## Method

For each instrument the audit relies on, ask what class of input it handles and
whether that class is complete:

1. **The shared mmCIF parser** (`loops()`). #25 was a two-serialisation bug.
   Does the other parser have it? Measure across every category actually read.
2. **The regression guard** (`verify_claims.py`). It reports pass/fail; what
   does a pass actually prove? Compare every tolerance to the precision of the
   value it guards.
3. Verify each fix **adversarially** — a guard that has never rejected anything
   has not been tested.

## Step-locked order

| Step | Target | Outcome |
|---|---|---|
| C25 | Every mmCIF category read, in both serialisations | **defect #26**: 9 rows in 9 structures |
| C26 | Fix `loops()`; re-run the affected analyses | +1 step, +1 disorder record; all headlines unchanged |
| C27 | Propagate (56 replacements, 8 files); re-render; rebuild | 40pp / 0 boxes, blueprint v25 |
| C28 | Compare all 54 tolerances to quoted precision | **defect #27**: 45 of 54 too wide |
| C29 | Derive the default from how the value is written | all 54 still pass |
| C30 | **Inject the two values that previously slipped through** | both now fail, exit 1 |
| C31 | Document; rebuild; republish | 40pp / 0 boxes, blueprint v26 |
