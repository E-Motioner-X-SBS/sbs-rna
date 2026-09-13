# Plan — Cycle 7: audit the cycle-6 audit

## Why this framing

Cycle 6 closed with an open question it judged harmless: 18 of 180 structures in
a non-redundant *RNA* list appeared to contain no RNA. The reasoning for
deferring it was "they contribute zero residues under either definition, so no
published number depends on it".

**That reasoning assumes the zero is real.** If the zero is a parse failure, the
structures are silently absent from every statistic computed over the set — and
an absence that correlates with a property (here: being a small isolated RNA)
biases every number conditioned on that property.

So cycle 7 asks: **is the audit's own tooling correct?** Cycles 0-6 audited the
measurements, the derivations, the specification and the propagation. Nothing had
audited the instrument.

## Method

1. Take the 18 structures and look at the raw file, not the parser output.
2. Distinguish *absence* from *failure to parse*.
3. If a failure: characterise what class of file it hits, and whether that class
   is random with respect to any published claim.
4. Fix, re-derive, re-propagate — and re-run the guards written in cycle 6,
   which exist precisely to catch propagation errors of this kind.
5. Extend the test suite so the failure class is asserted against, not merely
   fixed.

## Step-locked order

| Step | Target | Outcome |
|---|---|---|
| C15 | Inspect the 18 raw files | 16 return **zero `_entity_poly` rows** — a parse failure |
| C16 | Identify the class | **defect #25**: mmCIF's key-value serialisation, used for single-row categories |
| C17 | Is the class random? | **No** — one polymer entity means a small isolated RNA, exactly G2's population |
| C18 | Handle both serialisations | 162 -> 178 structures, 306,857 -> 309,191 residues |
| C19 | Resolve hybrid chains per residue by `O2'` | 178 -> **179**; only 7PU7 remains, modelled all-DNA |
| C20 | Re-derive every G-finding | **G2's "strengthening" retracted**: 99.70% -> 98.95%, i.e. the published 98.96% |
| C21 | Re-propagate; rebuild; extend guards and tests | 39pp / 0 boxes, blueprint v24, 20 properties, ALL CLAIMS REPRODUCE |
