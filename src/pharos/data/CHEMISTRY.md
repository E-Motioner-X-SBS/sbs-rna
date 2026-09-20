# Per-nucleotide chemistry vector — the 24 features

Specification for `pharos/data/chemistry.py`, introduced in ARCHITECTURE v0.2 §4.
**Implemented** — see `chemistry.py` and `test_chemistry.py` (34 properties).

## Why this exists

v0.1 fed the trunk nucleotide identity and little else, then asked a physics
module to reason about hydrogen bonding, protonation and stacking. The physics
had no chemical substrate to work from. This file supplies it.

**Design constraint that shapes everything below.** A fully attributed
nucleotide carries **58.9 bits**; a 512-dim bf16 token holds **8,192** — a 139×
over-provisioning (measured, `audit_token_attributes.py`). So `d_model` is
*compute* space, not storage: chemistry enters through **one projection** into
the existing width, and never motivates widening it, because FFN cost is
quadratic in `d` while an input projection is linear.

## The separation that must not be violated

| | Available at inference | Supervision target only |
|---|---|---|
| identity, ring class, H-bond capacity, pKa, pucker propensity, stacking priors, backbone charge, modification class, local GC | **yes — all 24 below** | — |
| Saenger class, Leontis–Westhof class, normalised B-factor, Mg²⁺ distance, disorder, SHAPE/DMS reactivity | — | **yes** |

Feeding a structural observable in as a feature leaks the answer.
`chemistry.py` and `attributes.py` keep the two sets in physically separate
arrays so the mistake cannot be made by accident.

---

## The 24 features

### Identity and ring class (6 dims)

| Dim | Feature | A | C | G | U |
|---|---|---|---|---|---|
| 0–4 | one-hot A/C/G/U/MOD | | | | |
| 5 | purine = 1, pyrimidine = 0 | 1 | 0 | 1 | 0 |

`MOD` fires for any residue outside {A,C,G,U}; the modification class (dims
20–22) then says which kind, and the `MOD` embedding keyed by mmCIF `comp_id`
supplies the rest.

### Hydrogen-bond capacity, per edge (6 dims) **[lit]**

RNA pairs on three distinct faces, and which face is available is the single
most useful chemical fact about a base. Donors (D) and acceptors (A) counted per
edge in the neutral, dominant tautomer.

| Dim | Edge | A | C | G | U |
|---|---|---|---|---|---|
| 6 | Watson–Crick donors | 1 (N6-H₂ counts 1 site) | 1 (N4-H₂) | 1 (N1-H, N2-H₂) | 1 (N3-H) |
| 7 | Watson–Crick acceptors | 1 (N1) | 2 (N3, O2) | 1 (O6) | 2 (O2, O4) |
| 8 | Hoogsteen / CH donors | 1 (N6-H) | 0 | 1 (N2-H via sugar edge) | 0 |
| 9 | Hoogsteen / CH acceptors | 1 (N7) | 1 (O2) | 1 (N7, O6) | 1 (O2) |
| 10 | sugar-edge donors | 0 | 0 | 1 (N2-H₂) | 0 |
| 11 | sugar-edge acceptors | 1 (N3) | 1 (O2) | 1 (N3) | 1 (O2) |

The 2′-OH is a donor *and* acceptor on every residue and is carried separately
at dim 19, because it is the feature that distinguishes RNA from DNA and drives
A-form geometry, ribose-zipper motifs and much of tertiary contact chemistry.

### Protonation (2 dims) **[lit]**

| Dim | Feature | A | C | G | U |
|---|---|---|---|---|---|
| 12 | pKa of the ring nitrogen that protonates | 3.5 (N1) | 4.2 (N3) | 9.2 (N1) | 9.2 (N3) |
| 13 | shifted-pKa flag | 0 | 0 | 0 | 0 |

Free-nucleoside values. Dim 13 is set when a structural context is known to
shift the pKa toward neutrality — notably **A⁺·C wobble pairs** and protonated
A in i-motif-like and pseudoknot contexts, where N1 pKa rises several units.
This is the mechanism behind pH-dependent RNA structure, and it is why pH is an
input alongside ionic strength.

### Sugar pucker propensity (2 dims) **[lit]**

| Dim | Feature | value |
|---|---|---|
| 14 | C3′-endo (A-form, north) propensity | 0.75–0.85 for RNA |
| 15 | C2′-endo (B-like, south) propensity | 1 − dim 14 |

Purines sit slightly further north than pyrimidines; 2′-O-methylation pushes
strongly north. The pucker equilibrium is what the 2′-OH controls, which is why
dims 14–15 and dim 19 are correlated by construction rather than independent.

### Stacking (2 dims) **[lit]**

| Dim | Feature | A | C | G | U |
|---|---|---|---|---|---|
| 16 | relative polarisability | 0.82 | 0.61 | 1.00 | 0.58 |
| 17 | stacking-energy prior | 0.85 | 0.63 | 1.00 | 0.55 |

Normalised to G = 1. Purine–purine stacks are the strongest, pyrimidine–
pyrimidine the weakest; this ordering (G > A > C > U) is the dominant term in
helix stability after base pairing and is the reason GC content is not a
sufficient stability proxy.

### Backbone (2 dims)

| Dim | Feature | value |
|---|---|---|
| 18 | phosphate charge state | −1 (1 for the 5′ terminus with a free OH) |
| 19 | 2′-OH present | 1 for ribo, 0 for deoxy |

Dim 18 is what Manning condensation acts on. The measured axial charge spacing
is **b = 1.40 Å**, giving ξ = 5.11, θ = 0.804 and q_eff = −0.196 for A-RNA — see
`physics/manning.py`. Dim 19 distinguishes the DNA residues that appear in
hybrid duplexes, which raw PDB entries genuinely contain.

### Modification class (3 dims) **[measured on the whole archive]**

| Dim | Class | residues | share of modified |
|---|---|---|---|
| 20 | methylation | 42,555 | **56.4%** |
| 21 | pseudouridylation | 18,458 | **24.5%** |
| 22 | other | 14,421 | 19.1% |

**Measured over all 10,520 RNA-bearing PDB entries: 75,434 of 13,348,166
residues are modified (0.565%), across 370 distinct species.** The derivative
corpora carry 0.025%, so these dims are only meaningful when training consumes
raw entries. (The 1.005% quoted elsewhere is over *all* polymer residues in the
ion-rigidity pass; 0.565% is over canonical RNA chains — different denominators,
both correct.)

The three-way split is not a guess about what the archive contains; it is what
it contains. Pseudouridine is a quarter of all modifications on its own, which
is why it has a dim rather than a share of "other", and it changes H-bond
capacity (adds an N1-H donor on the Hoogsteen face) — `chemistry.py` applies
that at dim 8.

**Resolution is by dictionary, not by name.** The parent base of a modified
residue is `_chem_comp.mon_nstd_parent_comp_id`, and that field is **not in the
entry files**: they carry only id / type / mon_nstd_flag / name / synonyms /
formula / formula_weight. It lives in the PDB Chemical Component Dictionary.
`scripts/sampling/resolve_ccd_parents.py` reads the CCD once and emits
`ccd_parents.json`; all 370 species are found there and **88.6% of modified
residues resolve to a standard A/C/G/U parent**. The 11.4% that do not are
genuinely parentless — inosine, UNK, L-nucleotides (0G, 0C), locked and
fluorinated synthetic analogues — and are reported as parent `N` rather than
assigned one.

Classification uses the CCD's systematic `name` ("5-METHYLURIDINE
5'-MONOPHOSPHATE"), which is chemistry, rather than a list of component codes,
which is the curated list defect #22 was about. **A2M is the test case** in
`test_chemistry.py`: nothing in the string says adenosine, and only the
dictionary knows that it is 2'-O-methyladenosine.

### Context (1 dim)

| Dim | Feature |
|---|---|
| 23 | local GC fraction, ±16 window |

---

## What is deliberately absent

**Thermodynamic nearest-neighbour ΔG.** It is a property of a *dinucleotide
step*, not a nucleotide, so it belongs in the pair track, not here. Adding it
per-residue would either duplicate the stacking prior or smear a pair quantity
across two tokens.

**Solvent accessibility, base-pair class, reactivity.** All structural
observables. Targets, not inputs.

**Tautomer state.** Rare enough that a flag would be almost always zero, and
unmeasurable from sequence.

---

## Provenance and honesty

Dims 6–17 are **literature values compiled into a static table**, not measured
from our corpus — they are properties of the chemistry, not of the dataset, and
a table is the right representation. They should be cited, not re-derived.

Dims 20–22 are **measured**: `modification_census.py` gives 75,434 modified
residues of 13,348,166 across 370 species over the whole archive, and
`resolve_ccd_parents.py` maps every one of them to a parent and a class through
the Chemical Component Dictionary.

The one place to be careful: dim 13 (shifted pKa) is set from *context*, and the
contexts that shift it are structural. At inference, before any structure is
predicted, it must default to 0 and may only be updated on recycles ≥ 1 — the
same circularity that affects router features at recycle 0
(ARCHITECTURE v0.2 §5.3). Setting it from a known structure during training and
from nothing at inference would be a train/test mismatch.
