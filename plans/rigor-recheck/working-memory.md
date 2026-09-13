# Working memory

## Decisions
- DEC-1: Rigor state lives in `plans/rigor-recheck/` to avoid colliding with the
  repo's existing data-acquisition plans in `plans/`.
- DEC-2: Every headline number is re-derived by an INDEPENDENT reimplementation,
  not by re-running the original script (which would reproduce its own bugs).

## Discoveries
- DISC-1 [VERIFIED]: Ground-truth experimental method over the 180 sampled
  structures is 117 cryo-EM / 63 X-ray. Counted directly by exact-tag match on
  `_exptl.method`, first occurrence, header only.
- DISC-2 [VERIFIED]: The rigidity analysis uses 31 of those 63. The other 32 are
  dropped by a `len(res) < 30` minimum-residue guard, not by method filtering.
  All 32 verified to hold 2-28 RNA residues.
- DISC-3 [DEFECT, documentation]: All three deliverables describe the rigidity
  sample as "X-ray only (31 structures)" and attribute the exclusion to cryo-EM
  ADP incomparability alone. The >=30-residue filter is unstated. This
  understates the selection applied and must be corrected.

## Reversals
- REV-1 [MAJOR]: The coevolution depth-gate claim is walked back.
  FROM: "Neff/L >= 1 -> 0.975 vs < 1 -> 0.609. The decisive pattern... the
        strongest argument for MoE in the whole design."
  TO:   a suggestive but statistically weak association (exact permutation
        p=0.0455 on a POST-HOC threshold, Spearman only +0.224 over n=12, deep
        arm n=2, and one shallow family reaching precision 1.000).
  REASON: T8d. The mean difference is real but the evidence cannot carry the
        weight placed on it. The architectural case for depth-gated routing now
        rests on THEORY and the published literature (which is unambiguous that
        shallow RNA MSAs degrade coupling analysis), with this measurement as
        weak supporting evidence only -- not as proof.

## Additional discoveries (cycle 1 cont.)
- DISC-10 [VERIFIED]: All parameter-budget, attention-saving and memory
  arithmetic reproduces exactly. 910.5M/382.0M vs documented 911M/382M.
- DISC-11 [DEFECT, PHYSICS]: The stated axial charge spacing b ~ 5.9-7.0 A is
  wrong and internally inconsistent with the stated theta ~ 0.76. Manning's b is
  the AXIAL projection (1.70 A for B-DNA, 1.40 A for A-form RNA), not the P-P
  contour distance. Correct RNA values: b=1.40 A, xi=5.11, theta=0.804.
  theta~0.76 is the B-DNA number. Verified against literature this session.
- DISC-4 [VERIFIED]: 30 A centroid prefilter is provably safe (bound 22.20 A,
  observed max 18.59 A, zero missed contacts). All contact-derived numbers stand.
- DISC-5 [VERIFIED]: Mg/rigidity gradient reproduces exactly at 1.760 sigma and is
  monotonic. Partial correlation controlling for packing density is +0.372 vs
  +0.420 raw -- the signal is NOT merely restating that folded cores are ordered.
  Within-structure effect +1.514 sigma, positive in 4/4 structures.
- DISC-7 [VERIFIED]: effective_c naive accounting overstates by 1.029x
  (17.61 vs 17.12 strict). Conservative; no correction needed.
- DISC-8 [DEFECT, conservative]: coevolution ground truth omits 53 pseudoknot
  pairs (9.87%) because WUSS alphabetic brackets are unparsed. This DEFLATES
  measured precision, so 0.670 is a lower bound. Must be disclosed.
- DISC-9 [DEFECT, OVERCLAIM]: the depth split is far weaker than reported.
  See REV-1.
- DISC-6 [DEFECT, documentation]: The Mg gradient is measured over **15** X-ray
  structures (those containing Mg2+), not 31. The report and blueprint both say
  "24,623 nucleotides in 31 X-ray structures", conflating two different sample
  definitions: 31 is the X-ray set used for the density/GNRA analyses; 15 is the
  Mg-containing subset. The nucleotide count 24,623 is correct for the 15.

## Cycle 2 discoveries
- DISC-12 [DEFECT #11, INFLATION]: the 143,871 unobserved-residue count is not
  RNA. `_pdbx_unobs_or_zero_occ_residues` covers every polymer in the entry, and
  `extract_basepair_geometry.py:163` counts rows without filtering residue type.
  Split: protein/other 97,131 (67.5%), **RNA 46,447 (32.3%)**, DNA 293.
  The figure had been promoted to a novelty claim ("free per-residue disorder
  labels"). Overstated 3.1x. The claim stands at 46,447 — still the largest
  free flexibility-label source we have, but a third of what was published.
  Top contributor 8FKW: 8,932 unobserved rows of which only 3,896 are RNA.

## Open Questions
- OQ-1 [**ANSWERED — NO BIAS**]: confound controlled (partial corr survives), and
  the <30-residue exclusion is now tested directly
  (`scripts/sampling/test_residue_guard_bias.py`). Recomputing the gradient at
  guards 30/20/15/10/5/1: the span moves from **1.760 to 1.757 sigma**, a change of
  **0.003 sigma**, and stays monotonic at every threshold. Dropping the guard
  entirely adds only 4 structures and 43 nucleotides (24,623 -> 24,666, +0.17%),
  because the excluded structures hold 2-28 residues each. The guard is a
  variance control, not a source of bias. Headline 1.76 sigma stands.
- OQ-1-orig: Does the <30-residue exclusion bias the rigidity result? Small RNAs are
  exactly where Mg2+ inner-sphere fraction was measured lowest (0.296). Excluding
  them could inflate the Mg-rigidity gradient. MUST TEST (-> T11).
