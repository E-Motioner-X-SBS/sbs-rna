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

- DISC-13 [ARCHITECTURE, G2, MOST SERIOUS]: 98.96% of RNA residues in the
  sample sit in entries containing protein (159/180 structures, median 10
  protein chains). The design predicts single chains from sequence, so it would
  learn folds that are partner-stabilised. Undocumented before cycle 2.
- DISC-14 [ARCHITECTURE, G3]: 93.07% of RNA residues come from 61 ribosome-like
  entries (33.9% of structures). Every residue-weighted statistic is primarily
  ribosomal. The length-binned tables happen to stratify it, and show the c=20
  budget is safe at both ends, but the headline 1.34% occupancy is a ribosome
  number.
- DISC-15 [ARCHITECTURE, G7]: 27,437 residues (8.90%) fall outside {A,C,G,U} --
  17,767 DNA, 7,036 UNK, 2,634 inosine/other. Vocab-5 cannot represent any of
  them. Caution: my first pass mislabelled DNA as "modification"; corrected.
- DISC-16 [ARCHITECTURE, G6, design gap]: the router reads structural features
  produced by the pair track, which runs after the trunk, so routing at
  recycle 0 is undefined. B_elec has an explicit first-pass rule; the router
  does not.

- DISC-17 [DEFECT #12, METHOD]: `measure_stiffness_headroom.py` fits every
  group Gaussian in-sample and scores each model on its own covered subset
  (M0 103,964 / M1 90,098 / M2 78,076 steps). Finer partitioning lowers
  in-sample NLL mechanically, and M2's subset is the well-populated, more
  regular steps. Both biases inflate M2.

## Reversals (cycle 2)
- REV-2 [MAJOR]: "structural context contributes more than sequence context".
  FROM: +3.0282 nats structure-over-sequence vs +2.1443 sequence-over-global,
        used to justify the learned stiffness encoder over a lookup table and
        quoted in ARCHITECTURE.md, main.tex, blueprint and prior-art 07/08.
  TO:   on the common 78,076-step subset with 2-fold held-out scoring,
        structure-over-sequence is **+1.0336** and sequence-over-global is
        **+1.7847** — the order INVERTS and the gain shrinks 2.9x.
  SURVIVES: structural context still adds a real +1.03 nats beyond sequence.
        Sequence-alone (+1.7847) and structure-alone (+1.7409) are near-equal
        and complementary (3.53 if independent vs 2.82 actual), so an encoder
        reading BOTH is still correct and a sequence-only lookup table still
        leaves ~1.03 nats unused. The architectural decision stands; only the
        comparative superlative is retracted.
  ALSO:  the §7c acceptance gate "below 14.551 nats/step" was an in-sample
        number on a favourable subset. Fair held-out M2 is **15.3424**.

## Cycle 3 reversals
- REV-3 [MAJOR]: token budget 323B -> staged 25B. The over-training defence was
  argued at 269M active (1,201 tok/param); at 61M it is 5,295 = 265x Chinchilla,
  the QiD worst case (arXiv 2411.17691). The corpus is redundant, not rich:
  measured RNA entropy 2.0167 bits/nt vs ~11 for an English token.
- REV-4 [defect #15]: cost quoted in A100-hours with a 1.60x FP8 lever applied.
  A100 (SM80) has no FP8 tensor cores. "79 A100-hours" existed on no machine.
  Corrected to hardware-explicit: 126 h A100 bf16 / 20 h H100 fp8 at 323B;
  ~10 h / ~1.6 h at the decided 25B.
- REV-5 [MAJOR]: 4-bit dropped at this size. NF4 cannot train from scratch
  (W4A16, frozen base, adapters only). NVFP4 is a real pretraining format but
  validated at 125 tok/param against our 5,295.

## Cycle 3 discoveries
- DISC-18 [VERIFIED]: elDORS composition entropy 2.0167 bits/nt. A 512-dim bf16
  token is 4,062x that; a fully attributed nucleotide (58.9 bits) is still 139x
  over-provisioned. d_model is compute space, not storage.
- DISC-19 [VERIFIED]: only 26.0 of 58.9 bits are available at INFERENCE. The
  rest are supervision targets; feeding them in would leak the answer.
- DISC-20 [VERIFIED]: at 3.10 A median resolution, fp16 coordinates are 3,100x
  finer than the experimental noise floor.

## Cycle 4 discoveries
- DISC-21 [DEFECT #16]: the ladder's "loops" column mixed train-time (Small: 8)
  with serve-time (Base-v2: 3). At matched serve-3, Small is 48 effective layers
  vs Base-v2's 96 -- the depth claim reverses at inference.
- DISC-22 [DEFECT #17, LARGEST]: refinement loops were never in the FLOP budget.
  Every cost understated by its loop count (Small 8x). "4.4x smaller" is really
  1.65x by effective compute; the lever chain is 5.6x not 24x; the corrected
  cost at 25B is 78 A100-h bf16 / 12 H100-h fp8. The activation-memory half of
  the claim survives -- only the compute half was wrong.

## Cycle 5 discoveries
- DISC-23 [DEFECT #20]: B_motif has the same recycle-0 circularity as the router
  and B_elec -- it enters the trunk's bias but is keyed on the pair track's
  interaction graph, produced after the trunk. Only B_elec had a first-pass
  rule. All three now share an explicit attention_bias_recycle_schedule.
- DISC-24 [DEFECT #21]: the MoE copied DeepSeekMoE's per-expert width (0.250 vs
  0.286) but not its ACTIVE capacity (1.500x vs 2.571x d) -- 1.7x below the
  cited template and 2.7x below a dense FFN. Never examined. top-8 fixes it for
  +12.6M active and ~0 total.
- DISC-25 [PROCESS]: 10 of 37 components were unspecified. Seven were filled
  with a PROCEDURE rather than an invented number, on the view that a guessed
  hyperparameter is worse than an acknowledged gap.

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
