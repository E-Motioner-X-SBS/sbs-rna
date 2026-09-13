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

## Cycle 6 decisions
- D8: **The RNA-residue definition comes from the format, not from a curated
  list.** A hardcoded residue-name list can only ever be as complete as the
  modifications its author knew about, and RNA has >170. `_entity_poly.type` is
  a declaration the depositor made, so it is complete by construction.
  Alternatives rejected: (a) extend the hardcoded list — fails on the next novel
  modification; (b) chemical-component-dictionary lookup — needs a 500 MB
  external file and still disagrees on hybrids.
- D9: **Hybrid chains excluded.** Measured impact 6 residues, 0.002%. If a future
  corpus carries many, resolve per-residue instead. Recorded in the resolver.
- D10: **Every repair substitution is exact-match and count-asserted.** After
  three occurrences of the unguarded-replace failure mode (#18b, and twice in
  #24), no global `replace()` is used on a deliverable without asserting the
  match count first.

## Cycle 6 reversals
- REV-6 [MAJOR]: **G7's headline was 8.90% of residues outside A/C/G/U; it is
  1.04%.** The published figure was 90% contamination — DNA from hybrid duplexes
  and UNK records that are not in RNA chains at all. The *conclusion* (vocab-5
  cannot represent real RNA) is unchanged, but the evidence for it is now "76
  distinct modification types led by pseudouridine", not a percentage.
- REV-7: **G2 strengthens rather than weakens.** 98.96% -> 99.70% of RNA residues
  sit in protein-containing entries. The largest generalization hazard in the
  design is slightly larger than reported, not smaller.
- REV-8: the structure-count denominator is **162, not 180**. 18 of the sampled
  structures contain no RNA polymer entity at all.

## Cycle 6 discoveries
- DISC-26 [DEFECT #22]: the project had **no definition of its own unit of
  measurement**. Every script invented one, and they disagreed on 72 of 180
  structures in both directions. This had been visible as a two-line discrepancy
  since cycle 2 and was never chased.
- DISC-27 [DEFECT #23]: water and ions share the auth chain of the RNA they
  solvate. A naive entity-based selection therefore over-counts by ~24%
  (404,205 vs 306,857). `label_seq_id` is the polymer test that separates them.
- DISC-28 [DEFECT #24]: **a repair introduced its own defect, again.** 4 of 11
  affected numbers were propagated, and an unguarded replace corrupted the
  published column of the very table recording the correction. Cycle 4 flagged
  this exact failure mode as "#18b"; it recurred despite being flagged. The
  durable fix is not vigilance but the two new guards in `verify_claims.py` —
  a collapsed correction table now fails the build.
- DISC-29 [SCOPE]: cycle 6 re-derived only the **G-findings**. Every other
  residue-weighted measurement — 103,964 base-pair steps, 44,708 Mg coordination
  records, block occupancy, contact scaling — still comes from a script with its
  own invented definition. Their exposure is **unmeasured**, not zero. -> C13/C14.

## Open Questions
- OQ-2 [OPEN -> cycle 7]: do the residue-weighted measurements change under the
  canonical definition? The G-findings moved by −0.5% (counts) to 8.6x (G7).
  The base-pair-step and ion-coordination numbers are keyed on different mmCIF
  categories (`_ndb_struct_na_base_pair_step`, `_struct_conn`) that carry their
  own chain references, so the exposure is probably smaller — but "probably" is
  exactly the word cycle 6 exists to eliminate. Priority HIGH.
- OQ-3 [OPEN -> cycle 7]: 18 of 180 sampled structures declare no RNA polymer
  entity. What are they? If the BGSU non-redundant RNA list contains entries with
  no RNA entity, either the sampler or the parse of `_entity_poly` is wrong for
  those files. Priority MEDIUM — it does not affect any published number
  (they contribute 0 residues either way) but it is an unexplained 10%.

## Cycle 7 decisions
- D11: **An unexplained residual is not a harmless residual.** Cycle 6 deferred
  OQ-3 because "those structures contribute zero residues either way". That
  reasoning silently assumes the zero is *real* rather than a measurement
  failure. A zero produced by a broken instrument is indistinguishable from a
  true zero in the output, and differs entirely in the input. New rule: an
  unexplained count is chased before the cycle closes, not deferred on the
  grounds that it changes no number — because that is exactly what it would look
  like if it changed every number.
- D12: **Hybrid chains resolved structurally, not by name.** `O2'` presence is a
  property of the coordinates. It classified BRU (5-bromo-deoxyuridine)
  correctly as DNA, which a name-based rule would likely have got wrong.
- D13: **The record of a correction is a claim under guard.** The cycle-6 values
  (306,857 / 99.70%) are pinned in `verify_claims.py` alongside the cycle-7 ones,
  so a future edit that tidies away the mistake fails the build.

## Cycle 7 reversals
- REV-9 [MAJOR, retracts a cycle-6 finding]: **"G2 strengthens to 99.70%" is
  withdrawn.** The true canonical value is **98.95%**, materially identical to the
  originally published 98.96%. The apparent strengthening was defect #25 dropping
  16 small isolated RNAs from the denominator. **G2 is unchanged by the canonical
  re-derivation.**
- REV-10: G3 residues-ribosomal moves 93.35% -> **92.65%** (published: 93.07%).
  Also unchanged in substance.
- REV-11: the canonical structure count is **179**, not 162 (cycle 6) and not 180
  (published). 7PU7 is the single genuine exception and is explained.
- REV-12: G7's overstatement is **8.5x**, not 8.6x; 82 distinct modification
  types, not 76; 3,237 instances, not 3,189. The direction and the argument are
  unchanged.

## Cycle 7 discoveries
- DISC-30 [DEFECT #25]: **the audit's own instrument parsed half the format.**
  mmCIF has a key-value serialisation for single-row categories; the resolver
  handled only `loop_`. 16 of 180 structures silently became zero.
- DISC-31 [THE IMPORTANT ONE]: **the defect produced a false finding that the
  audit published.** Cycle 6 reported "G2 strengthens" as a result. It was an
  artifact of the bug introduced in the same cycle. Six cycles had audited
  measurements, derivations, specifications and propagation; **nothing had
  audited the tooling doing the auditing.** That is now the standing question
  for every cycle that builds a new instrument.
- DISC-32 [BIAS SHAPE]: the failure class was not random — it hit exactly the
  structures with one polymer entity, i.e. small isolated RNAs, i.e. the
  population the affected claim is about. A parse failure that correlates with
  the measured property is worse than noise; it is a systematic bias that
  strengthens the claim being tested.
- DISC-33 [NEW MEASUREMENT]: 21 of 179 structures contain no protein, holding
  3,257 residues = **1.05% of the corpus**. Isolated RNA is 11.7% of structures
  but ~1% of residues. Any "restrict training to autonomous folds" mitigation for
  G2 operates on roughly one percent of the available supervision — which makes
  option (c), reporting performance split by complexed vs isolated, the only one
  of the three that is actually affordable.

## Open Questions
- OQ-2 [OPEN, priority HIGH -> C14]: do the residue-weighted measurements change
  under the canonical definition? Still unanswered, and cycle 7 raises the stakes:
  the resolver has now changed twice, so any script carrying a copied definition
  is drifting from canonical in a way nothing currently detects.
- OQ-3 [**ANSWERED — defect #25**]: the 18 structures were 16 parse failures plus
  2 genuine hybrids. Now 179/180 with the single exception explained.
- OQ-4 [OPEN, priority MEDIUM]: are there other mmCIF categories the analysis
  scripts read where the same two-serialisation problem applies? `_exptl`,
  `_entity_poly_seq`, `_ndb_struct_na_base_pair_step` and `_struct_conn` are all
  read by loop-oriented parsers. Any of them written in key-value form in a
  small structure would fail identically and silently. **This is the direct
  generalisation of #25 and should open cycle 8.**

## Cycle 8 decisions
- D14: **Fix a defect whose measured impact is negligible, and say so plainly.**
  #26 moves one number by +1 in 103,965. The case for fixing it is not the
  magnitude here but that the sample is 180 files and the corpus is not — and the
  *identical* bug in `_entity_poly` moved a published finding by 0.75 points. The
  report states the negligible impact rather than dressing the fix up as material.
- D15: **Integer-count claims are pinned at `tol=0`.** The guard's default 0.5%
  tolerance let `103,965` pass against an expected `103,964` — it reported OK for
  a value it should have rejected. A count is exact or it is wrong.

## Cycle 8 discoveries
- DISC-34 [DEFECT #26]: `loops()` had the same two-serialisation bug as
  `entity_poly_types()`. Found by generalising #25 rather than by any symptom —
  nothing was visibly wrong. **9 rows across 9 structures**; every headline
  unchanged.
- DISC-35 [THE GUARD WAS LYING]: the regression guard *passed* on the corrected
  data while its expected value was stale, because `chk()` defaults to a 0.5%
  tolerance and the change was 1 part in 103,964. A guard with a tolerance wider
  than the effect it is meant to detect is not a guard. Four count claims moved to
  `tol=0` and a fifth was added.
- DISC-36 [THE SCARY HYPOTHESIS WAS WRONG]: `_exptl_crystal_grow` is key-value in
  all 64 files carrying it and feeds the ionic audit — which looked like a large
  exposure. It is not: that audit uses a key-value-aware reader. **The alarming
  reading of the evidence was the false one**, and only checking distinguished
  them. Worth recording because the cycle-6 failure was the opposite error —
  assuming a residual was harmless — and the discipline that catches both is the
  same one: measure, don't reason about it.

## Open Questions
- OQ-4 [**ANSWERED — defect #26**]: nine rows, nine structures, every headline
  unchanged. The two categories with the largest key-value footprint (`_exptl`
  180 files, `_exptl_crystal_grow` 64) are read by a key-value-aware parser and
  were never affected.
- OQ-5 [OPEN -> cycle 9, priority HIGH]: **are there other guards whose tolerance
  exceeds the effect they check?** DISC-35 found one by accident. `verify_claims.py`
  has ~40 `chk()` calls and most inherit the 0.5% default, including ratios and
  percentages where 0.5% is larger than several of the corrections this audit has
  made. This is the direct generalisation of #26's discovery and should open
  cycle 9.

## Cycle 9 decisions
- D16: **A tolerance is a claim about precision and must be justified like one.**
  The default is now derived from how the value is written rather than chosen: an
  integer matches exactly, a decimal to half a unit in its last quoted place. A
  looser window requires an explicit `abs_tol` and a reason in the comment.
- D17: **A guard is verified by making it fail.** Passing a correct value proves
  nothing about a guard. Cycle 9 injects the two specific wrong values that
  previously slipped through and asserts exit 1, then restores. Any future guard
  gets the same treatment.

## Cycle 9 discoveries
- DISC-37 [DEFECT #27]: **45 of 54 numeric claims had a tolerance wider than the
  precision they quote.** `G7 frac of RNA residues` was checked as 0.01047 ± 0.01
  — accepting 0.0005 to 0.0205 — a vacuous guard on the single figure this audit
  corrected by 8.5x.
- DISC-38 [NEAR MISS, quantified]: `G2 frac RNA res in complexes` allowed ±0.005.
  The cycle-6 value that cycle 7 retracted differs by 0.0075. **The largest
  retraction of the project came within a factor of 1.5 of passing its own
  regression guard undetected.** Defect #25 would then have been invisible to
  every automated check, and the false "G2 strengthens" finding would have stood.
- DISC-39 [THE GOOD NEWS]: tightening every tolerance to exact quoted precision,
  **all 54 claims still pass**. The numbers were accurate to their stated digits
  through nine cycles; only the *proof* was weak. "ALL CLAIMS REPRODUCE" now
  carries substantially more information than it did.
- DISC-40 [PATTERN, cycles 7-9]: three consecutive defects in the *verification
  apparatus* rather than in the work — the resolver (#25), the shared mmCIF
  parser (#26), the regression guard (#27). Cycles 0-6 audited the work; cycles
  7-9 audited the instruments and found one defect in each. **The tooling had
  never been audited at all**, and it had a 100% defect rate when it finally was.

## Open Questions
- OQ-5 [**ANSWERED — defect #27**]: 45 of 54; fixed; adversarially verified.
- OQ-6 [OPEN -> cycle 10, priority HIGH]: the guard checks *cross-document token
  presence* with `tok in text`. That proves a string appears somewhere, not that
  it appears in the right claim — "179" would be satisfied by any stray 179 in
  the document. Given DISC-40's pattern, the token check is the remaining
  unaudited instrument and is weaker than it looks by construction.
- OQ-2 / C13 / C14 [STILL OPEN]: 13 analysis scripts still carry their own RNA
  definitions, and only the G-findings have been re-derived canonically. Three
  cycles have now passed with these open; they are the oldest outstanding items.

## Cycle 10 decisions
- D18: **A guard must state what it proved.** Every token check now prints
  `(boundary)` or `(anchored)`, so a weak check is visible in the output instead
  of being indistinguishable from a strong one. The failure mode of #27 and #28
  was identical: a check that *looked* like evidence.
- D19: **Adversarial tests delete every occurrence, not a representative one.**
  The first attempt at C33 removed two of several instances, the new check passed,
  and it passed *correctly* — the claim was still in the document. The test was
  wrong, not the guard. Logged rather than quietly re-run, because it is the same
  error class the cycle is about.

## Cycle 10 discoveries
- DISC-41 [DEFECT #28]: `tok in text` proved the wrong thing. **3 of 5** guards
  tested adversarially were vacuous — `61` was satisfied by a contact-sparsity
  table row, `153` by "153.1 -> 153.2M", `78` by "1.78" in a density table. The
  suite exited **0** with the guarded claims deleted outright.
- DISC-42 [LATENT, found by the fix]: the tokens `15.342` and `16.376` were
  written truncated and had only ever matched as **prefixes** of 15.3424 and
  16.3760. They passed nine cycles while guarding nothing exactly. Tightening a
  check found defects that no amount of re-running the loose version could.
- DISC-43 [MY OWN TOOL HAD THE BUG]: the throwaway script that inventoried the
  token list used `re.findall(r'"([^"]+)"', ...)` and picked up a **comment**
  string, reporting a non-existent token `"99.70 -> 99.70"` as missing from all
  three documents. The analysis tool written to audit an instrument had the same
  class of defect as the instrument. Recorded because it is the cleanest possible
  illustration of DISC-40.
- DISC-44 [THE PATTERN COMPLETES]: cycles 7-10 audited four instruments — the
  resolver, the shared mmCIF parser, the guard's tolerances, the guard's token
  check — and found a defect in **every one**. Four for four.

## Open Questions
- OQ-6 [**ANSWERED — defect #28**]: vacuous for 3 of 5 tokens tested; fixed and
  adversarially verified; two latent token defects exposed as a side effect.
- OQ-2 / C13 / C14 [STILL OPEN, now four cycles old]: the analysis scripts have
  still not been migrated onto the shared resolver, and only the G-findings are
  canonical. **These are now the oldest outstanding items by a wide margin and
  should take priority over opening any further instrument audit.**

## Cycle 11 decisions
- D20: **Corpus statistics and per-example correctness are different standards.**
  The ACGU filter changes 34% of chains and the worst by +27.9%, yet the corpus
  medians move ~1% and the c=20 budget is untouched. Both facts are true and they
  license different actions: the **published statistics stand**, and the
  **training pipeline must use the canonical loader**. This is the first point in
  eleven cycles where the two standards diverge, and collapsing them either way
  would be wrong — re-running every statistic would be churn, and shipping an
  ACGU filter into the data pipeline would be a defect.
- D21: **Check the excluded population rather than inheriting the exclusion.**
  The published block-sparsity analysis caps at `MAX_L=3000`, which drops the 24
  longest chains — exactly where a per-chain budget is most at risk. Those were
  run separately (max effective c 18.44, both definitions) instead of being
  quietly carried over.
- D22: **An analysis that does not terminate is indistinguishable from one that
  was never run.** The first sensitivity implementation was O(pairs x atoms^2) in
  Python and did not finish. Rewritten with a cKDTree; all 179 structures in
  under two minutes. Logged rather than silently replaced.

## Cycle 11 discoveries
- DISC-45 [C14, MEASURED]: the ACGU filter changes **34.1% of chains** (31 of 91)
  and restores **703 residues**. Worst single chain 7VNV, **L 61 -> 78 (+27.9%)**;
  worst contacts/nt **+46.1%**.
- DISC-46 [THE DECISION SURVIVES]: max effective c is **19.03 published vs 19.04
  canonical**, and **no chain breaches the c=20 budget under either definition**
  — including the 24 longest, where it is 18.44 either way. The sparse-track
  sizing, which is the only design decision these numbers feed, is unaffected.
- DISC-47 [THE HEADROOM IS THINNER THAN ASSUMED]: the worst chain sits at 19.04
  against a budget of 20 — **4.8% headroom**, not the comfortable margin the
  "mean 17.2 for long chains" figure suggests. A mean was being read as if it
  bounded the maximum. Worth stating in the spec: c=20 is adequate for this
  sample and has little room for a corpus with more modified residues.
- DISC-48 [C13 DONE]: `rna_chain_coords()` / `longest_rna_chain()` added to the
  shared module as the single entry point for RNA geometry, mirroring
  `rna_residues()` for counts. Suite now **27 properties**, all pass.

## Open Questions
- OQ-7 [OPEN, priority MEDIUM]: c=20 has only **4.8% headroom** at the worst
  chain in this sample (19.04). The sample is 180 BGSU structures; a corpus with
  more heavily-modified RNA (tRNA-rich, or rRNA from organisms with denser
  modification) would push individual chains higher. Either widen the budget or
  measure the distribution on a larger sample before fixing c.
- OQ-2 / C13 / C14 [**ANSWERED**]: definition sensitivity measured on all 179
  structures; shared geometry loader added and tested.
