# Final Verification Report — Cycle 7 (auditing the instrument)

| Field | Value |
|---|---|
| Cycles | 7 |
| Defects this cycle | **1** (#25) — plus **#26 identified and measured**, opening cycle 8 |
| **Total defects** | **25 confirmed, 26th measured** |
| Macro-audit | **FAIL — 2 open items (C13, C14), by design** |
| Tests | 3 suites (20 + 6 + N properties) all pass; `verify_claims.py` **ALL CLAIMS REPRODUCE** |

## The question this cycle asked

Cycles 0-6 audited the measurements, the derivations, the specification, and the
propagation of corrections. **Nothing had audited the instrument.**

Cycle 6 closed with one residual, logged as OQ-3 and deferred: 18 of 180
structures in a non-redundant *RNA* list appeared to contain no RNA. The stated
reason for deferring was that they "contribute zero residues under either
definition, so no published number depends on it."

**That reasoning assumes the zero is real.** A zero produced by a broken
instrument is indistinguishable from a true zero in the output, and completely
different in the input. It was not real.

## Defect #25 — half the format was never parsed

16 of the 18 returned **zero `_entity_poly` rows**. The raw file:

```
_entity_poly.entity_id                      1
_entity_poly.type                           polyribonucleotide
_entity_poly.pdbx_strand_id                 A,B
```

mmCIF serialises a category in **two** forms: the `loop_` form, and a key-value
form used when the category has exactly one row. The resolver handled only the
first.

**The failure class is not random.** One polymer entity means one RNA chain set
and no protein partner — a *small isolated RNA*. That is precisely the population
G2 quantifies. The bug therefore removed, from the denominator, the very
structures that would weaken the claim.

| | cycle 6 | **cycle 7** | published |
|---|---|---|---|
| structures with countable RNA | 162 | **179** | 180 |
| RNA residues | 306,857 | **309,197** | 308,370 |
| **G2 RNA residues in complexes** | **99.70%** | **98.95%** | 98.96% |

## The finding that matters most in seven cycles

**Cycle 6 published a false finding produced by a defect it introduced in the
same cycle.** Its headline was "G2 strengthens: 98.96% → 99.70%". G2 does not
strengthen. It is 98.95%, which is the published figure. The apparent
strengthening was 16 missing isolated RNAs.

That result shipped: committed, written into ARCHITECTURE.md, main.tex, the
blueprint, and the regression guard, one commit before this cycle caught it. It
was caught only because cycle 6 *recorded* the anomaly it chose to defer, rather
than discarding it — which is the single practice that saved the result.

**REV-9**: "G2 strengthens" is withdrawn. **REV-10**: G3 is 92.65%, not 93.35%.
**REV-11**: the structure count is 179. **REV-12**: G7 is 8.5x, 82 types, 3,237
instances.

## What genuinely survives the canonical re-derivation

Only **G7** moves. Its published 8.90% counted DNA from hybrid duplexes (DT
4,608, DG 4,556, DA 4,440, DC 4,135) and UNK records (7,036) — **24,775 of 27,437
instances, 90% contamination**. The corrected figure is 1.05%, and the argument
is stronger than the one it replaces: **3,237 modified RNA residues across 82
chemically distinct types**, led by pseudouridine (1,003), the most abundant
modification in cellular RNA.

**G2 and G3 are unchanged from the originally published figures.** Two cycles of
work on the RNA-residue definition confirmed them rather than correcting them.
That is a real result, and it is worth more than the false improvement it
replaced.

## New measurement

**21 of 179 structures contain no protein at all, holding 3,257 residues — 1.05%
of the corpus.** Isolated RNA is 11.7% of *structures* but ~1% of *residues*,
because isolated RNAs are small. This sharpens G2's mitigation options: of the
three (restrict to autonomous folds / supply partner context / report split by
complexed-vs-isolated), the first operates on roughly one percent of the
available supervision. **Only the third is affordable.**

## Hybrid chains: resolved, not excluded

Cycle 6 excluded hybrid DNA/RNA chains and measured the loss. Cycle 7 resolves
them with a *structural* test that does not reintroduce defect #22's
curated-list problem: **ribose has an `O2'` atom, deoxyribose does not.** It
classifies BRU (5-bromo-deoxyuridine) correctly as DNA despite its uridine-like
name — which a name list would plausibly have got wrong. Coverage closes at
**179/180**, and the one exception (7PU7, whose only nucleic entity is modelled
entirely as DNA) is explained rather than unexplained.

## The guard now protects the retraction

`verify_claims.py` pins the *wrong* cycle-6 values alongside the right ones:

```
both columns present: cycle-6 vs cycle-7 totals       306,857 -> 309,197
both columns present: the retracted G2 strengthening    99.70 -> 98.95
```

A future edit that tidies the mistake out of the record fails the build. **The
record of a correction is itself a claim under guard.**

## Confidence

**MEDIUM-HIGH measurements · MEDIUM-LOW derived claims · LOW outcomes.**

The level is unchanged but the basis has shifted again. Cycle 6 argued that
repeated re-derivation partly re-applied the same undefined rule. Cycle 7 shows
something sharper: **re-derivation through a shared instrument propagates that
instrument's defects into every result at once**, and does so in a way that looks
like agreement. The three G-findings that have now survived *two* independent
definitions of the instrument deserve MEDIUM-HIGH. Nothing that has been through
the resolver only once does.

## Cycle 8 opens with a measured target — defect #26

OQ-4 asked whether other categories suffer the same two-serialisation problem.
**They do, and it is measured, not hypothesised.** `loops()` in
`extract_basepair_geometry.py` — shared by the geometry, stiffness and ionic
scripts — does `cols.append(tag)` and **discards any value on the same line**, so
a key-value category yields zero rows exactly as `_entity_poly` did:

| category | files with it | `loop_` | **key-value (lost)** |
|---|---|---|---|
| `_exptl_crystal_grow` | 64 | 0 | **64** |
| `_struct_conn` | 169 | 166 | **3** |
| `_pdbx_unobs_or_zero_occ_residues` | 165 | 162 | **3** |
| `_ndb_struct_na_base_pair` | 156 | 155 | **1** |
| `_ndb_struct_na_base_pair_step` | 155 | 154 | **1** |
| `_em_buffer_component` | 32 | 31 | **1** |
| `_entity_poly_seq` | 180 | 180 | 0 |
| `_atom_site` | 180 | 180 | 0 |

`_exptl_crystal_grow` is key-value in **every one of the 64 files that carry
it** — it is the crystallisation-condition category, the source for the salt
concentrations behind the ionic-metadata audit. If that audit reads it through
`loops()`, it has been reading nothing.

## Defect base rate, seven cycles

| Cycle | Defects | Character |
|---|---|---|
| 0 (build) | 5 | silent parser/logic bugs |
| 1 (audit) | 4 | wrong physics, overclaim, disclosure |
| pre-2 | 1 | precision/rounding |
| 2 | 4 | inflated count, unfair comparison, unimplemented code |
| 3 | 1 | hardware incoherence |
| 4 | 5 | derived quantities: cost, depth, budget |
| 5 | 2 | components never specified at all |
| 6 | 3 | no definition of the unit of measurement; a repair that damaged its own record |
| **7** | **1** | **the audit's instrument, which published a false finding** |

**Twenty-five defects across seven cycles, and the rate has not fallen.** The
trajectory of *character* is the actual finding: wrong values (0-2) → wrong
derivations (3-4) → missing specifications (5) → missing definitions (6) →
**defective instrumentation (7)**. Each level was invisible from the one below.
Cycle 7's defect could not have been found by re-checking numbers, because the
numbers were self-consistent; it took asking why a residual existed at all.

## Known limitations carried forward

- **C13 / OQ-2 still open.** 13 analysis scripts carry their own RNA definitions.
  The resolver has now changed twice, so any copied definition is drifting from
  canonical with nothing to detect it.
- **C14 still open.** Only the G-findings have been re-derived canonically. The
  103,964 base-pair steps, 44,708 Mg records, block occupancy and contact
  scaling have not — and defect #26 means at least one structure's steps and
  three structures' connectivity are missing from them outright.
- **Defect #26 is measured but not yet fixed.** That is cycle 8.

---

# Final Verification Report — Cycle 6 (the unit of measurement)

| Field | Value |
|---|---|
| Cycles | 6 |
| Defects this cycle | **3** (#22, #23, #24) |
| **Total defects** | **24** |
| Macro-audit | **8/8 YES** |
| Tests run | 3 suites, 15 + 6 + N properties — **all pass**; `verify_claims.py` **ALL CLAIMS REPRODUCE** |
| Open to-dos | 0 open, 3 deferred, **2 opened for cycle 7** (C13, C14) |

## The question this cycle asked

Cycle 5 concluded that the newest defects were *absences* rather than errors, and
argued the audit should target absences. Cycle 6 targeted the most basic absence
available: **does the project have a definition of its own unit of measurement?**

It did not. Two scripts reading the same 180 files reported 307,965 and 308,370
RNA residues — a discrepancy visible since cycle 2 and never chased. Neither was
wrong about its own rule, because **there was no shared rule**. Every script had
invented one, and they disagreed on **72 of 180 structures, in both directions**.

Every residue-weighted number in the project — stiffness, ion coordination, block
occupancy, contact scaling, all four G-findings — is a count of RNA residues.

## Defect #22 — no definition

| script | residues | structures | rule |
|---|---|---|---|
| `analyze_ions_motifs.py` | 307,965 | 179 | hardcoded list of 16 residue names |
| `audit_generalization.py` | 308,370 | 180 | `len(comp) <= 3`, not water, `group=ATOM` |

7PKT: the hardcoded list is **767 short** (misses modifications not on the list).
8JDJ: the permissive rule is **156 short the other way** (drops HETATM-coded RNA).

Resolved by taking the definition from the format rather than from a curated
list: mmCIF *declares* polymer type in `_entity_poly.type`, so a residue is RNA
iff its chain is a `polyribonucleotide`. A depositor's declaration is complete by
construction; a residue-name list can only be as complete as its author's
knowledge, and RNA has >170 known modifications.
`src/pharos/data/mmcif_entities.py` is now the single shared resolver.

## Defect #23 — 24.87% "modified" was 90% contamination

The first canonical implementation gave **404,205 residues, 24.87% outside
A/C/G/U**. RNA modification rates are 1-2%, so that number diagnosed itself.
Composition dump: **HOH 37,857 and MG 9,997 in the first 60 files alone** —
water and ions sit in the *same auth chain* as the RNA they solvate. mmCIF
assigns `label_seq_id` only to polymer positions:

```python
if r.get("label_seq_id", ".") in (".", "?"):
    continue
```

**404,205 -> 306,857 residues; 24.87% -> 1.04%; solvent leakage 0.**

The same contamination class explains the published G7. Its 8.90% counted
**DT 4,608 + DG 4,556 + DA 4,440 + DC 4,135 + UNK 7,036 = 24,775 of 27,437**
instances — 90% of the "modified nucleotides" were DNA from hybrid duplexes and
unknown records not in RNA chains at all. **G7 was 8.6x too high.**

## Defect #24 — the repair carried its own defect, again

Cycle 4's #18b established that a repair can introduce a new defect. The cycle-6
propagation was therefore audited, and it had failed twice:

1. **Partial.** Only **4 of 11** affected numbers moved to the 162-structure
   basis. G2's protein row read `157/162` while the multi-chain row directly
   beneath it still read `150/180`; G3 carried the canonical *ratio* `93.35%`
   over the stale `286,990 / 308,370`. Three tables mixed two denominators inside
   a single row-set — invisible to a reader, because every individual number was
   defensible in isolation.
2. **Self-corrupting.** An unguarded global replace of the canonical values
   overwrote the **published** column of the correction table itself, so it read
   `99.70 -> 99.70` and `1.04 -> 1.04 (8.6x too high)`. Third occurrence of the
   unguarded-replace failure mode, and the first to destroy a table whose only
   purpose was to record a correction.

The durable fix is not vigilance — vigilance was already in place and failed —
but two new guards that make the failure fatal to the build:

- **Guard A**: each correction row must contain *both* columns
  (`308,370`&`306,857`, `98.96`&`99.70`, `93.07`&`93.35`, `8.90`&`1.04`).
  A replace that collapses one into the other now fails.
- **Guard B**: stale denominators (`150/180`, `61/180`, `159/180`) may appear
  **at most once per document** — legitimate in a published column, a failure
  anywhere else.

## What moved

| Finding | published | **canonical** | verdict |
|---|---|---|---|
| structures with a declared RNA entity | 180 | **162** | 18 had none |
| RNA residues | 308,370 | **306,857** | −0.5% |
| G1 longest chain median / max | 67 / 3,679 | **86.5 / 3,764** | median was dragged down by non-RNA chains |
| G2 structures with protein | 88.3% | **96.9%** | **stronger** |
| G2 structures with >1 RNA chain | 83.3% | **72.8%** | −10.5 pts |
| G2 RNA residues in complexes | 98.96% | **99.70%** | **stronger** |
| G3 ribosome-like entries | 33.9% | **37.0%** | +3.1 pts |
| G3 residues ribosomal | 93.07% | **93.35%** | unchanged |
| G7 outside A/C/G/U | 8.90% | **1.04%** | **8.6x too high** |
| G7 distinct modification types | 8 | **76** | undercounted 9.5x |

**G2 — the largest generalization hazard in the design — strengthens.** 99.70% of
RNA residues sit in protein-containing entries. **G3 is unchanged.** **G7's number
was wrong but its argument is now better founded**: not a percentage inflated by
DNA, but 76 chemically distinct modifications led by pseudouridine (999), OMG
(373), A2M (365), OMC/OMU (268 each), inosine (80). "A model that cannot
represent pseudouridine cannot be said to handle any RNA" is the stronger claim.

## What was tested, not asserted

The resolver is load-bearing for every count in the project, so it is now tested
rather than trusted — `src/pharos/data/test_mmcif_entities.py`, 15 properties over
all 180 files, **ALL TESTS PASS**: solvent excluded, DNA excluded, every residue
at a resolved polymer position, the five aggregate totals, PSU top, determinism,
and the `;`-delimited multi-line `_entity_poly` block parsed correctly on the
largest entry (7QVP, 162 chains).

Hybrid chains are excluded by the definition, which is a *choice*, so it was
measured: 3 entries, 59 residues, **6 of them A/C/G/U = 0.002%** of the total.
Immaterial here, documented in the resolver, with instructions for a corpus where
it would not be.

All three suites now run inside `verify_claims.py`.

## Confidence

**MEDIUM-HIGH measurements · MEDIUM-LOW derived claims · LOW outcomes.**

Unchanged in level, but the reasoning behind the first term has shifted. Six
cycles of independent re-derivation kept confirming the measurements, and that
was read as evidence they were sound. Cycle 6 shows part of it was evidence that
**the same undefined rule was being re-applied**. The measurements that have now
survived a *definitional* check — the four G-findings — deserve the MEDIUM-HIGH.
The rest inherit whichever rule their script invented, and that exposure is
**unmeasured rather than zero** (-> C13, C14).

## Known limitations carried out of cycle 6

- **Only the G-findings were re-derived canonically.** 103,964 base-pair steps,
  44,708 Mg coordination records, block occupancy and contact scaling still come
  from scripts with their own definitions. They key on different mmCIF categories
  that carry their own chain references, so the exposure is plausibly smaller —
  but "plausibly" is the word this cycle exists to eliminate. **OQ-2, priority
  HIGH.**
- **18 of 180 sampled structures declare no RNA polymer entity.** They contribute
  zero residues under either definition, so no published number depends on it,
  but a BGSU non-redundant *RNA* list containing entries with no RNA entity is
  unexplained. Either the sampler or the `_entity_poly` parse is wrong for those
  files. **OQ-3, priority MEDIUM.**

## Defect base rate, six cycles

| Cycle | Defects | Character |
|---|---|---|
| 0 (build) | 5 | silent parser/logic bugs |
| 1 (audit) | 4 | wrong physics, overclaim, disclosure |
| pre-2 | 1 | precision/rounding |
| 2 | 4 | inflated count, unfair comparison, unimplemented code |
| 3 | 1 | hardware incoherence |
| 4 | 5 | derived quantities: cost, depth, budget |
| 5 | 2 | components never specified at all |
| **6** | **3** | **no definition of the unit of measurement; and a repair that damaged its own record** |

**Twenty-four defects across six cycles. The rate has still not fallen**, and the
character has shifted once more — from *wrong values* (0-2), to *wrong
derivations* (3-4), to *missing specifications* (5), to **missing definitions and
self-inflicted regressions (6)**. Defect #24 is the first caused by a previous
cycle's repair, which is the argument for continuing: the corrections themselves
are now a defect source, and only a guard that fails the build catches that.

---

# Final Verification Report — Cycle 5 (buildability)

| Field | Value |
|---|---|
| Cycles | 5 |
| Defects this cycle | **2** (#20, #21) + 10 unspecified components |
| **Total defects** | **21** |
| Macro-audit | **8/8 YES** |
| Open to-dos | 0 (2 deferred to the 5B checkpoint) |

## The question this cycle asked

After #18 (attention heads) and #19 (decoder) — both components *never specified
at all*, then silently assumed small — cycle 5 asked of every component:
**could an engineer implement this from the spec, without inventing a number?**

**37 enumerated, 27 specified, 10 not.** Seven genuinely absent (learning rate,
batch size, warmup, the five loss weights, diffusion steps, MoE bias-update rate,
GDN state size); two present only in reference code or the catalogue, never in a
config; one exposed a circularity.

All closed or marked **PROVISIONAL with the sweep that fixes them**. A guessed
hyperparameter is worse than an acknowledged gap, and the 5B checkpoint costs
~1.5% of the run.

## Defect #20 — the circularity sweep had only ever been run on the router

G6 (cycle 2) found the router consumes features produced downstream of itself.
**That sweep was never run on anything else.** `B_motif` has the identical shape:
it enters the *trunk's* attention bias but is keyed on the *interaction graph*,
which is the pair track's output — produced after the trunk. `B_elec` had a
first-pass rule; `B_motif` never did; and G6's own fix was described in prose but
never written into the bias equation. All three now share one explicit
`attention_bias_recycle_schedule`.

## Defect #21 — the MoE copied one DeepSeek ratio and not the other

| | d_ff/d | active | **active FFN / d** |
|---|---|---|---|
| DeepSeek-V3 (verified) | 0.286 | 9 | **2.571x** |
| **PHAROS-Small** | **0.250** | **6** | **1.500x** |
| dense | — | — | 4.000x |

Per-expert width matches the cited template. **Active capacity is 1.7x below it
and 2.7x below a dense FFN.** The design copied the segmentation and not the
activation, and the ratio had never been examined. Fix is cheap — top-8 gives
2.50x for +12.6M active and ~0 total, since the experts already exist.

## Confidence

**MEDIUM-HIGH measurements · MEDIUM-LOW derived claims · LOW outcomes.**

Unchanged from cycle 4, and for the same reason: measurements keep surviving
independent re-derivation; things built on top of them keep not.

## Defect base rate, five cycles

| Cycle | Defects | Character |
|---|---|---|
| 0 (build) | 5 | silent parser/logic bugs |
| 1 (audit) | 4 | wrong physics, overclaim, disclosure |
| pre-2 | 1 | precision/rounding |
| 2 | 4 | inflated count, unfair comparison, unimplemented code |
| 3 | 1 | hardware incoherence |
| 4 | 5 | **derived quantities**: cost, depth, budget |
| **5** | **2** | **components never specified at all** |

**21 defects. The rate has not fallen across five cycles.** The *character* has
shifted though: cycles 0-2 found things that were wrong, cycles 4-5 found things
that were **missing or never checked against their own stated template**. That
is a meaningfully different failure mode and argues the audit should continue to
target *absences* rather than errors.

---

# Final Verification Report — Cycle 4

## Summary

| Field | Value |
|---|---|
| Cycles used | 4 |
| **Defects found (cycle 4)** | **4** (#16-#19) |
| Total defects, all cycles | **19** |
| Macro-audit | **6/6 YES** |
| Open to-dos | 0 (3 deferred: need hardware or training) |

Cycle 4 asked a question no earlier cycle had: after ~20 individually-correct
corrections, **does the specification still cohere as a whole?** It did not.

## Defects

| # | Defect | Impact |
|---|---|---|
| **16** | The ladder's `loops` column mixed **train-time** (Small: 8) with **serve-time** (Base-v2: 3) | At matched serve-3, Small is **48** effective layers vs Base-v2's **96** — the depth claim **reverses at inference** |
| **17** | **Refinement loops were never in the FLOP budget.** `6*N_active*T` costs one pass; Small runs 8 loops with deep supervision at each | Every cost understated by its loop count. **"4.4x smaller" is really 1.65x**; lever chain **24x -> 5.6x**; cost **78 A100-h** at 25B |
| **18** | Attention **head count never specified anywhere** | Spec gap; fixed at 8 x 64 |
| **18b** | **My own cycle-2 global replace** corrupted 5 parentheticals into "46,447 RNA all-polymer" | Repaired; the exact failure mode cycle 1 warned about, reintroduced while fixing a different defect |
| **19** | "Motif bank + heads + decoder ~10M" was a placeholder; the **decoder was never specified at all** | Itemised: heads 1.65M, bank 0.17M, decoder **12.59M** — the decoder alone exceeds the line. Totals **149M/61M -> ~153M/~65M** |

## The most consequential finding

Defect #17. The design stated that loops "cost compute but no additional
activation memory" — and then **omitted the compute**. The memory half was
right; the compute half was simply missing from every calculation.

| Model | loops | published h | real h |
|---|---|---|---|
| Micro | 16 | 61 | 976 |
| Mini | 12 | 148 | 1,776 |
| **Small** | **8** | **299** | **2,392** |
| Base-v2 | 3 | 1,325 | 3,975 |

Verified by **two independently written cost paths** agreeing at 78 A100-h for
the decided 25B budget.

Note what survives: the cycle-3 token cut (12.9x) was large enough that the
corrected absolute cost is still modest — **78 A100-h bf16, 12 H100-h fp8**.
The *claims* were wrong; the *plan* remains affordable.

## Confidence

**MEDIUM-HIGH on measurements; MEDIUM-LOW on derived claims; LOW on outcomes.**

Derived claims drop a notch: four of the last five defects were in *derived*
quantities (cost, depth, budget) rather than measurements. The measurements have
held up under independent re-derivation; the arithmetic built on top of them has
repeatedly not.

## Defect base rate

Cycle 0: 5 · cycle 1: 4 · pre-2: 1 · cycle 2: 4 · cycle 3: 1 · **cycle 4: 5**.
**Nineteen defects. The rate has not fallen.** Two were caused by formatting,
two by unfair statistical comparison, one by code that did not exist, and now
two by quantities that were never specified at all. The newest category is the
most concerning: **unspecified components silently assumed to be small.**

---

# Final Verification Report — Cycle 2

> Cycle 1's report follows below, unchanged. This covers the 6,259 lines added
> across 55 files *after* cycle 1 closed, plus the architecture-level audit the
> user requested mid-cycle ("check if all of our architecture is efficient and
> will be able to generalize").

## Summary

| Field | Value |
|---|---|
| Cycles used | 2 |
| Searches run (cycle 2) | 3 (Muon, HRM ablation, DeepSeek FP8) |
| Tests logged (cycle 2) | 8 (S1–S7 + G1/G2/G3/G7) |
| To-do completion | 14 done, 0 open, **1 DEFERRED** (G5, needs training) |
| Doubts | 3 logged, 3 resolved |
| **Defects found** | **4** (#11–#14) |
| **Major reversals** | **1** (REV-2) |
| Macro-audit | **9/9 YES** |

## Defects found in cycle 2

| # | Defect | Impact |
|---|---|---|
| **11** | `_pdbx_unobs_or_zero_occ_residues` counted **all polymers**. 143,871 rows are 67.5% protein; RNA-only is **46,447**. | A novelty claim overstated **3.1x**. Claim survives at one third the size. |
| **12** | Stiffness headroom fitted every group **in-sample** and scored each model on its **own covered subset** (103,964 / 90,098 / 78,076 steps). | **REV-2**: structure-over-sequence falls 3.0282 -> **1.0336** (2.9x) and **the order inverts**. |
| **13** | The auxiliary block-occupancy loss was documented as "implemented". Only a **hook** existed — no BCE anywhere in the repo. | The R1 downgrade rested on unwritten code. Now implemented and tested. |
| **14** | FP8 "<0.25% loss error **at 671B scale**" — the DeepSeek ablation was at **V2/V2-Lite scale (~1T tokens)**. Plus the HRM memorisation caveat existed in prior-art 07 but **never propagated** to where the argument is made. | Misattribution + a material caveat missing from the load-bearing citation. |

## The reversal (REV-2)

**Retracted**: "structural context contributes more than sequence context"
(+3.028 vs +2.144 nats), used to justify the learned stiffness encoder.

**Corrected**, held-out on the common 78,076-step subset:

| Model | in-sample | held-out |
|---|---|---|
| M0 global | 19.7235 | 18.1607 |
| M1 sequence | 17.5792 | 16.3760 (gain **1.7847**) |
| M_struct structure only | 17.8470 | 16.4198 (gain 1.7409) |
| M2 sequence x structure | 14.5510 | **15.3424** (gain over sequence **1.0336**) |

**Survives**: structure still adds a real **+1.03 nats beyond sequence**;
sequence-alone (+1.78) and structure-alone (+1.74) are near-equal and
complementary (3.53 if independent vs 2.82 actual). The learned-encoder decision
stands — a sequence-only lookup still leaves ~1.03 nats unused. The §7c
acceptance gate moved 14.551 -> **15.3424**.

## Architecture audit — the user's question

**"Is it efficient?" — YES, and it is the best-verified part.** The hierarchical
pair track runs at 0.96% of dense at L=4096 while the dense baseline cannot run
at all on a 14 GB machine; parameter and FLOP arithmetic reproduces exactly
(148.73M/60.65M against documented 149M/61M); the lever chain reproduces exactly
(1,883 -> 79 A100-h, 24.0x).

**"Will it generalize to any RNA?" — NOT AS SCOPED.** Four measured limits:

| ID | Finding | Status |
|---|---|---|
| **G2** | **98.96% of RNA residues sit in protein-containing entries** (159/180 structures, median 10 protein chains). The model predicts single chains but learns partner-stabilised folds. | **Largest hazard. Not fixable by tuning.** Split reporting now mandatory. |
| **G3** | **93.07% of RNA residues come from 61 ribosome-like entries.** Every residue-weighted statistic is primarily ribosomal. | Length-binned tables happen to stratify it; the c=20 budget is safe at both ends. Headline figures relabelled. |
| **G7** | **8.90% of polymer residues fall outside {A,C,G,U}** — 17,767 DNA, 7,036 UNK, 2,634 inosine/other. | Vocab-5 cannot represent any. "Any RNA" unsupportable without widening it. |
| **G1** | Every single chain fits 4096 (max 3,679), but total RNA per entry reaches **11,478**; 44/180 entries exceed the context. | Scope statement added. |
| **G6** | The router reads structural features produced by the pair track, which runs *after* the trunk — **routing at recycle 0 is undefined**. `B_elec` got an explicit first-pass rule; the router never did. | Design gap; fix specified. |

**Honest headline**: *a single-chain RNA structure model, up to 4,096 nt, trained
predominantly on ribosomal and complex-embedded RNA, which must report
performance split by isolated vs complexed context.*

## Confidence

**MEDIUM-HIGH on the measurements; MEDIUM on the design; LOW on outcomes.**

- *Measurements* — every headline number is now re-derived by an independently
  written check, and the regression guard re-derives them on demand. High.
- *Design* — internally consistent and the efficiency case is strong, but four
  generalization limits are now measured rather than hypothetical. Medium.
- *Outcomes* — **no model has been trained.** Nothing here predicts that PHAROS
  beats TM 0.55. Low, and should stay low until a checkpoint exists.

## Known limitations carried forward

1. **R1**: block detection is now a supervised target with a tested loss, but
   the test is a single-example synthetic overfit. Generalization untested.
2. **G5 [DEFERRED]**: 61M active vs ERNIE-RNA's 86M dense while doing strictly
   more tasks. An experiment, not an argument; first checkpoint ~1 GPU-day.
3. **G2** is unresolvable with current data — it bounds the claim permanently.
4. Muon 2x and FP8 1.6x are *reported* figures, verified as citations but
   **not benchmarked on our shapes**. The 24x cost reduction depends on them.
5. Thin bins persist: contact-sparsity n=3 and n=6; coevolution deep arm n=2;
   within-structure Mg estimate n=4.
6. Ionic titration data (RMDB) still not acquired — prerequisite for the
   headline ion-conditioning claim.

## Defect base rate

Cycle 0 (build) 5 · cycle 1 (audit) 4 · pre-cycle-2 1 · **cycle 2: 4**.
**Fourteen defects, every one producing a plausible number rather than an
error.** Two were caused by formatting rather than logic; two were unfair
statistical comparisons; one was code that did not exist. The base rate has not
fallen across cycles, which is the strongest argument for keeping
`verify_claims.py` in the loop.

---

# Final Verification Report — PHAROS empirical claims

**Date**: 2026-09-13 · **Cycles**: 1 (extended) · **Verdict**: claims stand after 6 corrections

## Scope
Every empirical claim in `research/` re-derived independently (separately-written
parsers, not re-runs of the original scripts), with statistical validity checks,
derived-arithmetic recomputation, literature verification, and a cross-document
consistency scan.

## Results

### PASSED unchanged (re-derived independently)
| Claim | Status |
|---|---|
| Mg²⁺ 17,428 / K⁺ 1,868 / 96 Mg-bearing structures | reproduces exactly |
| Inner-sphere OP1+OP2 share 0.83 | 0.8284 independent |
| Mg/rigidity gradient 1.760 sigma, monotonic | reproduces exactly |
| contacts/nt median 4.397; long-chain density 0.353% | reproduces |
| flat-proposal recall 0.200; random baseline 0.746 | reproduces |
| block occupancy 1.34%, effective c 17.24 | reproduces (17.12 strict) |
| coevolution mean precision@L/5 0.670 | reproduces |
| HPT benchmark: L=4096 at 0.96% of dense, c=19.6 | reproduces |
| parameter budget 910.5M / 382.0M | exact |
| 309 GB dense activations at L=2048 | exact |
| SOTA TM-scores (trRosettaRNA 0.548 etc.) | VERIFIED vs primary source (n=52 monomers) |

### DEFECTS FOUND AND CORRECTED

**1. PHYSICS ERROR (substantive).** Documents stated axial charge spacing
`b ~ 5.9-7.0 A` together with `theta ~ 0.76`. These are mutually inconsistent
(b=5.9 A gives theta=0.175). Manning's `b` is the **axial** charge spacing, not
the P-P contour distance. Corrected to A-form RNA values **b = 1.40 A,
xi = 5.11, theta = 0.804** (`q_eff = -0.196`); noted that 0.76 is the **B-DNA**
figure. Verified against literature (B-DNA xi=4.2/theta=0.76; A-RNA atmosphere
neutralises ~0.8 of phosphate charge).

**2. OVERCLAIM (substantive).** The coevolution depth split was described as
"the decisive pattern" and "the strongest argument for MoE in the whole design".
Testing shows: exact permutation p=0.0455 on a **post-hoc** threshold,
**Spearman +0.224** over n=12, deep arm **n=2**, and a *shallow* family reaching
precision 1.000. Retracted in all three deliverables; the architectural case now
rests on the literature and on measured between-family variance (0.333-1.000).

**3. SAMPLE MIS-STATEMENT.** The Mg/rigidity gradient was reported over
"31 X-ray structures". It is measured over **15** (the Mg-bearing subset); 31 is
the set used for density/sequence-context analyses. The `>=30-RNA-residue` guard
was also unstated. Both corrected.

**4. SILENT EDIT FAILURE.** An unguarded string replacement had silently failed,
leaving the report with **6 novelty rows while the other documents had 7**, and
leaving two retracted phrases in the report after a failed retry. Found by the
cross-document scan; fixed. All subsequent edits use asserted replacements.

### STRENGTHENED (new analysis, not previously reported)
- **Confound control on the headline 1.76 sigma claim**: partial correlation of
  z_B with Mg-distance controlling for packing density is **+0.372** vs **+0.420**
  raw, so the signal is largely independent of packing. Within-structure estimate
  **+1.514 sigma**, positive in 4/4 structures, bootstrap CI [+1.256, +1.661].
  This materially improves the claim and is now in all three deliverables.
- **30 A centroid prefilter proven safe**: theoretical bound 22.20 A, observed
  max 18.59 A, zero contacts missed in an exact no-prefilter test.
- **Ground-truth incompleteness disclosed**: the coevolution parser omits WUSS
  pseudoknot brackets, 53 of 537 pairs (9.87%). Bias is conservative (deflates
  precision), so 0.670 is a lower bound.

### KNOWN LIMITATIONS (flagged, not resolved)
- Contact-sparsity bins at L=100-200 (n=6), 200-500 (n=3) and 500-1500 (n=9) are
  thin. The O(L) conclusion rests on the well-populated ends (n=38 and n=61),
  with bootstrap CIs [1.335,2.063] and [4.757,4.957] respectively. The
  interpolating bins are illustrative, not load-bearing. [FLAGGED]
- Within-structure rigidity estimate rests on 4 structures. [FLAGGED]
- Minor ion-count drift between parsers (Zn 362 vs 359, ~0.8%) attributable to
  mmCIF loop-termination handling. Below tolerance; headline numbers unaffected. [FLAGGED]
- `effective_c` naive accounting overstates by 1.029x (17.61 vs 17.12 strict) —
  conservative, left as-is with a note. [ACCEPTED]
- "8x cheaper attention" counts only the quadratic term; including sliding-window
  cost the honest ratio is 7.53x. Footnoted in the report. [CORRECTED]
- **R1 remains open**: the reference implementation proves the hierarchical
  track's cost and structure, not that a trained scorer finds occupied blocks.

## Cycle 1 extension — mmCIF mining and the training design

After the audit closed, the scope extended to mining the data more fully and
designing the training. Two further defects surfaced, both in new code:

**5. mmCIF ROW WRAPPING.** Long mmCIF rows wrap across physical lines (a
43-column base-pair-step row arrives as 24 + 19 tokens). A parser requiring all
fields on one line yields **zero rows** for every wide category — silently. Fixed
with quote-aware token accumulation. This unlocked 103,964 annotated steps.

**6. METAL/LIGAND MISCLASSIFICATION.** A length-and-case heuristic classified the
nucleotides G, A, U and C as metals. Replaced with an explicit metal set.

### New measurements (all verified)
- **103,964 base-pair steps** across 155/180 structures; stiffness matrices
  `F = kT C^-1` for **76 contexts**. Validated: Watson-Crick means reproduce
  canonical A-form RNA (GG/CC rise 3.14 twist 29.98; AU/AU rise 2.81 twist 33.94).
- Stiffness spans **115x** (twist force constants; the earlier 134x came from
  4-decimal rounding that left the softest constants with one significant
  figure). GC content predicts rigidity (Pearson -0.314).
- **44,708 curated Mg²⁺ coordination records** whose coordinating-atom ranking
  (OP2 > OP1 > O6 > O4 > O2' > N7) **independently reproduces** the earlier
  distance-based result. Two methods, same conclusion.
- **Stiffness headroom**: M0 19.724 / M1 17.579 (sequence table) / M2 14.551
  (sequence x structure). Structural context contributes **more** than sequence
  (+3.028 vs +2.144 nats) => a learned encoder beats a lookup table. This
  changed the design from a tabulated force field to a learned stiffness encoder
  with a Gaussian-NLL objective and an explicit acceptance threshold.
- **Guard sensitivity (OQ-1 CLOSED)**: sweeping the >=30-residue guard down to
  none moves the Mg-gradient span only 1.760 -> 1.757 sigma, monotonic
  throughout. A concurrent session's claim to this effect was verified
  independently rather than accepted.

## Regression guard
`scripts/sampling/verify_claims.py` re-derives every headline number, checks the
derived arithmetic and closed-form physics, verifies all 14 shared tokens appear
in all three documents, and asserts the four retracted phrases are absent.
Exit 0 = reproduces. Currently: **ALL CLAIMS REPRODUCE**.

## Confidence assessment
**MEDIUM-HIGH.** The measurements are reproducible and now confound-controlled;
the two substantive defects (physics, overclaim) were in the *interpretation*,
not the data. Confidence is not HIGH because: the sample is 180 structures and
12 alignments; several length bins are thin; and the central architectural
claim (learned block detection) is still unproven by construction.

**Nine defects have now been found across this work** — seven in implementation
(BGSU HTML-as-CSV, `_exptl.method_details` overwrite, MI outer-product
broadcast, HPT L3 diagonal-only expansion, HPT L2 budget clamp, mmCIF row
wrapping, metal/ligand misclassification) and two in the write-up (the Manning
physics error and the coevolution overclaim), plus one silent unguarded string
replacement. Every one of them would have produced plausible-looking but wrong
output rather than an error.

That base rate is the central lesson: **in this kind of work the failure mode is
silence, not crashes.** Keep `verify_claims.py` in CI, re-run it before any
publication of these numbers, and treat any weak or surprising result as a
suspected bug until independently reproduced.
