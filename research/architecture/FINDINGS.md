# Findings register

Every defect found by the rigor audits, with its current status. The
narrative for each one — what it was, how it was measured, why it hid — is in
`AUDIT_2026-09-25.md` (1–32) and `AUDIT_2026-09-26.md` (33–62). This file is
the index, so "is that fixed?" is one lookup rather than a search.

**Status**

| | |
|---|---|
| `FIXED` | code changed, pinned by a test, verified |
| `FIXED+DATA` | also required rebuilding a derived artefact |
| `CORRECTED` | the claim was wrong, not the code; the claim was corrected |
| `DECLARED` | cannot be fixed now; declared in code and asserted by a test so the declaration cannot go stale |
| `OPEN` | not yet addressed |

Findings 1–32 are closed; see `AUDIT_2026-09-25.md` and the first half of
`AUDIT_2026-09-26.md`.

## 33–37 — the re-audit

| # | what | status |
|---|---|---|
| 33 | fp16 coordinates stored in the raw crystal frame | `FIXED+DATA` |
| 34 | nothing stopped a NaN, in the trainers or the monitor | `FIXED` |
| 35 | four heads emitted and trained by nothing | `FIXED` (fitness, via stage 6) / `DECLARED` (3) |
| 36 | `PharosHeads.loss` dead code disagreeing with the trainers | `FIXED` |
| 37 | two head numbering schemes, both live | `FIXED` — but see 54 |

## 38–42 — stage 6 and the fitness benchmark

| # | what | status |
|---|---|---|
| 38 | NABench and RNAGym are largely one dataset; 24 of 31 leaderboard assays shared | `FIXED` |
| 39 | a column named `DMS_score` holding sequencing read counts | `FIXED` |
| 40 | the consensus wild type is wrong for 9 of 31 assays | `FIXED` |
| 41 | two of three streams starving while the file claimed the opposite | `FIXED` |
| 42 | the leaderboard is largely a mutation counter; no floor was stated | `FIXED` |

## 43–49 — rechecking the pipeline

| # | what | status |
|---|---|---|
| 43 | block scorer's micro-average recall could never execute (`KeyError`) | `FIXED` |
| 44 | the fitness scorer printed a partial checkpoint load and scored anyway | `FIXED` |
| 45 | `r = 0.0` meant both "collapsed head" and "uncorrelated head" | `FIXED` |
| 46 | the NaN guard crashed stage 5 on its first optimiser step | `FIXED` |
| 47 | the verification gate could not be parsed by the declared Python floor | `FIXED` |
| 48 | four test modules existed, passed, and were never gated | `FIXED` |
| 49 | two of four trainers had no way to check that they start | `FIXED` |

## 50–58 — reading the architecture

| # | what | status |
|---|---|---|
| 50 | MoE balance loss divided a masked numerator by an unmasked mean width | `FIXED` |
| 51 | §6.2's electrostatic coupling has no caller | `FIXED` (63–65) |
| 52 | `Pharos.forward` built every pair from sequence 0 | `FIXED` |
| 53 | padding changed `fluctuation` at real positions by 0.206 | `FIXED` |
| 54 | finding 37's fix stopped at the trainers; `heads.py` still disagreed in 9 places | `FIXED` |
| 55 | 71% of stage-5 batches exceeded the token budget | `FIXED` |
| 56 | `fluctuation`'s target was shifted by the batch minimum | `FIXED` |
| 57 | Muon orthogonalised across the expert axis, not within each expert | `FIXED` |
| 58 | "singular values into [0.7, 1.3]" holds for 69% of the tensors | `CORRECTED` |

## 59–62 — the data layer and the GPU run

| # | what | status |
|---|---|---|
| 59 | one NMR model filtered, but not one conformer | `FIXED+DATA` |
| 59a | the obvious fix for 59 deleted a chain; caught by the rebuild, not the test | `FIXED` |
| 60 | the mutual-information joint did not sum to 1, error scaling with alignment depth | `FIXED+DATA` |
| 61 | the verifier's liveness check failed whenever training was live | `FIXED` |
| 62 | stages 2/3/6 batch by sequence count, at 3,478 tok/s against stage 1's 17,000 | `FIXED` — token-budgeted, 16,100 tok/s, 2 epochs 18.9 h → 5.5 h |

## 63–65 — wiring §6.2, and what wiring it exposed

Finding 51 stood as `DECLARED` because head 3 became a denoiser and stopped
emitting coordinates, so the bias had no distance estimate to screen. Wiring
it meant supplying one: `VirtualDistance` projects each residue to three
numbers and takes the pairwise distance -- `L^2 * 3`, not `L^2 * d_pair` --
initialised so the median pairwise separation starts near 20 A, the regime
the screened Coulomb was derived for. Two defects only became reachable once
the path existed, and both are the register's recurring class: a component
that exists, is measured, and whose output is never checked for validity.

| # | what | status |
|---|---|---|
| 63 | `ElectrostaticBias` carried a private copy of `b_elec`: it hardcoded `q_eff`, took a `kappa` where the canonical form takes a Debye length, and omitted the Bjerrum length entirely, making it **7.16x too small**. Its own docstring claimed the bias "is computed from `physics.manning`" -- the intent, not the code. Same class as finding 36, in the physics instead of the losses | `FIXED` |
| 64 | the trunk detached `pair_bias` before the differentiated pass, so `vdist.proj` and `elec.log_scale` were **absent from the graph** while still perturbing the forward pass: the physics was measurable (property 7e saw 4.7e-04 of salt sensitivity) and **frozen at its random init for the whole run**. `grad is None` caught it only because the detach removed them entirely; once the gate was the only thing zeroing them, it would not have | `FIXED` |
| 65 | `ElectrostaticBias.forward` passed `lam=float(torch.exp(self.log_scale))`. The `float()` severs the graph, so the one learned quantity in §6.2 could never receive a gradient however the bias was used | `FIXED` |

The fix for 64 recomputes the bias from the **detached** iterate outside
`no_grad`, so §6.2's parameters get a gradient through the final pass while
the recurrence stays one-step. Property 7f now asserts the magnitude, not
merely the presence, of that gradient, and separately that the zero-init
gate opens itself -- verified end to end: `bias_scale` 0 -> 0.01 at step 1,
`vdist` and `log_scale` moving from step 2. Before the fix they stayed at
exactly 0.000e+00 indefinitely.

## 66–67 — probing the RNA-file channels

Every channel the corpus carries from the mmCIF files, checked for whether it
holds information rather than merely existing. The backbone passed outright:
P-C4' 3.90 +/- 0.13 A, C4'-N 3.38 +/- 0.10 A, consecutive P-P 5.88 +/- 0.78 A
over 2.2M residues -- textbook A-form values, and tight enough to prove the
`(P, C4', glycosidic N)` triple is correctly assigned, since a mis-ordered
atom would scatter them. Two channels did not pass.

| # | what | status |
|---|---|---|
| 66 | `b_factor_z` took its mean and sd from the OBSERVED B-factors (`bf > 0`) and then applied the z to **all** residues, so a residue whose B was never set received `-mean/std` -- a fabricated extreme claiming it is the most rigid in the structure. 3,408 residues below z = -5 across 35 chains; one 271-residue chain had 213 of them at exactly **-7.17**. Those 0.107% inflated the channel's sd from **0.94 to 3.33** | `FIXED` |
| 67 | 4 of the 24 chemistry dims are constant across all 2.2M residues: dims 6, 9, 11 (`wcD`, `hgA`, `sgA`) because the literature table gives all four bases the same value, and dim 13 (`shift`) because nothing outside a unit test ever sets `shifted_pka`. They are exactly redundant with the bias of `nn.Linear(24, d)` | `RECORDED` |

Finding 66's giveaway is that every extreme value **within a chain is one
repeated number**, which a continuous measurement does not produce; and that
the negative tail sat flat at 0.09% out past z = -50 while the positive tail
decayed smoothly to nothing past 20. Excluding the fabricated block the
channel reads mean -0.013, sd 0.944 -- a z-score, as intended.

The same fabrication existed at **chain scale** and was found by writing the
test rather than by inspection: when a chain had no usable statistic -- every
B equal, or none set at all, which is routine in cryo-EM depositions -- the
`else` branch returned `zeros_like(bf)`, telling every residue it was exactly
average rigidity. `3J8G` chain B is 2,874 residues of it. The test asserts the
invariant (a finite z must come from a positive B, and the finite values must
be standardised) rather than a specimen's numbers, which is why it caught a
case the residue-scale fix had not.

The parser now writes **NaN** for an unset B-factor, and the collate rejects
non-finite targets and `z < -6` from `rigidity_mask`. Mapping them to 0 would
have been the same defect one step quieter: the residue would train as exactly
average. The positive tail is kept in full. The guard works on the corpus that
exists today, where the fabricated values are finite, and becomes a no-op once
the corpus is rebuilt.

## 68–71 — running the 3D stage, and reading the router conditioning

Three of these are reachable only by running stage 5 end to end or by reading
every call site side by side; none would fail a unit test, and all four are
the register's class.

| # | what | status |
|---|---|---|
| 68 | `neff_over_l` is read at `moe.py:106` into `extra[:, 0]` of the router conditioning and is **set by no caller anywhere** -- not stage 5, not stages 2/3/6, not inference, not the router audit. One conditioning dimension is permanently zero | `DECLARED` |
| 69 | `predict_structure.py` built `RouterFeatures(length=...)` alone while stage 5 trains with the pooled chemistry too, so **6 of 14 conditioning dims were zero at inference**. `chem` is computed four lines above the call, so the omission bought nothing. On a randomly-initialised model the representation moves **35.3% of signal scale** | `FIXED` |
| 70 | `length_bin_max` was defined inside one trainer, so every other caller silently got the default octave bins. The same chain then lands in a **different router bin** at inference than it trained in: 800, 1,200 and 2,000 nt all move, which is exactly the long-chain regime the hierarchical pair track exists for. The constant now lives beside the binning it parametrises | `FIXED` |
| 71 | stage 5's `evaluate` passed `torch.Generator()` -- a CPU generator whatever the tensors are -- into `random_rigid`, so every CUDA run trained to the end of the epoch and then **died on the first eval batch**: `Expected a 'cuda' device type for generator but found 'cpu'`. Stage-5 validation had never once reported a number on GPU | `FIXED` |

Finding 71 sat in a structural blind spot. The gate runs every suite with
`CUDA_VISIBLE_DEVICES=""`, deliberately -- a neighbour's OOM once turned a
passing suite into DRIFT DETECTED -- and the cost is that **no
device-placement bug can ever be caught by it**. The probe now carries a
source-level check for the class, because a source check runs on CPU, which
is where the gate runs. It reports only real code: comments and string
literals that mention the pattern are tokenised away, after the first
version of the check reported itself twice.

Stage 5 otherwise learns on every head over 150 steps from the stage-1
checkpoint: `mg_auroc` 0.692 -> 0.839, `rigidity_r` **0.134 -> 0.645** on the
b-factor targets finding 66 corrected, `lw_lift` 0.000 -> +0.074, `motif_lift`
-0.010 -> +0.190, contact 0.535 -> 0.373, structure 1.028 -> 0.911.

## Unnumbered, same class

| what | status |
|---|---|
| the monitor alarmed at its own deliberate pause | `FIXED` |
| stage 6's per-assay lists broke the history-equals-csv invariant | `FIXED` |
| four stale "13 test suites" counts against a gate running 17 of 21 | `FIXED` |
| a `--assays` subset printed a full leaderboard ranking with no PARTIAL banner | `FIXED` |
| numpy views pinning 360 MB to retain 27 MB in the masked-marginal scorer | `FIXED` |
| the fitness corpus digest covered composition but not content | `FIXED` |
| distance bins documented as "2–40 Å" actually span 2–41 Å | `CORRECTED` |
| `prewarm` divided `ru_maxrss` by 1e6 and called the result GiB | `FIXED` |
| Muon had no test module at all | `FIXED` |

## Open questions, recorded and not acted on

| what | why it is open |
|---|---|
| are production gradients orthogonalisable? | a 3,840-token probe says mostly not, but production runs 180,224 a step; needs a production-scale batch. A first probe's "0 of 168" is **withdrawn** — 128 tokens against 2,304-wide matrices makes the ranks zero by arithmetic |
| the diffusion violation term is a high-sigma penalty | it sits outside the EDM sigma-weighting; defensible, worth one A/B once stage 5 runs |
| `disorder_logit` is emitted and untrained | needs `entity_poly_seq` in the corpus: a build change, not a loss term |
| `splice_logits` is emitted and untrained | no RNA splice-site corpus has been acquired |
| `ensemble_state_logits` is emitted and untrained | no observable in the corpus distinguishes the K states |
| the supervised fitness head barely transfers across construct families | one bounded epoch from the 34%-budget stage-1 checkpoint reads val rho **+0.604** over 12/12 assays (unseen VARIANTS of seen constructs) against transfer rho **+0.085** over 5/5 (the Townshend aptamer family held out entirely). The split exists to expose exactly this, and the leaderboard's macro-mean includes an aptamer category, so the gap is the number to watch as training continues rather than a defect to fix now |
| the block scorer's published numbers predate the split re-draw | re-measurable for the first time since finding 43; costs a GPU and ~30 epochs |
