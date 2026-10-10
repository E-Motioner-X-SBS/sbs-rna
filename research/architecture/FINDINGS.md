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

## 72–73 — the resume path, and the one head without a floor

| # | what | status |
|---|---|---|
| 72 | resuming stage 5 restored the scheduler to `step` and then replayed the partially-done epoch **from batch 0**, so OneCycleLR raised `Tried to step 151 times. The specified number of total steps is 150`. The checkpoint is written mid-epoch by design, so a mid-epoch resume is the normal case. Second time the schedule has killed a run that had finished its work; `_sched_step` now clamps | `FIXED` |
| 73 | head 2 reported **bare accuracy** while heads 9 and 10 report theirs against the majority rate, through the helper written for exactly that. The floor was added -- and on the first run that printed it, **head 2 is collapsed**: see 75 | `FIXED` |
| 75 | with the floor computed on the SAME batch, `dist_acc` tracks `dist_major` to the digit at every step -- 0.4830/0.4830, 0.4690/0.4690, 0.2900/0.2900 -- and `dist_macro` sits at 0.026-0.031 against 1/39 = 0.0256. Head 2 predicts **one bin and nothing else**. It had been reporting 0.47 accuracy for the whole of stage 5 and reading as a working head | **RESOLVED by 86** — see below: 41.88% of the supervised population was that one bin |

## 74 — the inference path reported a clean number for an impossible structure

| # | what | status |
|---|---|---|
| 74 | `predict_structure.py`'s only geometric check is `clash_score`, which excludes pairs within one residue of each other -- correct for a non-bonded clash, and blind to the BONDED geometry. A draw came back **`clash 0.000`** with a consecutive P-P median of **40.35 A** against a real 5.88, a 118x122x109 A bounding box for a 46-nt chain, and two atoms **0.59 A** apart: a structure that cannot exist, reported as clean, because every impossible distance was one the metric skips. Sample selection was `argmin(clash)`, which is inert in exactly that case -- an exploded draw has no neighbours to clash with, so every draw scores 0.000 and `argmin` returns draw 0 whatever the geometry | `FIXED` |

Inference now reports the bonded violation beside the clash, against the
bands the TRAINING loss already penalises (`BOND_C4_N`, `BOND_P_P`), so the
sampler is checked against the physics it was fitted to rather than a second
opinion; and draws are ranked on the bonded violation first.

It separates cleanly, which is the part worth stating: over 120 real corpus
chains the violation runs **0.000–0.034** with C4'-N 3.37–3.41 A and P-P
5.79–5.93 A, 119 of them under 0.10, while the undertrained draw scores
**0.989** and prints `BACKBONE NOT CONNECTED`. A check that always fails is
worth no more than one that never does.

(The draw itself is expected to be poor -- that checkpoint's structure head
is untrained. The defect is that nothing in the inference path said so.)

## 76–77 — the data path, and two statistics with one name

| # | what | status |
|---|---|---|
| 76 | **WITHDRAWN.** I reported the GPU idling at 28% because `to_device(tr.collate(idxs), device)` sat inline in the training loop. The 28% was measured while two of my OWN CPU-heavy jobs were running beside the trainer. Sampled on a quiet machine the same loop reads **mean 66%, median 66%, no zero samples**, and an interleaved A/B reads 117/117/116/117 s -- the prefetch buys nothing measurable. The epoch scaling I cited as proof (227 -> 251 -> 399 s) was my own contention, which is exactly what it looked like | `WITHDRAWN` |
| 77 | training reports `rigidity_r` as **per-batch Pearson averaged over batches**; validation reports `rigidity_r_pooled` as **pooled Pearson over all residues**. Near-identical names, different statistics, printed side by side -- 0.717 against 0.0874 reads as catastrophic overfitting. On the SAME data with the SAME model the per-batch mean is **+0.2075** and the pooled is **+0.0188**, an 11x gap from the statistic alone, with per-batch values ranging +0.006 to +0.482 | `RECORDED` |

Finding 77 is not a bug in either number -- the pooled one is the honest
summary and the per-batch one is a legitimate training signal. The defect is
that they are named as though they were comparable, so the obvious reading of
the training log is wrong. Measured rather than argued: two statistics, one
model, one set of batches.

Finding 76's fix carries its own hazard, because a prefetch can lose data
without raising -- a dropped batch is a smaller epoch, a reordered one breaks
nothing visible, and an exception swallowed on the worker hangs the consumer
forever. The probe asserts all three, and `--no-prefetch` keeps the ablation
runnable, since a speed claim with no way to turn it off is not a measurement.

**Finding 76 is withdrawn, and the way it failed is worth more than the
finding would have been.** The first A/B read 294.55 s inline against
117.46 s prefetched -- a 2.5x win. The verification gate finished between
the two runs and the load average fell from 9.45 to 2.96, so the second arm
ran on a quieter machine. Interleaved, the four runs read 117, 117, 116,
117 s: **no difference at all.**

The diagnosis was contaminated the same way. The 28% utilisation and the
227 -> 251 -> 399 s epoch scaling were both measured while my own rigidity
job and the verification gate were running beside the trainer. On a quiet
machine the same loop samples at mean 66%, median 66%, with no zero
readings. I measured my own interference and attributed it to the code.

A third and fourth design agree. The 40-step runs are startup-heavy (a 6.46 GiB corpus
and a 4.7 GB checkpoint), which could in principle hide a step-time
difference, so both arms were rerun at 200 steps to difference the startup
out: **912.5 ms/step prefetched against 900.0 ms/step inline**, i.e. 1.4%
*slower*. Three measurements, two designs, one answer.

The last defence of the `Prefetcher` was that it guards against exactly the
contention that produced the false reading, which on a box with 73 logged-in
users is a real operating condition. That was an assumption, so it was
measured too: under **sixteen CPU burners**, at load 16.7-18.8, the arms
read **117 s prefetched against 116 s inline**. It does not help there
either -- the burners starve the worker thread and the main thread alike, so
there is no spare core for the overlap to use.

**The `Prefetcher` is reverted.** Four measurements in three conditions, no
benefit in any, and marginally negative on the one with the tightest error
bars. It was a thread, a bounded queue, and three tests' worth of
concurrency failure surface added to a training loop on the strength of a
reading that turned out to be my own interference. Keeping it because it
might help somewhere unmeasured is the habit this register exists to break.

What the numbers actually say: the step is **~900 ms** and collate is a
small part of it, so the 66% utilisation is ordinary per-step overhead --
optimiser, metrics, Python -- not a data stall. There is no data-loading
problem in stage 5 to fix.

### A correction, and what it cost

I first put head 2's lift at **+0.2761** and wrote that the head was
learning. That was wrong, and wrong in the way this register keeps finding:
the 0.47 came from a training run and the 0.19 floor came from a majority
share I had sampled MYSELF over different batches, where random negatives
land in the catch-all bin at a different rate. Two numbers from two
samplings, subtracted. The floor is only a floor when it is computed on the
same data as the accuracy -- which is precisely why `_class_metrics`
computes `*_major` per batch, and precisely the mistake the helper exists to
prevent. Once head 2 went through the helper the lift read 0.000.

Head 2's collapse is left `OPEN` rather than guessed at. What is established:
the head predicts a single bin; `dist_macro` is at chance for 39 classes; at
random init `dist_lift` is -0.1195, so it *moved* to the prior rather than
starting there; and the contact head reads 1.78x lift from the SAME `pair`
tensor, so the representation it is given carries signal. What is not yet
established is why. One more measurement narrows it: the distance LOSS does
improve, 2.546 -> 2.205 by step 200, against a bin-prior entropy of 2.872
nats -- so the predicted DISTRIBUTION carries information while the argmax
stays degenerate. That is a head spreading mass over neighbouring bins
without ever peaking off the majority, which points at the 40-bin
discretisation and the negative sampler (the catch-all bin beyond 41 A takes
21-48% of pairs depending on batch composition) rather than at a severed
gradient. Accuracy was the wrong statistic for this head either way: the
loss and the macro recall disagree with it, and only the macro recall and
the floor say what it is doing.

A caution for reading any of the per-step numbers above: they are
single-batch, and batch composition swings hard -- `dist_major` alone ranges
0.29 to 0.48 across steps. The pooled epoch validation is the signal, and it
improved on every head.

## 78–79 — adding a parameter broke every resume and every evaluation

Running the curriculum end to end from the 3.340B-token stage-1 checkpoint
found both of these in the first ninety seconds, and neither is visible
without actually running it. Wiring §6.2 (finding 63) added one parameter,
`vdist.proj.weight`, 2,304 of 394,780,316. Every checkpoint in the tree was
written before it existed.

| # | what | status |
|---|---|---|
| 78 | `train_sequence_stages.py` and `train_pharos.py` both load the RESUME path with `model.load_state_dict(resume["model"])` -- **strict** -- three lines above an `--init-from` path that uses `strict=False` and reports what it loaded. One new parameter therefore turned every resumable checkpoint in the tree into a bare `RuntimeError: Missing key(s) in state_dict: "vdist.proj.weight"`, which says nothing about how much of the model is affected or whether continuing is sound. A project that adds parameters as it goes cannot have its resume path be the one that breaks | `FIXED` |
| 79 | `_check_load` refuses to score a checkpoint missing any non-allowlisted tensor. Correct in form, wrong here: in all these checkpoints `bias_scale` is stored as **exactly 0.0** -- the zero-initialised gate the §6.2 wiring was built around -- so `vdist.proj.weight` is multiplied by zero and cannot move a logit. The guard counted tensors where the question was reachability, and stopped the only published-comparable number this model has | `FIXED` |

**78's fix is not `strict=False`.** That is worse than the crash: a resume
that silently re-randomises half the trunk reports as a resume and is not
one, which is the exact failure `--init-from`'s own guard exists to stop, on
the path that never had it. `pharos.train.checkpoint.load_resume` loads
non-strictly, **names** every fresh tensor in the log, and refuses past 1% of
the model -- measured in PARAMETERS, not tensors, because one missing tensor
can be a 2,304-element projection or a 285M-element embedding table and the
tensor count cannot tell them apart. The test asserts both directions: a
0.0972% addition resumes and is named, a 99.5% one is refused with the tensor
named and the fraction stated.

**79's fix is not an allowlist entry either.** Adding `vdist.proj.weight` to
`_MAY_BE_MISSING` would wave it through on a checkpoint where the gate HAD
opened -- a real defect hidden, to silence one that does not exist. The
question is empirical, so `_provably_inert` measures it: one forward pass at
the loaded weights, then three more with the named tensors filled from
`randn * 10`, in fp32 rather than the evaluator's bf16 so the comparison is
as sensitive as the hardware allows. Bitwise-identical logits is a proof for
THAT checkpoint and claims nothing about any other, which is the property an
allowlist cannot have.

The control matters more than the result. Run against four tensors of the
same checkpoint:

| tensor | verdict | evidence |
|---|---|---|
| `vdist.proj.weight` | inert | 3 draws at 10x, logits bitwise identical |
| `elec.log_scale` | inert | same -- it is behind the same zero gate |
| `trunk.blocks.7.mixer.bias_scale` | **reachable** | draw 1 moved the logits by 6.790e-03 |
| `mlm_norm.weight` | **reachable** | draw 1 moved the logits by 3.886e+01 |

The gate itself is reachable at 6.8e-3 while the tensor it gates is not
reachable at all. That is the whole §6.2 design stated as a measurement, and
it is also the discrimination that makes the proof worth trusting: a check
that returned "inert" for everything would have passed this checkpoint too.

## 80–81 — what an independent implementation of the same physics said

`md_rnaions` (Ryan Hayes, the reference code behind generalized Manning
condensation) arrived as a reference for §6.2. Checking our closed form
against it is the only check of that form that has ever been worth
anything: `test_manning.py` compares our constants against numbers we
derived ourselves, which catches arithmetic and nothing else, because a
wrong formula reproduces its own wrong value perfectly.

| # | what | status |
|---|---|---|
| 80 | **NOT a defect, and worth as much as one.** Two implementations sharing nothing but the physics — ours from SI constants and the Malmberg–Maryott cubic, Hayes' in C from `eps = exp(5.71455988 - 0.004540698*T)` — agree on ε_r to 1.6e−4, on l_B to 1.2e−4, and on λ_D to **6.2e−5 at every concentration from 25 to 300 mM**. §6.2's closed form is independently confirmed | `CONFIRMED` |
| 81 | `IonicCondition.screening()` evaluates `condensed_fraction` and `effective_charge` at **z = 1 whatever is in solution**. Raising Mg²⁺ from 0 to 15 mM shortens λ_D 9.6078 → 7.9788 Å and leaves θ and `q_eff` **bit-identical**. Manning gives θ = 1 − 1/(zξ), so a divalent condenses 0.9022 against 0.8044 and `q_eff` halves — and `B_elec` goes as `q_eff²`, so the omitted channel is **4× against the 1.2× the screening term supplies**. The model's whole response to Mg²⁺ ran through the smaller of the two | `FIXED` |

The two models also disagree on λ_D with Mg²⁺ present, by 20.9% (11.918 Å
against 9.857 Å), and that one is **not** an error in either. Hayes runs Mg
as explicit particles, so they must not also appear in the implicit
screening — his `c_Cl` rises by `2·c_Mg` for electroneutrality and the Mg
itself is left out of κ. PHAROS has no explicit ions, so Mg belongs in the
ionic strength with z² = 4. It is asserted in the test module rather than
left to be rediscovered, because anyone comparing a PHAROS ionic response
against a number from that paper is comparing two different definitions of
the screening length.

**81's fix is not z = 2.** The solution is a mixture and there is no closed
form for one — which is precisely why that paper exists: it gives every
phosphate its own dynamical θᵢ and finds it by minimising a free energy
with Debye–Hückel shell terms, a mixing entropy and a soft constraint.
`ElectrostaticBias.theta_head` predicts θᵢ from the representation instead,
and `b_elec_sitewise` uses `qᵢ·qⱼ` where the global form used `q_eff²`.

Two properties make it a generalisation rather than a replacement:

* θᵢ is squashed into **[0.80440, 0.90220]** — Manning's own bounds for z=1
  and z=2 — so the head cannot leave the interval on which "condensed
  fraction" means anything. Asserted at sigmoid saturation with inputs
  scaled by 100, not on a typical input.
* `site_scale` is zero-initialised, so at init every residue carries the rod
  charge and the bias is **bitwise identical** to the global form, max |Δ|
  0.000e+00. No loaded checkpoint's forward pass moves.

The head is inert until the gate opens — at `site_scale = 0` the derivative
with respect to `theta_head` is exactly zero, which is finding 64's shape.
Here it is deliberate, and it is **declared and asserted in both
directions** (gate takes gradient at init 1.8e+00; head takes none; head
takes 2.4e−01 once the gate is nudged) rather than discovered later by a
gradient probe. The dead window is short: the sibling gate `bias_scale` is
measured reaching 1.04e−02 within 200 steps of stage 2/3.

### A correction, recorded because it changed a conclusion I had already given

I reported that §6.2 was "inert everywhere except after stage 5" and
contributing 0.0006%, from the gate values in three checkpoints. Two of
those three **predate the wiring in-process**: stage 1 finished 2026-10-04,
the wiring landed 2026-10-06 16:32, and the seqstages checkpoint written at
16:48 carries 536 tensors — no `vdist.proj.weight` — so that process had the
pre-wiring code loaded. §6.2 had never once been trained in a sequence
stage. The first run that has gives, after **200 steps**:

| | before | whole stage-5 run | seqstages, 200 steps |
|---|---|---|---|
| `bias_scale` blk 7 | 0 | 2.80e−03 | **1.04e−02** |
| `bias_scale` blk 15 | 0 | 1.73e−03 | **6.39e−03** |
| `elec.log_scale` | 0 | 4.67e−05 | **9.53e−04** |

The physics was not quiet by design. It had never been reached. Reading a
parameter's value out of a checkpoint says what that RUN did, and a run is
identified by the code it had loaded, not by the commit that exists now.

## 82–84 — the full run, end to end, and what it exposed

The curriculum ran from the 3.340B-token stage-1 checkpoint to RNA-Puzzles
scores in 4h24m: gate 611 s, zero-shot 2,095 s, stages 2/3/6 10,596 s,
stage 5 1,062 s, prediction 38 s, scoring 73 s.

| # | what | status |
|---|---|---|
| 82 | `probe_channels.py` perturbed `mod_ids` with `m2[0, 0] = 1` against a `mod` drawn from `randint(1, 4)`. When the draw was already 1 the "perturbed" input WAS the original and the delta was exactly 0.00e+00 — a dead channel reported for a live one. `torch.manual_seed(0)` is set three lines before the MODEL is built, and initialisation consumes the stream, so adding head 11 and `theta_head` shifted every subsequent draw, `mod[0,0]` became 1, and a probe that had passed all night started failing the gate and blocking the pipeline. **The seed made it deterministic, not correct** | `FIXED` |
| 83 | stage 5 runs **317 optimiser steps over 8 epochs** — 40 per epoch, 1,062 s in total. A denoising diffusion decoder is being asked to learn a generative model of RNA backbone geometry in three hundred steps. At inference the consequence is unambiguous: consecutive P–P distances come out at **16–19 Å against a 5.88 Å target**, bond violation 0.79–0.93, and the structures are not connected chains | **ADDRESSED** — 3,069 steps since, and P–P came to 5.67–6.69 Å; the step count is no longer the binding constraint (93) |
| 84 | head 11 learns on the training batches and **does not generalise**: `tors_theta_lift` reaches **+9.79°** at step 250 and the same epoch's validation reads **−0.74°**; η +5.71 train against −1.27 val. On 317 steps and 6,585 chains that is the expected shape, but it is recorded rather than assumed, and the family-disjoint test agrees (η −0.60, θ −0.26, χ̃ **+0.45**, the only one positive) | `OPEN` |

Finding 83 is the one that matters, and the inference guard of finding 74 is
what made it visible rather than flattering: every one of the seventeen
predictions carries `BACKBONE NOT CONNECTED`, and the clash score — the
metric that would have been reported without that guard — reads a clean
**0.001** on all of them, because `clash_score` skips pairs within one
residue and every impossible distance here is exactly such a pair.

**The blind-test result, stated plainly.** 17 RNA-Puzzles targets, scored
against the 885 submitted competitor models:

| | PHAROS | field best | field median |
|---|---|---|---|
| mean TM | **0.0546** | 0.4592 | 0.3013 |
| best single target | 0.0725 (rp05) | 0.7629 (rp04) | — |
| mean lDDT | 0.125 | — | — |

That is random-coil territory and it is the honest number. The sequence-level
heads are a different story on the same checkpoint — contact AP 0.8060
against a 0.4541 base rate (1.77× lift), Mg AP lift 2.92×, and #6 of 14 on
the RNAGym ncRNA leaderboard — so what is failing is specifically the
generative decoder, not the representation it reads.

### What stage 5 actually produced, epoch 0 → 7 (validation)

| metric | ep 0 | ep 7 | floor |
|---|---|---|---|
| contact_ap | 0.5907 | **0.8060** | 0.4541 |
| mg_ap_lift | 1.28× | **2.92×** | 1.0× |
| lw_macro | 0.0887 | **0.1465** | — |
| structure_violation | 1.1566 | **0.5087** | — |
| structure_bond_pp | 0.571 | **0.2192** | — |
| rigidity_r_pooled | 0.0384 | **0.1180** | 0.0 |
| motif_lift | −0.0112 | −0.0102 | 0.0 |
| dist_lift | 0.000 | 0.003 | 0.0 |

Family-disjoint test: contact_ap 0.6755 against a 0.3475 base rate (1.94×,
*higher* lift than validation), mg_ap_lift 1.98×, lw_lift +0.0181.

Two heads are flat on validation and both are already on the register:
`motif_lift` is −0.01 on val while training reaches +0.20, and head 2 is
still collapsed — `dist_acc` 0.480 against `dist_major` 0.477, macro 0.030
against a 1/39 = 0.026 chance. Finding 75 reproduces exactly.

## 85 — the denoiser was asked for 0.035% of its task

The structure head produces disconnected backbones (finding 83): P–P at
16–19 Å against 5.95 Å, TM 0.0546 against a field median of 0.3013. The
obvious explanation was the 317 optimiser steps. It is not the whole
explanation, and the real one is a two-character omission.

| # | what | status |
|---|---|---|
| 85 | `DiffusionConfig` raised `sigma_data` from the image-diffusion default 0.5 to **16.0** for ångström coordinates — correctly — and left `p_mean` at Karras's **−1.2**, which is the value *for* `sigma_data = 0.5`. The training noise is set in units of the data's own scale, so the median `sigma/sigma_data` fell by the same 32× it was raised by: from 0.602 to **0.0188** | `FIXED` |

**What that does is visible in the preconditioning, not in the loss.**
`D(x;σ) = c_skip(σ)·x + c_out(σ)·F(…)` with `c_skip = σ_d²/(σ² + σ_d²)`, so
`c_skip` is the fraction of the answer handed to the network for free:

| | median σ | median `c_skip` | draws asking >10% of the work | draws above σ=40 |
|---|---|---|---|---|
| shipped | 0.301 | **0.99965** | 2.78% | **0.06%** |
| derived | 9.626 | 0.73426 | 68.9% | 11.7% |

At the median training draw the network was asked for **0.035%** of the
task. The loss it reported was therefore honest and meaningless: a near-
identity is a very good denoiser at σ = 0.3.

**Measured on the trained checkpoint**, denoising real validation
structures at a sweep of σ:

| σ | RMSE after denoising | RMSE of the noise alone | reduction | P–P median |
|---|---|---|---|---|
| 0.1 | 0.100 | 0.100 | **1.00×** | 5.96 |
| 0.3 | 0.295 | 0.300 | **1.02×** | 5.94 |
| 3.0 | 2.217 | 3.009 | 1.36× | 6.08 |
| 20 | 8.369 | 20.08 | 2.40× | 7.51 |
| 160 | 12.932 | 160.7 | 12.43× | **18.46** |

(true P–P on that batch: 5.95 Å)

At σ = 0.3, where half the training mass sits, it removes **2%** of the
error. At σ = 160, where the sampler starts, one step leaves P–P at 18.46 Å
— which is exactly the 16–19 Å the seventeen RNA-Puzzles predictions came
out at. **It learned to polish a structure that was already there and never
learned to make one**, because it was almost never asked to.

**The fix is a derivation, not a number.** `p_mean` is now
`ln(0.6016 · sigma_data)` in `__post_init__` and cannot be typed beside
`sigma_data` again; `p_std = 1.2` is Karras's and is scale-free, being a
width in log-σ. At `sigma_data = 0.5` the derivation returns **−1.2013**,
reproducing the paper, which is the check that it is the right derivation
and not a fitted constant. An explicit `p_mean` still overrides, for the
ablation.

The test asserts the coupling at four scales, asserts the live config's
median `c_skip` is in `(0.5, 0.85)` — **and asserts that the configuration
that shipped would fail that check**, at `c_skip` 0.99965. A test that only
passes on the fixed code proves nothing about the bug.

No checkpoint changes numerically: `p_mean` enters only the training noise
draw, never the forward pass. It does mean `pharos_shared400.pt` was
trained against the wrong distribution and its structure head has to be
retrained.

## 75 RESOLVED, and 86 — the floor that was built into the bins

Finding 75 sat `OPEN` for two sessions: head 2 predicts one bin, `dist_acc`
tracks `dist_major` to the digit, `dist_macro` is at 1/39. What was never
established was why. Measuring the target distribution settles it.

Over **891,804 supervised pairs from 394 training chains** — the population
head 2 is actually scored on, which is every true contact plus the contact
head's uniform random negatives at |i−j| ≥ 4:

| bin | range | share | from contacts | from negatives |
|---|---|---|---|---|
| 10 | 12–13 Å | 3.74% | 28,778 | 4,574 |
| … | every other bin | ≤ 3.1% | | |
| **39** | **≥ 41 Å** | **41.88%** | **0** | **374,241** |

**The floor was built into the binning.** `floor(d − 2)` clamped to 2–41 Å
puts 41.88% of the population in one catch-all bin, every bit of it a
negative, and a head that predicts that bin scores the majority rate. The
observed `dist_major` of 0.42–0.478 is this number. Nothing was wrong with
the head, the gradient or the representation — the task had a 42% majority
class and no reason not to take it.

| # | what | status |
|---|---|---|
| 75 | head 2 collapsed onto one bin | **RESOLVED** — 41.88% of the supervised population was that bin |
| 86 | the obvious fix is the wrong one. Narrowing to 2–22 Å at 0.5 Å — "concentrate resolution where the contacts are" — pushes *more* of the sampled population into the catch-all and takes the majority to **68.83%**, worse than the scheme it replaces | `RECORDED` |

Six schemes scored on the measured distribution:

| scheme | majority | entropy / max | contacts resolved |
|---|---|---|---|
| current, 2–41 Å uniform | 41.88% | 2.707 / 3.689 | 100% |
| 2–22 Å, 0.5 Å | **68.83%** | 1.653 | 99.91% |
| 2–32 Å, 0.8 Å | 52.53% | 2.316 | 100% |
| 40 equal-frequency | **2.50%** | **3.689 / 3.689** | 100% |
| **8 fine (3–10 Å) + 32 equal-frequency** | **3.01%** | 3.558 / 3.689 | 100% |

Equal-frequency reaches the theoretical maximum `ln(40)` and by construction
has no class to collapse onto. The shipped scheme is the hybrid: pure
quantiles leave a **5.88 Å first bin**, and a Watson–Crick pair has a
characteristic C4′–C4′ distance that a 5.88 Å bin cannot see. Eight fixed
0.875 Å bins over 3–10 Å cost 3.5% of the maximum entropy and buy 6.7× the
resolution exactly where base pairing lives.

The edges are **fixed constants derived once from the training split** —
recomputing them per batch would make the label mean a different distance
from one step to the next, and deriving them from anything but train would
leak — and `distance_bin()` is the one definition, because the trainer
inlining its own copy of a binning rule is how the label drifts from what
the head was told it meant.

### Also fixed here

`tors_*_mae` was the name of two different statistics: the training loop's
circular mean of **one batch's** targets, and `evaluate`'s circular mean of
**the whole split's**. Printed side by side, that reads as a generalisation
gap partly made of a change in baseline — finding 77 exactly. The eval keys
are now `tors_*_{mae,base,lift}_pooled`, following `rigidity_r_pooled`, and
the test asserts the two name sets are disjoint.

The gap itself survives the rename and is real: training θ error 17.0° at
step 250 against 29.8° on validation in the same epoch. On 317 steps that is
finding 84, not a metric artefact.

## 87 — a column declared, and still nothing writing it

The training watch, on its own, on the live run. Three of the four
warnings it raised in this window were false and are fixed in the check;
the fourth was real and was **mine, from an hour earlier**.

| # | what | status |
|---|---|---|
| 87 | `part_n_pairs` was declared in `_PART_SCALARS` (fixing "computed every step, no column") and the `log()` call still read `for k, v in parts.items() if k != "n_pairs"`. The key had been filtered out at the call site *because* it had no column — RunLog says an undeclared key out loud, so it was silenced rather than declared — and declaring the column without removing the filter left a column nothing writes. The same defect, one layer along, introduced by the fix for it | `FIXED` |

The test now **parses the call site** rather than trusting it: it extracts
the `**{f"part_{k}": ...}` expression out of `main`'s source and asserts
there is no ` if ` in it. Asserting "every key `step_losses` returns has a
column" passes in both the broken and the fixed state, which is why it did
not catch this.

The run in flight has the old code loaded, so `part_n_pairs` stays empty
for it and populates on the next.

### Three false positives in the watch, and what each was

| reported | verdict | fix |
|---|---|---|
| `base_acc` tracks `base_major`, 1.0000 vs 1.0000 | **false** — `base_n_class` is 1. The base-identity head is supervised only where identity was unassigned, and a batch can hold one base. "Predicts the majority class" is vacuous with one class | skip `n_class <= 1` |
| five heads "macro at chance" | **false** — first reading of a restarted run, where every head is at chance | require three readings, as the lift check already did |
| `lw_acc` tracks `lw_major` over 2 readings | **false** — same, two steps into a restart | require three readings |

All three are the same mistake: reporting a true statement that is not yet
a finding. Finding 75 held for an entire stage, so nothing real is lost by
waiting two steps, and a watcher that fires on ordinary start-up behaviour
is one nobody reads.

## 88 — a fifth of the run spent writing a file with the card idle

The user noticed the GPU utilisation fluctuating. It does, and the cause is
exact rather than jittery.

Sampling the A100 at 1 Hz for 253 s during stage 5, alongside the
checkpoint file's mtime, on a quiet machine with the trainer as the only
GPU process and a load average of 2.50:

| stall start | duration below 50% | checkpoint rewritten |
|---|---|---|
| t = 18 s | 10 s | t = 27 s |
| t = 100 s | 11 s | t = 111 s |
| t = 219 s | 13 s | t = 232 s |

Three stalls, three writes, every pair aligned.

| # | what | status |
|---|---|---|
| 88 | `atomic_save` writes a **4.73 GB** checkpoint — model plus optimiser state plus scheduler — with an `fsync`, **on the training thread**, every `--ckpt-every 100` steps. Measured: **20.4% of wall-clock below 50% GPU and 6.2% at exactly 0%**, mean utilisation 71.5% where the busy stretches run at 86–100% | `FIXED` |

Not thermal and not contention: SM clock held 1335–1410 MHz against a
1410 nominal, power averaged 207 W of a 300 W cap, and `nvidia-smi` showed
one compute process. The trace is unambiguous:

```
#++#+#########+##+........C##+####+######+#+###.######+##+#+###+#####+##########
#++#+###++#+++#..........C#######+#+######+####+####+##+++####+##+#+#######++#+#
####+##+#+##+##+##+#+#+####+###+####.######+#+#+...........C##+##+######+#++####
```

(`#` ≥75%, `+` ≥25%, `.` <25%, `C` the second the checkpoint was rewritten)

**`BackgroundSaver` moves the write off the training thread**, and two
things make it safe where finding 76's prefetch thread was not:

* **the state is snapshotted to CPU before the thread starts.** A
  `state_dict()` handed to another thread is a view of tensors the
  optimiser is about to update, so the file would be a mixture of two
  steps. The test asserts exactly this: it mutates the live tensor the
  instant `save()` returns and requires the file to hold the value as of
  the call — and separately asserts the mutation really happened, because
  otherwise the first assertion proves nothing.
* **`atomic_save` already renames into place**, so a reader sees one whole
  version whichever thread wrote it.

One writer at a time; a save due while one is running is **dropped and
counted**, because queueing lets a slow disk grow memory without bound and
a checkpoint is worth exactly as much as the next one. `close()` is called
before any exit path that follows training — a daemon thread would be
killed at interpreter shutdown and the last checkpoint would be the one the
process did not finish writing, which is worse than the stall, because it
is silent.

**The 20.4% is measured; the improvement is not yet.** `--sync-ckpt` is the
ablation arm and the A/B has to be run on a quiet card against a run that
is not already in flight. Finding 76 was withdrawn for claiming a speed-up
measured against my own interference, and this entry will not repeat it:
what is established is the cost of the current behaviour, not the size of
the saving.

## 89 — P–P is not a stage-1 problem, and the data is not the problem either

The P–P bond is the chain: consecutive phosphates at 16–19 Å instead of
5.9 Å is not a bad structure, it is not a structure. Two hypotheses were
worth testing before blaming the head, and **both are refuted by
measurement**.

**Hypothesis 1: chain breaks are poisoning the target.** The violation
mask is `mask[1:] & mask[:-1]` — the *residue* mask — which never asks
whether residues i and i+1 are consecutive in the chain, and the corpus
carries `unobserved_seq_id`. If unmodelled gaps were being taught as
bonds, the loss would be pushing 20 Å apart pairs toward 6 Å.

Measured over **2,227,344 array-adjacent P–P pairs from 3,072 chains**:

| | |
|---|---|
| median / mean / sd | 5.88 / 5.96 / 0.78 Å |
| inside the flat bottom [4.51, 7.51] | **99.09%** |
| below | 0.59% |
| above | 0.32% |
| above 10 Å (real breaks) | **0.11%** |

The targets are clean. **REFUTED.**

Worth keeping from it: a *perfect* prediction scores `structure_bond_pp` =
**0.0149 Å**, not 0, because 0.32% of true pairs sit outside the band and
contribute 90.2% of that floor. The metric has a floor and it is now
written down.

**Hypothesis 2: it is stage 1 — the representation does not carry what the
bond needs.** This is testable without touching the model, because if P–P
is a geometric constant then no representation is needed to predict it.
Over 2,224,936 pairs, grouped by dinucleotide — the only sequence
information the head could use:

| | |
|---|---|
| overall mean / sd | 5.9465 / 0.5508 Å |
| pooled within-dinucleotide sd | 0.5485 Å |
| between-dinucleotide sd | 0.0527 Å |
| **variance explained by sequence** | **0.9151%** |

A constant predictor of 5.947 Å scores RMSE 0.5508 Å on every pair in the
corpus. The best possible sequence-conditioned predictor reaches 0.5485 Å
— an improvement of **0.0023 Å**. The sixteen dinucleotide means span
5.845 to 6.032 Å against a within-group sd ten times larger.

**P–P is a geometric constant. Stage 1 cannot be the cause, because there
is nothing in the representation for it to get right.** REFUTED.

What is left is the head's own training, and that is findings 85 and 83:
the denoiser was trained almost entirely at σ where the input already had
correct bonds, so it never had to *create* them, and it had 317 steps in
which not to. The run with both fixed has taken `structure_bond_pp` from
2.538 at epoch 0 to 0.307 at epoch 21 and it is still falling — against a
floor of 0.0149.

The prediction that follows, and that the next blind-test run will settle:
if 85 and 83 were the cause, P–P at inference comes down with it; if it
does not, the remaining suspect is the sampler, not the representation and
not the corpus.

## 90 — the sampler was never told which residues are adjacent

Finding 89 ruled out the corpus and ruled out stage 1, and named the
sampler as the remaining suspect. It is the sampler.

| # | what | status |
|---|---|---|
| 90 | `predict_structure._sample_with` is a **second implementation** of `DiffusionStructureHead.sample`, written to control the noise seed, and it hardcoded `pair=None`. Training calls `structure.loss(coords, hidden, diff_pair(hidden, coev), mask)`; inference called `denoise(x, σ, single, None, mask)`. The decoder was trained with pair conditioning and sampled without it | `FIXED` |

`DiffusionPairFeatures` exists for one reason and its own docstring says
so: *"**Relative position**, clamped to ±max_rel. Without it the decoder
has to infer every geometric relationship from per-residue embeddings."*
Sampling with `None` asked the model to build a chain **without telling it
which residues are adjacent**, which is precisely the information a P–P
bond is. It produced exactly that: consecutive phosphates at 16–19 Å
against a true 5.95, on all seventeen targets, while the training-time
violation looked fine because training had the features.

Two implementations of one thing, and the public one was the broken one —
the same shape as the `pair_index` defect already recorded in
`Pharos.forward`, where the only caller of the broken path was the test
that validates the interface.

Measured after the fix, same noise, gates opened: threading `pair` through
changes the sampled structure by **19.70 Å** maximum. It is not a
refinement.

### Two false positives in my own test, both worth keeping

**The first draft tested a freshly constructed head and reported max |Δ| =
0.0000.** `DenoiseBlock.out.weight` and `CoordDenoiser.out.weight` are
zero-initialised *by design* — a deep denoiser starts as a shallow one —
so on an untrained head the attention output is exactly zero and `pair`
provably cannot matter. The test now opens those gates first, the way
property 7f opens `bias_scale`. Testing a zero-init model for whether an
input matters is vacuous, and it reported "the sampler ignores pair" for a
sampler that does not.

**The second draft reported the sampler non-deterministic.** The repeat
call sat *outside* the `torch.no_grad()` block, so autograd picked
different kernels and the last bits differed. Verified separately: two
identical calls under `no_grad` give `torch.equal` True. The sampler is
bit-exact.

The adjacency claim is checked rather than asserted: adjacent pairs embed
2.16 apart from |i−j| = 7, and with `rel.weight` zeroed that collapses to
0.52 — so it is the relative-position table carrying it, not the single
projections.

## 91 — and the re-run's result

| # | what | status |
|---|---|---|
| 91 | `--sync-ckpt`'s help text, added with finding 88, contained a literal `%` ("the 20.4% measured"). argparse %-formats help strings, so `train_pharos.py --help` exited 1. `test_head_metrics.py` caught it on the post-training re-verify and blocked the gate — which is the whole reason that check exists: "a literal % raises only when --help is run, which nobody does on a trainer" | `FIXED` |

### The re-run, against the baseline

Findings 75, 83, 85, 86 and 90 fixed; stage 5 restarted from
`seqstages_shared400.pt` with 3,069 optimiser steps against 317.

**P–P, which was the question:**

| | before | after | true |
|---|---|---|---|
| median over 17 targets | 16.26 – 19.31 Å | **5.67 – 6.69 Å** | 5.95 Å |

Every one of the seventeen. The backbone is a chain.

**RNA-Puzzles, 17 targets against 885 submitted competitor models:**

| | before | after | field median | field best |
|---|---|---|---|---|
| mean TM | 0.0546 | **0.0913** | 0.3013 | 0.4592 |
| mean lDDT | 0.125 | **0.2765** | — | — |
| targets beating the field median | 0 | **0** | — | — |

TM up 67%, lDDT up 121%, **and still nowhere near the field.** Best single
target is rp15 at 0.1145 against a 0.2271 median on that target. The
geometry is now locally correct and globally wrong: bonds right, fold
wrong, which is what an lDDT of 0.2765 against a TM of 0.0913 says — local
environments are starting to be recovered and the global superposition is
not.

What that leaves, in order: the fold itself needs the pair track and the
contact head feeding the decoder (contact AP is 0.885, and none of that
reaches head 3 except through `single` and `diff_pair`), and 3,069 steps
on 6,585 chains is still a small run for a generative decoder.

## 92 — the model predicts contacts well and never uses them to build

Why is TM 0.0913 while lDDT is 0.2765 and contact AP is 0.885? The
attention maps answer it.

**Only 2 of the trunk's 18 blocks can see globally at all.** The §5.1
period-8 pattern gives `Counter({'gdn': 12, 'swa': 4, 'full': 2})` — twelve
linear-attention blocks, four sliding-window at 128, and **blocks 7 and 15**
are the only full softmax attention. Those two are also the only ones
carrying §6.2's `bias_scale`.

Measured on 12 held-out chains, AUROC of the symmetrised attention weight
against the true contact map, with the pair bias included in the logits
(chance = 0.5 whatever the imbalance):

| attention map | all \|i−j\|≥4 | long >24 | very long >64 |
|---|---|---|---|
| **trunk 7** (full) | 0.5039 | 0.4523 | 0.5213 |
| **trunk 15** (full) | 0.4962 | 0.4452 | 0.4439 |
| decoder 0 | **0.5967** | 0.5498 | **0.5875** |
| decoder 1 | 0.5546 | 0.4967 | 0.4841 |
| decoder 2 | 0.5450 | 0.4586 | 0.4059 |
| decoder 3 | 0.5198 | 0.5066 | 0.4747 |
| decoder 4 | 0.4823 | 0.4141 | 0.4852 |
| decoder 5 | **0.5929** | **0.5579** | 0.5392 |

| # | what | status |
|---|---|---|
| 92 | **Both globally-connected trunk blocks attend at chance with respect to the contact map, at every separation.** The decoder's first and last blocks carry weak signal (0.59) and its middle four are at chance. Meanwhile the contact head reaches **AP 0.885 against a 0.4541 base rate**. The model knows the contacts and no attention layer routes information along them | `OPEN` |

The decoder's signal is **the pair bias, not learned content attention**.
Recomputing the same logits without the bias term drops decoder 5 from
0.5929 to 0.4700 and decoder 1 from 0.5546 to 0.4883, while the trunk
numbers do not move at all — `bias_scale` is ~1e-2, so §6.2's bias is
present (verified) and numerically negligible against the content term.

**And the contact prediction never reaches the decoder.** Head 1 reads
`pair_proj(cat(h_i, h_j))`; the decoder reads `diff_pair(single, coev)`,
whose inputs are the hidden states and coevolution and nothing else. The
two pair representations are built separately from the same `h` and never
meet. AlphaFold's structure module reads the same pair representation the
distogram head does; here they are disjoint.

That is the shape of the TM/lDDT gap, stated as architecture rather than
as a training deficit: the decoder is biased by relative **sequence**
position, which is what gives it correct bonds and an lDDT of 0.2765, and
by nothing that encodes predicted **spatial** proximity, which is what a
TM of 0.0913 reflects. Local geometry is solved; global topology has no
path into the decoder.

**Not yet acted on.** Wiring the contact/distance logits into `diff_pair`
is the obvious move and it is a change to the architecture, not a fix to a
defect, so it is recorded here and not made silently. What is established
is the measurement; that a fix would help is a prediction.

A note on the first draft of this probe: it reconstructed `q @ k.T` only
and omitted the bias both block types add before the softmax. That
understated the decoder by up to 0.12 AUROC and changed nothing for the
trunk. Measuring the content term alone and calling it "the attention" is
wrong by construction when the bias is where structure would enter.

## 93 — the decoder got the contacts and the fold did not improve

Finding 92's prediction, tested and **refuted**. The experiment is
reverted; it is recorded because a negative result that cost two training
runs is worth more than the guess it replaced.

**The design.** `DiffusionPairFeatures.contact`, a readout `d_pair -> 1`
on the tensor the decoder already materialises, supervised with the
contact targets head 1 already uses. A readout and not a new input, so the
forward pass is unchanged — verified at weight 0.0 against 0.5 that the
totals differ, `dec_contact` is reported only in the second, and **no
other part differs by more than 1e-9**. The A/B isolates the gradient.

**Both arms re-run**, identical code, seed, 40 epochs, 3,069 steps,
16,384-token budget; the only difference is `--decoder-contact-weight`.
This morning's finished run was not reused as the baseline, because adding
the readout shifts the initialisation RNG and a baseline that differs in
init is not a baseline.

**The mechanism worked.** The decoder's pair representation went from not
predicting contacts at all to predicting them at **AUROC 0.611 -> 0.926 ->
0.938**, against the 0.41–0.52 finding 92 measured in the baseline
decoder's attention. The information is unambiguously in there, and
`DenoiseBlock` reads that same tensor into its attention logits.

**The fold did not move.**

| | arm A (control) | arm B (readout) | delta |
|---|---|---|---|
| mean TM, 17 targets | 0.0890 | 0.0914 | **+0.0025** |
| mean lDDT | 0.2550 | 0.2648 | +0.0097 |
| targets where B wins | — | **9 of 17** | sign test **p = 0.500** |

+0.0025 against a noise floor of **0.0023**, measured independently: arm A
scored 0.0890 where this morning's identical-settings run scored 0.0913,
and nothing differed between those two but the init RNG. Nine wins out of
seventeen is a coin flip to three decimal places.

**What it refines.** Finding 92 said the decoder has no path to the
contacts. That was true and it was not the binding constraint. Giving the
decoder's pair representation the contacts — provably, at AUROC 0.94 —
buys nothing, so the deficit is not *information*, it is the decoder's
ability to **use** it: six `DenoiseBlock` layers reading a pair bias, with
no iterative refinement of the pair track and no structure-aware update
between them. The next hypothesis has to be about capacity or about the
sampler, and it is not this one.

Reverted in full. `--decoder-contact-weight`, the readout, the loss, the
three declared scalars and the test's explicit exercise of it are all
gone: a flag that defaults to off and does nothing is the dead-parameter
class this register exists for.

## 94–95 — the venv was rebuilt and one dependency did not come back

| # | what | status |
|---|---|---|
| 94 | `gemmi` is imported by `pharos.eval.structure` and `pharos.eval.base_pairs` — the entire blind-test path — and was **not declared in `pyproject.toml`**. On 2026-10-09 16:30 the venv was rebuilt from cpython **3.13.5 to 3.14.2**; the old `lib/python3.13/` tree is gone. torch, numpy, pandas, pyarrow and scipy all came back. gemmi did not, because nothing listed it | `FIXED` |
| 95 | The pipeline driver reported `FAIL gate rc=0` **and trained anyway**, defeating the one guard whose job is to stop training against a tree whose checks fail | `FIXED` |

**94 was caught by the gate and by nothing else.** `test_metrics.py` is the
only module in the tree that reads a deposition on CPU, so it is the only
thing that touches gemmi before the GPU stages do. Had the gate not run,
the failure would have surfaced ninety minutes later at
`predict_structure`, after a full stage-5 run.

Two more facts worth keeping: `requires-python` was `>=3.10,<3.14` while
the venv is now 3.14.2, so the rebuild put the project on an interpreter
its own manifest excludes — the bound is widened rather than the venv
moved, because everything imported works on it. And **torch (39 files),
scipy (5) and matplotlib (2) are still undeclared**. torch is left that way
deliberately: a plain `"torch"` spec resolves to the CPU wheel on a machine
whose venv must carry `+cu130`, so declaring it without an index pin would
quietly break every GPU run. It is written into `pyproject.toml` as a
comment rather than left silent.

`uv pip install gemmi` was used rather than `uv add`, because **`uv add`
runs a sync and a sync prunes undeclared packages** — with torch
undeclared and a stage-5 run live on the card, that would have deleted
torch out from under a training job.

### 95, and why `if` is the wrong shape for this

```bash
if "$@" >> "$log" 2>&1; then
    say "OK"; return 0
fi
local rc=$?          # <-- 0, always
```

A compound `if` whose condition fails and which has **no `else`** returns
**0**. So `$?` after `fi` is 0, `return $rc` returns 0, and the caller's
`|| exit 1` never fires. Reproduced directly:

```
=== old shape, command exits 1 ===     === new shape, command exits 1 ===
START gate                             START gate
FAIL gate rc=0                         FAIL gate rc=1
                                         -> caller saw failure
```

The fix is to capture the status before any compound command touches it:
`"$@" >> "$log" 2>&1; local rc=$?` then branch on `$rc`.

**The production runner does not have this bug** — `gpu_cron_runner.sh`
uses `if cmd; then ...; else ...; exit 1; fi`, which is correct. This was
my scratchpad driver, and it had already mis-reported `FAIL reverify2
rc=0` the day before; I read that as cosmetic and it was not.

## 96–97 — auditing stages 2, 3 and 4

§12.1's **stage 4 is "physics — closed form, no stage"**: there is no
stage-4 training run, and that term is already independently validated
against `md_rnaions` (findings 80–81, agreement to 1 part in 10⁴). So the
audit is of stages 2, 3 and the co-trained 6.

**Both working heads check out.**

| stage | head | result | floor | verdict |
|---|---|---|---|---|
| 2 secondary structure | 4 | val acc **0.5897**, macro **0.2810**, gap **−0.0197** | majority 0.5321, chance macro 0.1250 | **lift +0.0576, macro 2.25× chance**, and the negative gap means val beat train |
| 3 chemical probing | 5 | Pearson **0.4262** over 1,813 batches | — | learning |

Measured directly off the corpus to settle the macro: the bpRNA validation
split is 1,330 rows / 173,729 characters with **all 8 symbols present**
(`.` 53.7%, `(` and `)` 22.5% each, `[`/`]` 0.63%, `{`/`}` 0.02%, `<`
0.005%), so chance is 0.1250. The run's reported `ss_val_majority` of
0.5321 against a true 0.5368 is sampling over 7 validation batches.

`probing_passes: 0` against 1,813 batches pulled is the documented
artefact, not a defect: a pass is credited only when a generator runs out,
and the longest stream finishes exactly as the loop stops.

| # | what | status |
|---|---|---|
| 96 | `ss_val_macro_recall` is reported with **no class count**, while every other head in the tree reports `*_n_class` beside its macro — `lw_n_class`, `motif_n_class`, `dist_n_class`, `base_n_class`. Chance is 1/n_class, so the metric is unreadable alone: **0.2810 is 2.25× chance over 8 classes and BELOW chance over 3**, and nothing in the log said which. `ss_step` already computes `present = cnt > 0` — the mask it averages over — so the number existed and was discarded | `FIXED` |
| 97 | Stage 6's fitness head is **worse after the second epoch than after the first**: val Spearman **+0.6044 → +0.5073**, transfer **+0.0850 → +0.0068**. ~~Not overfitting — train rho 0.4266 is *below* val 0.5073~~ — **see finding 108: that comparison is invalid and overfitting is back on the table** | `OPEN` |

96 is fixed by returning the count `ss_step` already had and declaring
`ss_n_class` / `ss_val_n_class`. Verified on synthetic input: 3 distinct
symbols → `n_class` 3, 7 → 7.

**97's most consistent explanation is the warm restart**, and I am not
claiming more than that. Epoch 1 *is* the restarted epoch — the run
resumed a checkpoint that had annealed a ~541-step cosine to 3.4e-06 into
a 3,626-step schedule evaluating 2.8e-04 at that step. The within-epoch
trajectory matches: fitness rho fell +0.6617 → +0.2475 by step 900 and
recovered only to ~+0.53 by step 1400. So 1,813 steps did not buy back
what the restart cost, and the transfer number — already the weakest thing
in the model — took the worst of it, falling 12×.

What would settle it is re-running epoch 1 from the epoch-0 checkpoint
without a schedule change, which needs the card. The schedule guard added
with finding 95's sibling now prints the before/after rate and the ratio,
so a future resume of this shape announces itself instead of being
reconstructed afterwards from three metrics drifting.

## 98 — the register contradicted itself, and one declared defect is still live

Asked whether stages 1–4 are clean, I went through every finding whose
status is not closed. Two of them were **stale**, which makes the register
itself unreliable in the direction that matters — it was overstating how
much is broken, and a register nobody trusts is the thing this file exists
to avoid.

| # | was | now | why |
|---|---|---|---|
| 75 | `OPEN` on its original row while a later row said **RESOLVED** | RESOLVED by 86 | the file contradicted itself across 370 lines |
| 83 | `OPEN` — "317 optimiser steps … structures are not connected chains" | ADDRESSED | 3,069 steps since, P–P 5.67–6.69 Å against a true 5.95 |

**And one declared defect is still live.** Finding 68, rechecked today:

```
moe.py:79    neff_over_l: Optional[torch.Tensor] = None   # (B,) MSA depth
moe.py:115   if self.neff_over_l is not None:
moe.py:116       extra[:, 0] = self.neff_over_l.to(device).float()
```

Grepping every caller in `src/` and `scripts/` for an assignment: **none**.
So `extra[:, 0]` of the router conditioning is permanently zero in every
stage — stage 1, 2/3, 5, and the block scorer alike. The router has a
declared MSA-depth input that no run has ever supplied, and the field is
the first column of the conditioning vector, so the capacity is allocated
and the signal is absent. Still `DECLARED`, not fixed: supplying it means
computing Neff/L per chain from the coevolution data, which is a corpus
change rather than a line.

### So: what is actually clean

| stage | verdict |
|---|---|
| 1 MLM | no open finding of its own; shares 68 |
| 2 secondary structure | results verified — lift +0.0576, macro 2.25× chance, negative generalisation gap |
| 3 chemical probing | results verified — Pearson 0.4262 |
| 4 physics | **no code**; the closed form reproduces an independent implementation to 1 part in 10⁴ |
| 6 fitness (co-trained) | **97 OPEN** — worse after epoch 2 than epoch 1 |
| all stages | **68 DECLARED** — one router conditioning channel is permanently zero |

## 99 — editing a shell script while a shell is running it

| # | what | status |
|---|---|---|
| 99 | The pipeline driver died **after stage 5 completed**, before prediction, with `unexpected EOF while looking for matching '"'`. The file on disk parsed clean. The cause was that I applied finding 95's fix to `run_fresh.sh` **while bash was executing it** | `FIXED` |

Bash does not read a script into memory — it reads it incrementally by
**byte offset**, re-seeking as it goes. An in-place rewrite moves every
byte after the edit, so when the running interpreter next reads, it
resumes at a position that is now the middle of a different token. Stage 5
took 6,205 s; the edit landed somewhere in the middle of that, and the
script only tripped over it when the long command finally returned and
bash went looking for the next one.

Nothing was lost — the checkpoint (10:32), the results JSON (10:33) and
the family-disjoint test evaluation all landed before the crash, because
they are the trainer's work and not the driver's. The remaining steps were
run from a **new** script rather than by repairing the old one, which is
the rule: a script a shell is currently executing is immutable until that
shell exits.

It is worth separating from finding 95, which it looks like but is not.
95 was a logic error in the driver — `if cmd; then ... fi` returning 0 and
swallowing a failed gate. 99 is the *act of fixing* 95 breaking the run it
was fixing. The fix was right; applying it live was not.

### The run it interrupted, completed

Stage 5, 40/40 epochs, 3,069 optimiser steps, zero OOMs, on the
post-revert architecture. 17 RNA-Puzzles targets:

| | this run | arm A | first run | field median |
|---|---|---|---|---|
| mean TM | **0.0936** | 0.0890 | 0.0913 | 0.3013 |
| mean lDDT | **0.2690** | 0.2550 | 0.2765 | — |
| P–P median | 5.53–6.43 Å | — | — | true 5.95 |
| beats field median | 0/17 | 0/17 | 0/17 | — |

Three runs at identical settings now read 0.0890 / 0.0913 / 0.0936 — a
spread of 0.0046, against the ~0.0023 init-RNG noise floor measured
between two of them. **Nothing has moved.** Local geometry is right and
the fold is wrong, which is where finding 93 left it and where it stays
until the decoder-capacity or sampler hypothesis is tested.

Re-verify after the run: **ALL CLAIMS REPRODUCE**, zero failures.

## Checked and not defects

Recorded so the same ground is not re-covered. Each looked like the register's
class and was not, which is the other half of this work: a probe that reports
a false positive costs exactly as much as one that misses a real defect.

| what it looked like | what it is |
|---|---|
| `w = t["weights"]` assigned and seemingly unused | it reaches the contact and distance losses as a proper weighted mean, weight in numerator AND denominator. The weights are live: 0.30/0.55/0.80/1.00 over 5.0/8.6/28.3/58.1% of chains |
| `method` is None for 78.7% of the loader's per-chain meta, and it drives `rigidity_valid` | `rigidity_valid` reads **32.9%**, matching the documented 32%: the flag was computed at build time where the method was known, and 2,345 chains carry `method=None` with `rigidity_valid=True` |
| `in_complex` comes from `m.get("has_protein")`, a `.get()` that yields 0.0 forever if the key is misnamed | `has_protein` is a real meta key; `in_complex` reads mean 0.85 with both values present |
| `coev_frac` read 0.0 on a real batch | the first six chains carry no Rfam family. Over random batches the rate is **3.99%** against 3.54% expected |
| the structure loss did not move when coordinates were shifted +7 A | a rigid translation is one the loss must ignore. It moves under scaling and per-atom noise, and the invariance is now pinned as a property |
| four chemistry dims constant, three parameters without a gradient, a `pair` key absent, `embed.mod` with no gradient | all four were my own probe's errors, not the model's -- `padding_idx=0` against all-zero ids, a key that does not exist by design, a head unreachable without its own loss, and a gate that is *supposed* to start closed |

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

## 100 — the decoder never looked at where the atoms were

| # | what | status |
|---|---|---|
| 100 | The diffusion decoder had **no mechanism by which one residue's position could influence another's update**. Coordinates entered only through `self.in_proj(x_noisy.reshape(B, L, N_ATOM * 3))`, a per-residue linear map of that residue's own nine numbers, and the attention bias came from `pair`, built once from hidden states plus relative *sequence* position. Residue i could attend to residue j because they are adjacent in sequence, and by no other route | `FIXED` |
| 100b | The fix, as first written, was **dead in production**: `GeometricBias` was fed `x_noisy`, which is `c_in * x`. Every test passed because every test called the module directly with angstroms | `FIXED` |

### 100

This is the mechanism behind 92 and 93, and it explains both results that
did not fit.

Bonds come out right because relative sequence position is sufficient for
a bond: i and i+1 are adjacent, and that is in `pair`. The fold comes out
wrong because nothing in the decoder couples residues by *proximity* —
two residues 200 apart in sequence and 6 Å apart in space are, to this
attention, exactly as unrelated as two residues 200 apart and 60 Å apart.
And feeding the decoder a predicted contact map (93) changed nothing,
because a static map is not current geometry: to decide how to move an
atom the model has to know where it is **now**, and that information was
never in the building.

The fix is `GeometricBias`: embed the pairwise C4′–C4′ distances of the
current iterate in a 16-centre radial basis over 2–40 Å and project to a
per-head additive attention bias. It is recomputed from the coordinates
on every call, so as the structure forms the attention follows it — the
iterative part of iterative refinement. `proj` is zero-initialised, so a
loaded checkpoint's forward pass is bit-identical until the bias earns
its way in. **+128 parameters** at every scale — a `Linear(16, 8,
bias=False)` — which is the whole cost.

Properties pinned by test: zero at init; symmetric; moving one C4′ by
25 Å moves the bias (1.9044); **moving one residue changes the predicted
update for the OTHERS** (4.2504, and this is the one that was absent);
and SE(3) invariant to 2.03e-06, which matters because the loss augments
with random rotations and a bias that moved under one would fight its own
training signal.

### 100b — and the same class of defect, in the fix for it

The first version was fed `x_noisy`. That is not the structure in
angstroms; it is `c_in * x` with `c_in = 1/sqrt(sigma^2 + sigma_data^2)`,
the EDM input preconditioner. At the median training sigma that divides
every distance by **18.7**:

| true gap | what the bias saw | nearest centre |
|---|---|---|
| 5.95 Å (a P–P bond) | 0.32 | `mu_0` = 2.0 |
| 23.2 Å (contact limit) | 1.24 | `mu_0` = 2.0 |
| 40 Å (basis maximum) | 2.14 | `mu_0` = 2.0 |

The entire informative range collapsed below the first centre, inside a
single basis width of 2.375. Channel 0 read 0.778 at 6 Å and 0.998 at
40 Å. Summed per-channel variation over 5.95–40 Å: **0.72, against 11.42
in angstroms — a 15.8× loss of resolving power.** It would have trained
to approximately nothing, and the A/B would have come back "no effect",
and the honest conclusion from that A/B would have been *wrong*.

Nothing caught it because the module was correct and its *caller* was
wrong, and all six tests called the module directly, with angstroms. The
register's recurring class is a component that exists, is measured, and
whose output is never checked for validity; this is its sharper form —
a component checked thoroughly **in a frame the program never uses**.

Two tests now cover the seam rather than the module. One registers a
forward pre-hook on the bias, runs `denoise()`, and asserts the tensor
that arrives is the unscaled coordinates. The other measures summed
per-channel variation and requires > 10.0, which the scaled version fails
at 0.72.

Physical units also buy the sigma behaviour for free, and it is now
pinned: at noise scale 160 the pairwise distances run off the top of the
basis and mean |bias| falls to 0.0107 against 0.5179 at scale 8, a 48×
decay. The decoder ignores geometry exactly when there is no geometry to
read, with no sigma-conditioning needed to tell it so.

`denoise()` is the only entry point into the network, and all five
callers — the training loss, the internal sampler, and
`predict_structure.py` — go through it, so the correction is universal.

**Not yet measured against the baseline.** Three runs at identical
settings read TM 0.0890 / 0.0913 / 0.0936 against a ~0.0023 noise floor;
whether this moves them is the next experiment, and it is the first
change since 90 with a mechanism that predicts it should.

## 102 — the active parameter count assumed a width the repo had already measured

| # | what | status |
|---|---|---|
| 102 | `SharedMoE.n_active_params` costed a token at `max_k // 4` experts, which for shared400's `max_k = 512` is **128**. The router audit in the same repository measured the mean nucleus width at **15.28**. An 8.4x overstatement, carried in every training log, every budget table and every tokens-per-active-parameter figure | `FIXED` |

The docstring four lines above the constant says it returns "parameters
touched by a token at the AVERAGE routing width", and the body of the
same property then used a constant fraction of a cap. Nothing ever
compared the two, although both numbers were already on disk:
`data/samples/analysis/router_audit.json` records `mean_width` 15.2750 at
step 7,000 (992.8M tokens, shared400, 512 experts, `max_k` 512), and the
gate has pinned it as claim R2 since it was measured.

| | assumed | measured |
|---|---|---|
| routing width | 128 | **15.28** |
| per-adapter parameters | 9,984 | 9,984 |
| over-count per block | — | 1,125,446 |
| over-count, 18 blocks | — | **20,258,027** |

| | reported | corrected |
|---|---|---|
| active parameters | 326,258,698 | **306,001,570** |
| tokens/active-param at 3.340B | 10.24 | **10.92** |
| at the 8e9 budget | 24.5 | **26.1** |
| Chinchilla-optimal (20/param) | 6.53B tokens | **6.12B** |

Only the shared path is affected. `mini`, `small` and `base400` route
top-k, where exactly `top_k` experts fire and the count is exact.

`typical_width` is now a config field defaulting to the measured constant,
and `test_shared_moe.py` reads `router_audit.json` and fails if the two
drift apart. The width is a measurement that has to be re-taken when the
router changes, not a number anyone may type.

Nothing in the gate or the tests pinned the old figure; it appears only in
historical run records, which are left as they were written.

## 103 — two thirds of the trunk cannot see to its right

| # | what | status |
|---|---|---|
| 103 | 12 of the trunk's 18 blocks are `GatedDeltaNet`, which is **strictly causal**. PHAROS is an encoder, not a language model, and nothing in the task calls for a causal mask | `FIX BUILT, OFF BY DEFAULT` — the fact is measured, the consequence is not |

`BLOCK_PATTERN` is `(gdn, gdn, swa, gdn, gdn, swa, gdn, full)` tiled to 18,
giving `G G S G G S G F G G S G G S G F G G`: twelve `gdn`, four `swa`
banded at ±128, and two `full` at indices **7 and 15**.

Measured directly — scramble the right half of the input and watch the
left half of the output:

| mixer | left-half change |
|---|---|
| `gdn` | **0.000e+00** |
| `swa` (w=8) | 2.012e-01 |
| `full` | 1.445e-01 |

Exactly zero, as the `tril` masks and the chunk-ordered state recurrence
require. So a nucleotide can be paired with a partner more than 128 nt to
its **right** only through the two `full` blocks — and those are the two
blocks finding 92 measured at chance against the contact map (AUROC
0.44–0.52).

**Why this is DECLARED and not a defect.** The contact head reaches
AP 0.885, so the information plainly arrives. Two routes explain that
without any help from `gdn`: the pair features are an outer sum
`A s_i + B s_j`, which sees both endpoints whatever the attention did, and
the trunk is run three times with the previous pass recycled in, so
right-context gathered by `swa`/`full` in loop 0 reaches left positions in
loop 1. The causal mask is an unforced restriction inherited from
language-model architectures, and removing it is a strict increase in
capacity — but "unforced" is not "harmful", and this register does not
promote a hypothesis to a defect without an A/B. That experiment is
queued behind finding 100's.

## 104 — the OOM guard covered training and not validation

| # | what | status |
|---|---|---|
| 104 | `evaluate()` had no `OutOfMemoryError` handler. The training loop beside it skips an OOM batch and carries on; validation did not, so the same unlucky batch that costs one gradient during training **killed the run** | `FIXED` |
| 104b | `GeometricBias` expanded its radial basis in fp32 inside a bf16 autocast region, because `self.proj.weight.dtype` is fp32 there — autocast casts at the operator, not on the parameter. Twice the activation memory it needed | `FIXED` |

The finding-100 A/B died at **epoch 25 of 40**, 2h in, with

```
File "scripts/train_pharos.py", line 1142, in evaluate
File "src/pharos/model/diffusion.py", line 544, in forward
torch.OutOfMemoryError: Tried to allocate 6.63 GiB
```

after **25 validations had already succeeded**, and in a run whose
training loop had caught and skipped five OOMs. The comment on that
handler states the reason it exists — "chain length here runs to 4,298
and the batches are packed to a token budget, so one long-chain batch can
cost several times the median. Skipping it costs one gradient; dying
costs the epoch" — and every word of it applies equally to the
evaluation that runs after each epoch. The guard existed, was exercised,
and covered one of the two places that needed it.

The whole batch body of `evaluate` is now inside the handler, which
counts skips and reports `eval_oom_batches` in the returned metrics
rather than silently averaging over fewer batches.

The test is structural, because the defect is: it parses
`scripts/train_pharos.py`, finds the batch loop in `evaluate`, and
asserts both that a `torch.OutOfMemoryError` handler is there and that
**nothing in the loop body sits outside it** — 16 statements inside, 0
after. No amount of running short evaluations on small batches would have
found this.

### 104b — and the memory that provoked it

`GeometricBias` is pure activation memory at `(B, h, L, L)`. It asked
`self.proj.weight.dtype` for its working width, and inside a bf16
autocast region that answer is **fp32**, so both the
`(B, rows, L, n_rbf)` basis and the output were double width. It now asks
`torch.is_autocast_enabled` / `get_autocast_dtype` instead, which is the
only thing that actually knows.

Measured on the path that died — `eval()`, `no_grad`, `L = 1024`:

| batch | fp32 basis | bf16 basis | saved |
|---|---|---|---|
| B = 10 | 0.633 GiB | 0.321 GiB | **1.97x** |
| B = 16 | 1.008 GiB | 0.508 GiB | **1.98x** |

Distances are still computed in fp32 and cast afterwards: `cdist` in bf16
on coordinates reaching hundreds of angstroms loses low bits that matter,
while a *distance* in bf16 is good to about 0.1 Å against a basis width
of 2.375.

Taken together with finding 100's earlier 10.89 → 2.03 GiB chunking, the
decoder's geometric bias now costs about a twentieth of what the first
working version did, for a bit-identical result.

## 105 — the watch said "still taking them" about a run that had been dead for four hours

| # | what | status |
|---|---|---|
| 105 | `check_oom` decided whether a run was *still* taking OOMs by comparing the last two **rows of a csv**. That is a property of the file, not of the clock, so once a run ended on a rising count the test stayed true forever | `FIXED` |

The report at 16:13 read:

```
**1 CRIT, 0 WARN, 7 INFO**
## CRIT
- oom — stage5_3d/20261010T054625Z has taken 46 OOMs and is still taking them
## INFO
- liveness — no trainer process is running
- run — stage5_3d/20261010T054625Z: 109 rows, last written 231 min ago
```

A CRIT contradicted by two INFO lines in its own report. The check's
comment states the intent exactly — *"Only while it is still happening. A
finished run's OOM count is history, and re-reporting it every half hour
forever is noise"* — and the implementation never expressed it. Same
class as the rest of this register, in the monitor instead of the model,
and the monitor is where it costs most: 22 of this script's first 23
warnings were false, and a CRIT that is wrong teaches its reader to skip
the one that is right.

`fresh` is now `rising AND live`, where `live` means the run's telemetry
moved within `LIVE_MIN = 30` minutes — several missed rows at stage 5's
~150 s per epoch. A run that ended on a rising count now reads
`INFO ... and ended on a rising count (247 min ago)`, which is what was
true. The ALERT cleared on the next pass.

**The 46 OOMs themselves were real** and are finding 104: that run took
them because `GeometricBias` was double-width, and it then died of one.

`scripts/test_training_watch.py` is new — the watch had **no test module
at all**, which for a script with that false-positive record is its own
finding. Seven assertions, each on a *severity* rather than a message:
live+rising+large is CRIT, the same rows from a dead run are INFO, a flat
count is INFO, zero OOMs is silence, and live+rising+small is WARN.

## 106 — the physics bias is wired, trains, and the model has switched it off

| # | what | status |
|---|---|---|
| 106 | The screened-Coulomb pair bias reaches the attention logits at a magnitude of **2.6e-04**. It is not broken and not frozen; the model has driven it to numerically nothing | `MEASURED` — not a defect |

Every gate on the physics path is open, and all of them are tiny:

| parameter | trained value |
|---|---|
| `elec.log_scale` | +0.00738, so amplitude `exp(·)` = **1.0074** |
| `elec.site_scale` | −0.00335 |
| `trunk.blocks.{7,15}.mixer.bias_scale` | mean 0.0088, max 0.0269 |
| `B_elec` itself (documented range) | −0.0097 to −0.0064 |

The product is what reaches a logit:

| | contribution |
|---|---|
| low salt, largest head | **2.63e-04** |
| high salt, largest head | 1.73e-04 |

A logit perturbation of 2.6e-04 multiplies an attention weight by
`exp(2.6e-04) = 1.00026`. The physics module changes where the model
attends by **0.03%**.

### Why "rescale it" is the wrong fix, and would have been a false positive

The tempting reading is a scale handicap: `B_elec` is O(1e-2) while
other gated features are O(1), so `bias_scale`'s gradient is ~100x
smaller and the gate cannot open. Normalise `B_elec` to unit scale and
let it compete fairly.

That argument is wrong, and checking it is the point of writing it down.
Stage 5 optimises with **AdamW** (`train_pharos.py:1434`), whose update
is `lr · m̂/(√v̂ + ε)` — **invariant to a uniform rescaling of the
gradient**. Multiplying `B_elec` by 100 would leave the step size on
`bias_scale` unchanged and AdamW would simply settle at a `bias_scale`
100x smaller, for an identical product. The fix is a no-op.

Nor is weight decay suppressing it: decoupled decay pulls `bias_scale`
by `lr·wd = 2e-4 × 0.01 = 2e-6` per step, a factor of 0.994 over all
3,069 steps.

What is left is the honest reading: given a scale-invariant optimiser,
3,069 steps and a gradient it demonstrably receives, the model put the
physics bias at 2.6e-04 of a logit **because that is where it wanted
it**. The docstring's claim that `bias_scale` "reaches 1e-2 within 200
steps" is true and was read as the gate opening; the longer measurement
is that it reaches 1e-2 and then **stops**, for the remaining 2,800.

This does not say the physics is wrong — `manning.py` is validated
against `md_rnaions` to 1.2e-4 on the Bjerrum length. It says the
screened-Coulomb term, as delivered through two full-attention blocks of
eighteen, is not something this model finds useful at this scale of
training. Recorded so the next person does not rediscover the tempting
fix and ship it.

### 103, continued — the fix, built and not yet enabled

`GatedDeltaNet(bidirectional=True)` runs the same scan over the reversed
sequence and adds it back through a per-head gate. The projections are
**shared** between the two directions, because a direction is a reading
order and not a different feature set, so the entire cost is `n_heads`
parameters per `gdn` block — **+144** at shared400, taking the total from
395,295,754 to 395,295,898. The output gate and output projection apply
once, to the combined signal.

Four properties, all pinned in `test_attention.py`:

| property | measured |
|---|---|
| bit-identical at its zero init | `0.000e+00` |
| genuinely bidirectional once open | left-half change `1.21e-01` against exactly 0 |
| exactly padding-safe | `0.000e+00` padded vs unpadded |
| gate takes gradient at closed init | `5.52` |

Padding needs no special handling: the scan already multiplies `b` by the
mask so a pad writes nothing and forces `a` to 1 so a pad forgets
nothing, which makes right-padding-reversed-to-left-padding a no-op.

`bidirectional_gdn` defaults to **False** on `PharosConfig` and
`TrunkConfig`. It is off deliberately: finding 100's A/B is mid-flight at
epoch 24 of 40, and enabling a second architecture change now would
confound it. The A/B for this one is queued behind it, and the honest
position remains the one above — an encoder has no use for a causal mask,
but *unforced* is not *harmful* until measured.

## 107 — the contact head's headline number is measured on a 45%-positive set

| # | what | status |
|---|---|---|
| 107 | `contact_ap = 0.8901` is computed on pairs from `sample_pairs`, which takes **every** positive plus at most 512 sampled negatives. That set is **45.4% positive**. The task — every pair with \|i−j\| ≥ 4 — is **9.6% positive** on this corpus. Average precision is strongly base-rate dependent, so the two numbers are not the same quantity | `FIXED` (the real metric is now computed) |

This matters more than a mis-stated metric usually would, because
**0.885–0.890 is the number the project's central diagnosis rests on**:
finding 92 is titled *"the model predicts contacts well and never uses
them to build"*, and the gap between that and TM 0.094 is what sent the
search into the decoder. If the head is not in fact predicting contacts
well on the real task, the diagnosis needs revisiting.

The sampled block's own comment says *"the positives are a few percent
of sampled pairs"*. They are 45.4%. The code and its description had
drifted, in the direction that flatters.

### What the floor actually is

Measured on 200 test chains (mean length 80), all pairs with \|i−j\| ≥ 4:

| predictor | mean AP |
|---|---|
| base rate (chance) | 0.0961 |
| −\|i−j\| — sequence separation only | 0.1209 |
| `deg_i · deg_j` — how many partners each residue has, not **who** | **0.3005** |
| `deg_i·deg_j/(1+\|i−j\|)` — both, still partner-blind | 0.2354 |

This was built to test a hypothesis — that the head scores well by
predicting *which residues are paired* without knowing *with whom*,
which an outer-sum pair feature `A s_i + B s_j` can do and which would
have explained AP 0.89 beside TM 0.09 neatly. **The hypothesis is
refuted.** A partner-blind oracle, given the true degrees, reaches only
0.30. The head's 0.89 is far above that, so it does carry genuine
pairing-partner information. Recorded because a refuted hypothesis costs
the same to test as a confirmed one and is worth exactly as much.

### The fix

`evaluate()` now computes the same head's score on **every** pair with
\|i−j\| ≥ 4 for chains up to `FULL_PAIR_MAX_L = 320`, through a closure
that shares the sampled path's feature construction exactly — pair
projection, coevolution lookup and motif mixture — so the two cannot
drift by building features differently. It reports `contact_ap_full`,
`contact_base_rate_full`, `contact_ap_lift_full` and `contact_n_full`
beside the sampled four. Bounded by length rather than sampled again, so
it is an exact measurement of a real subset instead of a second
estimate; 320 is 50,560 pairs and the test split's mean length is 80.

**The number itself is not yet known** — it needs a GPU, and the card is
occupied. `contact_ap_full` on the existing checkpoint is the first thing
to read when one frees, and it is the number that decides whether
finding 92 still says what it says.

## 108 — finding 97 ruled out overfitting by comparing two different statistics

| # | what | status |
|---|---|---|
| 108 | Finding 97 rejected overfitting because "train rho 0.4266 is *below* val 0.5073". Those are not the same quantity: `fitness_spearman` is a running mean of **per-batch** Spearman over batches that **mix assays**, while `fitness_val_spearman` is **per-assay**, then averaged over assays | `FIXED` — a comparable train number is now computed |

`load_fitness`'s own docstring states the problem exactly, two hundred
lines above the comparison:

> the metric is a WITHIN-ASSAY Spearman: the DMS score of a tRNA assay
> and of a ribozyme assay are different quantities measured on different
> instruments, and a correlation computed across a mixed batch would
> mostly measure which assay a sequence came from. `target` is already
> rank-normalised within its assay by the builder, so the regression
> loss is comparable across a mixed batch **even though the correlation
> is not**.

Three things make the training number smaller regardless of fit:
targets are rank-normalised *within* each assay, so cross-assay
variation in a mixed batch is pure noise; the batch is small, so the
correlation is attenuated by restricted range; and NaN batches are
dropped. A badly overfit model would also show train-below-val here.
**The comparison is no evidence either way, so finding 97's stated reason
for excluding overfitting does not hold.**

And the shape of the data points the other way. Over epochs 1→2:

| split | epoch 1 | epoch 2 | what it holds out |
|---|---|---|---|
| val | +0.6044 | +0.5073 | unseen **variants** of seen constructs |
| transfer | +0.0850 | **+0.0068** | an entire unseen **construct family** |

Val falls by 16% and transfer by **92%**. Degrading fastest on the split
that shares least with training is the signature of over-specialisation,
which is what finding 97 ruled out.

### The fix

`score_fitness(model, "train", ...)` now runs beside val and transfer,
scored identically — per assay, then averaged — and is reported as
`fitness_train_split_spearman`, with `fitness_overfit_gap` = train − val
as a number a reader may actually subtract. `train.parquet` was already
on disk; nothing needed building. The existing `fitness_spearman` stays,
with a comment at both ends saying what it is and why it is not the one
to compare.

This does not resolve 97 — it removes a wrong reason for an answer, and
supplies the instrument that can give a right one on the next stage-6
run. 97 stays `OPEN`.
