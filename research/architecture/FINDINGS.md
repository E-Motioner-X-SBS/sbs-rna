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
| 75 | with the floor computed on the SAME batch, `dist_acc` tracks `dist_major` to the digit at every step -- 0.4830/0.4830, 0.4690/0.4690, 0.2900/0.2900 -- and `dist_macro` sits at 0.026-0.031 against 1/39 = 0.0256. Head 2 predicts **one bin and nothing else**. It had been reporting 0.47 accuracy for the whole of stage 5 and reading as a working head | `OPEN` |

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
| 83 | stage 5 runs **317 optimiser steps over 8 epochs** — 40 per epoch, 1,062 s in total. A denoising diffusion decoder is being asked to learn a generative model of RNA backbone geometry in three hundred steps. At inference the consequence is unambiguous: consecutive P–P distances come out at **16–19 Å against a 5.88 Å target**, bond violation 0.79–0.93, and the structures are not connected chains | `OPEN` |
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
