# Defects — found, and fixed

Things that were wrong. Each entry: the symptom, the cause, and how it was
caught. The last column matters most; nearly every one of these was found by a
guard written after a similar failure, not by inspection.

The v0.1 audit found **28 defects across 11 cycles**; those are tabulated with
their published-versus-canonical values in
[`ARCHITECTURE_v0.1_with_corrections.md`](ARCHITECTURE_v0.1_with_corrections.md).
What follows is everything found since, in the order it was found.

---

## Data and parsing

### C15 — the entry counter read 3,254 entries as empty

**Symptom.** Entry-level composition statistics were computed over far fewer
entries than the archive contains, and nothing complained.

**Cause.** mmCIF carries two chain identifiers. `_entity_poly` declares
entities against `auth_asym_id`; the counter keyed on `label_asym_id`. Where
the two disagree — 3,254 entries — the join produced nothing and the entry was
silently counted as holding no RNA.

**Caught by** requiring two independent code paths, chain geometry and entry
composition, to agree on every entry-level claim. They did not.

### C16 — NMR ensembles stacked 20 models into one residue

**Symptom.** 1ARJ reported 424 atoms per residue.

**Cause.** No filter on `pdbx_PDB_model_num`, so every deposited model of an NMR
ensemble was read as additional atoms of the same residue.

**Caught by** a per-residue atom-count sanity bound. A nucleotide has ~20 heavy
atoms; 424 is not a subtle error, but nothing had been looking.

### The pretraining corpus was a length band, not a sample

**Symptom.** None visible. The corpus held 10M sequences, which sounds like
plenty drawn from 1.32 billion.

**Cause.** elDORS ships as twenty chunks and **is sorted by length**. The
starter took chunks 001-008. Those average 850 nt at GC 0.474; the chunks it
skipped average 233 nt at GC 0.572. Mean length differed by 3.7×, GC by ten
points. Stage 1 had seen the long, AT-rich end and none of the short, GC-rich
majority — and the `20 ≤ L ≤ 1024` filter then discarded much of even that,
since the first three chunks have a p90 above 2,700.

**Caught by** asking whether the chunks were interchangeable instead of assuming
a chunk index was a byte offset. Fixed by rebuilding across all twenty (25M
sequences, median 522 → 261) and shuffling shard order, since the shards are
named after their chunk and reading them sorted walks the corpus
longest-to-shortest.

### The MLM chemistry leak

**Symptom.** Masked-token accuracy 0.948 and loss 0.392 nats = 0.57 bits at step
100 — *below* the 2.0165 bits/nt entropy of the corpus. A model scoring below
the entropy of its own data has not learned anything.

**Cause.** Chemistry dims 0-4 are a one-hot of base identity, and the chemistry
was computed from the *original* sequence while only the token stream was
masked. The answer was sitting in another channel.

**Caught by** the number being too good. Fixed by deriving chemistry from the
masked input, and guarded by a test that constructs the leak deliberately and
asserts the safe path does not reproduce it.

---

## Measurement and reporting

### MFU was never measured

**Symptom.** Every A100-hour figure in the architecture descended from
`MFU = 0.35`, written into four scripts as "a realistic MoE figure".

**Cause.** Nobody had run the model and checked. Measured, it was **3.6%** —
roughly 10× optimistic. A profile put only ~18% of GPU time in tensor-core
GEMMs.

**Caught by** measuring it. Now reported every logging interval by the trainer
itself, so the cost model reads an observation rather than an assumption.

### The FLOP model overcharged training by 1.8×

**Symptom.** Cost estimates disagreed with what the trainer implied.

**Cause.** `train_flops` charged every refinement loop a full forward and
backward, `loops × 6N`. The trunk runs loops 1..N-1 under `no_grad` — that is
what the one-step gradient means — so an intermediate loop costs 2N. The correct
figure is `6N + 2N(loops-1)`: 10N at three loops, not 18N.

**Caught by** making the trainer and the cost model use one convention, at which
point they could no longer disagree silently.

### The reported loss was not the loss

**Symptom.** Stage 1's curve flattened at ~1.45 with accuracy stuck near 0.39,
apparently above the corpus entropy.

**Cause.** The reported number was cross-entropy **plus the MoE balance term**.
The Switch load-balance loss is `n_experts · Σ(frac · pbar)`, which is **1.0 at
perfect uniformity, not 0** — so at weight 0.01 across 16 blocks it contributes
a floor of ~0.16 nats that never goes away. Dividing the sum by ln 2 inflated
bits/token by ~0.23 and made a model that had gone *below* corpus entropy look
stuck above it.

**Caught by** the floor being suspiciously constant. CE and the balance term are
now tracked separately and bits/token comes from CE alone.

### The throughput number averaged in the warm-up

**Symptom.** Reported tok/s climbed for hundreds of steps and never settled.

**Cause.** The rate was cumulative since process start, so it folded in
`torch.compile` warm-up across every quantised width.

**Caught by** the number still rising at step 650. Now reported per interval.

---

## Behaviour that documentation claimed and code did not implement

### Sliding-window attention was never banded

The docstring described a chunked banded implementation for `L > 2·window`, "so
the cost really is linear in L". There has never been one in the code: it is a
dense masked attention at every length. The result is identical; the cost is
O(L²) with a materialised `(B, 1, L, L)` mask. The docstring now says so.

### The curriculum did not chain

§12.1 describes stage 5 as fine-tuning what stages 1-3 build. Only
`train_sequence_stages.py` accepted `--init-from`; the others each started from
random weights. Stage 5 run that way produced r = 0.049 on unseen folds, which
was read as a weak result rather than as an unchained run.

### Three of four trainers could not resume

Stage 1 resumed. The other three saved at **epoch boundaries, without optimiser
state**, and nothing read it back — and because the runner passes
`--init-from <previous stage>` on every fire, an interrupted stage did not merely
lose its epoch, it restarted from the previous stage's weights and discarded
everything it had done. An epoch of stages 2-3 is ~10,500 steps.

This is the defect stage 1's own docstring names — *"an unattended pretraining
script that restarts from zero is not unattended, it is a loop that makes no
progress"* — left in place in three other scripts.

### Two of four trainers had no learning-rate schedule

Constant rate from first step to last: no warm-up, no decay. The completeness
audit listed `train/schedule.py` under "deliberately not built — settled as a
decision (D2)", which conflated D2's *token budget* with the *learning rate*.
The audit's own note is what made the absence look intentional.

### Telemetry was written once, at the end

Every trainer accumulated history in a list and serialised it when the run
completed. On a shared card runs do not complete: stage 1 reached 188.6M tokens
across several fires and produced **no history file at all**.

---

## Throughput

### 42.9% of every stage-1 step was padding

The packer appended sequences in corpus order and cut on the token budget, so a
20-nt sequence and a 1,024-nt one shared a batch and both padded to 1,024. It
was never a token budget; it was a padded-token budget.

### The chemistry was computed twice per step and used once

`encode_batch` built the 24-dim vector from the original sequence;
the MLM path rebuilt it from the masked tokens — it has to, see the leak above
— and discarded the first. Both were Python loops between two GPU kernels: 273
ms for a 26×950 batch, so 545 ms of a ~1.26 s step.

### The span mask synchronised the device once per sequence

It took GPU tensors and read `int(mask[b].sum())` inside the batch loop. The row
lengths were already known from packing.

---

## Operational

### Peak memory was modelled as one constant per token

The auto-budget fitted 2.01 GiB per 1,000 tokens from three ~350-step
benchmarks and predicted a 72.0 GiB peak at budget 35,840. The run reached
**79.14 GiB and OOMed** at 72.4M tokens.

Peak depends on the batch **shape**, not its token count: SWA and FULL
materialise a `(B, 1, L, L)` mask, so at 24,576 tokens a 24×1024 batch costs
61.2 GiB where 192×128 costs 39.2. Short benchmarks never draw the worst pool in
the corpus; a 2,300-step run does. The constant is now the worst observed cost
plus margin, and — more to the point — the trainer survives an OOM rather than
dying of one.

### `status.json` reported the last thing that went right

A killed runner left the file saying `running` indefinitely, because nothing
wrote a status on the way out. Checking the pipeline reported a stage that had
not been alive for twenty minutes. A signal trap now writes `interrupted`.

### `--restart` was unrecoverable, and took 70.8M tokens with it

The flag overwrote the checkpoint in place, silently. Smoke-testing the OOM
handler with it against the default path destroyed 70.8M tokens of stage 1. The
flag did exactly what it documented; what it documented was unrecoverable. It
now renames, and `--ckpt` lets a test write elsewhere.

### A test with `--epochs 0` clobbered eight epochs of pinned results

Trainers wrote their report unconditionally, so a run that trained nothing
overwrote the last real one with an empty history and turned four pinned checks
red. All three now refuse to overwrite a results file with an empty one.

### Indexing a growing history by position

`verify_claims.py` pinned a router measurement as `history[0]`. When a second
probe landed, `[0]` was a different measurement and four checks drifted. Pins
are now keyed by token count.
