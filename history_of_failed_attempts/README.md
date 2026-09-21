# History of failed attempts

Everything that did not work, kept on purpose.

`research/architecture/ARCHITECTURE.md` states the architecture as it is. It
says nothing about how it got there, and it should not — a specification that
narrates its own corrections is hard to read and easy to misread. This folder
is where that material lives instead, in full, because the record of what was
wrong is the more useful half when the question is *why is it like this*.

Three kinds of thing are here.

## 1. Defects — [`DEFECTS.md`](DEFECTS.md)

Things that were **wrong**, found, and fixed: measurement errors, silent data
corruption, unimplemented behaviour that documentation claimed, assumptions
that were never checked. Each entry records what the symptom was, what the
cause turned out to be, and how it was caught — the last being the part worth
reading, because most of these were found by a guard that someone wrote after
being burned by a similar one.

## 2. Negative results — [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md)

Things that were **tried and measured and were worse**. These are not defects.
The obvious optimisation that turns out to be slower, the larger model that
generalises no better, the mask fix that needs a retrain before it helps. They
are recorded so nobody spends an afternoon rediscovering them, and because a
design decision is only justified if the alternative was actually measured.

## 3. Superseded documents

| file | what it is |
|---|---|
| [`ARCHITECTURE_v0.1_with_corrections.md`](ARCHITECTURE_v0.1_with_corrections.md) | the original specification, carrying its correction tables — published value beside canonical value, for 28 defects across 11 audit cycles |
| [`ARCHITECTURE_v0.2_revision_trail.md`](ARCHITECTURE_v0.2_revision_trail.md) | the revision narrative: what v0.1 claimed, what re-measurement found, and what changed as a result |

**These two are still under `verify_claims.py`.** The guards that protect them
are not ceremony: a global search-and-replace once overwrote the *published*
column of a correction table with the canonical values, so a row that recorded
`98.96 -> 98.95` came to read `98.95 -> 98.95` and the correction vanished. The
checks now require **both** columns of every row, and require the retracted
phrasings to stay absent. Flattening the record of an error is a way of
committing it again.

## Why keep any of this

Three reasons, all of them practical.

**It stops the same mistake twice.** Fixed expert capacity is the textbook fix
for the device synchronisation in the MoE. It is also 25% slower here, for
reasons specific to this implementation, and the measurement is in
`NEGATIVE_RESULTS.md` with the numbers.

**It justifies the design.** "PHAROS-Small is the default because the task is
data-limited" is an assertion. "A 14× larger block scorer was no better on
held-out data, and a 153M one was much worse for step-count reasons" is
evidence, and it is here.

**It calibrates the numbers that remain.** Several figures in the current
specification are corrections of earlier figures that were confidently wrong —
92.65% ribosomal became 85.94%, a 1.76 σ coupling became 1.523 σ on a properly
restricted population. Knowing which numbers moved, and by how much, is part of
knowing how much to trust the ones that have not been re-measured yet.
