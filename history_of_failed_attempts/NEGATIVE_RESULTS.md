# Negative results — tried, measured, worse

Not defects. These are designs that were implemented, measured against the
alternative, and rejected on the evidence. They are here so the same afternoon
is not spent twice, and because a design choice is only justified when the
alternative was actually tried.

---

## Fixed expert capacity in the MoE — 25% slower

**The idea.** `cap = int(counts.max())` in the grouped-expert permutation is a
device synchronisation, and under `torch.compile` a graph break — 32 times a
step at 16 blocks and 2 loops. The textbook fix is a fixed capacity
`ceil(f · M / E)` with the overflow dropped, which needs nothing read back from
the device.

**Measured**, compiled, 48×512:

| capacity factor | s/step | drop fraction |
|---|---|---|
| exact (`counts.max()`) | **0.289** | 0% |
| 1.25 | 0.377 | 0.00% |
| 1.5 | 0.364 | 0.00% |
| 2.0 | 0.389 | 0.00% |

**Why it loses.** Selecting the survivors is itself data-dependent —
`e_sorted[keep]`, `row[order][keep]`, `weight[order][keep]` are three boolean
gathers with their own synchronisations and allocations. It trades one sync for
several. And there was nothing to win: the drop fraction was 0.00% at every
factor, because the load-balance loss keeps the groups near-uniform and
`counts.max()` is already close to `M / E`.

**Kept:** the exact form. No token is ever dropped, and the graph break is
cheaper than the cure.

---

## Admitting the L1 block diagonal without retraining — worse at the shipped budget

**The idea.** The pair track's cascade ceilings at 0.811 rather than 1.0 because
18.9% of true L2 blocks sit inside a single L1 block: `valid_mask` clamps
separation to at least one block, which is 4 residues at L2 and **16** at L1, so
contacts 4-15 residues apart have no valid parent at any budget. Admitting the
diagonal is the obvious fix.

**Measured**, same weights, same budgets:

| keep_frac_l1 | diagonal excluded | diagonal admitted |
|---|---|---|
| 0.12 (shipped) | **0.264** | 0.247 |
| 0.25 | 0.367 | **0.396** |

**Why it loses at the tight budget.** The scorer was trained with the diagonal
masked out, so it has never been asked to score a diagonal block; its scores
there are arbitrary and displace off-diagonal blocks it ranks well. At a looser
budget the extra reachability wins.

**Conclusion:** the fix is a mask change **plus a retrain**, not a switch. The
mask is now a config option defaulting to the current behaviour, so the existing
measurements stay reproducible.

---

## A larger block scorer — no better held out, and much worse when the step count fell

**The idea.** R1's selector is the pair track's load-bearing assumption. More
capacity should help.

**Measured**, matched optimiser steps:

| parameters | val L2 recall | test L2 recall |
|---|---|---|
| 4.4M | 0.871 | **0.840** |
| 61.9M | **0.890** | 0.835 |
| 153M, 14× batch | 0.666 | — |

The 61.9M model is better on validation and *not* better on the held-out test
set — a tie with a hint of overfitting. The 153M attempt was much worse, and for
a reason unrelated to capacity: the larger batch cut optimiser steps from
~11,700 to 2,448, and raising the learning rate to 1e-3 did not rescue it. That
is a step-count failure, not a capacity ceiling.

**Conclusion:** direct evidence that the structure task is **data-limited, not
capacity-limited**, and the reason PHAROS-Small is the default.

---

## Flat top-K contact proposal — the reason the pair track exists

A flat top-K proposer keeping K = 32L recovers **0.200** of true contacts on
500-1,200 nt chains, where a *random* scorer gets **0.746** on the same set.
Ranking across all L² pairs is dominated by the diagonal. A banded fallback does
not rescue it either: at 1,500-3,000 nt even ±512 misses ~15%.

**Conclusion:** contacts must be selected as blocks, not pairs. This is §7.1 of
the specification.

---

## A 90/5/5 family-disjoint split — does not exist

**The idea.** Standard practice: hold out 5% by family.

**Why it is impossible here.** Four rRNA families are **84.4%** of all
structural residues. A 5% quota is 658,650 residues and the *smallest* of the
four is 2.4× an entire held-out bucket. Keeping a family whole means placing it
somewhere, and wherever it goes the target fractions are gone.

**Kept instead:** family-disjoint on families small enough to hold out, with the
four dominant families forced into training and measured by a separate
**entry-disjoint** `test_ribosomal` set that is never averaged with the
family-disjoint test set.

---

## Training on CPU while waiting for the GPU

Considered and rejected on request, and correctly: the trunk at d=512 over a
25M-sequence corpus is not a CPU workload, and a number produced that way would
not have been comparable to anything else measured here.
