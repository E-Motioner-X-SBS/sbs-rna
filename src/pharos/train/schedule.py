#!/usr/bin/env python3
"""Learning-rate schedules, in one place, with the measurement behind each.

Four trainers in this tree set a learning rate and three of them wrote their
own rule. `pretrain_mlm.lr_at` keys off TOKENS against a budget,
`train_sequence_stages.lr_at` keys off STEPS against an estimate, and stage 5
and the block scorer run `OneCycleLR`. Three implementations of "warm up then
decay" drift, which is the defect `ElectrostaticBias` was fixed for in the
physics and `PharosHeads.loss` in the losses.

WHY COSINE IS THE WRONG SHAPE FOR THIS PROJECT
----------------------------------------------
Not on quality grounds. On a structural one, and it has already cost a run.

A cosine's value at step *t* is a function of the TOTAL, and in this setup
the total is neither known at the start nor stable between runs:

* training is preemptible — `gpu_cron_runner.sh` resumes whenever the shared
  A100 frees, and stage 1's own history shows the learning rate climbing
  from 4.28e-04 back to 5.33e-04 around step 12,700, which is a restart;
* `estimate_steps` derives the total from parquet row counts and a CSV, so
  it moves when the corpus does;
* `--epochs` is a flag, and it changed from 1 to 2 between two runs of
  stage 2/3.

The consequence, measured: a checkpoint that had annealed a ~541-step cosine
to 3.4e-06 was resumed with `--epochs 2`, which estimates 3,626 steps, and
`lr_at(541, 3626, peak)` returned 2.8e-04. An 82x jump onto a model sitting
in a sharp minimum. Over the next 500 steps `ss_accuracy` went 0.5871 ->
0.5588, `probing_pearson` 0.4619 -> 0.3997 and `fitness_spearman` +0.6617 ->
+0.3337, and nothing in the log said a restart had happened.

WHY WSD IS THE RIGHT ONE
------------------------
Warmup -> Stable -> Decay. During the stable phase the learning rate is
`peak`, **a constant that does not depend on the total at all**, so a resume
with a different total is a no-op over the bulk of training rather than a
warm restart. The schedule stops being a function of a number this setup
cannot pin down.

It also fits what the loss surface here actually looks like, measured on
stage 1's 252 logged points over 25,250 steps:

* **the signal-to-noise is terrible.** Batch-to-batch residual sd is
  0.090-0.117 nats against a drift of 0.0013-0.0041 nats per 100 steps —
  a noise-to-signal ratio of **22x to 89x** depending on the window. A
  schedule shaped finely against the loss is shaping against noise, and
  anything loss-reactive (ReduceLROnPlateau) would be driven by it.
  A constant plus a terminal anneal is as good as any curve and far more
  robust.
* **the spikes are a high-LR phenomenon.** Four jumps above 4 sd of the
  step-to-step difference, at steps 3,700 / 7,200 / 8,200 / 9,300, every one
  of them at lr >= 4.6e-04 (up to 5.83e-04), and none after the rate fell
  below that. A stable phase set under the spike threshold removes them;
  a cosine spends its early-middle above it by construction.
* **the anneal is cheap and can be run whenever you like.** WSD lets the
  stable-phase checkpoint be kept and decayed from for ANY budget, which is
  exactly what "train until somebody else needs the card" wants.

The decay shape is `1 - sqrt(progress)`, which is what Hagele et al. (2024),
*Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*,
measured as the best of the shapes they compared. It is offered beside
linear and cosine rather than asserted, because this project has not
measured it on this corpus.
"""
from __future__ import annotations

import math

SHAPES = ("1-sqrt", "linear", "cosine")


def warmup_stable_decay(step: float, total: float, peak: float, *,
                        warmup_steps: int = 200,
                        decay_frac: float = 0.2,
                        floor_frac: float = 0.0,
                        shape: str = "1-sqrt") -> float:
    """Warmup -> Stable -> Decay. Clamps past `total` rather than raising.

    `warmup_steps` is an ABSOLUTE COUNT, not a fraction, and that is not a
    detail. A fractional warmup depends on the total exactly the way the
    cosine does: at `warmup_frac = 0.01` a 2,000-step run warms for 20 steps
    and a 50,000-step run warms for 500, so the same step in the same run
    gets a different rate the moment the total is re-estimated. The first
    draft of this function had it, and the test that asserts the stable
    phase is total-independent caught it at eleven step/total pairs. Warmup
    is a property of the optimiser's state, not of how long you intend to
    train.

    `floor_frac = 0.0` decays to zero, which is what the WSD papers do and
    what the cosine here never did: `lr_at`'s floor is 0.1, so stage 1 has
    never annealed below 4.0e-05 and has spent its whole life at a rate that
    still moves the weights.
    """
    if shape not in SHAPES:
        raise ValueError(f"unknown decay shape {shape!r}; expected one of {SHAPES}")
    step = max(float(step), 0.0)
    total = max(float(total), 1.0)
    if step < warmup_steps:
        return peak * step / max(float(warmup_steps), 1e-9)
    decay_start = total * (1.0 - decay_frac)
    if step <= decay_start:
        return peak
    # Snapped to [0, 1] so the endpoint is EXACTLY the floor: without the
    # clamp, (step/total - 0.8)/0.2 lands on 1.0000000000000002 and
    # 1 - sqrt(that) is -1.1e-16, which is a negative learning rate.
    y = min(max((step - decay_start) / max(total - decay_start, 1e-9), 0.0), 1.0)
    if y >= 1.0:
        return peak * floor_frac
    if shape == "1-sqrt":
        f = 1.0 - math.sqrt(y)
    elif shape == "linear":
        f = 1.0 - y
    else:
        f = 0.5 * (1.0 + math.cos(math.pi * y))
    return peak * (floor_frac + (1.0 - floor_frac) * f)


def cosine(step: float, total: float, peak: float, *,
           warmup_frac: float = 0.01, floor_frac: float = 0.1) -> float:
    """The incumbent, kept so the two are comparable and switchable.

    Byte-for-byte the rule `train_sequence_stages.lr_at` and
    `pretrain_mlm.lr_at` implement, so moving a trainer onto this module
    changes nothing until `--lr-schedule wsd` is passed.
    """
    total = max(float(total), 1.0)
    x = min(max(float(step) / total, 0.0), 1.0)
    if x < warmup_frac:
        return peak * x / max(warmup_frac, 1e-9)
    y = (x - warmup_frac) / max(1.0 - warmup_frac, 1e-9)
    return peak * (floor_frac + (1.0 - floor_frac) * 0.5
                   * (1.0 + math.cos(math.pi * y)))


def lr_for(name: str, step: float, total: float, peak: float, **kw) -> float:
    """Dispatch by name, so a trainer takes `--lr-schedule` and nothing else."""
    if name == "wsd":
        return warmup_stable_decay(step, total, peak, **kw)
    if name == "cosine":
        return cosine(step, total, peak,
                      **{k: v for k, v in kw.items()
                         if k in ("warmup_frac", "floor_frac")})
    raise ValueError(f"unknown schedule {name!r}; expected 'wsd' or 'cosine'")
