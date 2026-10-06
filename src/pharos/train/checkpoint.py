#!/usr/bin/env python3
"""Checkpoint writes that cannot leave a half-written file behind.

`torch.save(obj, path)` truncates `path` and streams into it. For a 2.85 GB
stage-1 checkpoint that is seconds during which the file on disk is neither the
old checkpoint nor the new one. Two things go wrong in that window, and one of
them is fatal:

* a reader sees a corrupt file. The router probe hit exactly this --
  `PytorchStreamReader failed reading zip archive: failed finding central
  directory` -- reading a checkpoint that was mid-save. Harmless; retry works.
* **the process dies and the only resume point is gone.** Stage 1 saves every
  250 steps and had 573M tokens in that one file. A kill landing in the write
  window would have cost all of it, and the trainer would have restarted from
  nothing while reporting that it resumed.

Writing to a temporary file in the same directory and renaming fixes both:
`os.replace` is atomic on POSIX within a filesystem, so a reader sees either
the old checkpoint or the new one, never a mixture, and a crash leaves the
previous checkpoint intact.

`fsync` before the rename, because the rename being atomic does not help if the
data behind it is still in the page cache when the machine goes down.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def atomic_save(obj: Any, path: Path | str, *, fsync: bool = True) -> None:
    """`torch.save`, but the destination is never partially written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    try:
        with tmp.open("wb") as fh:
            torch.save(obj, fh)
            fh.flush()
            if fsync:
                os.fsync(fh.fileno())
        os.replace(tmp, path)          # atomic within a filesystem
    except BaseException:
        # a failed save must not leave the temp file lying around pretending to
        # be a checkpoint, and must not have touched the real one
        tmp.unlink(missing_ok=True)
        raise


def load_resume(model, state: dict, *, what: str,
                max_fresh_frac: float = 0.01) -> dict:
    """Load a checkpoint into `model`, tolerating a deliberate architecture
    change but refusing a silent re-randomisation.

    The resume paths in `train_sequence_stages.py` and `train_pharos.py` both
    called `model.load_state_dict(resume["model"])` with the default
    `strict=True`, while the `--init-from` path three lines below used
    `strict=False` and printed what it had loaded. Wiring §6.2 added one
    parameter -- `vdist.proj.weight`, 2,304 of 394,780,316, behind a
    `bias_scale` that every one of these checkpoints stores as exactly zero,
    so it cannot change a single output -- and that turned every resumable
    checkpoint in the tree into

        RuntimeError: Missing key(s) in state_dict: "vdist.proj.weight".

    which does not say what changed, how much of the model it is, or whether
    continuing would be sound. A project that adds parameters as it goes
    cannot have its resume path be the one that breaks.

    `strict=False` alone would be worse than the crash: a resume that quietly
    re-initialises half the trunk reports as a resume and is not one -- the
    exact failure `--init-from`'s own guard exists to stop, on the path that
    never had it. So the tolerance is bounded, and measured in PARAMETERS
    rather than tensors: one missing tensor can be a 2,304-element projection
    or a 285M-element embedding table, and the tensor count cannot tell them
    apart.

    Returns the report so the caller can print it in its own voice.
    """
    sd = state["model"] if "model" in state else state
    res = model.load_state_dict(sd, strict=False)
    shapes = {k: v for k, v in model.state_dict().items()}
    n_total = sum(v.numel() for v in shapes.values())
    fresh = {k: shapes[k].numel() for k in res.missing_keys if k in shapes}
    n_fresh = sum(fresh.values())
    frac = n_fresh / max(n_total, 1)
    report = {"fresh": fresh, "n_fresh": n_fresh, "n_total": n_total,
              "frac": frac, "ignored": list(res.unexpected_keys),
              "loaded": len(sd) - len(res.unexpected_keys)}
    if frac > max_fresh_frac:
        named = ", ".join(sorted(fresh)[:6]) + (" ..." if len(fresh) > 6 else "")
        raise SystemExit(
            f"{what}: {len(fresh)} tensors ({n_fresh:,} parameters, "
            f"{frac:.2%} of the model) are absent from the checkpoint and "
            f"would resume at random initialisation -- {named}\n"
            f"  That is past the {max_fresh_frac:.0%} this path tolerates for "
            f"an architecture change. Resuming would report as a resume and "
            f"not be one. Use --restart to begin a new run, or --init-from to "
            f"treat it as initialisation rather than continuation.")
    return report


def load_optimizer(opt, saved: dict, model, *, what: str,
                   absent=()) -> str:
    """Load optimiser state across a deliberate architecture change.

    `load_resume` fixed the model load and stopped one line short. Two lines
    below it every trainer does `opt.load_state_dict(resume["opt"])`, and
    torch keys optimiser state by the PARAMETER'S POSITION in
    `opt.param_groups[i]["params"]`. Insert one parameter anywhere but the
    very end and every index after it shifts, so the load fails with

        ValueError: loaded state dict contains a parameter group that
        doesn't match the size of optimizer's group

    which, like the model-side crash, says nothing about what changed.

    Dropping the state instead would be quiet and wrong: Adam's moments are
    most of what a resume is FOR, and a run that silently restarts them
    reports as a resume while behaving like a warm restart -- a step-541
    resume of a 3,626-step cosine would take a visible loss spike that
    nothing in the log would explain.

    So the state is transplanted by NAME. The old ordering is not stored, but
    it is reconstructible: parameters are registered in module order, so the
    names the optimiser saw are exactly this model's names minus the ones the
    checkpoint did not carry. That reconstruction is CHECKED against the
    saved group sizes rather than assumed -- if the arithmetic does not land,
    the state is dropped loudly instead of transplanted wrongly.
    """
    new_names = [n for n, _ in model.named_parameters()]
    absent = set(absent)
    old_names = [n for n in new_names if n not in absent]
    n_saved = sum(len(g["params"]) for g in saved.get("param_groups", []))
    if n_saved == len(new_names):
        opt.load_state_dict(saved)                  # nothing moved
        return f"optimiser state loaded unchanged ({n_saved:,} parameters)"
    if n_saved != len(old_names):
        opt.state = type(opt.state)()
        return (f"OPTIMISER STATE DROPPED: it holds {n_saved:,} parameters, "
                f"the model has {len(new_names):,}, and removing the "
                f"{len(absent)} tensor(s) the checkpoint lacks leaves "
                f"{len(old_names):,} -- the two do not reconcile, so the "
                f"moments are not transplanted. Expect a brief loss "
                f"transient while Adam re-estimates them.")
    pos = {n: i for i, n in enumerate(new_names)}
    remap = {old_i: pos[n] for old_i, n in enumerate(old_names)}
    out = {"state": {remap[k]: v for k, v in saved["state"].items()
                     if k in remap},
           "param_groups": []}
    for g in saved["param_groups"]:
        g2 = dict(g)
        g2["params"] = [remap[p] for p in g["params"]]
        out["param_groups"].append(g2)
    # The fresh parameters have no moments and must still be IN a group, or
    # they would silently stop being optimised -- a parameter that exists,
    # takes gradient, and is never stepped.
    covered = {p for g in out["param_groups"] for p in g["params"]}
    missing = [i for i in range(len(new_names)) if i not in covered]
    if missing:
        out["param_groups"][0]["params"] = sorted(
            out["param_groups"][0]["params"] + missing)
    opt.load_state_dict(out)
    return (f"optimiser state transplanted by name: {len(remap):,} parameters "
            f"keep their moments, {len(missing)} start fresh "
            f"({', '.join(sorted(absent)) if absent else 'none named'})")
