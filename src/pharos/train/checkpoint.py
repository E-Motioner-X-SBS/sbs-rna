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
