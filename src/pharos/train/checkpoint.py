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
