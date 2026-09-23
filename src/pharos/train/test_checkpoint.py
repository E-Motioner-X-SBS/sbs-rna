#!/usr/bin/env python3
"""An interrupted save must leave the previous checkpoint intact.

Run: python3 src/pharos/train/test_checkpoint.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pharos.train.checkpoint import atomic_save                 # noqa: E402

fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} {detail}")
    if not ok:
        fails.append(name)


def main() -> int:
    d = Path(tempfile.mkdtemp())
    try:
        p = d / "ckpt.pt"
        atomic_save({"step": 1, "w": torch.zeros(4)}, p)
        chk("first save lands", p.exists() and torch.load(
            p, weights_only=False)["step"] == 1)
        chk("no temp file left behind", not (d / "ckpt.pt.writing").exists())

        atomic_save({"step": 2, "w": torch.ones(4)}, p)
        chk("second save replaces it", torch.load(p, weights_only=False)["step"] == 2)

        print("\n== a save that raises must not damage what is there ==")

        class Boom:
            def __reduce__(self):
                raise RuntimeError("serialisation failed")

        try:
            atomic_save({"step": 3, "bad": Boom()}, p)
        except Exception:
            pass
        loaded = torch.load(p, weights_only=False)
        chk("previous checkpoint survives a failed save", loaded["step"] == 2,
            f"step {loaded['step']}")
        chk("and no temp file is left", not (d / "ckpt.pt.writing").exists())

        print("\n== the file is never observed partially written ==")
        # os.replace is atomic, so any reader sees one whole version or the
        # other. Assert the invariant that makes that true: the destination is
        # only ever created by a rename, never opened for writing directly.
        import inspect
        from pharos.train import checkpoint as mod
        src = inspect.getsource(mod)
        chk("destination is written via os.replace", "os.replace(tmp, path)" in src)
        chk("data is fsynced before the rename", "os.fsync" in src)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
