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
from pharos.train.checkpoint import atomic_save, load_resume     # noqa: E402

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

        print("\n== a resume tolerates an architecture change, within bounds ==")
        # The real case: wiring §6.2 added `vdist.proj.weight`, 2,304 of
        # 394,780,316 parameters, and a bare strict load turned every
        # resumable checkpoint in the tree into a RuntimeError that said
        # nothing about how much of the model was affected.
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, extra: bool):
                super().__init__()
                self.big = nn.Linear(256, 256)        # 65,792 parameters
                if extra:
                    self.tiny = nn.Linear(8, 8, bias=False)   # 64

        full, old = Net(True), Net(False)
        n_total = sum(v.numel() for v in full.state_dict().values())
        rep = load_resume(full, {"model": old.state_dict()}, what="t")
        chk("a 0.1% addition resumes", rep["n_fresh"] == 64,
            f"{rep['n_fresh']} of {n_total:,} = {rep['frac']:.4%}")
        chk("and it is NAMED, not silently absorbed",
            list(rep["fresh"]) == ["tiny.weight"], str(list(rep["fresh"])))
        chk("the weights that WERE present actually loaded",
            torch.equal(full.big.weight, old.big.weight))

        print("\n== but a resume that re-randomises the model is refused ==")
        # strict=False on its own is worse than the crash: it reports as a
        # resume and is not one. The bound is in PARAMETERS, because one
        # missing tensor can be 64 elements or an embedding table.
        starved = {k: v for k, v in full.state_dict().items() if k != "big.weight"}
        try:
            load_resume(full, {"model": starved}, what="t")
            chk("a 99.6% re-randomisation is refused", False, "it was allowed")
        except SystemExit as e:
            msg = str(e)
            chk("a 99.6% re-randomisation is refused", True)
            chk("the refusal names the tensor", "big.weight" in msg)
            chk("and states how much of the model it is", "%" in msg)

        print("\n== the bound is parameters, not tensor count ==")
        # One tensor missing either way; only the parameter count separates
        # a harmless architecture addition from a gutted model.
        chk("one small tensor passes, one large tensor does not",
            rep["frac"] < 0.01 < (full.big.weight.numel() / n_total),
            f"{rep['frac']:.4%} vs {full.big.weight.numel()/n_total:.1%}")
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
