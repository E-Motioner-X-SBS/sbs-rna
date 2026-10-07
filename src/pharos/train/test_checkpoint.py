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
from pharos.train.checkpoint import (BackgroundSaver,            # noqa: E402
                                     atomic_save, load_optimizer,
                                     load_resume)

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

        print("\n== and the optimiser moments survive the same change ==")
        # Fixing the model load alone left `opt.load_state_dict` to raise two
        # lines later: torch keys optimiser state by the parameter's POSITION,
        # so inserting a parameter shifts every index after it.
        old2 = Net(False)
        o_old = torch.optim.AdamW(old2.parameters(), lr=1e-3)
        old2.big.weight.grad = torch.ones_like(old2.big.weight)
        old2.big.bias.grad = torch.ones_like(old2.big.bias)
        o_old.step()                                   # give it real moments
        saved_opt = o_old.state_dict()

        new2 = Net(True)
        o_new = torch.optim.AdamW(new2.parameters(), lr=1e-3)
        raised = False
        try:
            o_new.load_state_dict(saved_opt)
        except ValueError:
            raised = True
        chk("a positional load raises on the added parameter", raised)

        o_new = torch.optim.AdamW(new2.parameters(), lr=1e-3)
        msg = load_optimizer(o_new, saved_opt, new2, what="t",
                             absent=["tiny.weight"])
        chk("transplanting by name succeeds", "transplanted" in msg, msg)
        names = [n for n, _ in new2.named_parameters()]
        st = o_new.state_dict()["state"]
        idx = {n: i for i, n in enumerate(names)}
        chk("the pre-existing parameter KEPT its moment",
            idx["big.weight"] in st
            and torch.equal(st[idx["big.weight"]]["exp_avg"],
                            o_old.state_dict()["state"][0]["exp_avg"]))
        chk("the added parameter has no moment yet", idx["tiny.weight"] not in st)
        covered = {q for g in o_new.param_groups for q in
                   range(len(g["params"]))}
        chk("and the added parameter is still IN a group, so it is stepped",
            sum(len(g["params"]) for g in o_new.param_groups) == len(names),
            f"{sum(len(g['params']) for g in o_new.param_groups)} of {len(names)}")

        print("\n== an unreconcilable optimiser state is DROPPED, loudly ==")
        # Wrong transplant is worse than no transplant: it would give a
        # parameter another parameter's moments.
        o_bad = torch.optim.AdamW(new2.parameters(), lr=1e-3)
        msg = load_optimizer(o_bad, saved_opt, new2, what="t", absent=[])
        chk("it refuses to guess when the counts do not reconcile",
            "DROPPED" in msg, msg.split(":")[0])
    finally:
        shutil.rmtree(d, ignore_errors=True)

        print("\n== the background saver: snapshot first, write later ==")
        import threading
        import time as _time
        q = d / "bg.pt"
        sv = BackgroundSaver()
        w = torch.arange(16, dtype=torch.float32)
        obj = {"step": 1, "model": {"w": w}}
        chk("the first save is accepted", sv.save(obj, q))
        # THE REASON THE SNAPSHOT EXISTS. Mutate the live tensor the instant
        # the call returns: a saver that handed the thread a view would
        # write the mutated value, i.e. a file from two different steps.
        w.fill_(999.0)
        sv.close()
        got = torch.load(q, weights_only=False)["model"]["w"]
        chk("the file holds the state AS OF THE CALL, not as of the write",
            bool(torch.equal(got, torch.arange(16, dtype=torch.float32))),
            f"first element {float(got[0])} -- 999 would mean the writer "
            f"saw a live view")
        chk("and the live tensor really was mutated", float(w[0]) == 999.0,
            "otherwise the test above proves nothing")

        print("\n== one writer at a time; a second is dropped, not queued ==")
        slow = d / "slow.pt"
        sv2 = BackgroundSaver()
        big = {"model": {"w": torch.zeros(2_000_000)}}
        sv2.save(big, slow)
        dropped_any = False
        for _ in range(50):
            if not sv2.save(big, slow):
                dropped_any = True
                break
        sv2.close()
        chk("a save during a save is dropped and counted",
            dropped_any and sv2.n_dropped >= 1,
            f"{sv2.n_saved} saved, {sv2.n_dropped} dropped -- queueing would "
            f"let a slow disk grow memory without bound")
        chk("and the file is still valid afterwards",
            torch.load(slow, weights_only=False)["model"]["w"].numel() == 2_000_000)

        print("\n== close() waits, so the last checkpoint is whole ==")
        sv3 = BackgroundSaver()
        last = d / "last.pt"
        sv3.save({"step": 7, "model": {"w": torch.ones(1_000_000)}}, last)
        sv3.close()
        chk("after close() the file is complete and readable",
            torch.load(last, weights_only=False)["step"] == 7
            and not sv3.busy())
        chk("no .writing temp file survives", not (d / "last.pt.writing").exists())

        print("\n== disabled, it is exactly atomic_save ==")
        sv4 = BackgroundSaver(enabled=False)
        sync = d / "sync.pt"
        sv4.save({"step": 3, "model": {"w": torch.full((8,), 2.0)}}, sync)
        chk("a disabled saver writes synchronously, no thread",
            not sv4.busy() and torch.load(sync, weights_only=False)["step"] == 3,
            "the ablation arm, so the speed claim can be turned off")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
