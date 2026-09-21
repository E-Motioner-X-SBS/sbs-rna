#!/usr/bin/env python3
"""Train the block-detection scorer and measure recall at the operating budget.

This closes R1, the largest unvalidated assumption in PHAROS: ARCHITECTURE v0.2
§7 selects contact-bearing blocks instead of ranking pairs, and every recall
number published for it so far came from **random weights**. The design is
sound only if a model can actually find the occupied blocks from sequence.

The result is meaningless without baselines, because a scorer that knows only
sequence separation already does well -- contacts concentrate near the diagonal.
Four numbers are reported at the identical budget, on a family-disjoint test
split:

    random          blocks drawn uniformly from the valid triangle
    separation      the |i-j| prior alone, no sequence
    learned         the trained scorer
    learned, ablated  the trained scorer with its separation prior switched off

The interesting quantity is `learned - separation`: how much the sequence
contributes over the geometry of a chain. If that gap is small, §7's selector
should be replaced by a fixed banded prior and the compute spent elsewhere --
which is a real possible outcome of this experiment and is why it is run.

GPU only. The block hierarchy makes this cheap (288 blocks at b1 for a 4,608-nt
chain), but the residue encoder still runs at full length, and CPU training
would take long enough to discourage the ablations that give the result meaning.

Usage:
    /store/shuvam/.venv/bin/python scripts/train_block_scorer.py --epochs 8
    /store/shuvam/.venv/bin/python scripts/train_block_scorer.py --eval-only --ckpt ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/derived/pharos3d"
OUT = ROOT / "data/samples/analysis"
CKPT = ROOT / "data/derived/checkpoints"

#: See train_pharos.CKPT_FORMAT. `block_scorer.pt` as it stands predates this
#: and holds only the BEST epoch's weights -- no optimiser, no schedule, no
#: step count -- so it is a result, not training state, and is not resumed from.
CKPT_FORMAT = 2
sys.path.insert(0, str(ROOT / "src"))

from pharos.data.loader import Pharos3DDataset                      # noqa: E402
from pharos.train.telemetry import RunLog
from pharos.model.block_scorer import (BlockScorer, ScorerConfig,   # noqa: E402
                                       l1_budget, l2_budget, occupancy_labels,
                                       occupancy_loss, recall_at_budget,
                                       valid_mask)


def enable_gpu_fast_paths() -> None:
    """A100 fast paths that are free and off by default.

    TF32 gives the Ampere tensor cores a 10-bit mantissa on fp32 matmuls and
    convolutions. For a model already training in bf16 autocast -- 8-bit
    mantissa -- refusing TF32 on the fp32 residue is precision theatre that
    costs real throughput. `high` keeps fp32 accumulation.
    """
    import torch as _t
    if not _t.cuda.is_available():
        return
    _t.backends.cuda.matmul.allow_tf32 = True
    _t.backends.cudnn.allow_tf32 = True
    _t.backends.cudnn.benchmark = True
    _t.set_float32_matmul_precision("high")


def to_device(batch: Dict, device) -> Dict:
    t = {
        "tokens": torch.as_tensor(batch["tokens"], dtype=torch.long, device=device),
        "mod_ids": torch.as_tensor(batch["mod_ids"], dtype=torch.long, device=device),
        "chem": torch.as_tensor(batch["chem"], dtype=torch.float32, device=device),
        "mask": torch.as_tensor(batch["mask"], dtype=torch.bool, device=device),
        "lengths": batch["lengths"],
        "weights": torch.as_tensor(batch["weights"], dtype=torch.float32, device=device),
    }
    t["contacts"] = [torch.as_tensor(c, dtype=torch.long, device=device)
                     for c in batch["contacts"]]
    return t


@torch.no_grad()
def evaluate(model: Optional[BlockScorer], ds: Pharos3DDataset, cfg: ScorerConfig,
             device, mode: str = "learned", max_batches: Optional[int] = None,
             token_budget: int = 16384, seed: int = 0) -> Dict:
    """Recall and precision at budget, per level, plus a length breakdown.

    `mode` selects what does the scoring: `learned`, `learned_noprior`,
    `separation` (the prior alone) or `random`. All four see the same batches,
    the same valid masks and the same budgets, so the numbers are comparable by
    construction rather than by assertion.
    """
    if model is not None:
        model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    acc: Dict[str, List] = {f"{lvl}_{m}": [] for lvl in ("l1", "l2")
                            for m in ("recall", "precision", "pos", "kept")}
    by_len: Dict[str, List] = {}
    n_chains = 0

    for bi, batch in enumerate(ds.iter_batches(token_budget=token_budget,
                                               shuffle=False, seed=seed)):
        if max_batches is not None and bi >= max_batches:
            break
        t = to_device(batch, device)
        if mode in ("learned", "learned_noprior"):
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                out = model(t["tokens"], t["mod_ids"], t["chem"], t["mask"],
                            use_sep_prior=(mode == "learned"))
            out = {k: (v.float() if v.dtype != torch.bool else v)
                   for k, v in out.items()}
        else:
            # scores that need no model: build them at the right shapes
            out = {}
            for lvl, b in (("l1", cfg.b1), ("l2", cfg.b2)):
                B, L = t["tokens"].shape
                n = (L + b - 1) // b
                bm = torch.zeros(B, n, dtype=torch.bool, device=device)
                for i, Li in enumerate(t["lengths"]):
                    bm[i, :((int(Li) + b - 1) // b)] = True
                if mode == "separation":
                    idx = torch.arange(n, device=device, dtype=torch.float32)
                    sep = (idx[None, :] - idx[:, None]).abs()
                    s = (-torch.log1p(sep))[None].expand(B, n, n).contiguous()
                else:
                    s = torch.rand(B, n, n, device=device, generator=None)
                out[f"{lvl}_scores"], out[f"{lvl}_bmask"] = s, bm

        for i, L in enumerate(t["lengths"]):
            L = int(L)
            n_chains += 1
            for lvl, b in (("l1", cfg.b1), ("l2", cfg.b2)):
                s = out[f"{lvl}_scores"][i]
                bm = out[f"{lvl}_bmask"][i]
                n = s.shape[0]
                v = valid_mask(n, cfg.sep_blocks(lvl), bm)
                if not bool(v.any()):
                    continue
                lab = occupancy_labels(t["contacts"][i], b, n, device)
                k = (l1_budget(n, v, cfg) if lvl == "l1"
                     else min(l2_budget(L, cfg), int(v.sum())))
                r, p, kk = recall_at_budget(s, lab, v, k)
                if math.isnan(r):
                    continue
                acc[f"{lvl}_recall"].append(r)
                acc[f"{lvl}_precision"].append(p)
                acc[f"{lvl}_pos"].append(float((lab * v).sum()))
                acc[f"{lvl}_kept"].append(kk)
                band = ("<128" if L < 128 else "128-512" if L < 512
                        else "512-1500" if L < 1500 else ">=1500")
                by_len.setdefault(f"{lvl}_{band}", []).append(r)

    res = {"mode": mode, "n_chains": n_chains}
    for k, v in acc.items():
        res[k] = round(float(np.mean(v)), 4) if v else None
    res["recall_by_length"] = {k: round(float(np.mean(v)), 4)
                               for k, v in sorted(by_len.items())}
    return res


def gpu_free_gib(index: int = 0) -> Optional[tuple]:
    """`(free, total)` GiB, read WITHOUT creating a CUDA context.

    `torch.cuda.mem_get_info` needs a context, and on a GPU that is already
    full, creating one is itself what fails -- so the check meant to produce a
    clean "come back later" instead produced a CUDA OOM traceback. nvidia-smi
    reads the same counters from outside the driver's allocator.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}",
             "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        f, t = (float(x) for x in out.stdout.strip().split(",")[:2])
        return f / 1024.0, t / 1024.0
    except Exception:                                        # noqa: BLE001
        return None


def require_gpu(args) -> torch.device:
    """Refuse to start rather than fall back to CPU or crash on a busy GPU."""
    if not args.device.startswith("cuda"):
        raise SystemExit(
            "This trainer is GPU-only by design (see the module docstring): the "
            "ablations that give the result its meaning are four full passes, "
            "and on CPU that is long enough to discourage running them. "
            "Pass --device cuda once a GPU is free.")
    mem = gpu_free_gib(int(args.device.split(":")[-1]) if ":" in args.device else 0)
    if mem is None:
        raise SystemExit("no GPU visible to nvidia-smi")
    free, total = mem
    print(f"[bs] GPU {free:.1f} of {total:.1f} GiB free")
    if free < args.min_free_gib:
        raise SystemExit(
            f"only {free:.1f} GiB free of {total:.1f}; another job holds this GPU. "
            f"Nothing has been started. Re-run when it frees "
            f"(or lower --min-free-gib, currently {args.min_free_gib}).")
    return torch.device(args.device)


def train(args) -> None:
    device = require_gpu(args)
    enable_gpu_fast_paths()

    cfg = ScorerConfig(d_model=args.d_model, d_block=args.d_block,
                       n_conv=args.n_conv, n_attn=args.n_attn,
                       target_c=args.target_c,
                       l1_sep_blocks=args.l1_sep_blocks)
    tr = Pharos3DDataset(args.data, split="train")
    va = Pharos3DDataset(args.data, split="val")
    te = Pharos3DDataset(args.data, split="test")
    print(f"[bs] train {len(tr):,} | val {len(va):,} | test {len(te):,} chains")
    # decompress every shard once up front; profiled at 53% of step time when
    # left to happen inside the loop
    for d in (tr, va, te):
        d.prewarm()
    print(f"[bs] shards resident: {tr.prewarm(verbose=True):.2f} GiB RSS")

    model = BlockScorer(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[bs] {n_par:,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    # Exact, not len(one epoch) x epochs -- greedy token-budget packing gives a
    # different batch count per shuffle, and OneCycleLR raises on the step past
    # its total. See the note in train_pharos.py.
    epoch_batches = [tr.length_batches(token_budget=args.token_budget, seed=ep)
                     for ep in range(args.epochs)]
    total_steps = max(1, sum(len(b) for b in epoch_batches))
    print(f"[bs] {total_steps:,} optimiser steps over {args.epochs} epochs",
          flush=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

    CKPT.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history: List[Dict] = []
    step, start_ep, n_oom = 0, 0, 0
    # Resume state lives BESIDE the result. `block_scorer.pt` is written only
    # when validation improves, so it is the best epoch and not the last one --
    # resuming from it would silently rewind training to whenever that was.
    state = args.ckpt.with_suffix(".state.pt")
    if state.exists() and args.restart:
        keep = state.with_suffix(".superseded.pt")
        state.replace(keep)
        print(f"[bs] --restart: {state.name} moved to {keep.name}", flush=True)
    resume = torch.load(state, map_location=device) if state.exists() else None
    if resume is not None and resume.get("format") != CKPT_FORMAT:
        print(f"[bs] {state.name} predates resumable state; ignoring", flush=True)
        resume = None
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        step = int(resume.get("step", 0))
        best = float(resume.get("best", -1.0))
        history = list(resume.get("history", []))
        start_ep = int(resume.get("epoch", 0)) + (1 if resume.get("epoch_done") else 0)
        if resume.get("total_steps") == total_steps and "sched" in resume:
            sched.load_state_dict(resume["sched"])
        else:
            print(f"[bs] schedule changed ({resume.get('total_steps')} -> "
                  f"{total_steps}); fast-forwarding a fresh one", flush=True)
            for _ in range(min(step, total_steps - 1)):
                sched.step()
        print(f"[bs] resumed: epoch {resume.get('epoch')}, step {step:,}, "
              f"best val L2 {best:.4f}", flush=True)
        if start_ep >= args.epochs:
            print(f"[bs] all {args.epochs} epochs already done", flush=True)

    runlog = RunLog(ROOT, "r1_block_scorer", [
        "lr", "loss", "n_oom", "best_val_l2", "note",
        "val_l1_recall", "val_l1_precision", "val_l2_recall", "val_l2_precision",
    ], manifest={
        "config": cfg.__dict__, "n_parameters": n_par, "epochs": args.epochs,
        "token_budget": args.token_budget, "lr_peak": args.lr,
        "total_steps": total_steps,
        "batches_per_epoch": [len(b) for b in epoch_batches],
        "splits": {"train": len(tr), "val": len(va), "test": len(te)},
        "resumed_from_epoch": start_ep, "resumed_from_step": step,
        "checkpoint": str(args.ckpt), "state": str(state),
    })
    if step:
        runlog.event("resumed", epoch=start_ep, step=step)

    def save_state(ep: int, done: bool) -> None:
        torch.save({"format": CKPT_FORMAT, "cfg": cfg.__dict__,
                    "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "total_steps": total_steps,
                    "epoch": ep, "epoch_done": done, "step": step,
                    "best": best, "history": history}, state)

    for ep in range(start_ep, args.epochs):
        model.train()
        t0, run = time.time(), []
        for bidx in epoch_batches[ep]:
            t = to_device(tr.collate(bidx), device)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(t["tokens"], t["mod_ids"], t["chem"], t["mask"])
                    out = {k: (v.float() if v.dtype != torch.bool else v)
                           for k, v in out.items()}
                    loss, parts = occupancy_loss(out, t["contacts"],
                                                 t["lengths"], cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            except torch.OutOfMemoryError:
                n_oom += 1
                opt.zero_grad(set_to_none=True)
                del t
                torch.cuda.empty_cache()
                sched.step()       # keep the curve aligned with the batch count
                step += 1
                if n_oom <= 5 or n_oom % 50 == 0:
                    print(f"[bs] OOM #{n_oom} at step {step}; batch skipped",
                          flush=True)
                runlog.event(f"OOM #{n_oom}", epoch=ep, step=step, n_oom=n_oom)
                continue
            sched.step()
            run.append(float(loss.detach()))
            step += 1
            if step % args.ckpt_every == 0:
                save_state(ep, False)
            if step % args.log_every == 0:
                runlog.log("step", epoch=ep, step=step,
                           lr=sched.get_last_lr()[0], n_oom=n_oom,
                           loss=round(float(np.mean(run[-args.log_every:])), 6))
                print(f"[bs] ep{ep} step{step} loss {np.mean(run[-args.log_every:]):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        ev = evaluate(model, va, cfg, device, "learned",
                      max_batches=args.eval_batches)
        history.append({"epoch": ep, "loss": float(np.mean(run)), **ev})
        print(f"[bs] epoch {ep}: loss {np.mean(run):.4f}  "
              f"val L1 recall {ev['l1_recall']}  L2 recall {ev['l2_recall']}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        runlog.log("epoch", epoch=ep, step=step,
                   loss=round(float(np.mean(run)), 6), n_oom=n_oom,
                   lr=sched.get_last_lr()[0], best_val_l2=best,
                   val_l1_recall=ev.get("l1_recall"),
                   val_l1_precision=ev.get("l1_precision"),
                   val_l2_recall=ev.get("l2_recall"),
                   val_l2_precision=ev.get("l2_precision"))
        score = (ev["l2_recall"] or 0.0)
        if score > best:
            best = score
            # the RESULT: best epoch only, and deliberately without optimiser
            # state, because this is what evaluation and the cascade read
            torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                        "epoch": ep, "val": ev}, args.ckpt)
        save_state(ep, True)          # the RESUME STATE: last epoch, always

    # ---- the comparison R1 is actually about --------------------------
    print("\n[bs] test-set comparison at the identical budget")
    sd = torch.load(CKPT / "block_scorer.pt", map_location=device)
    model.load_state_dict(sd["model"])
    if not history:
        # see train_pharos.py: a run that trained nothing must not clobber the
        # last real results file
        print("[bs] no epochs ran; leaving the existing results file alone",
              flush=True)
        return
    report = {"config": cfg.__dict__, "n_parameters": n_par,
              "best_epoch": sd["epoch"], "history": history,
              "splits": {"train": len(tr), "val": len(va), "test": len(te)},
              "test": {}}
    for mode in ("random", "separation", "learned_noprior", "learned"):
        r = evaluate(model, te, cfg, device, mode, max_batches=args.eval_batches)
        report["test"][mode] = r
        print(f"  {mode:16s} L1 recall {r['l1_recall']}  L2 recall {r['l2_recall']}"
              f"  (L2 precision {r['l2_precision']})")
    sep = report["test"]["separation"]
    lrn = report["test"]["learned"]
    for lvl in ("l1", "l2"):
        if sep[f"{lvl}_recall"] is not None and lrn[f"{lvl}_recall"] is not None:
            report[f"{lvl}_sequence_gain"] = round(
                lrn[f"{lvl}_recall"] - sep[f"{lvl}_recall"], 4)
    print(f"\n  sequence gain over the separation prior: "
          f"L1 {report.get('l1_sequence_gain')}  L2 {report.get('l2_sequence_gain')}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "block_scorer_results.json").write_text(json.dumps(report, indent=1))
    print(f"\n[bs] -> {OUT / 'block_scorer_results.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-free-gib", type=float, default=20.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-block", type=int, default=192)
    ap.add_argument("--n-conv", type=int, default=6)
    ap.add_argument("--n-attn", type=int, default=3)
    ap.add_argument("--target-c", type=float, default=24.0)
    ap.add_argument("--l1-sep-blocks", type=int, default=1,
                    help="minimum L1 separation IN L1 BLOCKS. 1 (the default) "
                         "excludes the L1 diagonal, which makes 18.9%% of true "
                         "L2 blocks unreachable at any budget because an L1 "
                         "block is 16 residues and L2's separation is 4. 0 "
                         "admits it -- and only helps if the scorer is trained "
                         "that way, so this is a retrain flag, not a switch")
    ap.add_argument("--eval-batches", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=200,
                    help="steps between resume-state saves")
    ap.add_argument("--restart", action="store_true",
                    help="ignore existing resume state; it is RENAMED")
    ap.add_argument("--ckpt", type=Path, default=CKPT / "block_scorer.pt")
    args = ap.parse_args()

    if args.eval_only:
        device = require_gpu(args)
        sd = torch.load(args.ckpt, map_location=device)
        cfg = ScorerConfig(**sd["cfg"])
        model = BlockScorer(cfg).to(device)
        model.load_state_dict(sd["model"])
        te = Pharos3DDataset(args.data, split="test")
        for mode in ("random", "separation", "learned_noprior", "learned"):
            r = evaluate(model, te, cfg, device, mode, max_batches=args.eval_batches)
            print(f"  {mode:16s} L1 {r['l1_recall']}  L2 {r['l2_recall']}")
        return
    train(args)


if __name__ == "__main__":
    main()
