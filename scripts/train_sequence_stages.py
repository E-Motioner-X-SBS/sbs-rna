#!/usr/bin/env python3
"""Stages 2, 3 and 6 of the §12.1 curriculum — the sequence-supervised heads.

    stage 2  secondary structure   head 4   bpRNA-SPOT, 10,934 train
    stage 3  chemical probing      head 7   Ribonanza, 335,616 profiles
    stage 6  fitness (auxiliary)   head 8   NABench + RNAGym

All three read sequence and predict something per residue or per sequence, so
they share a trainer and are **co-trained** rather than run one after another.
The curriculum orders them because the *representation* matures in that order,
not because the losses conflict, and stage 6 is explicitly auxiliary.

Why probing is weighted above secondary structure
-------------------------------------------------
§12.1 calls Ribonanza "the largest labelled channel the model will see":
335,616 profiles against 673 clean 3D sequences, **499x**. Secondary structure
is 10,934 examples. If the two are summed unweighted, 2D dominates the gradient
by virtue of being read more often per epoch rather than by carrying more
information, so each stage's loss is scaled by an explicit weight that is
recorded rather than implied.

Reactivity is a masked target
-----------------------------
Ribonanza profiles are 206 columns against sequences of ~177 nt, and positions
outside the read-out window are `NaN` -- not zero. A zero there is a claim that
the base is unreactive, which is a different statement from "not measured", and
training on it teaches the model that the ends of every construct are
protected. Every reactivity loss is masked to the finite entries.

GPU only. Refuses to start on a busy card rather than OOM mid-run.

Usage:
    /store/shuvam/.venv/bin/python scripts/train_sequence_stages.py --epochs 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmarks"
OUT = ROOT / "data/samples/analysis"
CKPT = ROOT / "data/derived/checkpoints"

#: See train_pharos.CKPT_FORMAT. A checkpoint without it predates resumable
#: checkpoints and carries no optimiser state, so it is weights, not training
#: state, and must not be resumed from.
CKPT_FORMAT = 2
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pharos.data.chemistry_torch import BatchChemistry               # noqa: E402
from pharos.data.vocab import PAD_ID, SYMBOLS, encode_chain          # noqa: E402
from pharos.model.moe import RouterFeatures                          # noqa: E402
from pharos.model.pharos import Pharos, PharosConfig                 # noqa: E402
from pharos.train.telemetry import RunLog                            # noqa: E402
from pharos.train.checkpoint import atomic_save
from train_block_scorer import gpu_free_gib                          # noqa: E402

#: dot-bracket alphabet, matching HeadConfig.n_ss_symbols = 8
SS_SYMBOLS = ".()[]{}<"
SS_INDEX = {c: i for i, c in enumerate(SS_SYMBOLS)}
#: loss weights, stated rather than implied -- see the module docstring
STAGE_WEIGHTS = {"ss": 0.5, "reactivity": 1.0, "fitness": 0.3}


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


@lru_cache(maxsize=4)
def _batch_chem(device_str: str) -> BatchChemistry:
    return BatchChemistry(SYMBOLS, torch.device(device_str))


def _m(xs, n):
    """Mean of the last `n`, or None when there is nothing to average."""
    return round(float(np.mean(xs[-n:])), 6) if xs else None


def lr_at(step: float, total: float, peak: float,
          warmup_frac: float = 0.01, floor_frac: float = 0.1) -> float:
    """Linear warm-up then cosine decay. Same rule stage 1 uses.

    Stages 2 and 3 had no schedule either -- a constant 3e-4 throughout. The
    block scorer and stage 5 both run `OneCycleLR`; these two were the gap.

    This CLAMPS past `total` instead of raising, which `OneCycleLR` does not:
    `estimate_steps` reads row counts from parquet metadata and a CSV, and the
    loop drops malformed rows as it goes, so the estimate is close but not
    exact. A schedule that throws on the step past its total is how stage 5's
    off-by-N crashed a finished epoch.
    """
    x = min(max(step / max(total, 1.0), 0.0), 1.0)
    if x < warmup_frac:
        return peak * x / max(warmup_frac, 1e-9)
    y = (x - warmup_frac) / max(1.0 - warmup_frac, 1e-9)
    return peak * (floor_frac + (1.0 - floor_frac) * 0.5 * (1.0 + math.cos(math.pi * y)))


def estimate_steps(args) -> int:
    """Total optimiser steps, from row counts rather than by running an epoch.

    The loop pulls one batch from each stage per step and continues until both
    generators are exhausted, so an epoch is `max` of the two batch counts, not
    the sum. Parquet carries its row count in the footer, so stage 2 is free;
    stage 3 is a CSV and is counted by lines, once.
    """
    import pyarrow.parquet as pq
    n_ss = 0
    f = BENCH / "secondary_structure/bprna_spot/train.parquet"
    if f.exists():
        n_ss = pq.ParquetFile(f).metadata.num_rows
    n_pr = 0
    g = BENCH / "chemical_probing/ribonanza_train_quickstart.csv"
    if g.exists():
        with g.open() as fh:
            n_pr = max(sum(1 for _ in fh) - 1, 0)
    if args.probing_limit:
        n_pr = min(n_pr, args.probing_limit)
    per_epoch = max(-(-n_ss // args.batch), -(-n_pr // args.batch), 1)
    return per_epoch * max(args.epochs, 1)


def encode(seqs: List[str], device) -> Dict[str, torch.Tensor]:
    """Tokens, chemistry and mask for a padded batch.

    The chemistry used to be `chain_chemistry` per sequence on the host -- a
    Python loop between two GPU kernels, measured at 273 ms for a 26x950 batch.
    `BatchChemistry` is the same function as two device-side lookups and a
    cumulative sum, asserted equal to `chain_chemistry` in
    `test_chemistry_torch.py`. Deriving it from the encoded tokens rather than
    the letters is equivalent: a letter outside the vocabulary encodes to UNK,
    and `residue_chemistry` gives UNK and any unresolvable component the same
    row -- parent N, class other.
    """
    L = max(len(s) for s in seqs)
    B = len(seqs)
    tok = np.full((B, L), PAD_ID, dtype=np.int64)
    mask = np.zeros((B, L), dtype=bool)
    for i, s in enumerate(seqs):
        e, _ = encode_chain(list(s.upper().replace("T", "U")))
        n = len(e)
        tok[i, :n] = e
        mask[i, :n] = True
    tokens = torch.as_tensor(tok, device=device)
    msk = torch.as_tensor(mask, device=device)
    return {"tokens": tokens,
            "mod_ids": torch.zeros((B, L), dtype=torch.long, device=device),
            "chem": _batch_chem(str(device))(tokens, msk),
            "mask": msk}


# ---------------------------------------------------------------- stage 2
def iter_ss(split: str, batch: int) -> Iterator[Tuple[List[str], List[str]]]:
    import pyarrow.parquet as pq
    f = BENCH / f"secondary_structure/bprna_spot/{split}.parquet"
    if not f.exists():
        return
    seqs: List[str] = []
    ss: List[str] = []
    for b in pq.ParquetFile(f).iter_batches(
            batch_size=512, columns=["sequence", "secondary_structure"]):
        d = b.to_pylist()
        for r in d:
            s, t = r["sequence"], r["secondary_structure"]
            if not s or not t or len(s) != len(t):
                continue
            seqs.append(s)
            ss.append(t)
            if len(seqs) >= batch:
                yield seqs, ss
                seqs, ss = [], []
    if seqs:
        yield seqs, ss


def ss_targets(ss: List[str], L: int, device) -> torch.Tensor:
    y = np.full((len(ss), L), -100, dtype=np.int64)
    for i, t in enumerate(ss):
        for j, c in enumerate(t[:L]):
            y[i, j] = SS_INDEX.get(c, 0)
    return torch.as_tensor(y, device=device)


# ---------------------------------------------------------------- stage 3
def iter_probing(batch: int, limit: Optional[int] = None
                 ) -> Iterator[Tuple[List[str], np.ndarray, List[str]]]:
    """Ribonanza rows: sequence, 206 reactivity columns, experiment type."""
    import csv
    f = BENCH / "chemical_probing/ribonanza_train_quickstart.csv"
    if not f.exists():
        return
    seqs: List[str] = []
    rows: List[List[float]] = []
    kinds: List[str] = []
    n = 0
    with f.open(newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        rcols = [i for i, h in enumerate(header) if h.startswith("reactivity_")
                 and "error" not in h]
        i_seq = header.index("sequence")
        i_exp = header.index("experiment_type")
        for r in rd:
            seqs.append(r[i_seq])
            rows.append([float(r[c]) if r[c] not in ("", "NaN") else np.nan
                         for c in rcols])
            kinds.append(r[i_exp])
            n += 1
            if len(seqs) >= batch:
                yield seqs, np.asarray(rows, dtype=np.float32), kinds
                seqs, rows, kinds = [], [], []
            if limit and n >= limit:
                return
    if seqs:
        yield seqs, np.asarray(rows, dtype=np.float32), kinds


def require_gpu(args) -> torch.device:
    if not args.device.startswith("cuda"):
        raise SystemExit("GPU only; pass --device cuda once one is free.")
    mem = gpu_free_gib()
    if mem is None:
        raise SystemExit("no GPU visible to nvidia-smi")
    free, total = mem
    print(f"[seq] GPU {free:.1f} of {total:.1f} GiB free")
    if free < args.min_free_gib:
        raise SystemExit(f"only {free:.1f} GiB free; nothing started.")
    return torch.device(args.device)


def ss_step(model, seqs, ss, device, feats_fn) -> Tuple[torch.Tensor, float]:
    b = encode(seqs, device)
    L = b["tokens"].shape[1]
    y = ss_targets(ss, L, device)
    out = model(b["tokens"], b["mod_ids"], b["chem"], b["mask"],
                feats=feats_fn(b), n_loops=2)
    lg = out["ss_logits"].float()
    valid = (y >= 0) & b["mask"]
    if not bool(valid.any()):
        return None, 0.0
    loss = F.cross_entropy(lg[valid], y[valid])
    acc = float((lg[valid].argmax(-1) == y[valid]).float().mean())
    return loss + out["aux"]["balance_loss"], acc


def probing_step(model, seqs, react, kinds, device, feats_fn):
    b = encode(seqs, device)
    L = b["tokens"].shape[1]
    tgt = np.full((len(seqs), L), np.nan, dtype=np.float32)
    w = min(react.shape[1], L)
    tgt[:, :w] = react[:, :w]
    t = torch.as_tensor(tgt, device=device)
    # NaN means NOT MEASURED, which is not the same as unreactive; training on
    # a zero there teaches the model that construct ends are protected
    valid = torch.isfinite(t) & b["mask"]
    if not bool(valid.any()):
        return None, 0.0
    out = model(b["tokens"], b["mod_ids"], b["chem"], b["mask"],
                feats=feats_fn(b), n_loops=2)
    # head 7 emits two channels: DMS and SHAPE-like are different chemistries
    chan = torch.tensor([0 if "DMS" in k.upper() else 1 for k in kinds],
                        device=device)
    pred = out["reactivity"].float().gather(
        2, chan.view(-1, 1, 1).expand(-1, L, 1)).squeeze(-1)
    loss = F.smooth_l1_loss(pred[valid], t[valid])
    with torch.no_grad():
        p, q = pred[valid].cpu().numpy(), t[valid].cpu().numpy()
        r = float(np.corrcoef(p, q)[0, 1]) if p.std() > 1e-6 and q.std() > 1e-6 else 0.0
    return loss + out["aux"]["balance_loss"], r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-free-gib", type=float, default=30.0)
    ap.add_argument("--size", default="small", choices=("mini", "small"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--probing-limit", type=int, default=None)
    ap.add_argument("--init-from", type=Path, default=None,
                    help="a stage-1 pretraining checkpoint")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=200,
                    help="steps between checkpoints. Epoch-end only was a "
                         "latent full-restart: an epoch here is ~10,500 steps")
    ap.add_argument("--restart", action="store_true",
                    help="ignore an existing checkpoint; it is RENAMED, not "
                         "overwritten")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint path; defaults to "
                         "data/derived/checkpoints/seqstages_<size>.pt")
    args = ap.parse_args()

    device = require_gpu(args)
    enable_gpu_fast_paths()
    cfg = PharosConfig.small() if args.size == "small" else PharosConfig.mini()
    model = Pharos(cfg).to(device)
    CKPT.mkdir(parents=True, exist_ok=True)
    ck = args.ckpt or (CKPT / f"seqstages_{args.size}.pt")
    if ck.exists() and args.restart:
        keep = ck.with_suffix(".superseded.pt")
        ck.replace(keep)
        print(f"[seq] --restart: {ck.name} moved to {keep.name}", flush=True)
    resume = torch.load(ck, map_location=device) if ck.exists() else None
    if resume is not None and resume.get("format") != CKPT_FORMAT:
        print(f"[seq] {ck.name} predates resumable checkpoints; ignoring it "
              f"and honouring --init-from", flush=True)
        resume = None
    if resume is not None:
        # RESUME BEATS --init-from. The cron runner passes --init-from on every
        # fire, so without this an interrupted stage restarts from stage 1's
        # weights and throws away everything it had done -- and an epoch here is
        # about 10,500 steps.
        model.load_state_dict(resume["model"])
        print(f"[seq] resumed from {ck.name}: epoch {resume.get('epoch', 0)}, "
              f"step {resume.get('gstep', 0):,}", flush=True)
    elif args.init_from and args.init_from.exists():
        sd = torch.load(args.init_from, map_location=device)
        missing = model.load_state_dict(sd["model"], strict=False)
        print(f"[seq] initialised from {args.init_from.name} "
              f"({sd.get('tokens', 0)/1e9:.2f}B tokens); "
              f"{len(missing.missing_keys)} keys fresh")
    pc = model.param_counts()
    print(f"[seq] {args.size}: {pc['total']:,} total / {pc['active']:,} active")
    print(f"[seq] stage weights {STAGE_WEIGHTS}")

    def feats_fn(b):
        return RouterFeatures(
            length=b["mask"].sum(1).float(),
            chem_summary=(b["chem"].sum(1)
                          / b["mask"].sum(1, keepdim=True).clamp(min=1))[:, :5])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    hist: List[Dict] = []
    total_steps = max(1, estimate_steps(args))
    print(f"[seq] schedule: warm-up 1% then cosine over ~{total_steps:,} steps "
          f"({args.epochs} epochs)", flush=True)
    gstep, start_ep, n_oom = 0, 0, 0
    if resume is not None:
        if "opt" in resume:
            opt.load_state_dict(resume["opt"])
        gstep = int(resume.get("gstep", 0))
        hist = list(resume.get("history", []))
        # A checkpoint written at the end of an epoch resumes at the next one;
        # one written mid-epoch redoes that epoch from its start, with the
        # weights and optimiser as saved. The data generators read files
        # sequentially and cannot seek, so a little repetition is the price --
        # far cheaper than the whole stage.
        start_ep = int(resume.get("epoch", 0)) + (1 if resume.get("epoch_done") else 0)
        if start_ep >= args.epochs:
            print(f"[seq] all {args.epochs} epochs already done; nothing to do",
                  flush=True)
            return

    runlog = RunLog(ROOT, "stage23_seq", [
        "gstep", "lr", "ss_loss", "ss_accuracy", "probing_loss",
        "probing_pearson", "n_oom", "note",
    ], manifest={
        "size": args.size, "config": cfg.__dict__, "params": pc,
        "epochs": args.epochs, "batch": args.batch, "lr_peak": args.lr,
        "total_steps_estimated": total_steps,
        "stage_weights": STAGE_WEIGHTS,
        "init_from": str(args.init_from) if args.init_from else None,
        "resumed_from_epoch": start_ep, "resumed_from_gstep": gstep,
        "checkpoint": str(ck),
    })
    if gstep:
        runlog.event("resumed", epoch=start_ep, step=gstep, gstep=gstep)

    def save(ep: int, done: bool) -> None:
        atomic_save({"format": CKPT_FORMAT, "cfg": cfg.__dict__,
                    "model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "epoch_done": done,
                    "gstep": gstep, "history": hist}, ck)

    for ep in range(start_ep, args.epochs):
        model.train()
        t0 = time.time()
        ss_loss, ss_acc, pr_loss, pr_r, step = [], [], [], [], 0
        gen_ss = iter_ss("train", args.batch)
        gen_pr = iter_probing(args.batch, args.probing_limit)
        done_ss = done_pr = False
        while not (done_ss and done_pr):
            total = None
            # stage 2
            try:
                seqs, ss = next(gen_ss)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l, a = ss_step(model, seqs, ss, device, feats_fn)
                if l is not None:
                    total = STAGE_WEIGHTS["ss"] * l
                    ss_loss.append(float(l.detach()))
                    ss_acc.append(a)
            except StopIteration:
                done_ss = True
            # stage 3
            try:
                seqs, react, kinds = next(gen_pr)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l, r = probing_step(model, seqs, react, kinds, device, feats_fn)
                if l is not None:
                    total = (STAGE_WEIGHTS["reactivity"] * l if total is None
                             else total + STAGE_WEIGHTS["reactivity"] * l)
                    pr_loss.append(float(l.detach()))
                    pr_r.append(r)
            except StopIteration:
                done_pr = True
            if total is None:
                continue
            lr_now = lr_at(gstep, total_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr_now
            gstep += 1
            try:
                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            except torch.OutOfMemoryError:
                # Same reasoning as stages 1 and 5: one batch is not worth the
                # stage. `gstep` has already advanced, so the cosine keeps its
                # place against the estimate rather than stretching.
                n_oom += 1
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if n_oom <= 5 or n_oom % 50 == 0:
                    print(f"[seq] OOM #{n_oom} at step {step}; batch skipped",
                          flush=True)
                runlog.event(f"OOM #{n_oom}", epoch=ep, step=step, n_oom=n_oom)
                continue
            step += 1
            if step % args.ckpt_every == 0:
                save(ep, False)
            if step % args.log_every == 0:
                runlog.log("step", epoch=ep, step=step, gstep=gstep, lr=lr_now,
                           ss_loss=_m(ss_loss, args.log_every),
                           ss_accuracy=_m(ss_acc, args.log_every),
                           probing_loss=_m(pr_loss, args.log_every),
                           probing_pearson=_m(pr_r, args.log_every),
                           n_oom=n_oom)
                print(f"[seq] ep{ep} step{step} lr {lr_now:.2e}  2D loss "
                      f"{np.mean(ss_loss[-args.log_every:]):.4f} acc "
                      f"{np.mean(ss_acc[-args.log_every:]):.4f}  |  probing loss "
                      f"{np.mean(pr_loss[-args.log_every:]):.4f} r "
                      f"{np.mean(pr_r[-args.log_every:]):.4f}", flush=True)
        hist.append({"epoch": ep, "steps": step,
                     "ss_loss": float(np.mean(ss_loss)) if ss_loss else None,
                     "ss_accuracy": float(np.mean(ss_acc)) if ss_acc else None,
                     "probing_loss": float(np.mean(pr_loss)) if pr_loss else None,
                     "probing_pearson": float(np.mean(pr_r)) if pr_r else None})
        print(f"[seq] epoch {ep}: {hist[-1]}  ({time.time()-t0:.0f}s)", flush=True)
        runlog.log("epoch", epoch=ep, step=step, gstep=gstep,
                   ss_loss=hist[-1]["ss_loss"], ss_accuracy=hist[-1]["ss_accuracy"],
                   probing_loss=hist[-1]["probing_loss"],
                   probing_pearson=hist[-1]["probing_pearson"], n_oom=n_oom)
        save(ep, True)

    if not hist:
        # see train_pharos.py: a run that trained nothing must not clobber the
        # last real results file
        print("[seq] no epochs ran; leaving the existing results file alone",
              flush=True)
        return
    report = {"size": args.size, "params": pc, "stage_weights": STAGE_WEIGHTS,
              "history": hist}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"seqstages_{args.size}_results.json").write_text(json.dumps(report, indent=1))
    print(f"\n[seq] -> {OUT / f'seqstages_{args.size}_results.json'}")


if __name__ == "__main__":
    main()
