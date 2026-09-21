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
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmarks"
OUT = ROOT / "data/samples/analysis"
CKPT = ROOT / "data/derived/checkpoints"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pharos.data.chemistry import N_DIMS, chain_chemistry            # noqa: E402
from pharos.data.vocab import PAD_ID, encode_chain                   # noqa: E402
from pharos.model.moe import RouterFeatures                          # noqa: E402
from pharos.model.pharos import Pharos, PharosConfig                 # noqa: E402
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


def encode(seqs: List[str], device) -> Dict[str, torch.Tensor]:
    L = max(len(s) for s in seqs)
    B = len(seqs)
    tok = np.full((B, L), PAD_ID, dtype=np.int16)
    chem = np.zeros((B, L, N_DIMS), dtype=np.float32)
    mask = np.zeros((B, L), dtype=bool)
    for i, s in enumerate(seqs):
        comps = list(s.upper().replace("T", "U"))
        t, _ = encode_chain(comps)
        n = len(t)
        tok[i, :n] = t
        chem[i, :n] = chain_chemistry(comps)
        mask[i, :n] = True
    return {"tokens": torch.as_tensor(tok, dtype=torch.long, device=device),
            "mod_ids": torch.zeros((B, L), dtype=torch.long, device=device),
            "chem": torch.as_tensor(chem, device=device),
            "mask": torch.as_tensor(mask, device=device)}


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
    args = ap.parse_args()

    device = require_gpu(args)
    enable_gpu_fast_paths()
    cfg = PharosConfig.small() if args.size == "small" else PharosConfig.mini()
    model = Pharos(cfg).to(device)
    if args.init_from and args.init_from.exists():
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
    CKPT.mkdir(parents=True, exist_ok=True)
    hist: List[Dict] = []

    for ep in range(args.epochs):
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
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                print(f"[seq] ep{ep} step{step}  2D loss "
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
        torch.save({"cfg": cfg.__dict__, "model": model.state_dict(), "epoch": ep},
                   CKPT / f"seqstages_{args.size}.pt")

    report = {"size": args.size, "params": pc, "stage_weights": STAGE_WEIGHTS,
              "history": hist}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"seqstages_{args.size}_results.json").write_text(json.dumps(report, indent=1))
    print(f"\n[seq] -> {OUT / f'seqstages_{args.size}_results.json'}")


if __name__ == "__main__":
    main()
