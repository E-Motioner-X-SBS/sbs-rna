#!/usr/bin/env python3
"""Multi-task training for PHAROS on the 3D corpus (stage 5 of §12.1).

Stages 1-3 of the curriculum (sequence MLM, 2D, probing) run on corpora that
are not the 3D set; this trains the structural stage and the heads whose labels
come out of the deposited files themselves. It is the stage that uses everything
`build_dataset.py` extracts:

    head 1   contact          contact set, 8 A, |i-j| >= 4
    head 5   Mg2+ sites       residues within 3 A of a magnesium
    head 6   rigidity         normalised B-factor -- X-RAY ONLY (D12)
    head 10  base identity    the N_struct residues
    disorder disorder        _pdbx_unobs_or_zero_occ_residues
    ensemble fluctuation      supervised through the rigidity target

Three things the data forces, each of which is a way to get this wrong.

**Heads are masked, never zero-filled.** The corpus is heterogeneous by
construction: B-factors are meaningful only for X-ray (62% of entries are
cryo-EM, where per-atom B is a fitted display parameter on an entirely
different scale -- 7PAS reads a mean of 987 against 1VY7's 72), `N_struct`
residues are 0.025% of the corpus, and Mg2+ is absent from most cryo-EM
depositions. A head with no labels in a batch contributes nothing, rather than
contributing a zero that quietly trains it toward the mean.

**Quality weights the loss, it does not filter the data** (D16). A resolution
cutoff would discard the cryo-EM majority.

**Evaluation is split three ways** (D25): `val`/`test` are family-disjoint and
measure generalisation to unseen folds; `test_ribosomal` is entry-disjoint only
and is reported under its own name, never averaged in.

GPU only, for the same reason as the block scorer: see `require_gpu`.

Usage:
    /store/shuvam/.venv/bin/python scripts/train_pharos.py --epochs 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/derived/pharos3d"
OUT = ROOT / "data/samples/analysis"
CKPT = ROOT / "data/derived/checkpoints"
sys.path.insert(0, str(ROOT / "src"))

from pharos.data.loader import Pharos3DDataset                    # noqa: E402
from pharos.model.moe import RouterFeatures                       # noqa: E402
from pharos.model.pharos import Pharos, PharosConfig              # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from train_block_scorer import gpu_free_gib                       # noqa: E402


def require_gpu(args) -> torch.device:
    if not args.device.startswith("cuda"):
        raise SystemExit("GPU only; pass --device cuda once one is free.")
    mem = gpu_free_gib()
    if mem is None:
        raise SystemExit("no GPU visible to nvidia-smi")
    free, total = mem
    print(f"[pharos] GPU {free:.1f} of {total:.1f} GiB free")
    if free < args.min_free_gib:
        raise SystemExit(f"only {free:.1f} GiB free; nothing started.")
    return torch.device(args.device)


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


def to_device(b: Dict, device) -> Dict:
    t = {k: torch.as_tensor(v, device=device)
         for k, v in b.items()
         if k in ("tokens", "mod_ids", "chem", "mask", "mg_site", "b_factor_z",
                  "unknown_base", "rigidity_mask", "base_mask", "weights")}
    t["tokens"] = t["tokens"].long()
    t["mod_ids"] = t["mod_ids"].long()
    t["chem"] = t["chem"].float()
    t["lengths"] = b["lengths"]
    t["contacts"] = [torch.as_tensor(c, dtype=torch.long, device=device)
                     for c in b["contacts"]]
    t["meta"] = b["meta"]
    return t


def router_features(t: Dict, recycle: int = 0) -> RouterFeatures:
    """§5.3's conditioning, from what the batch actually knows.

    `Neff/L` is not in the 3D set, so it is left absent -- which the router
    reads as zero -- rather than invented. In-complex comes from the entry's
    composition, which is measured (§11.2).
    """
    dev = t["tokens"].device
    B = t["tokens"].shape[0]
    in_cx = torch.tensor([1.0 if m.get("has_protein") else 0.0 for m in t["meta"]],
                         device=dev)
    chem_summary = t["chem"].sum(1) / t["mask"].sum(1, keepdim=True).clamp(min=1)
    return RouterFeatures(
        length=torch.as_tensor(t["lengths"], dtype=torch.float32, device=dev),
        in_complex=in_cx, chem_summary=chem_summary[:, :5], recycle=recycle)


def sample_pairs(contacts: torch.Tensor, L: int, n_neg: int,
                 device, min_sep: int = 4) -> tuple:
    """Positive contacts plus sampled negatives, for the contact head.

    The contact head is evaluated on the pairs the track selected; during this
    stage the track is not yet trained, so pairs are sampled directly. Negatives
    are drawn from the same |i-j| >= 4 population as the positives so the head
    cannot win by learning the separation prior alone -- which is the same trap
    the block-scorer baselines exist to expose.
    """
    if len(contacts) == 0:
        return None, None, None
    pos = contacts
    n_neg = max(1, min(n_neg, 8 * len(pos)))
    i = torch.randint(0, max(L - min_sep, 1), (n_neg,), device=device)
    off = torch.randint(min_sep, max(L - 1, min_sep + 1), (n_neg,), device=device)
    j = (i + off).clamp(max=L - 1)
    keep = (j - i) >= min_sep
    i, j = i[keep], j[keep]
    ii = torch.cat([pos[:, 0], i])
    jj = torch.cat([pos[:, 1], j])
    y = torch.cat([torch.ones(len(pos), device=device),
                   torch.zeros(len(i), device=device)])
    return ii, jj, y


def step_losses(model: Pharos, t: Dict, cfg, n_neg: int,
                n_loops: Optional[int] = None) -> tuple:
    """One forward pass and every head that this batch can supervise.

    Two throughput decisions live here, both measured.

    **Recycling is sampled, not fixed.** The trunk at the configured 8 loops
    costs 5,192 ms of an 8,029 ms step; one loop costs 635 ms, so the loops
    *are* the step. Sampling the count per step is the standard recycling
    recipe and it is better than a fixed 8 in two ways: the expected cost falls
    to about half, and the model is trained to produce a usable answer at every
    recycle count rather than only at the one it always saw.

    **The ensemble is computed only when something supervises it.** §10's
    selected inversion is O(L) sequential 6x6 solves -- 1,241 ms at L=981 --
    and its only training signal is the X-ray B-factor target, which D12
    restricts to 32% of chains. On a batch with no X-ray chain it was a second
    of arithmetic nobody read.
    """
    dev = t["tokens"].device
    B, L = t["tokens"].shape
    want_dyn = bool(t["rigidity_mask"].any())
    out = model(t["tokens"], t["mod_ids"], t["chem"], t["mask"],
                feats=router_features(t), n_loops=n_loops, dynamics=want_dyn)
    parts: Dict[str, float] = {}
    total = torch.zeros((), device=dev)
    w = t["weights"]

    # head 5 -- Mg2+ sites. Heavily imbalanced (4.7% positive), and a missed
    # site is a missed ion, so positives are up-weighted by the batch ratio.
    m = t["mask"]
    if m.any():
        tgt = t["mg_site"].float()
        pos = tgt[m].sum().clamp(min=1.0)
        pw = ((m.sum() - pos) / pos).clamp(1.0, 100.0)
        l = F.binary_cross_entropy_with_logits(out["mg_logit"][m], tgt[m],
                                               pos_weight=pw)
        total = total + 0.3 * l
        parts["mg"] = float(l.detach())

    # head 6 -- rigidity. D12: X-ray only, and the mask is what enforces it.
    rm = t["rigidity_mask"]
    if want_dyn and rm.any():
        l = F.smooth_l1_loss(out["rigidity"][rm], t["b_factor_z"][rm].float())
        total = total + 0.3 * l
        parts["rigidity"] = float(l.detach())
        # the ensemble's fluctuation amplitude predicts the same observable, so
        # it is supervised by it -- that is what makes the stiffness field
        # trainable without any measured stiffness
        fl = out["fluctuation"][rm]
        l2 = F.smooth_l1_loss(fl, (t["b_factor_z"][rm].float() -
                                   t["b_factor_z"][rm].float().min()).clamp(min=0))
        total = total + 0.1 * l2
        parts["fluctuation"] = float(l2.detach())

    # head 10 -- base identity, exactly where it was never assigned
    bm = t["base_mask"]
    if bm.any():
        l = F.cross_entropy(out["base_logits"][bm], t["tokens"][bm].clamp(max=3))
        total = total + 0.2 * l
        parts["base"] = float(l.detach())

    # head 1 -- contacts, on sampled pairs.
    #
    # ONE call, not one per chain. The per-chain version ran pair_proj, the
    # motif bank and the contact head B times sequentially; with the batch cap
    # lifted to 512 chains that is 512 sequential sub-forwards per step, each
    # too small to fill the card, and it is why utilisation sat at 32% after
    # the trunk was already fixed. Pairs from every chain are gathered into one
    # flat tensor, scored together, and the per-chain quality weight is applied
    # per pair -- which is what weighting the per-chain mean amounted to.
    ii_all, jj_all, y_all, wt_all, bi_all = [], [], [], [], []
    for b in range(B):
        Lb = int(t["lengths"][b])
        ii, jj, y = sample_pairs(t["contacts"][b], Lb, n_neg, dev)
        if ii is None or len(ii) == 0:
            continue
        ii_all.append(ii)
        jj_all.append(jj)
        y_all.append(y)
        wt_all.append(w[b].expand(len(ii)))
        bi_all.append(torch.full((len(ii),), b, dtype=torch.long, device=dev))
    n_pairs = 0
    if ii_all:
        ii = torch.cat(ii_all); jj = torch.cat(jj_all)
        y = torch.cat(y_all); wt = torch.cat(wt_all); bidx = torch.cat(bi_all)
        n_pairs = int(len(ii))
        h = out["hidden"]
        pair = model.pair_proj(torch.cat([h[bidx, ii], h[bidx, jj]], dim=-1))
        if model.motifs is not None:
            r, _ = model.motifs(pair)
            pair = pair + model.motif_mix(r)
        logit = model.heads.pair.contact(pair).squeeze(-1)
        per = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        l = (per * wt).sum() / wt.sum().clamp(min=1e-6)
        total = total + 1.0 * l
        parts["contact"] = float(l.detach())

    total = total + out["aux"]["balance_loss"]
    parts["balance"] = float(out["aux"]["balance_loss"].detach())
    parts["n_pairs"] = n_pairs
    return total, parts, out


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the precision-recall curve, threshold-free.

    The metric a rare-positive detection task needs. Its random baseline is the
    positive base rate, so it is interpretable without a companion number.
    """
    if label.sum() == 0 or label.sum() == len(label):
        return float("nan")
    order = np.argsort(-score)
    y = label[order].astype(np.float64)
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


@torch.no_grad()
def evaluate(model: Pharos, ds: Pharos3DDataset, device, cfg,
             max_batches: Optional[int] = None, token_budget: int = 8192) -> Dict:
    """Threshold-free where the task is imbalanced, pooled where it is thin.

    Two corrections over the first version, both of which made a working head
    look broken.

    **The Mg head was scored at the wrong threshold.** It is trained with
    `pos_weight = neg/pos` (~17.5 at a 5.41% base rate), and BCE with a positive
    weight moves the decision boundary: the minimiser has `sigma(z) = w.p /
    (w.p + 1 - p)`, so `p > 0.5` corresponds to `z > log(w)` -- about 2.86, not
    0. Thresholding at 0 reported precision **0.026 against a 5.41% base rate**,
    i.e. apparently worse than chance, for a head that had simply been asked to
    over-predict. Average precision is reported instead, with the base rate
    beside it, plus precision at the calibrated threshold.

    **Correlations were averaged per batch.** A mean of per-batch Pearson `r`
    is not the `r` of the split, and on batches where the rigidity target is a
    handful of X-ray residues it is mostly noise. Scores are pooled across the
    split and correlated once.
    """
    model.eval()
    acc: Dict[str, List[float]] = {}
    pool: Dict[str, List[np.ndarray]] = {}
    for bi, batch in enumerate(ds.iter_batches(token_budget=token_budget,
                                               shuffle=False)):
        if max_batches is not None and bi >= max_batches:
            break
        t = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(t["tokens"], t["mod_ids"], t["chem"], t["mask"],
                        feats=router_features(t),
                        dynamics=bool(t["rigidity_mask"].any()))
        m = t["mask"]
        if m.any():
            pool.setdefault("mg_score", []).append(
                out["mg_logit"][m].float().cpu().numpy())
            pool.setdefault("mg_label", []).append(
                (t["mg_site"][m] > 0).cpu().numpy())
        rm = t["rigidity_mask"]
        if rm.any():
            pool.setdefault("rig_pred", []).append(
                out["rigidity"][rm].float().cpu().numpy())
            pool.setdefault("rig_true", []).append(
                t["b_factor_z"][rm].float().cpu().numpy())

    res: Dict = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
    if "mg_score" in pool:
        sc = np.concatenate(pool["mg_score"])
        yy = np.concatenate(pool["mg_label"])
        base = float(yy.mean())
        res["mg_base_rate"] = round(base, 4)
        res["mg_average_precision"] = round(average_precision(sc, yy), 4)
        # lift over chance is the number that says whether the head learned
        res["mg_ap_lift"] = (round(res["mg_average_precision"] / base, 2)
                             if base > 0 else None)
        # and at the threshold the pos_weight actually implies
        pw = (1 - base) / max(base, 1e-9)
        thr = float(np.log(max(pw, 1.0)))
        p = sc > thr
        tp = float((p & yy).sum()); fp = float((p & ~yy).sum())
        fn = float((~p & yy).sum())
        res["mg_precision_at_calibrated_thr"] = round(tp / max(tp + fp, 1), 4)
        res["mg_recall_at_calibrated_thr"] = round(tp / max(tp + fn, 1), 4)
    if "rig_pred" in pool:
        pr = np.concatenate(pool["rig_pred"])
        tg = np.concatenate(pool["rig_true"])
        res["rigidity_r_pooled"] = (round(float(np.corrcoef(pr, tg)[0, 1]), 4)
                                    if pr.std() > 1e-6 and tg.std() > 1e-6 else None)
        res["rigidity_n"] = int(len(pr))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-free-gib", type=float, default=30.0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--n-neg", type=int, default=2048)
    ap.add_argument("--size", default="mini", choices=("mini", "small"))
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--sample-loops", action="store_true", default=True,
                    help="sample the recycle count per step (default on)")
    ap.add_argument("--fixed-loops", dest="sample_loops", action="store_false")
    args = ap.parse_args()

    device = require_gpu(args)
    enable_gpu_fast_paths()
    rng = np.random.default_rng(0)
    cfg = (PharosConfig.small() if args.size == "small" else PharosConfig.mini())
    tr = Pharos3DDataset(args.data, split="train", max_length=args.max_length)
    va = Pharos3DDataset(args.data, split="val", max_length=args.max_length)
    te = Pharos3DDataset(args.data, split="test", max_length=args.max_length)
    tb = Pharos3DDataset(args.data, split="test_ribosomal", max_length=args.max_length)
    print(f"[pharos] train {len(tr):,} | val {len(va):,} | test {len(te):,} "
          f"| test_ribosomal {len(tb):,}")
    for d in (tr, va, te, tb):
        d.prewarm()
    print(f"[pharos] shards resident: {tr.prewarm(verbose=True):.2f} GiB RSS")

    model = Pharos(cfg).to(device)
    pc = model.param_counts()
    print(f"[pharos] {args.size}: {pc['total']:,} total, {pc['active']:,} active")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    # Precompute each epoch's batches and total them EXACTLY. Multiplying one
    # epoch's count by the epoch count is wrong here: batches are packed greedily
    # to a token budget, so a different shuffle yields a different number of them
    # -- measured 39/40/40/40/39/40/39/40 over eight seeds, 317 against the 312
    # the multiplication predicts. OneCycleLR raises on the step past its total,
    # which killed a completed 8-epoch run at its very last step, after the
    # training was done and before the test evaluation ran.
    epoch_batches = [tr.length_batches(token_budget=args.token_budget, seed=ep)
                     for ep in range(args.epochs)]
    n_steps = max(1, sum(len(b) for b in epoch_batches))
    print(f"[pharos] {n_steps:,} optimiser steps over {args.epochs} epochs "
          f"({[len(b) for b in epoch_batches]})", flush=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=n_steps, pct_start=0.05)
    CKPT.mkdir(parents=True, exist_ok=True)
    history: List[Dict] = []
    step = 0
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), []
        for idxs in epoch_batches[ep]:
            t = to_device(tr.collate(idxs), device)
            # sampled recycling: 1..max_loops, uniform
            nl = int(rng.integers(1, cfg.n_loops + 1)) if args.sample_loops \
                else cfg.n_loops
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts, _ = step_losses(model, t, cfg, args.n_neg, n_loops=nl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run.append(float(loss.detach()))
            step += 1
            if step % args.log_every == 0:
                print(f"[pharos] ep{ep} step{step} loss "
                      f"{np.mean(run[-args.log_every:]):.4f} "
                      f"{ {k: round(v, 3) for k, v in parts.items() if k != 'n_pairs'} }",
                      flush=True)
        ev = evaluate(model, va, device, cfg, args.eval_batches, args.token_budget)
        history.append({"epoch": ep, "loss": float(np.mean(run)), "val": ev})
        print(f"[pharos] epoch {ep}: loss {np.mean(run):.4f} val {ev} "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                    "epoch": ep, "val": ev}, CKPT / f"pharos_{args.size}.pt")

    report = {"size": args.size, "config": cfg.__dict__, "params": pc,
              "history": history,
              "splits": {"train": len(tr), "val": len(va), "test": len(te),
                         "test_ribosomal": len(tb)},
              "test": evaluate(model, te, device, cfg, args.eval_batches,
                               args.token_budget),
              # D25: reported under its own name, never averaged with `test`
              "test_ribosomal": evaluate(model, tb, device, cfg,
                                         args.eval_batches, args.token_budget)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pharos_{args.size}_results.json").write_text(json.dumps(report, indent=1))
    print(f"\ntest (family-disjoint) : {report['test']}")
    print(f"test_ribosomal (entry-disjoint only, homolog-rich): "
          f"{report['test_ribosomal']}")
    print(f"\n[pharos] -> {OUT / f'pharos_{args.size}_results.json'}")


if __name__ == "__main__":
    main()
