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

#: Bumped when the checkpoint gains fields a resume depends on. A file without
#: it predates resumable checkpoints -- weights and an epoch number and nothing
#: else -- and must not be read as training state.
CKPT_FORMAT = 2
sys.path.insert(0, str(ROOT / "src"))

from pharos.data.loader import Pharos3DDataset                    # noqa: E402
from pharos.model.moe import RouterFeatures                       # noqa: E402
from pharos.model.pharos import Pharos, PharosConfig
from pharos.train.telemetry import RunLog
from pharos.train.checkpoint import atomic_save              # noqa: E402

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
                  "unknown_base", "rigidity_mask", "base_mask", "weights",
                  "coords", "coord_mask", "coord_residue_mask",
                  "coev_key", "coev_val")}
    t["tokens"] = t["tokens"].long()
    t["mod_ids"] = t["mod_ids"].long()
    t["chem"] = t["chem"].float()
    if "coords" in t:
        t["coords"] = t["coords"].float()
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


def assert_structure_supervision(ds, weight: float, n: int = 64) -> None:
    """Refuse to train head 3 on a corpus that carries no coordinates.

    The shards written before the diffusion head existed have no `coords`
    array, and every layer below here degrades politely when it is missing:
    `ShardReader` skips the field, `pad_batch` leaves the mask false, and
    `step_losses` sees nothing to supervise. The result is a run that looks
    healthy, reports a falling loss, and never trains the head that produces
    the structure -- which is the failure this project has the least chance of
    noticing, because the other nine heads keep improving.

    So it is checked once, loudly, at startup.
    """
    if weight <= 0:
        return
    seen = 0
    for i in range(min(n, len(ds))):
        m = ds[i].get("coord_mask")
        if m is not None and bool(np.asarray(m).all(-1).any()):
            seen += 1
    if seen == 0:
        raise SystemExit(
            f"[pharos] --structure-weight {weight} but none of the first {n} "
            "chains carries backbone coordinates. The shards predate the "
            "diffusion head; rebuild them with scripts/build_dataset.py, or "
            "pass --structure-weight 0 to train the other heads only.")
    print(f"[pharos] structure supervision: {seen}/{min(n, len(ds))} of the "
          f"sampled chains carry backbone coordinates", flush=True)


def lookup_coevolution(t: Dict, bidx: torch.Tensor, ii: torch.Tensor,
                       jj: torch.Tensor, L: int) -> Optional[torch.Tensor]:
    """The coupling score at each sampled pair, 0 where there is none.

    A dense (B, L, L) coupling tensor would be 79 MB for the corpus's longest
    chain alone, so the batch carries a SORTED flat key array instead and this
    is a binary search into it -- one `searchsorted` for every pair in the
    batch, on GPU, rather than a Python lookup per pair.

    Absent is zero, which is only safe because `coev_proj` is zero-initialised
    and the score is non-negative: "no coupling measured" and "a coupling of
    zero" enter the model identically, and neither is allowed to look like
    evidence against contact.
    """
    key = t.get("coev_key")
    if key is None or key.numel() == 0:
        return None
    val = t["coev_val"]
    lo = torch.minimum(ii, jj).to(torch.int64)
    hi = torch.maximum(ii, jj).to(torch.int64)
    want = bidx.to(torch.int64) * L * L + lo * L + hi
    pos = torch.searchsorted(key, want).clamp(max=key.numel() - 1)
    hit = key[pos] == want
    return torch.where(hit, val[pos], torch.zeros_like(val[pos]))


def step_losses(model: Pharos, t: Dict, cfg, n_neg: int,
                n_loops: Optional[int] = None,
                structure_weight: float = 1.0) -> tuple:
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
        cv = lookup_coevolution(t, bidx, ii, jj, L)
        if cv is not None:
            pair = pair + model.coev_proj(cv.unsqueeze(-1).to(pair.dtype))
        if model.motifs is not None:
            r, _ = model.motifs(pair)
            pair = pair + model.motif_mix(r)
        logit = model.heads.pair.contact(pair).squeeze(-1)
        per = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        l = (per * wt).sum() / wt.sum().clamp(min=1e-6)
        total = total + 1.0 * l
        parts["contact"] = float(l.detach())

    # head 3 -- the backbone itself, by denoising diffusion.
    #
    # This is the head that was missing. Everything above supervises a PROPERTY
    # of the structure -- which residues touch, where the magnesium sits, how
    # rigid a region is -- and none of it produces coordinates. A contact map is
    # not a structure: many geometries satisfy the same contacts, and the
    # ambiguity is exactly what the other heads cannot resolve.
    #
    # Diffusion rather than regressing coordinates directly, because a
    # structure is defined only up to a rigid motion, so there is no single
    # correct coordinate to regress towards; a squared error against one
    # arbitrary frame trains the model towards the mean of the orbit, which is
    # the centroid and not a structure. The denoiser is trained on randomly
    # rotated copies (SO(3), never a reflection -- a mirrored RNA has a
    # left-handed helix and preserves every distance, so a distance check
    # cannot catch it), so the orbit is the thing it learns.
    cm = t.get("coord_residue_mask")
    if cm is not None and bool(cm.any()):
        # per-chain quality weighting, the same weight the contact head uses:
        # a 3.5 A structure is not evidence in the way a 1.9 A one is
        dl = model.heads.structure.loss(
            t["coords"].to(out["hidden"].dtype), out["hidden"], None, cm)
        total = total + structure_weight * (dl["loss"] * w.mean())
        parts["structure"] = float(dl["loss"].detach())
        parts["structure_mse"] = float(dl["mse"])

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
    ap.add_argument("--structure-weight", type=float, default=1.0,
                    help="weight on the diffusion backbone loss (head 3). The "
                         "EDM sigma-weighting already normalises across noise "
                         "scales, so this only trades head 3 against the "
                         "property heads.")
    ap.add_argument("--n-neg", type=int, default=2048)
    ap.add_argument("--size", default="mini",
                    choices=("mini", "small", "base400", "shared400"),
                    help="ignored when --init-from names a checkpoint that "
                         "records its own config, which is the curriculum case")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=100,
                    help="steps between checkpoints. Epoch-end only meant an "
                         "interrupted epoch lost everything, and the next fire "
                         "restarted from the PREVIOUS stage's weights")
    ap.add_argument("--restart", action="store_true",
                    help="ignore an existing checkpoint; it is RENAMED")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint path; default pharos_<size>.pt")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="a checkpoint from an earlier curriculum stage")
    ap.add_argument("--from-scratch", action="store_true",
                    help="train stage 5 from random weights. This is a known "
                         "bad configuration -- it produced r = 0.049 on unseen "
                         "folds -- so it has to be asked for explicitly.")
    ap.add_argument("--sample-loops", action="store_true", default=True,
                    help="sample the recycle count per step (default on)")
    ap.add_argument("--fixed-loops", dest="sample_loops", action="store_false")
    args = ap.parse_args()

    device = require_gpu(args)
    enable_gpu_fast_paths()
    rng = np.random.default_rng(0)
    # THE ARCHITECTURE COMES FROM THE CHECKPOINT WE ARE FINE-TUNING.
    #
    # Stage 5 is the last step of a curriculum: it is supposed to refine what
    # stage 1 built, so its architecture is not a free choice. Hardcoding
    # `small` here while stage 1 trained `shared400` meant --init-from could
    # only ever fail -- 149M/61M against 394M/302M, every shape different --
    # and the cron runner passes `--size small` on every fire. The existing
    # guard would have caught it as "too many missing keys", loudly and
    # uselessly, at the start of a job queued behind sixty hours of stage 1.
    #
    # Stage 1 records `cfg` in its checkpoint. That is the authority.
    cfg = getattr(PharosConfig, args.size)()
    if args.init_from and Path(args.init_from).exists():
        try:
            _saved = torch.load(args.init_from, map_location="cpu",
                                weights_only=False).get("cfg")
        except Exception:                                    # noqa: BLE001
            _saved = None
        if isinstance(_saved, dict):
            _from_ckpt = PharosConfig(**{k: v for k, v in _saved.items()
                                         if k in PharosConfig.__dataclass_fields__})
            for k, v in _saved.items():                      # non-field attrs
                if not hasattr(_from_ckpt, k):
                    setattr(_from_ckpt, k, v)
            if _from_ckpt.__dict__ != cfg.__dict__:
                print(f"[pharos] --size {args.size} overridden by the config in "
                      f"{Path(args.init_from).name}: d_model {_from_ckpt.d_model}, "
                      f"{_from_ckpt.n_blocks} blocks, {_from_ckpt.n_experts} experts",
                      flush=True)
            cfg = _from_ckpt
    tr = Pharos3DDataset(args.data, split="train", max_length=args.max_length)
    va = Pharos3DDataset(args.data, split="val", max_length=args.max_length)
    te = Pharos3DDataset(args.data, split="test", max_length=args.max_length)
    tb = Pharos3DDataset(args.data, split="test_ribosomal", max_length=args.max_length)
    print(f"[pharos] train {len(tr):,} | val {len(va):,} | test {len(te):,} "
          f"| test_ribosomal {len(tb):,}")
    for d in (tr, va, te, tb):
        d.prewarm()
    print(f"[pharos] shards resident: {tr.prewarm(verbose=True):.2f} GiB RSS")
    assert_structure_supervision(tr, args.structure_weight)

    model = Pharos(cfg).to(device)
    # §12.1 is a CURRICULUM: stage 5 is meant to fine-tune the representation
    # stages 1-3 built, and it is explicitly "last and briefest". Run from
    # random weights it is neither -- it is a short run on 7,653 chains with no
    # representation to build on, which is what produced r = 0.049 on unseen
    # folds. Chaining is the difference between a curriculum and four unrelated
    # runs.
    CKPT.mkdir(parents=True, exist_ok=True)
    ck = args.ckpt or (CKPT / f"pharos_{args.size}.pt")
    if ck.exists() and args.restart:
        keep = ck.with_suffix(".superseded.pt")
        ck.replace(keep)
        print(f"[pharos] --restart: {ck.name} moved to {keep.name}", flush=True)
    resume = torch.load(ck, map_location=device) if ck.exists() else None
    if resume is not None and resume.get("format") != CKPT_FORMAT:
        # A checkpoint written before resume existed carries weights and an
        # epoch number and nothing else. `pharos_small.pt` on this machine is a
        # COMPLETED 8-epoch run from the random-init experiment -- reading its
        # epoch would either skip stage 5 or resume the experiment the
        # curriculum was built to replace. Version the format and ignore what
        # predates it.
        print(f"[pharos] {ck.name} predates resumable checkpoints "
              f"(no format tag); ignoring it and honouring --init-from",
              flush=True)
        resume = None
    if resume is not None:
        # RESUME BEATS --init-from: the runner passes --init-from every fire.
        model.load_state_dict(resume["model"])
        print(f"[pharos] resumed from {ck.name}: epoch {resume.get('epoch')}, "
              f"step {resume.get('step', 0):,}", flush=True)
    elif args.init_from and Path(args.init_from).exists():
        sd = torch.load(args.init_from, map_location=device)
        res = model.load_state_dict(sd["model"], strict=False)
        n_loaded = len(sd["model"]) - len(res.unexpected_keys)
        print(f"[pharos] initialised from {Path(args.init_from).name}: "
              f"{n_loaded:,} tensors loaded, {len(res.missing_keys)} fresh, "
              f"{len(res.unexpected_keys)} ignored"
              + (f", after {sd['tokens']/1e9:.3f}B pretraining tokens"
                 if "tokens" in sd else ""), flush=True)
        if len(res.missing_keys) > 0.5 * len(list(model.state_dict())):
            raise SystemExit(
                f"{len(res.missing_keys)} of {len(list(model.state_dict()))} "
                f"tensors did not load -- that is a different architecture, not "
                f"a checkpoint. Refusing to train on a mostly-random model that "
                f"reports as initialised.")
    elif args.init_from and resume is None:
        raise SystemExit(f"--init-from {args.init_from} does not exist")
    elif not args.from_scratch:
        # NO RESUME AND NO PRETRAINED WEIGHTS. Stage 5 is the last step of a
        # curriculum and this is the one configuration it must never take
        # silently: this file's own §12.1 note records that stage 5 from random
        # weights is what produced r = 0.049 on unseen folds.
        #
        # It is not hypothetical. The cron runner sets
        #   PRETRAIN=.../pretrain_small.pt ; [ -f "$PRETRAIN" ] && INIT=...
        # and stage 1 at --size shared400 writes pretrain_shared400.pt, so the
        # test fails, INIT stays empty, and stage 5 starts from noise with no
        # error at all -- discarding every token of pretraining while looking
        # like it worked.
        cand = sorted(CKPT.glob("pretrain_*.pt"),
                      key=lambda f: f.stat().st_mtime, reverse=True)
        cand = [c for c in cand if "superseded" not in c.name]
        hint = (f"\n  the newest pretraining checkpoint is {cand[0]}"
                if cand else "\n  no pretrain_*.pt checkpoint exists yet")
        raise SystemExit(
            "stage 5 has no weights to fine-tune: no resumable checkpoint and "
            "no --init-from." + hint + "\n  pass --init-from <that file>, or "
            "--from-scratch if training from noise is genuinely intended.")
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
    history: List[Dict] = []
    step, start_ep, n_oom = 0, 0, 0
    if resume is not None:
        if "opt" in resume:
            opt.load_state_dict(resume["opt"])
        step = int(resume.get("step", 0))
        history = list(resume.get("history", []))
        start_ep = int(resume.get("epoch", 0)) + (1 if resume.get("epoch_done") else 0)
        # OneCycleLR carries an internal step count, so it has to be restored
        # rather than rebuilt -- and only if the schedule is the SAME schedule.
        # `n_steps` depends on the epoch count and the token budget, so a run
        # resumed with either changed would be stepping a different curve.
        if resume.get("n_steps") == n_steps and "sched" in resume:
            sched.load_state_dict(resume["sched"])
        else:
            print(f"[pharos] schedule changed ({resume.get('n_steps')} -> "
                  f"{n_steps} steps); fast-forwarding a fresh one instead",
                  flush=True)
            for _ in range(min(step, n_steps - 1)):
                sched.step()
        if start_ep >= args.epochs:
            print(f"[pharos] all {args.epochs} epochs already done", flush=True)
            return

    runlog = RunLog(ROOT, "stage5_3d", [
        "lr", "loss", "n_oom", "peak_gib", "note",
        "val_contact_ap", "val_mg_ap", "val_rigidity_r", "val_bfactor_r",
    ], manifest={
        "size": args.size, "config": cfg.__dict__, "params": pc,
        "epochs": args.epochs, "token_budget": args.token_budget,
        "lr_peak": args.lr, "n_neg": args.n_neg,
        "sample_loops": args.sample_loops, "n_steps_total": n_steps,
        "batches_per_epoch": [len(b) for b in epoch_batches],
        "splits": {"train": len(tr), "val": len(va)},
        "init_from": str(args.init_from) if args.init_from else None,
        "resumed_from_epoch": start_ep, "resumed_from_step": step,
        "checkpoint": str(ck),
    })
    if step:
        runlog.event("resumed", epoch=start_ep, step=step)

    def save(ep: int, done: bool, val=None) -> None:
        atomic_save({"format": CKPT_FORMAT, "cfg": cfg.__dict__,
                    "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "n_steps": n_steps,
                    "epoch": ep, "epoch_done": done, "step": step,
                    "history": history, "val": val}, ck)

    for ep in range(start_ep, args.epochs):
        model.train()
        t0, run = time.time(), []
        for idxs in epoch_batches[ep]:
            t = to_device(tr.collate(idxs), device)
            # sampled recycling: 1..max_loops, uniform
            nl = int(rng.integers(1, cfg.n_loops + 1)) if args.sample_loops \
                else cfg.n_loops
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, parts, _ = step_losses(
                        model, t, cfg, args.n_neg, n_loops=nl,
                        structure_weight=args.structure_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            except torch.OutOfMemoryError:
                # Chain length here runs to 4,298 and the batches are packed to
                # a token budget, so one long-chain batch can cost several times
                # the median. Skipping it costs one gradient; dying costs the
                # epoch, and -- before resume existed -- the whole stage.
                # `sched.step()` still runs, because OneCycleLR's total was
                # computed from the batch COUNT and skipping a batch must not
                # desynchronise the curve from it.
                n_oom += 1
                opt.zero_grad(set_to_none=True)
                del t
                torch.cuda.empty_cache()
                sched.step()
                step += 1
                if n_oom <= 5 or n_oom % 50 == 0:
                    print(f"[pharos] OOM #{n_oom} at step {step}; batch skipped",
                          flush=True)
                runlog.event(f"OOM #{n_oom}", epoch=ep, step=step, n_oom=n_oom)
                continue
            sched.step()
            run.append(float(loss.detach()))
            step += 1
            if step % args.ckpt_every == 0:
                save(ep, False)
            if step % args.log_every == 0:
                runlog.log("step", epoch=ep, step=step,
                           lr=sched.get_last_lr()[0],
                           loss=round(float(np.mean(run[-args.log_every:])), 6),
                           n_oom=n_oom,
                           peak_gib=round(
                               torch.cuda.max_memory_allocated() / 2**30, 2),
                           **{f"part_{k}": round(float(v), 6)
                              for k, v in parts.items() if k != "n_pairs"})
                print(f"[pharos] ep{ep} step{step} loss "
                      f"{np.mean(run[-args.log_every:]):.4f} "
                      f"{ {k: round(v, 3) for k, v in parts.items() if k != 'n_pairs'} }",
                      flush=True)
        ev = evaluate(model, va, device, cfg, args.eval_batches, args.token_budget)
        history.append({"epoch": ep, "loss": float(np.mean(run)), "val": ev})
        print(f"[pharos] epoch {ep}: loss {np.mean(run):.4f} val {ev} "
              f"({time.time()-t0:.0f}s)", flush=True)
        runlog.log("epoch", epoch=ep, step=step,
                   loss=round(float(np.mean(run)), 6), n_oom=n_oom,
                   lr=sched.get_last_lr()[0],
                   **{f"val_{k}": v for k, v in ev.items()
                      if isinstance(v, (int, float))})
        save(ep, True, ev)

    if not history:
        # A run that trained no epochs has nothing to report, and writing the
        # report anyway CLOBBERS the last real one. A guard test run with
        # --epochs 0 overwrote eight epochs of pinned stage-5 results with an
        # empty history, and four checks in verify_claims.py went red.
        print("[pharos] no epochs ran; leaving the existing results file alone",
              flush=True)
        return
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
