#!/usr/bin/env python3
"""Multi-task training for PHAROS on the 3D corpus (stage 5 of §12.1).

Stages 1-3 of the curriculum (sequence MLM, 2D, probing) run on corpora that
are not the 3D set; this trains the structural stage and the heads whose labels
come out of the deposited files themselves. It is the stage that uses everything
`build_dataset.py` extracts:

    head 1   contact          contact set, 8 A, |i-j| >= 4
    head 2   distance         binned C4'-C4', 2-40 A
    head 3   structure        backbone by EDM diffusion
    head 6   Mg2+ sites       residues within 3 A of a magnesium
    head 7   rigidity         normalised B-factor -- X-RAY ONLY (D12)
    head 9   geometry         Leontis-Westhof class, per pair
    head 10  motif            hairpin / internal / neither, per residue
    extra    base identity    the N_struct residues
    ensemble fluctuation      supervised through the rigidity target

Numbered per `pharos.model.heads.HEAD_SPEC`, which is §9's table as data.
This block used the numbering §9 ABANDONED -- 5 Mg, 6 rigidity, 10 base
identity -- so for every head it named, the number was wrong, and the two
schemes have been read against each other in this project more than once.

**`disorder_logit` is NOT trained here**, although this block used to say it
was and `unobserved_seq_id` is in every shard. It cannot be, as the corpus
stands: the label names polymer positions that were never modelled, and the
tokens are the modelled residues only, so a disorder target aligned to them
is vacuously zero -- which `mmcif_entities.residue_labels` says in as many
words. Supervising it needs the corpus to carry the full `entity_poly_seq`
and an observed/unobserved flag per position, which is a build change, not a
loss term. `splice_logits` is likewise emitted and untrained here, and so is
`ensemble_state_logits`; `src/pharos/model/test_pharos.py` property 7c pins
the whole emitted surface as trained / indirect / untrained-with-a-reason, so
the set cannot grow quietly. `fitness` was on that list and no longer is:
stage 6 trains it in `train_sequence_stages.py`, not here.

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
from pharos.model.moe import (LENGTH_BIN_MAX as _LENGTH_BIN_MAX,  # noqa: E402
                              RouterFeatures)
from pharos.model.diffusion import BOND_C4_N, BOND_P_P
from pharos.model.pharos import Pharos, PharosConfig
from pharos.train.telemetry import RunLog
from pharos.train.checkpoint import atomic_save              # noqa: E402
from pharos.train.guard import check_loss

sys.path.insert(0, str(ROOT / "scripts"))
from train_block_scorer import gpu_free_gib                       # noqa: E402


def require_gpu(args) -> torch.device:
    if not args.device.startswith("cuda"):
        # `--smoke` is the one exception, and it exists because stage 5 has
        # NEVER RUN. Every startup check in this file -- the structure-
        # supervision assertion, the coevolution-reach assertion, the config
        # read out of the checkpoint, the ten heads finding their targets --
        # is unexercised code guarding an unexercised path, and the only way
        # to find out whether it starts has been to displace a multi-day
        # pretraining run on the single card. A CPU smoke test that runs a
        # handful of steps at a toy batch costs nothing and answers the
        # question. It is not training: it refuses to write a checkpoint.
        if getattr(args, "smoke", 0):
            print("[pharos] SMOKE TEST on CPU: startup path only, "
                  "no checkpoint will be written", flush=True)
            return torch.device(args.device)
        raise SystemExit("GPU only; pass --device cuda once one is free. "
                         "For a startup check without a GPU use "
                         "--smoke N --device cpu.")
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
                  "coev_key", "coev_val", "lw_key", "lw_val",
                  "loop_class", "loop_mask")}
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

    `length_bin_max` matters here and not in stage 1. The default octave bins
    are `floor(log2(L/32))` clamped to `n_bins - 1`, which makes the top bin a
    catch-all for everything above 1,024 -- and this corpus runs to 4,450, so
    measured on the 3D training split that one bin takes **40.0% of chains**.
    The long-chain regime the hierarchical pair track exists for was a single
    indicator. Spread over [32, 4608] the occupancy is 15.7/39.9/3.0/1.2/21.7/
    18.6% and the conditioning entropy goes 1.890 -> 2.100 bits of the 2.585
    available.
    """
    dev = t["tokens"].device
    B = t["tokens"].shape[0]
    in_cx = torch.tensor([1.0 if m.get("has_protein") else 0.0 for m in t["meta"]],
                         device=dev)
    chem_summary = t["chem"].sum(1) / t["mask"].sum(1, keepdim=True).clamp(min=1)
    return RouterFeatures(
        length=torch.as_tensor(t["lengths"], dtype=torch.float32, device=dev),
        in_complex=in_cx, chem_summary=chem_summary[:, :5], recycle=recycle,
        length_bin_max=LENGTH_BIN_MAX)


#: Re-exported from `pharos.model.moe`, which is where the binning it
#: parametrises lives. Defining it here is what let inference diverge.
LENGTH_BIN_MAX = _LENGTH_BIN_MAX


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


def assert_coevolution_reaches_the_model(tr, n_batches: int = 4,
                                         n_chains: int = 12) -> None:
    """Refuse to claim coevolution is an input when no pair ever receives one.

    Same shape as `assert_structure_supervision`, for the feature with the
    quietest failure of any in the model. §4A says the pair track reads
    coevolutionary couplings; the path from that claim to the arithmetic runs
    through a bare `except Exception: return None`, a `searchsorted` whose
    misses are filled with zeros, and a zero-initialised projection. Every one
    of those degrades to "adds nothing" without raising, so the only difference
    between a working coevolution feature and a completely absent one is a
    number nobody was printing.

    Checked once at startup on real batches, and reported even when it passes,
    because a rate that drifts from 3.5% to 0.3% is the same failure arriving
    slowly.
    """
    rng = np.random.default_rng(0)
    tot = hit = 0
    for _ in range(n_batches):
        idxs = rng.choice(len(tr), min(n_chains, len(tr)), replace=False).tolist()
        t = to_device(tr.collate(idxs), torch.device("cpu"))
        B, L = t["tokens"].shape
        I, J, Bi = [], [], []
        for b in range(B):
            ii, jj, _ = sample_pairs(t["contacts"][b], int(t["lengths"][b]), 8,
                                     torch.device("cpu"))
            if ii is None or len(ii) == 0:
                continue
            I.append(ii)
            J.append(jj)
            Bi.append(torch.full((len(ii),), b, dtype=torch.long))
        if not I:
            continue
        cv = lookup_coevolution(t, torch.cat(Bi), torch.cat(I), torch.cat(J), L)
        tot += int(len(torch.cat(I)))
        if cv is not None:
            hit += int((cv != 0).sum())
    frac = hit / max(tot, 1)
    if hit == 0:
        raise SystemExit(
            f"[pharos] coevolution reaches NO sampled pair in {n_batches} "
            f"batches ({tot:,} pairs). Either the cache under "
            f"data/derived/coevolution is missing, the chains carry no Rfam "
            f"family, or `_coevolution_for` is swallowing an exception. "
            f"`coev_proj` is zero-initialised, so this trains silently and "
            f"looks identical to a working feature.")
    print(f"[pharos] coevolution: {hit:,} of {tot:,} sampled pairs carry a "
          f"coupling ({100 * frac:.2f}%; 3.54% on the built corpus)", flush=True)


def assert_supervised_heads_have_targets(tr, n_batches: int = 4,
                                         n_chains: int = 12) -> None:
    """Refuse to train heads 9 and 10 on nothing, or on one class.

    Both degrade silently. Head 9's loss is skipped entirely when no sampled
    pair carries an annotation; head 10's is skipped when `loop_mask` is empty.
    A skipped loss costs nothing, raises nothing, and leaves a head in the
    model that reports metrics on the batches where it does fire -- so a
    corpus rebuild that drops `lw_pairs`, or a mask that tightens too far,
    produces a run that looks complete and trains eight heads.

    The single-class check matters as much as the presence one. Head 10's
    mask used to be true for every residue of every chain, including the
    3,960 `pharos3d` chains with no base-pair annotation at all, whose
    3,715,593 residues all read "not in a loop": present, plentiful, and
    28.2% of them meaningless. Targets that are all one class are the same
    failure with a full tensor.
    """
    rng = np.random.default_rng(0)
    n_lw = n_loop = 0
    lw_classes: set = set()
    loop_classes: set = set()
    masked = total = 0
    for _ in range(n_batches):
        idxs = rng.choice(len(tr), min(n_chains, len(tr)), replace=False).tolist()
        t = to_device(tr.collate(idxs), torch.device("cpu"))
        lv = t.get("lw_val")
        if lv is not None and lv.numel():
            n_lw += int(lv.numel())
            lw_classes |= set(lv.unique().tolist())
        lm = t.get("loop_mask")
        if lm is not None:
            masked += int(lm.sum())
            total += int(t["mask"].sum())
            if bool(lm.any()):
                lc = t["loop_class"][lm]
                n_loop += int(lc.numel())
                loop_classes |= set(lc.unique().tolist())
    if n_lw == 0:
        raise SystemExit(
            "[pharos] head 9 (Leontis-Westhof) receives NO target in "
            f"{n_batches} batches. The corpus carries no `lw_pairs`, or "
            "`pad_batch` is not emitting `lw_key`/`lw_val`. The loss is "
            "skipped when this happens, so the run would complete with the "
            "head untrained and nothing said.")
    if n_loop == 0:
        raise SystemExit(
            "[pharos] head 10 (motif class) receives NO target in "
            f"{n_batches} batches: `loop_mask` is empty everywhere. It is "
            "gated on the chain carrying a cis Watson-Crick pair, so either "
            "the corpus lost its annotation or the gate is wrong.")
    if len(loop_classes) < 2:
        raise SystemExit(
            f"[pharos] head 10 sees ONE class ({loop_classes}) across "
            f"{n_loop:,} labelled residues. Cross-entropy against a constant "
            "target trains a bias and reads as high accuracy.")
    print(f"[pharos] head 9: {n_lw:,} pair targets, {len(lw_classes)} of 13 "
          f"classes present", flush=True)
    print(f"[pharos] head 10: {masked:,} of {total:,} residues labelled "
          f"({100 * masked / max(total, 1):.1f}%; unannotated chains are "
          f"masked out), {len(loop_classes)} of 3 classes present", flush=True)


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



def _class_metrics(logits: torch.Tensor, target: torch.Tensor, tag: str,
                   n_classes: int) -> Dict[str, float]:
    """Accuracy, the majority-class rate it must beat, and macro recall.

    Heads 9 and 10 reported bare accuracy, and bare accuracy on these targets
    is not a measurement. Over the corpus, 76.57% of annotated base pairs are
    one Leontis-Westhof class (cis Watson-Crick) and **64.4%** of residues
    whose loop class is actually known carry one; a head that has learned
    nothing except the prior scores those numbers and reads as a working head.

    64.4%, not 74.42%. The higher figure came from counting the 3,715,593
    residues of the 3,960 `pharos3d` chains that carry no base-pair annotation
    at all, every one of them labelled "not in a loop" by a mask that meant
    "this chain has a loop_class array" rather than "this label is known".
    `pad_batch` no longer labels them, so the floor this head is measured
    against dropped ten points -- in the direction that makes the head's job
    harder and the number honest. This is the same failure the
    rest of the audit has been chasing -- a component that exists, is measured,
    and whose output is never checked for basic validity -- so the metric is
    reported against its own floor.

    `*_major` is the majority rate ON THIS BATCH, which is the honest floor:
    the corpus figure would let a batch of unusual composition look good or
    bad for reasons that have nothing to do with the head. `*_lift` is what the
    head adds over it and is the number to watch; at or below zero the head is
    predicting the prior. `*_macro` is unweighted mean recall over the classes
    PRESENT in the batch, which collapses to 1/k for a prior-predicting head
    and is the metric the rare tertiary geometries actually live in.
    """
    with torch.no_grad():
        pred = logits.argmax(-1)
        acc = (pred == target).float().mean()
        cnt = torch.bincount(target, minlength=n_classes).float()
        major = (cnt.max() / cnt.sum()) if cnt.sum() > 0 else cnt.new_zeros(())
        present = cnt > 0
        hit = torch.bincount(target[pred == target], minlength=n_classes).float()
        macro = (hit[present] / cnt[present]).mean() if bool(present.any()) \
            else cnt.new_zeros(())
        return {f"{tag}_acc": float(acc),
                f"{tag}_major": float(major),
                f"{tag}_lift": float(acc - major),
                f"{tag}_macro": float(macro),
                f"{tag}_n_class": int(present.sum())}



#: Every column the two `runlog.log` calls below actually pass.
#:
#: Declared nine and passed forty. `extrasaction="ignore"` meant the step rows
#: carried `lr`, `loss`, `n_oom` and `peak_gib` and dropped all nineteen
#: `part_*` losses and every metric the 09-25 audit added; the four `val_*`
#: names that WERE declared -- `val_contact_ap`, `val_mg_ap`,
#: `val_rigidity_r`, `val_bfactor_r` -- match nothing `evaluate()` returns, so
#: they were four columns that could never have been filled. Both halves of
#: the mismatch in one list.
#:
#: Derived from the tags rather than typed, so adding a head to
#: `_TAGGED_CLASS` grows the csv instead of quietly not growing it.
_TAGGED_CLASS = ("lw", "motif", "base")      # _class_metrics
_TAGGED_BIN = ("mg",)                        # _binary_metrics
_TAGGED_REG = ("rigidity", "fluct")          # _regression_metrics
_PART_SCALARS = ("contact", "distance", "dist_acc", "structure",
                 "structure_mse", "mg", "rigidity", "fluctuation", "base",
                 "lw", "motif", "balance", "coev_frac", "coev_norm",
                 "motif_gate", "motif_eff", "motif_top_share", "n_lw")
_EVAL_KEYS = ("contact_ap", "contact_ap_lift", "contact_base_rate",
              "contact_n", "coev_frac", "motif_eff",
              "lw_acc", "lw_major", "lw_lift", "lw_macro", "lw_n_class",
              "structure_loss", "structure_mse", "structure_violation",
              "structure_bond_cn", "structure_bond_pp",
              "bond_cn_target_a", "bond_pp_target_a",
              "mg_base_rate", "mg_average_precision", "mg_ap_lift",
              "mg_precision_at_calibrated_thr", "mg_recall_at_calibrated_thr",
              "rigidity_r_pooled", "rigidity_n",
              "motif_acc", "motif_major", "motif_lift", "motif_macro",
              "motif_n_class")

def _stage5_fields() -> list:
    f = ["lr", "loss", "n_oom", "peak_gib", "note"]
    parts = list(_PART_SCALARS)
    for t in _TAGGED_CLASS:
        parts += [f"{t}_{x}" for x in ("acc", "major", "lift", "macro", "n_class")]
    for t in _TAGGED_BIN:
        parts += [f"{t}_{x}" for x in ("pos_rate", "auroc")]
    for t in _TAGGED_REG:
        parts += [f"{t}_{x}" for x in ("r", "base", "pred_sd")]
    f += [f"part_{k}" for k in dict.fromkeys(parts)]
    f += [f"val_{k}" for k in _EVAL_KEYS]
    return list(dict.fromkeys(f))


STAGE5_FIELDS = _stage5_fields()


def _binary_metrics(logit: torch.Tensor, target: torch.Tensor,
                    tag: str) -> Dict[str, float]:
    """AUROC and the positive rate, for a head whose loss has no floor.

    A weighted BCE on a 4.7%-positive target is not a measurement on its own:
    it falls when the head learns the prior, it falls again when `pos_weight`
    changes, and neither movement says the head found an Mg site. AUROC is a
    rank statistic -- chance is 0.5 whatever the imbalance and whatever the
    weighting -- so it is the floor this head never had. Computed by the
    Mann-Whitney identity rather than by sorting thresholds, which is one
    argsort.
    """
    with torch.no_grad():
        y = (target > 0.5)
        npos = int(y.sum())
        nneg = int((~y).sum())
        if npos == 0 or nneg == 0:
            return {f"{tag}_pos_rate": float(y.float().mean()),
                    f"{tag}_auroc": float("nan")}
        r = torch.empty_like(logit, dtype=torch.float)
        r[logit.argsort()] = torch.arange(logit.numel(), device=logit.device,
                                          dtype=torch.float) + 1.0
        auc = (r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
        return {f"{tag}_pos_rate": float(y.float().mean()),
                f"{tag}_auroc": float(auc)}


def _regression_metrics(pred: torch.Tensor, target: torch.Tensor,
                        tag: str) -> Dict[str, float]:
    """Correlation, the constant-predictor loss, and the prediction's spread.

    The rigidity target is a z-scored B-factor, so predicting zero everywhere
    is already a respectable smooth-L1 and the reported loss cannot distinguish
    that from a head that has learned something. `_r` is the Pearson
    correlation, which is 0 for any constant prediction however well tuned, and
    `_base` is the constant-mean predictor's loss so the reported figure has a
    number to be better than.

    `_pred_sd` is the third, and it is the one that says WHICH kind of zero a
    zero correlation is. When the prediction is constant the denominator
    vanishes and this function returns `r = 0.0` -- the same value it returns
    for a head that varies freely and happens to be uncorrelated. A collapsed
    head and an unlucky one were indistinguishable in the time series, which
    is the defect stage 6's `DEAD_PRED_SD` exists for, one stage over.
    `rigidity_r_pooled` in `evaluate()` already returns None rather than 0 in
    the same situation; the per-step metric did not.
    """
    with torch.no_grad():
        p = pred.float().flatten()
        y = target.float().flatten()
        base = float(F.smooth_l1_loss(y.mean().expand_as(y), y))
        pc = p - p.mean()
        yc = y - y.mean()
        den = pc.norm() * yc.norm()
        r = float((pc * yc).sum() / den) if float(den) > 1e-12 else 0.0
        return {f"{tag}_r": r, f"{tag}_base": base,
                f"{tag}_pred_sd": float(p.std()) if p.numel() > 1 else 0.0}


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

    # head 6 -- Mg2+ sites. Heavily imbalanced (4.7% positive), and a missed
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
        parts.update(_binary_metrics(out["mg_logit"][m].detach(), tgt[m], "mg"))

    # head 7 -- rigidity. D12: X-ray only, and the mask is what enforces it.
    rm = t["rigidity_mask"]
    if want_dyn and rm.any():
        l = F.smooth_l1_loss(out["rigidity"][rm], t["b_factor_z"][rm].float())
        total = total + 0.3 * l
        parts["rigidity"] = float(l.detach())
        parts.update(_regression_metrics(out["rigidity"][rm].detach(),
                                         t["b_factor_z"][rm], "rigidity"))
        # the ensemble's fluctuation amplitude predicts the same observable, so
        # it is supervised by it -- that is what makes the stiffness field
        # trainable without any measured stiffness
        # The target has to be NON-NEGATIVE, because a fluctuation amplitude
        # is. The shift that made it so was `- b_factor_z.min()` over the
        # CURRENT BATCH, so the same residue's target moved with whichever
        # other chains happened to share its batch: measured over 35 stage-5
        # batches the offset ran from -1.225 to -6.051, a spread of 4.83
        # against a signal whose standard deviation is 1.0 because
        # `b_factor_z` is z-scored per chain.
        #
        # It was invisible in the metric printed beside it. `fluct_r` is
        # Pearson, which is invariant to a constant offset, so the
        # correlation read clean while the loss could not converge below the
        # variance of the shift -- and `fluct_base`, the constant-predictor
        # floor, moved with the batch too, so the two drifted together and
        # their ratio looked stable.
        #
        # `softplus` is a FIXED map: monotone, non-negative, and defined by
        # the residue alone. Measured on 200,000 standard normals it gives
        # mean 0.806, sd 0.522, min 0.011 -- against the old shift's mean of
        # about 2.5 and a batch-dependent offset, so it is better conditioned
        # as well as well-defined. `rigidity` three lines above was always
        # batch-independent; these two targets are now both functions of
        # their own residue.
        fl = out["fluctuation"][rm]
        fl_target = F.softplus(t["b_factor_z"][rm].float())
        l2 = F.smooth_l1_loss(fl, fl_target)
        total = total + 0.1 * l2
        parts["fluctuation"] = float(l2.detach())
        parts.update(_regression_metrics(fl.detach(), fl_target, "fluct"))

    # base identity (unnumbered in §9), exactly where it was never assigned
    bm = t["base_mask"]
    if bm.any():
        l = F.cross_entropy(out["base_logits"][bm], t["tokens"][bm].clamp(max=3))
        total = total + 0.2 * l
        parts["base"] = float(l.detach())
        parts.update(_class_metrics(out["base_logits"][bm].detach(),
                                    t["tokens"][bm].clamp(max=3).long(),
                                    "base", 4))

    # head 10 -- motif class: hairpin loop, internal loop, or neither.
    #
    # The motif bank has always RETRIEVED a motif and mixed its descriptor into
    # the representation without ever committing to an answer that could be
    # scored. This makes the bank's contribution falsifiable. Targets come from
    # the canonical subset of the depositor's own base-pair annotation, and the
    # loop lengths they imply check out against biology: median 4, the
    # tetraloop, with 89% between 4 and 8 nucleotides.
    lm = t.get("loop_mask")
    if lm is not None and bool(lm.any()):
        ml = out["motif_logits"]
        l = F.cross_entropy(ml[lm], t["loop_class"][lm])
        total = total + 0.3 * l
        parts["motif"] = float(l.detach())
        # `ml.shape[-1]`, not `cfg.n_motif_classes`. That attribute lives on
        # HeadConfig, not on the PharosConfig this function is handed, so the
        # first version of this raised AttributeError on the first step -- in a
        # stage that has never run, so nothing caught it until `--smoke` did.
        # The logits' own width is the class count by construction and cannot
        # drift from the head.
        parts.update(_class_metrics(ml[lm], t["loop_class"][lm], "motif",
                                    ml.shape[-1]))

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
        # head 9 -- Leontis-Westhof geometry class, on the pairs that ARE
        # annotated base pairs. A contact says two residues touch and a
        # distance says how far apart; neither says whether the pair builds a
        # helix (cis WC/WC) or a tertiary contact, and those have the same
        # C4'-C4' distance. Unannotated pairs are excluded by the mask rather
        # than given a class: not-a-base-pair is not a kind of base pair.
        lwk = t.get("lw_key")
        if lwk is not None and lwk.numel():
            lo = torch.minimum(ii, jj).to(torch.int64)
            hi = torch.maximum(ii, jj).to(torch.int64)
            want = bidx.to(torch.int64) * L * L + lo * L + hi
            pos = torch.searchsorted(lwk, want).clamp(max=lwk.numel() - 1)
            hit = lwk[pos] == want
            if bool(hit.any()):
                lwl = model.heads.pair.geometry(pair[hit])
                tgt_lw = t["lw_val"][pos[hit]]
                l = F.cross_entropy(lwl, tgt_lw)
                total = total + 0.5 * l
                parts["lw"] = float(l.detach())
                parts.update(_class_metrics(lwl, tgt_lw, "lw",
                                            lwl.shape[-1]))
                parts["n_lw"] = int(hit.sum())

        cv = lookup_coevolution(t, bidx, ii, jj, L)
        # How often the lookup actually HITS, and whether the projection has
        # left zero. Neither was reported, and both have to be, because the
        # feature's failure mode is silence: `lookup_coevolution` returns zeros
        # where it finds nothing, `_coevolution_for` swallows every exception
        # and returns None, and `coev_proj` is zero-initialised -- so a broken
        # cache, a renamed Rfam family or a typo inside that try block all
        # produce exactly the same arithmetic as a working feature contributing
        # nothing, and the contact loss falls either way. Measured on the built
        # corpus the hit rate is 3.54% of sampled pairs; a run reading 0.00%
        # has lost coevolution entirely and nothing else would say so.
        if cv is not None:
            parts["coev_frac"] = float((cv != 0).float().mean().detach())
            parts["coev_norm"] = float(model.coev_proj.weight.detach().norm())
            pair = pair + model.coev_proj(cv.unsqueeze(-1).to(pair.dtype))
        else:
            parts["coev_frac"] = 0.0
        if model.motifs is not None:
            r, minfo = model.motifs(pair)
            # `r, _ = ...` is what this used to be. The bank computes its own
            # gate and confidence every call and the only caller that trains it
            # threw them away, so a bank that returned the same motif for every
            # pair -- a bias term wearing 667 descriptors -- would have looked
            # exactly like a working one.
            parts["motif_eff"] = float(minfo["effective_motifs"])
            parts["motif_top_share"] = float(minfo["top_motif_share"])
            parts["motif_gate"] = float(minfo["gate_mean"])
            pair = pair + model.motif_mix(r)
        logit = model.heads.pair.contact(pair).squeeze(-1)
        per = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        l = (per * wt).sum() / wt.sum().clamp(min=1e-6)
        total = total + 1.0 * l
        parts["contact"] = float(l.detach())

        # head 2 -- the distogram. It existed, its loss existed, and nothing
        # ever computed a target for it.
        #
        # A binary contact says "these two residues are within a cutoff". A
        # binned distance says how far apart they are, which is strictly more
        # information from the same coordinates and the same sampled pairs: it
        # is what AlphaFold trains its pair track on, and it is the difference
        # between a model that knows two residues touch and one that knows the
        # geometry they touch with. The targets were unavailable until the
        # corpus carried coordinates; now it does.
        cmask = t.get("coord_residue_mask")
        if cmask is not None and bool(cmask.any()):
            xyz = t["coords"][:, :, 1]          # C4', one point per residue
            ok = cmask[bidx, ii] & cmask[bidx, jj]
            if bool(ok.any()):
                dd = torch.linalg.norm(xyz[bidx, ii] - xyz[bidx, jj], dim=-1)
                nb = model.heads.cfg.n_distance_bins
                # bins 0..38 are 1 A wide spanning 2-41 A, bin 39 is
                # everything beyond -- see HeadConfig.n_distance_bins
                b = torch.clamp(((dd - 2.0)).floor().long(), 0, nb - 1)
                dl = model.heads.pair.distance(pair)
                ld = (F.cross_entropy(dl[ok], b[ok], reduction="none")
                      * wt[ok]).sum() / wt[ok].sum().clamp(min=1e-6)
                total = total + 0.5 * ld
                parts["distance"] = float(ld.detach())
                parts["dist_acc"] = float(
                    (dl[ok].argmax(-1) == b[ok]).float().mean().detach())

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
        # A dense (L, L) coupling map for the pair features. The contact head
        # reads coevolution at sampled pairs only; the denoiser attends over
        # everything, so it needs the full map. O(L^2) floats is affordable
        # here because stage 5 caps chains at --max-length, where the pair
        # tensor itself is already O(L^2 * d_pair).
        cv_dense = None
        key = t.get("coev_key")
        if key is not None and key.numel():
            cv_dense = torch.zeros(B, L, L, device=dev, dtype=out["hidden"].dtype)
            bb = torch.div(key, L * L, rounding_mode="floor")
            rem = key - bb * L * L
            ii2 = torch.div(rem, L, rounding_mode="floor")
            jj2 = rem - ii2 * L
            ok = (bb < B) & (ii2 < L) & (jj2 < L)
            v = t["coev_val"].to(cv_dense.dtype)
            cv_dense[bb[ok], ii2[ok], jj2[ok]] = v[ok]
            cv_dense = cv_dense + cv_dense.transpose(1, 2)   # couplings are symmetric
        pair_feat = model.diff_pair(out["hidden"], cv_dense)
        dl = model.heads.structure.loss(
            t["coords"].to(out["hidden"].dtype), out["hidden"], pair_feat, cm)
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
        # `enabled=`, not an unconditional cuda autocast. Every other autocast
        # in this file is guarded; this one was not, so a cpu run entered a
        # cuda autocast region on a machine that may have no cuda at all.
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
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

        # ---- HEAD 1, which had no validation metric either ----------------
        #
        # `val_contact_ap` was a DECLARED telemetry column for a number this
        # function never computed. Contacts are the head the pair track exists
        # for and the one every other head's features route through, and the
        # only thing said about it on held-out data was nothing at all.
        #
        # Scored exactly as training scores it -- the same sampled pairs, the
        # same coevolution lookup, the same motif mixture -- because a
        # validation path that builds the features differently measures a
        # different model. Average precision with the base rate beside it,
        # for the same reason the Mg head is: the positives are a few percent
        # of sampled pairs and accuracy at any threshold is uninformative.
        Bv, Lv = t["tokens"].shape
        ii_l, jj_l, y_l, bi_l = [], [], [], []
        for b in range(Bv):
            Lb = int(t["lengths"][b])
            ii, jj, y = sample_pairs(t["contacts"][b], Lb, 512, device)
            if ii is None or len(ii) == 0:
                continue
            ii_l.append(ii); jj_l.append(jj); y_l.append(y)
            bi_l.append(torch.full((len(ii),), b, dtype=torch.long,
                                   device=device))
        if ii_l:
            ii = torch.cat(ii_l); jj = torch.cat(jj_l)
            yv = torch.cat(y_l); bidx = torch.cat(bi_l)
            h = out["hidden"]
            pair = model.pair_proj(torch.cat([h[bidx, ii], h[bidx, jj]], -1))
            lwk = t.get("lw_key")
            if lwk is not None and lwk.numel():
                lo = torch.minimum(ii, jj).to(torch.int64)
                hi = torch.maximum(ii, jj).to(torch.int64)
                want = bidx.to(torch.int64) * Lv * Lv + lo * Lv + hi
                pos = torch.searchsorted(lwk, want).clamp(max=lwk.numel() - 1)
                hitlw = lwk[pos] == want
                if bool(hitlw.any()):
                    # `lwl.shape[-1]`, not a config attribute. `cfg.heads`
                    # does not exist -- it is `cfg.head_cfg()` -- and this is
                    # the second time in this file that reaching for the class
                    # count through the config has been wrong where the
                    # logits' own width is right by construction.
                    lwl = model.heads.pair.geometry(pair[hitlw])
                    for k, v in _class_metrics(lwl, t["lw_val"][pos[hitlw]],
                                               "lw", lwl.shape[-1]).items():
                        acc.setdefault(k, []).append(float(v))
            cv = lookup_coevolution(t, bidx, ii, jj, Lv)
            if cv is not None:
                acc.setdefault("coev_frac", []).append(
                    float((cv != 0).float().mean()))
                pair = pair + model.coev_proj(cv.unsqueeze(-1).to(pair.dtype))
            if model.motifs is not None:
                r, minfo = model.motifs(pair)
                acc.setdefault("motif_eff", []).append(
                    float(minfo["effective_motifs"]))
                pair = pair + model.motif_mix(r)
            pool.setdefault("contact_score", []).append(
                model.heads.pair.contact(pair).squeeze(-1).float().cpu().numpy())
            pool.setdefault("contact_label", []).append(
                (yv > 0.5).cpu().numpy())

        # ---- HEAD 3, which had no validation metric at all -----------------
        #
        # This function returned Mg and rigidity. Two heads of ten, and not the
        # one the project exists for: the structure head trains against a
        # denoising loss and its only report was that loss on TRAINING batches.
        # A stage whose deliverable is coordinates could not say whether the
        # coordinates were improving on held-out data.
        #
        # The EDM loss on val costs one extra forward through the decoder and
        # is the like-for-like number. The GEOMETRY terms matter more: `bond_cn`
        # and `bond_pp` are flat-bottomed violations in angstroms, and they are
        # the only quantity that asks whether the output is a CHAIN. The blind
        # test on rp01 reported 0.0% of bonds in tolerance with a median C4'-N
        # of 37.95 A against a true 3.38 -- a gas of points -- and TM-score and
        # lDDT cannot see that, because superposition metrics never ask whether
        # anything is bonded.
        crm = t.get("coord_residue_mask")
        if crm is not None and bool(crm.any()) and "coords" in t:
            sh = model.heads.structure
            # `diff_pair(hidden, coev_dense)`, not `(hidden, mask)`. The second
            # argument is a DENSE (B, L, L) coupling map, and passing the mask
            # got as far as a shape error only because L happened to differ
            # between the two -- with a square mask it would have silently fed
            # the wrong tensor into the pair features. Validation must use the
            # same couplings the training step does, or it is measuring a
            # different model.
            _B, _L = t["tokens"].shape
            cvd = torch.zeros(_B, _L, _L, device=device,
                              dtype=out["hidden"].dtype)
            _k = t.get("coev_key")
            if _k is not None and _k.numel():
                _bb = torch.div(_k, _L * _L, rounding_mode="floor")
                _rem = _k - _bb * _L * _L
                _ii = torch.div(_rem, _L, rounding_mode="floor")
                _jj = _rem - _ii * _L
                _ok = (_bb < _B) & (_ii < _L) & (_jj < _L)
                cvd[_bb[_ok], _ii[_ok], _jj[_ok]] = \
                    t["coev_val"].to(cvd.dtype)[_ok]
                cvd = cvd + cvd.transpose(1, 2)
            pair3 = model.diff_pair(out["hidden"], cvd)
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                # `torch.Generator()` is a CPU generator whatever the
                # tensors are, and `random_rigid` feeds it to a `torch.randn`
                # on the coordinates' device: on CUDA that raises "Expected a
                # 'cuda' device type for generator but found 'cpu'" at the
                # FIRST eval batch, so stage 5 trained and then died before
                # reporting a single validation number. The seed is per-batch
                # and fixed so the augmentation is reproducible; the device
                # has to follow the tensors for that to be reachable at all.
                dl = sh.loss(t["coords"].to(out["hidden"].dtype),
                             out["hidden"], pair3, crm,
                             generator=torch.Generator(device=device
                                                       ).manual_seed(1234 + bi))
            for k in ("loss", "mse", "violation", "bond_cn", "bond_pp"):
                if k in dl:
                    acc.setdefault(f"structure_{k}", []).append(
                        float(dl[k].detach()))

        # ---- head 10, motif class, on val ---------------------------------
        lm = t.get("loop_mask")
        if lm is not None and bool(lm.any()):
            ml = out["motif_logits"]
            for k, v in _class_metrics(ml[lm].detach(), t["loop_class"][lm],
                                       "motif", ml.shape[-1]).items():
                # no `val_` here. The epoch log prefixes every key this
                # function returns, so prefixing again produced
                # `val_val_motif_acc` -- a column no field list declares, in
                # the one metric block `evaluate` added for head 10.
                acc.setdefault(k, []).append(float(v))

    res: Dict = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
    # the geometry numbers carry their targets, so a reader does not have to
    # know that C4'-N is 3.38 A and P-P is 6.01 to see whether 0.4 is good
    if "structure_bond_cn" in res:
        res["bond_cn_target_a"] = BOND_C4_N[0]
        res["bond_pp_target_a"] = BOND_P_P[0]
    if "contact_score" in pool:
        sc = np.concatenate(pool["contact_score"])
        yy = np.concatenate(pool["contact_label"])
        base = float(yy.mean())
        res["contact_base_rate"] = round(base, 4)
        res["contact_ap"] = round(average_precision(sc, yy), 4)
        res["contact_ap_lift"] = (round(res["contact_ap"] / base, 2)
                                  if base > 0 else None)
        res["contact_n"] = int(len(sc))
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
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="run N steps and stop, writing no checkpoint. Works "
                         "on CPU. For checking that stage 5 STARTS -- which "
                         "has never been verified, because the only card is "
                         "busy with stage 1.")
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
    assert_coevolution_reaches_the_model(tr)
    assert_supervised_heads_have_targets(tr)

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

    runlog = RunLog(ROOT, "stage5_3d", STAGE5_FIELDS, manifest={
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
                # `step`, not `gstep`. Stage 5 has no `gstep`: its counter is
                # `step`, restored from the resume record at line 1212. The
                # guard this call implements was added to stop a NaN reaching
                # the optimiser, and as written it raised NameError on the
                # FIRST optimiser step of the stage -- so the check that
                # existed to prevent a silent failure was itself a hard one,
                # in a stage that has never run and therefore never reported
                # it. Found by running `--smoke 2 --from-scratch`, which is
                # what that flag is for.
                check_loss(loss, step, parts)
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
            if args.smoke:
                print(f"[pharos] smoke step {step}/{args.smoke} "
                      f"loss {float(loss.detach()):.4f} "
                      + " ".join(f"{k} {v:.4g}" if isinstance(v, float)
                                 else f"{k} {v}" for k, v in sorted(parts.items())),
                      flush=True)
                if step >= args.smoke:
                    # exercise the epoch-end VALIDATION before returning. It is
                    # where the structure and motif metrics were just added,
                    # and a smoke test that stops at step 1 never reaches it --
                    # which is exactly how `evaluate()` came to report two
                    # heads of ten without anyone noticing.
                    _ev = evaluate(model, va, device, cfg, 2, args.token_budget)
                    print(f"[pharos] smoke validation ({len(_ev)} metrics): "
                          + "  ".join(f"{k} {v}" for k, v in sorted(_ev.items())),
                          flush=True)
                    _need = ("structure_loss", "structure_bond_cn",
                             "structure_bond_pp")
                    _miss = [k for k in _need if k not in _ev]
                    if _miss:
                        print(f"[pharos] SMOKE TEST FAILED: validation is "
                              f"missing {_miss} -- head 3 has no held-out "
                              f"measurement", flush=True)
                        return 1
                    print("[pharos] SMOKE TEST PASSED: stage 5 starts, every "
                          "head this batch can supervise produced a loss, the "
                          "backward pass completes, and validation reports the "
                          "structure head. No checkpoint written.", flush=True)
                    return 0
                continue
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
