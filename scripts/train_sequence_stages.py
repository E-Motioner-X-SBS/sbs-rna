#!/usr/bin/env python3
"""Stages 2, 3 and 6 of the §12.1 curriculum — the sequence-supervised heads.

    stage 2  secondary structure   head 4      bpRNA-SPOT, 10,934 train
    stage 3  chemical probing      head 5      Ribonanza, 335,616 profiles
    stage 6  fitness (auxiliary)   `fitness`   139,837 variants, 12 assays

§9 numbers ten heads and `fitness` is not one of them: it is in
`EXTRA_HEAD_KEYS`, the outputs the renumbering left unnamed. This block
called it head 8, which is disorder.

All three read sequence and predict something per residue or per sequence, so
they share a trainer and are **co-trained** rather than run one after another.
The curriculum orders them because the *representation* matures in that order,
not because the losses conflict, and stage 6 is explicitly auxiliary.

Stage 6 and the benchmark it must not touch
-------------------------------------------
`build_fitness_dataset.py` splits the fitness data into a benchmark and a
training set, and the split is the point. 24 of the 31 assays on the RNAGym
ncRNA leaderboard are ALSO shipped by NABench, so "train on NABench, evaluate
on RNAGym" is one dataset wearing two names. This trainer reads only
`data/derived/fitness_v1/{train,val,transfer}.parquet`, which the builder
guarantees share no assay and no exact sequence with `benchmark.parquet`.
What stays comparable to a published table is `eval_fitness_zeroshot.py`,
which needs no fitness head at all.

Three fitness numbers come out, and they answer different questions:

    train      within-assay Spearman on variants the head has fitted
    val        within-assay Spearman on UNSEEN VARIANTS of the same 12 assays
    transfer   per-assay Spearman on the five Townshend aptamers, a construct
               family held out of training entirely

The gap between `val` and `transfer` is the honest statement of what a
supervised fitness head trained on two construct families can do. Reporting
only the first would be reporting the training accuracy, which is the defect
the bpRNA validation split below exists to stop.

Why probing is weighted above secondary structure
-------------------------------------------------
§12.1 calls Ribonanza "the largest labelled channel the model will see":
335,616 profiles against 673 clean 3D sequences, **499x**. Secondary structure
is 10,934 examples. If the three were summed unweighted the smaller channels
would be read as often as the largest -- see the cycling note below -- so each
stage's loss is scaled by an explicit weight that is recorded rather than
implied.

Every stream is read at every step
----------------------------------
This loop used to create one generator per stage per epoch and stop pulling
from a stage once its generator raised `StopIteration`. With 342 secondary
structure batches against 10,488 probing batches, head 4 therefore received a
gradient on 3.3% of the steps in an epoch and then sat idle for the other
96.7% -- while the file claimed the opposite, that "10,934 secondary-structure
examples are recycled about 31 times for every pass over Ribonanza", and
`STAGE_WEIGHTS["ss"] = 0.5` was set to hold back a channel that was in fact
starving. The weight and the comment described a loop that did not exist.

An epoch is now `max` over the streams' batch counts -- which is what
`estimate_steps` has always computed -- and the shorter streams restart within
it, so every step carries every task and the weights mean what they say. The
epoch report prints each stream's pass count so the ratio is on the page
rather than in a comment.

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
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmarks"
OUT = ROOT / "data/samples/analysis"
CKPT = ROOT / "data/derived/checkpoints"
#: stage 6's corpus. Built by `build_fitness_dataset.py`, which also writes the
#: benchmark this trainer is forbidden to read.
FITNESS = ROOT / "data/derived/fitness_v1"

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
from pharos.train.guard import check_loss
from train_block_scorer import gpu_free_gib                          # noqa: E402

#: dot-bracket alphabet, matching HeadConfig.n_ss_symbols = 8
SS_SYMBOLS = ".()[]{}<"
SS_INDEX = {c: i for i, c in enumerate(SS_SYMBOLS)}
#: loss weights, stated rather than implied -- see the module docstring.
#:
#: `"fitness": 0.3` sat here once before with nothing reading it: no loader, no
#: loss, no `out["fitness"]` anywhere in the file. It was printed at startup
#: and written into two report files, so every record of every run claimed a
#: third task that never ran. It was removed, and it is back now because the
#: loader, the loss and the three reported correlations exist below.
#:
#: 0.3 because stage 6 is auxiliary: it is the only one of the three whose
#: supervision is a single scalar per sequence, and its corpus is two
#: construct families rather than a sample of RNA. `test_stage6.py` asserts
#: that every weight here is read by the loop, which is the check that was
#: missing when the phantom weight survived.
STAGE_WEIGHTS = {"ss": 0.5, "reactivity": 1.0, "fitness": 0.3}

#: Below this spread the `fitness` head is not predicting at all, whatever its
#: Spearman says -- it is being correlated against float noise. See
#: `score_fitness`.
#:
#: 1e-5, not the 1e-3 this was first set to. The threshold has to separate
#: "collapsed" from "early in training", and those are two measured numbers,
#: not a guess: a head at fp32 noise has a spread around 1e-8, and the head
#: measured after 250 steps from a random init showed 0.0024 against a
#: unit-variance target. 1e-3 sits a factor of two under a live-but-untrained
#: head and would have fired on an ordinary first epoch, which is how a guard
#: gets switched off. 1e-5 has two orders of margin on each side.
DEAD_PRED_SD = 1e-5


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


def stream_batches(args) -> Dict[str, int]:
    """Batches per epoch in each stream, from row counts rather than a dry run.

    Split out of `estimate_steps` and printed at startup because the epoch is
    now BOUNDED by this arithmetic rather than merely paced by it. The loop
    used to run until every generator was exhausted and use the estimate only
    for the cosine schedule, so a wrong estimate skewed the learning rate; now
    the short streams restart and `steps_per_epoch` is the only thing that
    ends an epoch, so a wrong estimate silently truncates training. An
    estimate that decides how much training happens has to be on the page.
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
    n_fit = 0
    h = FITNESS / "train.parquet"
    if h.exists() and not getattr(args, "no_fitness", False):
        n_fit = pq.ParquetFile(h).metadata.num_rows
        if getattr(args, "fitness_limit", None):
            n_fit = min(n_fit, args.fitness_limit)
    # `max`, not `sum`: one batch is pulled from each stream per step and the
    # short streams restart inside the epoch, so an epoch is as long as the
    # longest stream. This matched the loop's INTENT before and not its
    # behaviour; now it matches both.
    b = getattr(args, "token_budget", 0)
    if not b:
        return {"ss": -(-n_ss // args.batch),
                "reactivity": -(-n_pr // args.batch),
                "fitness": -(-n_fit // args.batch)}
    # Under a token budget the batch count is total tokens / budget, to within
    # the padding each pack leaves. Mean lengths, measured: bpRNA 131 nt,
    # Ribonanza 177 (the constructs are fixed-length), the fitness corpus 45.
    per = max(1, b)
    return {"ss": -(-(n_ss * 131) // per),
            "reactivity": -(-(n_pr * 177) // per),
            "fitness": -(-(n_fit * 45) // per)}


def estimate_steps(args) -> int:
    """Total optimiser steps over the whole run.

    `max`, not `sum`: one batch is pulled from each stream per step and the
    short streams restart inside the epoch, so an epoch is as long as the
    longest stream. This matched the loop's INTENT before and not its
    behaviour; now it matches both.
    """
    return max(max(stream_batches(args).values()), 1) * max(args.epochs, 1)


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


def pack_by_tokens(lengths: Sequence[int], budget: int, max_batch: int,
                   rng: Optional[np.random.Generator] = None) -> List[List[int]]:
    """Indices grouped so each batch costs about `budget` padded tokens.

    Stages 2/3/6 batched by SEQUENCE COUNT while stages 1 and 5 both batch by
    token budget -- and both of those document at length why, stage 1's note
    reading "a fixed count is a latent OOM: 64 sequences is 1.3k tokens if
    they are 20 nt and 65k if they are 1,024". Measured on a free A100 at the
    cron runner's own `--batch 32`, with Ribonanza at ~177 nt, this stage ran
    **5,700 tokens a step in 11.5 of 80 GiB at ~1,750 tok/s** against stage
    1's ~17,000 on the same card. The budget was not the binding constraint;
    nothing was.

    Length-sorted before cutting, because a batch pads to its longest member
    and bpRNA runs from tens of nucleotides to over a thousand. The running
    maximum is tracked rather than the first element -- the same off-by-one
    that let 71% of stage-5 batches exceed their budget (finding 55).
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out: List[List[int]] = []
    cur: List[int] = []
    cur_max = 0
    for i in order:
        L = max(int(lengths[i]), 1)
        nxt = L if L > cur_max else cur_max
        if cur and ((len(cur) + 1) * nxt > budget or len(cur) >= max_batch):
            out.append(cur)
            cur, cur_max, nxt = [], 0, L
        cur.append(i)
        cur_max = nxt
    if cur:
        out.append(cur)
    if rng is not None:
        rng.shuffle(out)
    return out


# ---------------------------------------------------------------- stage 2
def iter_ss(split: str, batch: int, budget: int = 0,
            rng: Optional[np.random.Generator] = None
            ) -> Iterator[Tuple[List[str], List[str]]]:
    """bpRNA rows. Token-budgeted when `budget` is set, else `batch` rows.

    The whole split is 10,934 rows and fits in memory, so it is read once and
    packed, rather than streamed into fixed-size groups: packing needs the
    lengths, and the lengths are the point.
    """
    import pyarrow.parquet as pq
    f = BENCH / f"secondary_structure/bprna_spot/{split}.parquet"
    if not f.exists():
        return
    seqs: List[str] = []
    ss: List[str] = []
    for b in pq.ParquetFile(f).iter_batches(
            batch_size=512, columns=["sequence", "secondary_structure"]):
        for r in b.to_pylist():
            s, t = r["sequence"], r["secondary_structure"]
            if not s or not t or len(s) != len(t):
                continue
            seqs.append(s)
            ss.append(t)
    if not seqs:
        return
    if budget:
        for idx in pack_by_tokens([len(x) for x in seqs], budget, batch, rng):
            yield [seqs[i] for i in idx], [ss[i] for i in idx]
    else:
        for s0 in range(0, len(seqs), batch):
            yield seqs[s0:s0 + batch], ss[s0:s0 + batch]


def ss_targets(ss: List[str], L: int, device) -> torch.Tensor:
    y = np.full((len(ss), L), -100, dtype=np.int64)
    for i, t in enumerate(ss):
        for j, c in enumerate(t[:L]):
            y[i, j] = SS_INDEX.get(c, 0)
    return torch.as_tensor(y, device=device)


# ---------------------------------------------------------------- stage 3
def iter_probing(batch: int, limit: Optional[int] = None, budget: int = 0,
                 rng: Optional[np.random.Generator] = None
                 ) -> Iterator[Tuple[List[str], np.ndarray, List[str]]]:
    """Ribonanza rows: sequence, 206 reactivity columns, experiment type.

    Token-budgeted when `budget` is set. Streamed in POOLS rather than read
    whole: the file is 526 MB and 335,616 rows, so holding it costs more than
    the model does. A pool of `batch * 64` rows is enough for the packer to
    see a representative length spread -- these are ~177 nt constructs, so
    the spread is small and the pool only has to beat the batch size.
    """
    import csv
    f = BENCH / "chemical_probing/ribonanza_train_quickstart.csv"
    if not f.exists():
        return
    pool_rows = max(batch * 64, 1024) if budget else batch

    def _flush(seqs, rows, kinds):
        if not seqs:
            return
        if not budget:
            yield seqs, np.asarray(rows, dtype=np.float32), kinds
            return
        for idx in pack_by_tokens([len(x) for x in seqs], budget, batch, rng):
            yield ([seqs[i] for i in idx],
                   np.asarray([rows[i] for i in idx], dtype=np.float32),
                   [kinds[i] for i in idx])

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
            if len(seqs) >= pool_rows:
                yield from _flush(seqs, rows, kinds)
                seqs, rows, kinds = [], [], []
            if limit and n >= limit:
                break
    yield from _flush(seqs, rows, kinds)


# ---------------------------------------------------------------- stage 6
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Signed Spearman, or NaN when it is not defined.

    NaN when either side is constant, which is the case that matters: a head
    whose output has not moved off its initialisation predicts the same number
    for every variant, `spearmanr` returns NaN, and `np.nanmean` over the
    assays then reports the average of whatever was left. A dead head and a
    head that worked on two assays produce the same summary. Callers here
    count the NaNs and print the count.
    """
    from scipy.stats import spearmanr
    if len(a) < 3:
        return float("nan")
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else float("nan")


@lru_cache(maxsize=4)
def load_fitness(split: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """`[(assay, sequences, targets)]` for one split, grouped by assay.

    Grouped, and kept grouped, because the metric is a WITHIN-ASSAY Spearman:
    the DMS score of a tRNA assay and of a ribozyme assay are different
    quantities measured on different instruments, and a correlation computed
    across a mixed batch would mostly measure which assay a sequence came
    from. `target` is already rank-normalised within its assay by the builder,
    so the regression loss is comparable across a mixed batch even though the
    correlation is not.
    """
    import pyarrow.parquet as pq
    f = FITNESS / f"{split}.parquet"
    if not f.exists():
        return []
    t = pq.read_table(f, columns=["assay", "seq", "target"]).to_pydict()
    by: Dict[str, Tuple[List[str], List[float]]] = {}
    for a, q, y in zip(t["assay"], t["seq"], t["target"]):
        g = by.setdefault(a, ([], []))
        g[0].append(q)
        g[1].append(float(y))
    return [(a, np.array(q, dtype=object), np.asarray(y, dtype=np.float32))
            for a, (q, y) in sorted(by.items())]


def iter_fitness(split: str, batch: int, rng: np.random.Generator,
                 limit: Optional[int] = None, budget: int = 0
                 ) -> Iterator[Tuple[str, List[str], np.ndarray]]:
    """Batches of one assay each, in a shuffled order.

    One assay per batch so the within-batch Spearman the loop logs is the
    quantity the benchmark uses. Shuffled across assays so consecutive steps
    do not all come from `Rachapun_2022_f1u_ribozyme`, which is 65,536 of the
    139,837 rows and would otherwise own the first half of every epoch.
    """
    groups = load_fitness(split)
    if not groups:
        return
    chunks: List[Tuple[str, np.ndarray]] = []
    for assay, seqs, _ in groups:
        order = rng.permutation(len(seqs))
        # one assay per batch still, but sized by TOKENS. Every variant of an
        # assay is the same length -- they are mutants of one construct -- so
        # the budget divides exactly and no packing is needed: Rachapun is
        # 35 nt and Townshend 79, a factor of two the fixed count ignored.
        n_b = batch
        if budget:
            n_b = max(1, min(batch, budget // max(len(str(seqs[0])), 1)))
        for s0 in range(0, len(order), n_b):
            chunks.append((assay, order[s0:s0 + n_b]))
    index = {a: (q, y) for a, q, y in groups}
    n = 0
    for k in rng.permutation(len(chunks)):
        assay, idx = chunks[int(k)]
        q, y = index[assay]
        yield assay, [str(x) for x in q[idx]], y[idx]
        n += len(idx)
        if limit and n >= limit:
            return


def fitness_step(model, seqs: List[str], target: np.ndarray, device, feats_fn
                 ) -> Tuple[Optional[torch.Tensor], float, float]:
    """Loss, within-assay Spearman, and the spread of the predictions.

    The spread is returned because it is the one number that says the head is
    alive. `fitness` is a mean-pooled scalar: a one-nucleotide change in an
    87 nt construct moves the pooled representation by about 1/87 of one
    token's delta, and a head that collapsed to a constant would still produce
    a falling MSE -- it would be predicting the mean of a zero-mean target --
    while every correlation it was scored on came back NaN. Measured at
    initialisation and at 3.34B tokens the separation is 10^4 to 10^5 times
    fp32 spacing (`test_pharos.py`, property 7d), so a collapse here is a
    training failure rather than an architectural limit, and this is how it is
    seen.
    """
    b = encode(seqs, device)
    t = torch.as_tensor(np.asarray(target, dtype=np.float32), device=device)
    out = model(b["tokens"], b["mod_ids"], b["chem"], b["mask"],
                feats=feats_fn(b), n_loops=2)
    pred = out["fitness"].float()
    if pred.shape != t.shape:
        raise RuntimeError(f"fitness head gave {tuple(pred.shape)} for a "
                           f"target of {tuple(t.shape)}")
    loss = F.mse_loss(pred, t)
    with torch.no_grad():
        pn = pred.detach().cpu().numpy()
        rho = _spearman(pn, np.asarray(target))
        spread = float(pn.std())
    return loss + out["aux"]["balance_loss"], rho, spread


@torch.no_grad()
def score_fitness(model, split: str, device, feats_fn, batch: int,
                  max_rows: Optional[int] = None) -> Dict:
    """Per-assay Spearman over a whole split, with the NaNs counted.

    `max_rows` caps each assay, not the split: capping the split would score
    the first assays and silently drop the last, and the per-category means a
    reader compares would then be over different assay sets each epoch.
    """
    groups = load_fitness(split)
    rows: List[Dict] = []
    for assay, seqs, y in groups:
        n = len(seqs) if max_rows is None else min(len(seqs), max_rows)
        preds = np.empty(n, dtype=np.float32)
        for s0 in range(0, n, batch):
            chunk = [str(x) for x in seqs[s0:min(s0 + batch, n)]]
            b = encode(chunk, device)
            out = model(b["tokens"], b["mod_ids"], b["chem"], b["mask"],
                        feats=feats_fn(b), n_loops=2)
            preds[s0:s0 + len(chunk)] = out["fitness"].float().cpu().numpy()
        rows.append({"assay": assay, "n": int(n),
                     "spearman": _spearman(preds, y[:n]),
                     "pred_sd": float(preds.std()),
                     "distinct": int(len(np.unique(preds)))})
    rho = [r["spearman"] for r in rows if np.isfinite(r["spearman"])]
    # A head that has collapsed to EXACTLY one value scores NaN and is
    # caught. A head whose output varies by 1e-9 does not: `spearmanr` ranks
    # float noise and returns a perfectly finite, perfectly meaningless
    # number, and `n_scored` counts it as a scored assay. The first version
    # of this function had only the exact test, which is the weaker half of
    # the check it was written to be.
    #
    # The threshold is absolute because the TARGET is absolute: the builder
    # rank-normalises every assay to unit variance, so a prediction spread
    # below 1e-3 is a head explaining a thousandth of the signal it is being
    # correlated against, whatever its correlation says.
    dead = [r["assay"] for r in rows if r["pred_sd"] < DEAD_PRED_SD]
    # Reported as well as flagged: `min_pred_sd` is the number that says how
    # far above the floor the head is, and a flag that only fires at the
    # bottom says nothing on the way down.
    return {"split": split, "n_assays": len(rows), "n_scored": len(rho),
            "mean_spearman": float(np.mean(rho)) if rho else None,
            "min_pred_sd": (round(min(r["pred_sd"] for r in rows), 8)
                            if rows else None),
            "n_degenerate": len(dead), "degenerate": dead,
            "per_assay": rows}


def require_gpu(args) -> torch.device:
    if not args.device.startswith("cuda"):
        # Same escape as stage 5's, for the same reason: stages 2 and 3 have
        # never run either, and the only card is busy for days at a time. A
        # handful of CPU steps at a toy batch is the difference between
        # finding a startup bug now and finding it when the curriculum
        # finally reaches this file.
        if getattr(args, "smoke", 0):
            print("[seq] SMOKE TEST on CPU: startup path only, "
                  "no checkpoint will be written", flush=True)
            return torch.device(args.device)
        raise SystemExit("GPU only; pass --device cuda once one is free. "
                         "For a startup check without a GPU use "
                         "--smoke N --device cpu.")
    mem = gpu_free_gib()
    if mem is None:
        raise SystemExit("no GPU visible to nvidia-smi")
    free, total = mem
    print(f"[seq] GPU {free:.1f} of {total:.1f} GiB free")
    if free < args.min_free_gib:
        raise SystemExit(f"only {free:.1f} GiB free; nothing started.")
    return torch.device(args.device)


def ss_step(model, seqs, ss, device, feats_fn
            ) -> Tuple[Optional[torch.Tensor], float, float, float]:
    """Loss, accuracy, and the two numbers that make the accuracy mean something.

    Head 4 has 8 symbols, so chance is 0.125 -- but dot-bracket is dominated by
    unpaired positions, and an accuracy quoted without the majority rate cannot
    distinguish a head that has learned pairing from one that has learned that
    most bases are unpaired. Every other head in this project got a floor;
    this one had not.

    `major` is the batch's own majority-class rate and `macro` is unweighted
    recall over the classes present, which collapses to 1/k for a head that
    predicts one class and is where the rare bracket symbols live. At
    initialisation the smoke test reads accuracy 0.0645 against a chance of
    0.125 -- BELOW chance, which is what a fresh head that predicts one
    near-constant class scores: the frequency of whatever class it picked.
    """
    b = encode(seqs, device)
    L = b["tokens"].shape[1]
    y = ss_targets(ss, L, device)
    out = model(b["tokens"], b["mod_ids"], b["chem"], b["mask"],
                feats=feats_fn(b), n_loops=2)
    lg = out["ss_logits"].float()
    valid = (y >= 0) & b["mask"]
    if not bool(valid.any()):
        return None, 0.0, 0.0, 0.0
    loss = F.cross_entropy(lg[valid], y[valid])
    with torch.no_grad():
        pred, tgt = lg[valid].argmax(-1), y[valid]
        acc = float((pred == tgt).float().mean())
        k = lg.shape[-1]
        cnt = torch.bincount(tgt, minlength=k).float()
        major = float(cnt.max() / cnt.sum()) if float(cnt.sum()) > 0 else 0.0
        present = cnt > 0
        hit = torch.bincount(tgt[pred == tgt], minlength=k).float()
        macro = (float((hit[present] / cnt[present]).mean())
                 if bool(present.any()) else 0.0)
    return loss + out["aux"]["balance_loss"], acc, major, macro


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
    # head 5 emits two channels: DMS and SHAPE-like are different chemistries
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
    # shared400 was missing, so the curriculum could not reach this file at
    # all: the runner passes `--size shared400` and argparse rejected it
    # before a single line of training ran. Stages 2-3 have never run, so
    # nothing had ever discovered that. The choices now match stage 5's, and
    # the config is resolved by name rather than by an `if size == "small"`
    # that silently falls through to mini for everything else.
    ap.add_argument("--size", default="small",
                    choices=("mini", "small", "base400", "shared400"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--val-batches", type=int, default=64,
                    help="bpRNA validation batches scored at each epoch end. "
                         "The split shipped with the benchmark and was never "
                         "read; head 4's only reported accuracy was its "
                         "training accuracy, on 10,934 examples recycled ~31x "
                         "per Ribonanza epoch.")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=512,
                    help="MAXIMUM sequences in a batch. With --token-budget "
                         "this is a cap against a pathological batch of "
                         "thousands of 20-nt sequences, not the batch size.")
    ap.add_argument("--token-budget", type=int, default=32768,
                    help="padded tokens per batch, the way stages 1 and 5 "
                         "size theirs. 0 restores the old fixed-count "
                         "behaviour. At the previous --batch 32 this stage "
                         "ran 5,700 tokens a step in 11.5 of 80 GiB.")
    ap.add_argument("--probing-limit", type=int, default=None)
    ap.add_argument("--fitness-limit", type=int, default=None,
                    help="cap stage 6 rows per epoch")
    ap.add_argument("--no-fitness", action="store_true",
                    help="run stages 2-3 only. The run then says so in its "
                         "manifest rather than silently reporting two tasks "
                         "under a three-task weight table.")
    ap.add_argument("--fitness-val-rows", type=int, default=4096,
                    help="rows per assay scored at each epoch end for the "
                         "stage-6 val and transfer correlations. Per ASSAY, "
                         "not per split: a cap on the split would score the "
                         "first assays and drop the last.")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="a stage-1 pretraining checkpoint")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=200,
                    help="steps between checkpoints. Epoch-end only was a "
                         "latent full-restart: an epoch here is ~10,500 steps")
    ap.add_argument("--restart", action="store_true",
                    help="ignore an existing checkpoint; it is RENAMED, not "
                         "overwritten")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="run N steps and stop, writing no checkpoint. "
                         "Works on CPU. For checking that stages 2-3 START.")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint path; defaults to "
                         "data/derived/checkpoints/seqstages_<size>.pt")
    args = ap.parse_args()

    device = require_gpu(args)
    enable_gpu_fast_paths()
    cfg = getattr(PharosConfig, args.size)()
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

    # Stage 6 runs only if its corpus is on disk, and the run SAYS which of the
    # two it is. A weight table that lists a task the run is not performing is
    # the exact defect that let `"fitness": 0.3` sit here unread for the life
    # of the file; the weights printed below are the ones about to be applied.
    use_fitness = (not args.no_fitness) and (FITNESS / "train.parquet").exists()
    weights = dict(STAGE_WEIGHTS)
    if not use_fitness:
        weights.pop("fitness")
        why = ("--no-fitness" if args.no_fitness
               else f"{FITNESS / 'train.parquet'} is missing -- run "
                    f"scripts/build_fitness_dataset.py")
        print(f"[seq] stage 6 DISABLED: {why}")
    else:
        groups = load_fitness("train")
        print(f"[seq] stage 6: {sum(len(y) for _, _, y in groups):,} variants "
              f"over {len(groups)} assays, "
              f"val {sum(len(y) for _, _, y in load_fitness('val')):,}, "
              f"transfer {sum(len(y) for _, _, y in load_fitness('transfer')):,}")
    print(f"[seq] stage weights {weights}")

    def feats_fn(b):
        return RouterFeatures(
            length=b["mask"].sum(1).float(),
            chem_summary=(b["chem"].sum(1)
                          / b["mask"].sum(1, keepdim=True).clamp(min=1))[:, :5])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    hist: List[Dict] = []
    #: per-assay stage-6 breakdowns, kept out of `hist` so that `hist` stays
    #: exactly the set of csv columns. See the note where it is appended.
    fit_detail: List[Dict] = []
    total_steps = max(1, estimate_steps(args))
    sb = stream_batches(args)
    if not use_fitness:
        sb["fitness"] = 0
    per_epoch = max(1, total_steps // max(args.epochs, 1))
    print(f"[seq] schedule: warm-up 1% then cosine over ~{total_steps:,} steps "
          f"({args.epochs} epochs of {per_epoch:,})", flush=True)
    # The ratio the weights are set against, stated before a step is taken.
    # It lived in a comment that disagreed with the loop for the life of the
    # file: the comment said bpRNA was recycled ~31 times per Ribonanza pass
    # and the loop gave head 4 a gradient on 3.3% of the steps.
    print("[seq] per epoch: " + "  ".join(
        f"{k} {v:,} batches x{per_epoch / max(v, 1):.1f}" for k, v in sb.items()
        if v) + f"  (epoch = the longest, {per_epoch:,})", flush=True)
    if per_epoch < max(sb.values() or [0]):
        print(f"[seq] WARNING: an epoch is {per_epoch:,} steps but the longest "
              f"stream has {max(sb.values()):,} batches -- it will be "
              f"truncated", flush=True)
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

    # Everything `hist` records, because `hist` is what the epoch line prints
    # and what the results json keeps.
    #
    # The declared list was four names against a thirteen-field history: the
    # validation accuracy, the generalisation gap, the majority rate, the
    # macro recall and the val lift -- every metric the 09-25 audit added,
    # and the two numbers that say whether stage 2 is learning or memorising
    # -- were computed, stored, printed, written to json, and dropped on the
    # way to the csv. So the time series of them existed nowhere. Same defect
    # as stage 1's router scalars and stage 5's forty columns, one stage over.
    #
    # The epoch call below now logs `hist[-1]` itself rather than hand-picking
    # from it, so the record and the row cannot drift apart again.
    runlog = RunLog(ROOT, "stage23_seq", [
        "gstep", "lr", "ss_loss", "ss_accuracy",
        "ss_majority", "ss_macro_recall",
        "ss_val_loss", "ss_val_accuracy", "ss_val_majority",
        "ss_val_macro_recall", "ss_val_lift", "ss_generalisation_gap",
        "ss_val_batches", "probing_loss", "probing_pearson",
        # stage 6. `fitness_pred_sd` is in the time series on purpose: it is
        # the column that distinguishes a head that is learning slowly from a
        # head that has collapsed to a constant, and the two look identical in
        # the loss.
        "fitness_loss", "fitness_spearman", "fitness_pred_sd",
        "ss_batches", "probing_batches", "fitness_batches",
        "ss_passes", "probing_passes", "fitness_passes",
        "fitness_val_spearman", "fitness_val_assays",
        "fitness_val_degenerate", "fitness_val_min_pred_sd",
        "fitness_transfer_spearman", "fitness_transfer_assays",
        "fitness_transfer_degenerate",
        "fitness_generalisation_gap",
        "n_oom", "note",
    ], manifest={
        "size": args.size, "config": cfg.__dict__, "params": pc,
        "epochs": args.epochs, "batch": args.batch,
        "token_budget": args.token_budget, "lr_peak": args.lr,
        "total_steps_estimated": total_steps,
        "stage_weights": weights,
        "stage6_enabled": use_fitness,
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

    # Each stream restarts inside the epoch instead of going quiet when it
    # runs out, so every step carries every task. Both counters are reported,
    # because the ratio between the streams is the thing that used to live in
    # a comment and disagree with the code.
    #
    # `pulled` counts BATCHES and `passes` counts completed sweeps. Pulled is
    # the one to read: a pass is only credited when a generator runs out, so
    # the longest stream -- the one whose length defines the epoch -- finishes
    # its sweep exactly as the loop stops and would report 0 passes having
    # supplied every step. Counting the thing that happened rather than the
    # thing that completed is the difference.
    pulled = {"ss": 0, "reactivity": 0, "fitness": 0}
    passes = {"ss": 0, "reactivity": 0, "fitness": 0}

    def cycled(make, name):
        while True:
            got = False
            for item in make():
                got = True
                pulled[name] += 1
                yield item
            passes[name] += 1
            if not got:
                # an empty corpus must not spin: yield nothing, forever,
                # rather than loop on a generator that returns immediately
                return

    steps_per_epoch = max(1, total_steps // max(args.epochs, 1))
    for ep in range(start_ep, args.epochs):
        model.train()
        t0 = time.time()
        ss_loss, ss_acc, pr_loss, pr_r, step = [], [], [], [], 0
        ss_major: List[float] = []
        ss_macro: List[float] = []
        fit_loss: List[float] = []
        fit_r: List[float] = []
        fit_sd: List[float] = []
        fit_rng = np.random.default_rng(1234 + ep)
        for k in passes:
            passes[k] = pulled[k] = 0
        gen_ss = cycled(lambda: iter_ss("train", args.batch,
                                        args.token_budget, fit_rng), "ss")
        gen_pr = cycled(lambda: iter_probing(args.batch, args.probing_limit,
                                             args.token_budget, fit_rng),
                        "reactivity")
        gen_fit = (cycled(lambda: iter_fitness("train", args.batch, fit_rng,
                                               args.fitness_limit,
                                               args.token_budget), "fitness")
                   if use_fitness else iter(()))
        done_ss = done_pr = done_fit = False
        while step < steps_per_epoch and not (done_ss and done_pr and done_fit):
            total = None
            step_parts: dict = {}
            # stage 2
            try:
                seqs, ss = next(gen_ss)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l, a, a_maj, a_mac = ss_step(model, seqs, ss, device,
                                                 feats_fn)
                if l is not None:
                    total = STAGE_WEIGHTS["ss"] * l
                    step_parts["ss"] = float(l.detach())
                    ss_loss.append(float(l.detach()))
                    ss_acc.append(a)
                    ss_major.append(a_maj)
                    ss_macro.append(a_mac)
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
                    step_parts["reactivity"] = float(l.detach())
                    pr_loss.append(float(l.detach()))
                    pr_r.append(r)
            except StopIteration:
                done_pr = True
            # stage 6
            try:
                _assay, fseqs, ftgt = next(gen_fit)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l, r, sd = fitness_step(model, fseqs, ftgt, device, feats_fn)
                if l is not None:
                    total = (STAGE_WEIGHTS["fitness"] * l if total is None
                             else total + STAGE_WEIGHTS["fitness"] * l)
                    step_parts["fitness"] = float(l.detach())
                    fit_loss.append(float(l.detach()))
                    if np.isfinite(r):
                        fit_r.append(r)
                    fit_sd.append(sd)
            except StopIteration:
                done_fit = True
            if total is None:
                continue
            lr_now = lr_at(gstep, total_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr_now
            gstep += 1
            try:
                check_loss(total, gstep, step_parts)
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
            if args.smoke:
                _fz = (f" fitness {_m(fit_loss, 1):.4f} "
                       f"rho {(_m(fit_r, 1) if fit_r else float('nan')):+.4f} "
                       f"sd {_m(fit_sd, 1):.4f}" if fit_loss else " fitness -")
                print(f"[seq] smoke step {step}/{args.smoke} "
                      f"total {float(total.detach()):.4f} "
                      f"ss {_m(ss_loss, 1):.4f} acc {_m(ss_acc, 1):.4f} "
                      f"probing {_m(pr_loss, 1):.4f}{_fz}", flush=True)
                if step >= args.smoke:
                    # exercise the epoch-end VALIDATION before returning: it is
                    # new code on a path that has never run, and a smoke test
                    # that stops short of it proves nothing about it
                    _vl, _va, _vmaj, _vmac = [], [], [], []
                    _NSS = cfg.head_cfg().n_ss_symbols if hasattr(cfg, "head_cfg") else 8
                    model.eval()
                    with torch.no_grad():
                        for _vb, (_vs, _vd) in enumerate(
                                iter_ss("validation", args.batch,
                                        args.token_budget)):
                            if _vb >= 2:
                                break
                            _r = ss_step(model, _vs, _vd, device, feats_fn)
                            if _r is None or _r[0] is None:
                                continue
                            _vl.append(float(_r[0]))
                            _va.append(float(_r[1]))
                            _vmaj.append(float(_r[2]))
                            _vmac.append(float(_r[3]))
                    model.train()
                    if _va:
                        print(f"[seq] smoke validation: {len(_va)} batches, "
                              f"loss {np.mean(_vl):.4f} acc {np.mean(_va):.4f} "
                              f"majority {np.mean(_vmaj):.4f} "
                              f"lift {np.mean(_va) - np.mean(_vmaj):+.4f} "
                              f"macro {np.mean(_vmac):.4f} "
                              f"(chance is 1/{_NSS} = {1/_NSS:.4f})", flush=True)
                    else:
                        print("[seq] smoke validation: NO BATCHES -- the split "
                              "did not load", flush=True)
                    # and stage 6's two evaluation splits, on the same
                    # principle: a smoke test that stops before the new code
                    # proves nothing about the new code.
                    if use_fitness:
                        model.eval()
                        _fv = score_fitness(model, "val", device, feats_fn,
                                            args.batch, max_rows=64)
                        _ft = score_fitness(model, "transfer", device, feats_fn,
                                            args.batch, max_rows=64)
                        model.train()
                        for _nm, _r in (("val", _fv), ("transfer", _ft)):
                            print(f"[seq] smoke stage 6 {_nm:8s} "
                                  f"{_r['n_scored']}/{_r['n_assays']} assays "
                                  f"scored, rho {_r['mean_spearman']}, "
                                  f"smallest prediction sd {_r['min_pred_sd']}",
                                  flush=True)
                        if _fv["n_scored"] == 0:
                            print("[seq] SMOKE TEST FAILED: the fitness head "
                                  "returned a constant for every assay",
                                  flush=True)
                            return 1
                        if _fv["n_degenerate"] == _fv["n_assays"]:
                            print(f"[seq] SMOKE TEST FAILED: every assay's "
                                  f"prediction spread is under {DEAD_PRED_SD:g} "
                                  f"against a unit-variance target -- the head "
                                  f"is not predicting, whatever its Spearman "
                                  f"says", flush=True)
                            return 1
                    print(f"[seq] SMOKE TEST PASSED: stages 2-3"
                          f"{' and 6' if use_fitness else ''} start, every "
                          f"channel produced a loss, the backward pass "
                          f"completes, and the bpRNA validation split"
                          f"{' and both fitness splits' if use_fitness else ''}"
                          f" load and score. No checkpoint written.", flush=True)
                    return 0
                continue
            if step % args.ckpt_every == 0:
                save(ep, False)
            if step % args.log_every == 0:
                runlog.log("step", epoch=ep, step=step, gstep=gstep, lr=lr_now,
                           ss_loss=_m(ss_loss, args.log_every),
                           ss_accuracy=_m(ss_acc, args.log_every),
                           # the floor the accuracy above has to clear. 2D
                           # structure is mostly unpaired, so an accuracy
                           # without it is not a measurement -- which is why
                           # `ss_step` returns it, and it was going nowhere.
                           ss_majority=_m(ss_major, args.log_every),
                           ss_macro_recall=_m(ss_macro, args.log_every),
                           probing_loss=_m(pr_loss, args.log_every),
                           probing_pearson=_m(pr_r, args.log_every),
                           fitness_loss=_m(fit_loss, args.log_every),
                           fitness_spearman=_m(fit_r, args.log_every),
                           fitness_pred_sd=_m(fit_sd, args.log_every),
                           ss_batches=pulled["ss"],
                           probing_batches=pulled["reactivity"],
                           fitness_batches=pulled["fitness"],
                           ss_passes=passes["ss"],
                           probing_passes=passes["reactivity"],
                           fitness_passes=passes["fitness"],
                           n_oom=n_oom)
                fz = (f"  |  fitness loss {_m(fit_loss, args.log_every):.4f} "
                      f"rho {_m(fit_r, args.log_every):+.4f} "
                      f"sd {_m(fit_sd, args.log_every):.4f}"
                      if fit_loss else "")
                print(f"[seq] ep{ep} step{step} lr {lr_now:.2e}  2D loss "
                      f"{np.mean(ss_loss[-args.log_every:]):.4f} acc "
                      f"{np.mean(ss_acc[-args.log_every:]):.4f}  |  probing loss "
                      f"{np.mean(pr_loss[-args.log_every:]):.4f} r "
                      f"{np.mean(pr_r[-args.log_every:]):.4f}{fz}", flush=True)
        # ---- the validation split, which was sitting on disk unread --------
        #
        # `data/benchmarks/secondary_structure/bprna_spot/` ships train,
        # validation AND test parquets. Training read `train` and the epoch
        # report printed accuracy on it, so head 4's only published number was
        # its TRAINING accuracy -- and the loop is balanced per step, which
        # means 10,934 secondary-structure examples are recycled about 31
        # times for every pass over Ribonanza's 335,616 profiles. Overfitting
        # is the expected outcome of that ratio and would have been invisible:
        # the training accuracy rises either way.
        #
        # A split that exists and is never read is not a split. Run at every
        # epoch, capped so it costs a fraction of one, and reported beside the
        # training figure so the GAP is the thing on the page.
        vs_loss, vs_acc, vs_major, vs_macro = [], [], [], []
        model.eval()
        with torch.no_grad():
            for vb, (vseq, vdot) in enumerate(
                    iter_ss("validation", args.batch, args.token_budget)):
                if vb >= args.val_batches:
                    break
                r = ss_step(model, vseq, vdot, device, feats_fn)
                if r is None or r[0] is None:
                    continue
                vs_loss.append(float(r[0]))
                vs_acc.append(float(r[1]))
                vs_major.append(float(r[2]))
                vs_macro.append(float(r[3]))
        model.train()

        # ---- stage 6: unseen variants, then an unseen construct family ----
        #
        # Two numbers, because they answer different questions and only the
        # second one is about generalisation. `val` holds back 20% of the
        # variants of the SAME twelve assays, so a head that has memorised
        # each assay's landscape still scores well on it. `transfer` is the
        # five Townshend aptamers, a construct family no training step has
        # seen. Reporting the first alone would be reporting the training
        # accuracy under another name.
        fit_val = fit_tr = None
        if use_fitness:
            model.eval()
            fit_val = score_fitness(model, "val", device, feats_fn,
                                    args.batch, args.fitness_val_rows)
            fit_tr = score_fitness(model, "transfer", device, feats_fn,
                                   args.batch, args.fitness_val_rows)
            model.train()
            for nm, r in (("val", fit_val), ("transfer", fit_tr)):
                miss = r["n_assays"] - r["n_scored"]
                bits = []
                if miss:
                    bits.append(f"{miss} gave a CONSTANT prediction")
                if r["n_degenerate"]:
                    bits.append(f"{r['n_degenerate']} have a spread under "
                                f"{DEAD_PRED_SD:g} against a unit-variance "
                                f"target: {', '.join(r['degenerate'][:3])}")
                flag = ("  <<< " + "; ".join(bits)) if bits else ""
                mean = r["mean_spearman"]
                print(f"[seq] epoch {ep} stage 6 {nm:8s} "
                      f"rho {mean:+.4f} over {r['n_scored']}/{r['n_assays']} "
                      f"assays, smallest prediction sd {r['min_pred_sd']:.3g}"
                      f"{flag}" if mean is not None else
                      f"[seq] epoch {ep} stage 6 {nm:8s} NO ASSAY SCORED"
                      f" ({r['n_assays']} tried){flag}", flush=True)

        v_acc = float(np.mean(vs_acc)) if vs_acc else None
        t_acc = float(np.mean(ss_acc)) if ss_acc else None
        hist.append({"epoch": ep, "steps": step,
                     "ss_loss": float(np.mean(ss_loss)) if ss_loss else None,
                     "ss_accuracy": t_acc,
                     "ss_val_loss": float(np.mean(vs_loss)) if vs_loss else None,
                     "ss_val_accuracy": v_acc,
                     "ss_generalisation_gap": (None if (v_acc is None or t_acc is None)
                                               else round(t_acc - v_acc, 5)),
                     "ss_val_batches": len(vs_acc),
                     "ss_majority": float(np.mean(ss_major)) if ss_major else None,
                     "ss_macro_recall": float(np.mean(ss_macro)) if ss_macro else None,
                     "ss_val_majority": float(np.mean(vs_major)) if vs_major else None,
                     "ss_val_macro_recall": float(np.mean(vs_macro)) if vs_macro else None,
                     "ss_val_lift": (None if not (vs_acc and vs_major) else
                                     round(float(np.mean(vs_acc))
                                           - float(np.mean(vs_major)), 5)),
                     "probing_loss": float(np.mean(pr_loss)) if pr_loss else None,
                     "probing_pearson": float(np.mean(pr_r)) if pr_r else None,
                     "ss_batches": pulled["ss"],
                     "probing_batches": pulled["reactivity"],
                     "fitness_batches": pulled["fitness"],
                     "ss_passes": passes["ss"],
                     "probing_passes": passes["reactivity"],
                     "fitness_passes": passes["fitness"],
                     "fitness_loss": float(np.mean(fit_loss)) if fit_loss else None,
                     "fitness_spearman": float(np.mean(fit_r)) if fit_r else None,
                     "fitness_pred_sd": float(np.mean(fit_sd)) if fit_sd else None,
                     "fitness_val_spearman": (fit_val or {}).get("mean_spearman"),
                     "fitness_val_assays": (fit_val or {}).get("n_scored"),
                     "fitness_val_degenerate": (fit_val or {}).get("n_degenerate"),
                     "fitness_val_min_pred_sd": (fit_val or {}).get("min_pred_sd"),
                     "fitness_transfer_spearman": (fit_tr or {}).get("mean_spearman"),
                     "fitness_transfer_assays": (fit_tr or {}).get("n_scored"),
                     "fitness_transfer_degenerate": (fit_tr or {}).get("n_degenerate"),
                     # val minus transfer: how much of the head is the assay
                     # rather than the RNA. Positive means it learned the
                     # twelve assays it saw more than it learned fitness.
                     "fitness_generalisation_gap": (
                         None if not (fit_val and fit_tr)
                         or fit_val["mean_spearman"] is None
                         or fit_tr["mean_spearman"] is None
                         else round(fit_val["mean_spearman"]
                                    - fit_tr["mean_spearman"], 5)),
                     })
        # The per-assay breakdowns are LISTS and do not belong in a csv cell,
        # so they do not go in `hist`. `hist` is logged with `**hist[-1]` and
        # `test_head_metrics.py` asserts that every one of its keys is a
        # declared column -- an invariant that exists because stage 5 once
        # declared nine columns and passed forty. Putting them in `hist` and
        # then filtering them out at the `runlog.log` call satisfied neither
        # the csv nor the invariant; they live beside it instead, and reach
        # the results json by their own route.
        fit_detail.append({"epoch": ep,
                           "val": (fit_val or {}).get("per_assay"),
                           "transfer": (fit_tr or {}).get("per_assay")})
        if v_acc is not None and t_acc is not None:
            _vm = float(np.mean(vs_major)) if vs_major else float("nan")
            _vk = float(np.mean(vs_macro)) if vs_macro else float("nan")
            print(f"[seq] epoch {ep} head 4: train acc {t_acc:.4f}  "
                  f"VAL acc {v_acc:.4f}  gap {t_acc - v_acc:+.4f}  "
                  f"| val majority {_vm:.4f}  lift {v_acc - _vm:+.4f}  "
                  f"macro {_vk:.4f}  over {len(vs_acc)} batches", flush=True)
        print(f"[seq] epoch {ep} streams: "
              + "  ".join(f"{k} {pulled[k]:,} batches / {passes[k]} passes"
                          for k in ("ss", "reactivity", "fitness")), flush=True)
        print(f"[seq] epoch {ep}: {hist[-1]}  ({time.time()-t0:.0f}s)", flush=True)
        # `**hist[-1]`, not four of its thirteen fields picked out by hand.
        # `epoch` and `steps` are passed positionally above and would collide.
        runlog.log("epoch", epoch=ep, step=step, gstep=gstep, n_oom=n_oom,
                   **{k: v for k, v in hist[-1].items()
                      if k not in ("epoch", "steps")})
        save(ep, True)

    if not hist:
        # see train_pharos.py: a run that trained nothing must not clobber the
        # last real results file
        print("[seq] no epochs ran; leaving the existing results file alone",
              flush=True)
        return
    report = {"size": args.size, "params": pc, "stage_weights": weights,
              "stage6_enabled": use_fitness, "history": hist,
              "fitness_per_assay": fit_detail}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"seqstages_{args.size}_results.json").write_text(json.dumps(report, indent=1))
    print(f"\n[seq] -> {OUT / f'seqstages_{args.size}_results.json'}")


if __name__ == "__main__":
    main()
