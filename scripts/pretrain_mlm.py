#!/usr/bin/env python3
"""Stage 1 of the §12.1 curriculum: masked-language pretraining on elDORS.

The representation the structural stages fine-tune. 12M sequences are on disk as
48 parquet shards; the budget is **25B tokens, not 323B** (§12.2): at 63M active
parameters 323B is 5,295 tokens/parameter, 265x Chinchilla, the worst corner of
the quantisation-degradation curve, and for no information gain, because RNA
sequence entropy is **2.0165 bits/nt** — the corpus is redundant rather than
rich. 25B gives ~400 tok/param, about 20x Chinchilla.

Three things follow from the spec and are implemented rather than assumed.

**Five symbols, not twenty-five** (D6, VOCAB.md). elDORS is pre-normalised to
ACGTN, so the pretraining alphabet is five plus specials. NucleicBERT's vocab of
25 over the same alphabet is dead weight: tokens that never occur still occupy
embedding rows and softmax mass. The structural vocabulary is wider because
structures are *not* normalised, and the two meet in one model because the
output projection is tied to the input embedding.

**Span masking, not only single tokens.** A helix is locally periodic, so a
model can fill a single masked base from its immediate neighbours without
learning anything about structure. Masking contiguous spans forces the
prediction to come from further away, which is where the base-pairing signal is.

**Chemistry is available at pretraining too.** Every one of the 24 dims except
the shifted-pKa flag is computable from sequence, so stage 1 trains the same
chemistry projection the structural stages use rather than leaving it cold until
stage 5.

GPU only. Refuses to start on a busy card rather than OOM mid-run.

Usage:
    /store/shuvam/.venv/bin/python scripts/pretrain_mlm.py --tokens 5e9
"""
from __future__ import annotations

import argparse
import json
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/derived/parquet_starter"
OUT = ROOT / "data/samples/analysis"
CKPT = ROOT / "data/derived/checkpoints"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pharos.data.chemistry_torch import BatchChemistry               # noqa: E402
from pharos.train.telemetry import RunLog                            # noqa: E402
from pharos.train.checkpoint import atomic_save
from pharos.train.muon import Muon, muon_param_groups
from pharos.data.vocab import PAD_ID, SYM2ID, SYMBOLS, encode_chain  # noqa: E402
from pharos.model.moe import RouterFeatures                          # noqa: E402
from pharos.model.pharos import Pharos, PharosConfig                 # noqa: E402
from train_block_scorer import gpu_free_gib                          # noqa: E402

#: the pretraining alphabet: elDORS is normalised to these
PRETRAIN_SYMBOLS = ("A", "C", "G", "U", "N")
MASK_ID = SYM2ID["UNK"]          # reuse UNK as the [MASK] symbol
MASK_FRAC = 0.15
#: mean span length; geometric, SpanBERT-style
SPAN_MEAN = 3.0

#: A100 bf16 dense peak, FLOP/s. The cost model in §12.2 assumed 35% of this
#: and nothing ever checked; the trainer now measures it every log interval.
A100_BF16_PEAK = 312e12


def FLOPS_PER_PARAM_TOKEN(n_loops: int) -> float:
    """FLOPs per active parameter per token, for `n_loops` recycles.

    The last loop is forward and backward, 6; loops 1..N-1 run under
    `torch.no_grad()` (trunk.py, the one-step gradient), so each adds a forward
    only, 2. n_loops=2 is therefore 8, not 12.
    """
    return 6.0 + 2.0 * max(n_loops - 1, 0)


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


#: Shards reserved for evaluation and NEVER trained on.
#:
#: The held-out sample used to be drawn from the same stream the trainer reads,
#: by taking the first 512 sequences of a shuffled shard ordering. Nothing
#: excluded them from training, so the model trained on its own evaluation set
#: and the number went to pieces in the most flattering direction: between
#: steps 4750 and 5000 the in-loop "held-out" improved 1.4298 -> 0.9090 bits
#: and 0.5767 -> 0.7491 accuracy, while a genuinely separate sample moved
#: 1.8146 -> 1.8047 and 0.3997 -> 0.4064. A 0.9-bit gap, widening, entirely
#: memorisation.
#:
#: The fix is exclusion at the source: the last `N_HELDOUT_SHARDS` shards in
#: sorted order are removed from the training stream and are the ONLY place the
#: evaluation sample is drawn from. Sorted, not shuffled, so the split is the
#: same on every run and across restarts -- a held-out set that depends on a
#: seed is one a resumed run can silently train on.
N_HELDOUT_SHARDS = 2


def _corpus_files(corpus) -> List[Path]:
    """Every parquet shard under `corpus`, which may be dirs OR files.

    Accepting files matters: `heldout_files` returns shards, and passing those
    straight back in is how the evaluation pool is built. Globbing a file for
    `*.parquet` returns nothing, so the first version of this silently produced
    an EMPTY held-out set -- an evaluation that reports on zero sequences and
    raises nothing.
    """
    items = [corpus] if isinstance(corpus, (str, Path)) else list(corpus)
    out: List[Path] = []
    for it in items:
        q = Path(it)
        if q.is_dir():
            out.extend(q.glob("*.parquet"))
        elif q.suffix == ".parquet":
            out.append(q)
    return sorted(out)


def heldout_files(corpus) -> List[Path]:
    """The shards reserved for evaluation, `N_HELDOUT_SHARDS` PER CORPUS.

    Per corpus, not per run. `_corpus_files` returns one globally sorted list,
    and `data/derived/parquet_mars/*` sorts before `data/derived/parquet_starter/*`,
    so taking the last two of the combined list reserved
    `eldors_c020_shard0003` and `eldors_c020_shard0004` and reserved **nothing
    at all from MARS**: all 71 MARS shards were in the training stream and no
    held-out number said anything about them. It also made the held-out set the
    two shards of the SHORTEST chunk of a length-sorted corpus -- elDORS c020,
    mean 356 nt against 1,272 in c001 -- so the evaluation distribution was not
    the training distribution.

    Grouping by parent directory fixes both: each corpus contributes its own
    last shards, the mixture is represented, and the split is still a pure
    function of the file names, so a resumed run computes the same one.
    """
    files = _corpus_files(corpus)
    by_root: Dict[Path, List[Path]] = {}
    for f in files:
        by_root.setdefault(f.parent, []).append(f)
    out: List[Path] = []
    for root in sorted(by_root):
        grp = by_root[root]
        if len(grp) > N_HELDOUT_SHARDS:
            out.extend(grp[-N_HELDOUT_SHARDS:])
    return sorted(out)


def iter_sequences(corpus, min_len: int, max_len: int,
                   shards: Optional[int] = None,
                   rng: Optional[np.random.Generator] = None,
                   include_heldout: bool = False) -> Iterator[str]:
    """Sequences from the parquet corpus, optionally with the SHARD ORDER shuffled.

    elDORS is sorted by length -- chunk 001 averages 1,252 nt and chunk 020
    averages 162 -- and the shards are named after their chunk, so reading them
    in sorted order walks the corpus from longest to shortest. That is a
    length curriculum nobody specified, and a worse one than the within-pool
    sorting `_pack_pool` deliberately undoes, because it runs across the entire
    run rather than inside one pool.

    Passing `rng` shuffles which shard is read when, so a pool draws from across
    the corpus. `shards` still truncates deterministically, because the tests
    that use it want a fixed, reproducible subset.
    """
    import pyarrow.parquet as pq
    # `corpus` may be one directory or several. MARS lands in its own
    # directory rather than beside elDORS so the two stay separable -- turning
    # MARS off is not passing it, and a shard's provenance is its path.
    dirs = [corpus] if isinstance(corpus, (str, Path)) else list(corpus)
    files = _corpus_files(corpus)
    if not include_heldout:
        reserved = set(heldout_files(corpus))
        files = [f for f in files if f not in reserved]
    if shards:
        files = files[:shards]
    elif rng is not None:
        files = list(files)
        rng.shuffle(files)
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=8192, columns=["sequence"]):
            for s in batch.column("sequence").to_pylist():
                if s is None:
                    continue
                n = len(s)
                if min_len <= n <= max_len:
                    yield s.upper().replace("T", "U")


#: Sequence length is padded up to a multiple of this.
#:
#: Quantising at all is what makes `torch.compile` usable: the delta-rule chunk
#: loop is a Python loop, so inductor specialises on the chunk count, and an
#: unquantised L -- 969 distinct values over 400k sequences -- recompiles on
#: nearly every batch. Quantised there are a handful, compiled once each and
#: cached on disk between runs, and batch size stays dynamic.
#:
#: 128 was chosen when the corpus was chunks c001-c008 of elDORS, whose median
#: length is 522. The corpus now spans all twenty chunks and the median is 261,
#: and rounding 151 up to 256 is not the same bargain as rounding 522 up to 640.
#: Re-measured on the corrected corpus:
#:
#:     quantum   padding   widths   useful tokens/step
#:        none      0.0%      969        24,830
#:          32      4.1%       32        23,747
#:          64      8.1%       16        22,709      <- here
#:         128     15.1%        8        20,963
#:         256     25.1%        4        18,406
#:
#: 64 is the knee. 32 buys four more points of padding for twice the graphs, and
#: 128 gives back seven. It is still half a GDN chunk, so the scan's ragged tail
#: is at most half empty.
LEN_QUANTUM = 64


def _quantised(n: int, q: int = LEN_QUANTUM) -> int:
    return ((n + q - 1) // q) * q


def encode_batch(seqs: List[str], quantum: int = LEN_QUANTUM) -> tuple:
    """`(tokens, mask, lengths)` as numpy. No chemistry, and no device.

    It used to compute `chain_chemistry` for every sequence here and return it
    on the GPU -- and then `masked_chemistry` recomputed the whole thing from
    the masked token stream and threw this away. The chemistry was being built
    twice per step, on CPU, and used once. It is now built once, on the GPU,
    from the masked inputs, by `BatchChemistry`.
    """
    L = _quantised(max(len(s) for s in seqs), quantum)
    B = len(seqs)
    tok = np.full((B, L), PAD_ID, dtype=np.int64)
    mask = np.zeros((B, L), dtype=bool)
    lengths = np.zeros(B, dtype=np.int64)
    for i, s in enumerate(seqs):
        e, _ = encode_chain(list(s))
        n = len(e)
        tok[i, :n] = e
        mask[i, :n] = True
        lengths[i] = n
    return tok, mask, lengths


def apply_span_mask(tok: np.ndarray, lengths: np.ndarray,
                    rng: np.random.Generator, frac: float = MASK_FRAC,
                    span_mean: float = SPAN_MEAN):
    """BERT corruption with geometric spans. Returns (inputs, targets, selected).

    80/10/10 as usual: replace with [MASK], keep, or substitute a random base.
    The 10% kept and 10% substituted stop the model from learning that a
    prediction is only ever needed where it sees the mask symbol.

    Numpy, on the host, before anything is transferred. The previous version
    took `(B, L)` GPU tensors and read `int(mask[b].sum())` inside the batch
    loop, which is a device synchronisation per sequence per step -- the card
    idling while Python decided where to put the next span. The row lengths are
    already known from packing, so nothing has to be read back at all.
    """
    B, L = tok.shape
    sel = np.zeros((B, L), dtype=bool)
    p_geom = 1.0 / max(span_mean, 1.0)
    for b in range(B):
        n = int(lengths[b])
        if n < 8:
            continue
        want = max(1, int(frac * n))
        got = 0
        guard = 0
        while got < want and guard < 4 * want:
            guard += 1
            ln = min(int(rng.geometric(p_geom)), 10, max(want - got, 1))
            st = int(rng.integers(0, max(n - ln, 1)))
            if sel[b, st:st + ln].any():
                continue
            sel[b, st:st + ln] = True
            got += ln
    valid = np.arange(L)[None, :] < lengths[:, None]
    sel &= valid
    targets = np.where(sel, tok, -100)
    inputs = tok.copy()
    r = rng.random((B, L))
    inputs[sel & (r < 0.8)] = MASK_ID
    rand_pos = sel & (r >= 0.9)
    k = int(rand_pos.sum())
    if k:
        inputs[rand_pos] = rng.integers(0, 4, k)
    return inputs, targets, sel


def _pack_pool(seqs: List[str], token_budget: int, max_batch: int,
               rng: np.random.Generator,
               quantum: int = LEN_QUANTUM) -> List[List[str]]:
    """Sort a pool by length, then cut it into token-budgeted batches.

    Sorting is the whole point: a batch pads to its longest member, so mixing a
    20-nt sequence with a 1,024-nt one costs 1,004 tokens of compute that
    produce no gradient. Measured on 400,000 elDORS sequences at the shipped
    24,576-token budget, the stream-order packer spends **42.9% of every step
    on padding** and takes 14,985 steps to cover what length-sorted packing
    covers in 8,712 -- the same tokens, 1.72x the wall clock.

    The batch ORDER is then shuffled. A sorted pool fed in order would train
    short-to-long within each pool, which correlates batch length with
    optimiser step and is a curriculum nobody asked for.
    """
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    batches: List[List[str]] = []
    buf: List[str] = []
    for i in order:
        buf.append(seqs[i])
        # ascending length, so the last appended IS the longest; the batch is
        # padded to the QUANTISED width, which is what the budget must count
        if (len(buf) * _quantised(len(seqs[i]), quantum) >= token_budget
                or len(buf) >= max_batch):
            batches.append(buf)
            buf = []
    if buf:
        batches.append(buf)
    rng.shuffle(batches)
    return batches


def iter_batches(corpus, min_len: int, max_len: int, token_budget: int,
                 max_batch: int, rng: np.random.Generator,
                 shards: Optional[int] = None, pool: int = 131072,
                 quantum: int = LEN_QUANTUM) -> Iterator[List[str]]:
    """Length-bucketed batches over the corpus stream.

    A pool of `pool` sequences is buffered, sorted, packed and shuffled, then
    the next pool is read. The pool bounds how far a sequence can move from its
    position in the corpus, so this stays a stream -- 131,072 sequences is
    about 70 MB of Python strings, and large enough that the length histogram
    inside a pool matches the corpus.
    """
    buf: List[str] = []
    for s in iter_sequences(corpus, min_len, max_len, shards, rng):
        buf.append(s)
        if len(buf) >= pool:
            yield from _pack_pool(buf, token_budget, max_batch, rng, quantum)
            buf = []
    if buf:
        yield from _pack_pool(buf, token_budget, max_batch, rng, quantum)


#: Peak allocation per token of batch budget, GiB. Measured on PHAROS-Small,
#: compiled, with expandable segments, on the corrected corpus:
#:
#:     budget    peak      GiB/token    tok/s    MFU
#:     24,576   50.9 GiB   2.07e-3      36.5k    6.6%
#:     32,768   65.9 GiB   2.01e-3      38.8k    7.1%
#:     36,864   74.1 GiB   2.01e-3      41.1k    7.3%
#:
#: Those three runs were ~350 steps each and they all UNDER-report, because
#: peak memory is not a function of the token count alone. SWA and FULL both
#: materialise a `(B, 1, L, L)` mask, so at a fixed token budget a batch of
#: 35x1024 costs far more than one of 280x128 -- measured at 24,576 tokens,
#: 61.2 GiB at L=1024 against 39.2 GiB at L=128. A short benchmark never draws
#: the worst batch in the corpus; a long run does. At budget 35,840 the fitted
#: 2.01e-3 predicted a 72.0 GiB peak and the run reached **79.14 GiB and OOMed**
#: at 72.4M tokens.
#:
#: So the constant is the WORST observed cost per token (2.21e-3) plus margin,
#: not the mean, and `auto_token_budget` holds back more. The principled fix is
#: a cost model carrying the quadratic term -- allowed tokens at width w being
#: `target / (a + b*w)` -- which needs per-width calibration this run has not
#: done. Until then: a conservative constant, a bigger reserve, and a trainer
#: that survives an OOM instead of dying of one.
GIB_PER_TOKEN = 2.40e-3

#: Per size, when the architecture makes the constant above wrong. shared400
#: runs the trunk under gradient checkpointing, so it stores one block's
#: activations rather than eighteen: measured 12.0 GiB at 32,768 tokens with
#: L=512, i.e. 3.66e-4 GiB/token, which is 6.6x cheaper than the dense-MoE
#: constant. Using the shared constant for it would cap the budget at a
#: seventh of what fits and leave the card three-quarters idle -- the opposite
#: of the OOM the constant exists to prevent, and just as wasteful.
#: The margin here is 25%, the same worst-case-over-mean logic as above.
GIB_PER_TOKEN_BY_SIZE = {"shared400": 4.6e-4}


def auto_token_budget(free_gib: float, reserve_gib: float = 10.0,
                      lo: int = 8192, hi: int = 131072,
                      size: str = "") -> int:
    """The largest token budget that fits in `free_gib`, less a reserve.

    Hardcoding the budget means picking between leaving a third of the card
    unused and dying when a neighbour allocates. This card is shared: at 36,864
    tokens the run holds 74.1 of 79.3 GiB, and five spare gigabytes is not a
    margin on a machine where somebody else's job appears without warning.

    So the budget is computed from what is actually free when the run starts.
    An empty card gets ~36,800 tokens a step at the dense-MoE cost, and
    ~150,000 for a checkpointed shared400; a card with 20 GiB already taken
    gets proportionally less; below the floor the trainer refuses to start
    rather than thrash. `reserve_gib` covers the allocator's own slack and a
    small neighbour. The ceiling is 131,072 rather than 40,960 because
    gradient checkpointing moved the binding constraint from memory to
    throughput, and capping at the old value would leave the card idle.

    This does not protect against a neighbour arriving MID-run. Nothing short
    of a hard memory cap does, which is why the run checkpoints every 250 steps.
    """
    usable = max(free_gib - reserve_gib, 0.0)
    per = GIB_PER_TOKEN_BY_SIZE.get(size, GIB_PER_TOKEN)
    b = int(usable / per) // 1024 * 1024
    return max(lo, min(hi, b))


def lr_at(seen: float, budget: float, peak: float,
          warmup_frac: float = 0.01, floor_frac: float = 0.1) -> float:
    """Linear warm-up then cosine decay, as a function of TOKENS seen.

    Stage 1 had no schedule at all: a constant 6e-4 from the first step to the
    last, no warm-up and no decay. Warm-up matters because AdamW's second
    moment is meaningless for the first few hundred steps and a full-rate step
    against it is how a transformer diverges early; decay matters because a
    constant rate stops making progress long before the budget is spent, which
    is what the loss curve flattening at ~1.45 while accuracy sat at 0.388 was
    showing.

    Keyed on tokens rather than optimiser steps on purpose. Greedy packing puts
    a different number of sequences in every batch, so the step count for a
    given token budget is not known in advance and is not even stable across
    shuffles -- the same off-by-N that broke `OneCycleLR`'s `total_steps` in
    stage 5. Token progress is exact, and it resumes correctly for free.
    """
    x = min(max(seen / max(budget, 1.0), 0.0), 1.0)
    if x < warmup_frac:
        return peak * x / warmup_frac
    y = (x - warmup_frac) / max(1.0 - warmup_frac, 1e-9)
    return peak * (floor_frac + (1.0 - floor_frac) * 0.5 * (1.0 + math.cos(math.pi * y)))


def _clean_state(model) -> Dict[str, torch.Tensor]:
    """`state_dict` without `torch.compile`'s `_orig_mod.` prefix.

    `torch.compile` wraps the module, so every trunk tensor is saved as
    `trunk._orig_mod.blocks.0....`. Stage 5 loads this checkpoint into an
    uncompiled model, where those keys match nothing -- and the curriculum
    would silently start from random weights again, which is the exact failure
    `--init-from`'s load guard exists to catch. Strip the prefix at the source
    instead of relying on the guard to notice.
    """
    return {k.replace("._orig_mod.", "."): v
            for k, v in model.state_dict().items()}


def require_gpu(args) -> torch.device:
    if not args.device.startswith("cuda"):
        raise SystemExit("GPU only; pass --device cuda once one is free.")
    mem = gpu_free_gib()
    if mem is None:
        raise SystemExit("no GPU visible to nvidia-smi")
    free, total = mem
    print(f"[mlm] GPU {free:.1f} of {total:.1f} GiB free")
    if free < args.min_free_gib:
        raise SystemExit(f"only {free:.1f} GiB free; nothing started.")
    return torch.device(args.device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-free-gib", type=float, default=30.0)
    ap.add_argument("--tokens", type=float, default=5e9,
                    help="§12.2 stages 5B -> 25B; stop when metrics flatten")
    ap.add_argument("--size", default="small",
                    choices=("mini", "small", "base400", "shared400"))
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--optimizer", default="adamw", choices=("adamw", "muon"),
                    help="muon orthogonalises the update for 2-D weights and "
                         "leaves embeddings, norms, biases and the per-expert "
                         "modulation vectors on AdamW")
    ap.add_argument("--muon-lr", type=float, default=0.01,
                    help="Muon's rate is NOT comparable to AdamW's: an "
                         "orthogonal update has unit spectral norm, so its "
                         "natural scale is ~0.02 against AdamW's ~6e-4")
    ap.add_argument("--token-budget", type=int, default=0,
                    help="0 (the default) sizes it from free VRAM via "
                         "`auto_token_budget`; a positive value overrides")
    ap.add_argument("--vram-target", type=float, default=0.90,
                    help="fraction of the card the batch should occupy. The "
                         "budget grows toward it and shrinks off an OOM, so a "
                         "mis-estimated cost per token self-corrects instead of "
                         "leaving the card half empty or dying")
    ap.add_argument("--adapt-every", type=int, default=200,
                    help="steps between budget adjustments")
    ap.add_argument("--token-budget-reserve", type=float, default=10.0,
                    help="GiB held back from the automatic budget for the "
                         "allocator's slack and a small neighbour")
    ap.add_argument("--token-budget-fixed", type=int, default=24576,
                    help="padded tokens per step; PHAROS-Small peaks near 43 GiB "
                         "at 32k, so this leaves headroom on an 80 GiB card")
    ap.add_argument("--max-batch", type=int, default=512,
                    help="guard against a batch of thousands of 20-nt sequences")
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--n-loops", type=int, default=2,
                    help="loops give structural refinement depth; MLM needs few")
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--corpus", type=Path, nargs="*", default=None,
                    help="parquet directories to pretrain on. Default is "
                         "elDORS alone; add data/derived/parquet_mars to "
                         "include the filtered MARS structured-ncRNA shards")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--heldout-seq", type=int, default=512,
                    help="sequences in the fixed held-out sample scored at "
                         "every checkpoint; 0 disables it")
    ap.add_argument("--heldout-seed", type=int, default=1234)
    ap.add_argument("--ckpt-every", type=int, default=250,
                    help="steps between checkpoints; at ~2 s/step the old 2000 "
                         "meant losing up to 65 minutes to an interruption, on "
                         "a card that gets taken back without warning")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip torch.compile on the trunk; it costs ~3 min of "
                         "warm-up for 8 graphs and returns 1.65x on step time")
    ap.add_argument("--restart", action="store_true",
                    help="ignore an existing checkpoint and start from scratch; "
                         "the existing one is RENAMED, not overwritten")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint path; defaults to "
                         "data/derived/checkpoints/pretrain_<size>.pt. Point a "
                         "smoke test somewhere else and it cannot touch a real "
                         "run's weights")
    args = ap.parse_args()

    device = require_gpu(args)
    enable_gpu_fast_paths()
    # resolve() first: a relative --corpus is the natural thing to type and
    # `relative_to(ROOT)` throws on one, which turned a cosmetic log line into a
    # crash before the first step
    corpora = [Path(c).resolve() for c in (args.corpus or [CORPUS])]
    corpora = [c for c in corpora if c.is_dir()]
    if not corpora:
        raise SystemExit(f"no corpus directory found in {args.corpus or [CORPUS]}")

    def _rel(d: Path) -> str:
        try:
            return str(d.relative_to(ROOT))
        except ValueError:
            return str(d)

    _n = sum(1 for d in corpora for _ in d.glob("*.parquet"))
    print(f"[mlm] corpus: {_n} shards across "
          + ", ".join(_rel(d) for d in corpora), flush=True)
    if args.token_budget <= 0:
        mem = gpu_free_gib()
        free = mem[0] if mem else args.min_free_gib
        args.token_budget = auto_token_budget(free, args.token_budget_reserve,
                                              size=args.size)
        print(f"[mlm] {free:.1f} GiB free -> token budget {args.token_budget:,} "
              f"(~{args.token_budget * GIB_PER_TOKEN_BY_SIZE.get(args.size, GIB_PER_TOKEN):.1f}"
              " GiB predicted peak)",
              flush=True)
    cfg = {"small": PharosConfig.small, "mini": PharosConfig.mini,
           "base400": PharosConfig.base400,
           "shared400": PharosConfig.shared400}[args.size]()
    cfg.max_length = max(cfg.max_length, args.max_len)
    model = Pharos(cfg).to(device)
    pc = model.param_counts()
    # Compile the TRUNK, not the whole model: the trunk is 16 blocks run twice
    # per step and is where the elementwise work is -- the profile put only 18%
    # of GPU time in tensor-core GEMMs and the rest in copies, gating multiplies
    # and the delta-rule scan, which is exactly what inductor fuses. The heads
    # and the MLM projection are a small tail and compiling them only adds
    # graphs. Verified equal to eager to 1e-6 relative in fp32.
    if not args.no_compile:
        # dynamic=False, not True. On shared400 the dynamic-shape compile of an
        # 18-block checkpointed trunk did not finish in 25 minutes and produced
        # no graph at all; static shapes compile a bucket in two to three and
        # then run. Length quantisation already bounds the shape set to one
        # (B, L) per bucket, so the cache limit is raised to cover them -- the
        # default of 8 is what makes dynamo give up and fall back to eager.
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 256
        model.trunk = torch.compile(model.trunk, dynamic=False)
        print(f"[mlm] trunk compiled (static shapes, L quantised to "
              f"{LEN_QUANTUM}); first batch of each width pays the warm-up",
              flush=True)
    budget = int(args.tokens)
    print(f"[mlm] {args.size}: {pc['total']:,} total / {pc['active']:,} active")
    print(f"[mlm] budget {budget/1e9:.1f}B tokens = "
          f"{budget/max(pc['active'],1):.0f} tokens per active parameter "
          f"(~{budget/max(pc['active'],1)/20:.0f}x Chinchilla)")

    # Two optimisers, and the weight-decay split neither of them had before.
    #
    # Every trainer here passed `weight_decay=0.01` to `model.parameters()`
    # undifferentiated. Decaying a LayerNorm gain has no scale-invariance
    # argument behind it, and decaying the zero-initialised per-expert
    # modulation actively pulls the MoE's specialisation back toward the shared
    # network it is trying to differ from -- the `mod` tensor whose measured std
    # is 0.008 is what that decay was fighting. `muon_param_groups` does the
    # split for both paths.
    if args.optimizer == "muon":
        mp, ag = muon_param_groups(model, args.muon_lr, args.lr)
        opts = [Muon(mp, lr=args.muon_lr, momentum=0.95),
                torch.optim.AdamW(ag, lr=args.lr, betas=(0.9, 0.95))]
        print(f"[mlm] Muon on {len(mp)} matrices "
              f"({sum(p.numel() for p in mp)/1e6:.0f}M params) at lr "
              f"{args.muon_lr:g}; AdamW on the rest at {args.lr:g}", flush=True)
    else:
        mp, ag = muon_param_groups(model, args.muon_lr, args.lr)
        # AdamW over everything, but with decay only where it belongs: the
        # matrices Muon would have taken join the decay group, the norms and
        # biases stay at zero decay.
        groups = [{"params": mp, "weight_decay": 0.01}] + ag
        opts = [torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))]
        n_nd = sum(len(g["params"]) for g in ag if g["weight_decay"] == 0.0)
        print(f"[mlm] AdamW, weight decay on {len(mp) + len(ag[0]['params'])} "
              f"tensors and OFF for {n_nd} norms/biases/expert vectors",
              flush=True)
    opt = opts[0]
    rng = np.random.default_rng(0)
    CKPT.mkdir(parents=True, exist_ok=True)

    seen = step = resumed_padded = 0
    hist_resumed: List[Dict] = []
    # Resume, because this is the job that runs for days on a shared card and
    # will be interrupted. Without it every interruption discards everything and
    # the run can never finish -- an unattended pretraining script that restarts
    # from zero is not unattended, it is a loop that makes no progress.
    ck = args.ckpt or (CKPT / f"pretrain_{args.size}.pt")
    if ck.exists() and args.restart:
        # --restart used to overwrite in place, silently. A smoke test run with
        # it against the default path destroyed 70.8M tokens of stage 1 -- the
        # flag did exactly what it said, and what it said was not recoverable.
        # Renaming costs nothing and makes the mistake undoable.
        keep = ck.with_suffix(".superseded.pt")
        ck.replace(keep)
        print(f"[mlm] --restart: existing checkpoint moved to {keep.name} "
              f"rather than overwritten", flush=True)
    if ck.exists() and not args.restart:
        sd = torch.load(ck, map_location=device)
        want = set(model.state_dict())
        got = {k: v for k, v in sd["model"].items()}
        if not (set(got) <= want):      # saved uncompiled, loading compiled
            got = {k.replace("trunk.", "trunk._orig_mod.", 1)
                   if k.startswith("trunk.") and k not in want else k: v
                   for k, v in got.items()}
        # Tolerate parameters that did not exist when the checkpoint was
        # written, and nothing else.
        #
        # The architecture grows between runs -- `diff_pair` and `coev_proj`
        # were added after this run started -- and a strict load turns every
        # such addition into a refusal to resume a job with hundreds of
        # millions of tokens in it. But strict=False on its own would also
        # accept loading a shared400 checkpoint into a `small` model and
        # training a mostly-random network that reports as resumed, so the
        # tolerance is bounded: new tensors are allowed, and more than 5% of
        # the model missing is an error.
        res = model.load_state_dict(got, strict=False)
        if res.unexpected_keys:
            raise SystemExit(
                f"{len(res.unexpected_keys)} tensors in the checkpoint have no "
                f"home in this model, e.g. {res.unexpected_keys[:3]} -- that is "
                f"a different architecture, not a resume.")
        if res.missing_keys:
            frac = len(res.missing_keys) / max(len(model.state_dict()), 1)
            if frac > 0.05:
                raise SystemExit(
                    f"{len(res.missing_keys)} of {len(model.state_dict())} "
                    f"tensors ({100*frac:.0f}%) are absent from the checkpoint. "
                    f"Refusing to resume a mostly-random model.")
            print(f"[mlm] {len(res.missing_keys)} tensor(s) postdate this "
                  f"checkpoint and start fresh: "
                  f"{', '.join(res.missing_keys[:4])}"
                  f"{' ...' if len(res.missing_keys) > 4 else ''}", flush=True)
        # Restore EVERY optimiser, and refuse to restore across a change of
        # optimiser: AdamW's moments mean nothing to Muon, and loading them
        # would resume a run whose optimiser state is silently garbage.
        saved_kind = sd.get("optimizer", "adamw")
        if saved_kind != args.optimizer:
            print(f"[mlm] checkpoint was written with --optimizer {saved_kind} "
                  f"and this run is {args.optimizer}; keeping the WEIGHTS and "
                  f"starting fresh optimiser state", flush=True)
        elif "opts" in sd and len(sd["opts"]) == len(opts):
            for o, st in zip(opts, sd["opts"]):
                o.load_state_dict(st)
        elif "opt" in sd:
            opt.load_state_dict(sd["opt"])
        seen, step = int(sd.get("tokens", 0)), int(sd.get("step", 0))
        hist_resumed = list(sd.get("history", []))
        resumed_padded = int(sd.get("padded", 0))
        # advance the stream so a resumed run does not re-read the same
        # sequences it has already been trained on
        rng = np.random.default_rng(step)
        print(f"[mlm] resumed from {ck.name}: {seen/1e9:.3f}B tokens, "
              f"step {step:,}", flush=True)
        if seen >= budget:
            print(f"[mlm] budget already met ({seen/1e9:.2f}B >= "
                  f"{budget/1e9:.2f}B); nothing to do")
            return
    padded = int(resumed_padded)
    seen_mark, padded_mark = seen, padded
    t0 = tmark = time.time()
    # Telemetry is appended and flushed per row. `hist` used to reach disk only
    # in the report at the end of a run, and on a shared card runs do not end --
    # 188.6M tokens of stage 1 left no history file at all, only a text log.
    runlog = RunLog(ROOT, "stage1_mlm", [
        "tokens", "padded_tokens", "ce_nats", "bits_per_token", "balance",
        "total_loss", "lr", "masked_accuracy", "tok_per_s", "padded_per_s",
        "mfu", "pad_frac", "token_budget", "n_oom", "peak_gib", "note",
        # Perplexity, exp(CE) over masked positions. Cross-entropy and bits are
        # the same quantity on a log scale; perplexity is the same quantity
        # again, as an effective branching factor -- "the model is as uncertain
        # as if choosing uniformly among this many bases". On a 4-letter
        # alphabet that makes the scale immediately legible: 4.0 is no
        # knowledge, 1.0 is certainty, and the corpus entropy of 2.0165 bits is
        # ppl 4.05. A 0.02-bit move looks like nothing and is a 1.4% change in
        # branching factor.
        "perplexity",
    ], manifest={
        "size": args.size, "config": cfg.__dict__, "params": pc,
        "token_budget_requested": args.token_budget,
        "token_budget_total": budget, "n_loops": args.n_loops,
        "lr_peak": args.lr, "compiled": not args.no_compile,
        "len_quantum": LEN_QUANTUM, "gib_per_token": GIB_PER_TOKEN,
        "resumed_from_tokens": seen, "resumed_from_step": step,
        "checkpoint": str(ck),
    })
    if step:
        runlog.event("resumed", step=step, tokens=seen, padded_tokens=padded)
    run: List[float] = []
    bals: List[float] = []
    accs: List[float] = []
    hist: List[Dict] = list(hist_resumed)
    stop = False
    budget_tokens = args.token_budget
    n_oom = 0
    # The chemistry tables live on the device for the whole run; building them
    # is a few hundred calls to `residue_chemistry`, done once.
    batch_chem = BatchChemistry(SYMBOLS, device)

    # A fixed held-out sample, drawn once and reused for every checkpoint.
    # Built here rather than imported so the trainer has no dependency on the
    # standalone evaluator, and normalised the same way `iter_sequences`
    # normalises: the corpus is DNA-alphabet and an un-normalised sample scores
    # a healthy model at 3.42 bits against a 1.9964-bit unigram baseline.
    heldout = None
    if args.heldout_seq > 0:
        # islice, NOT list(): the corpus is 40.9M sequences and 20.1B tokens,
        # and materialising it to draw 512 would exhaust the machine before
        # step 0. The shard order is shuffled by the seeded rng, so taking a
        # prefix still samples across the corpus rather than one chunk.
        # Drawn ONLY from the reserved shards, which `iter_sequences` removes
        # from the training stream. Sampled across the whole reserved set
        # rather than as a prefix, so it is not one contiguous region of one
        # shard.
        # Reservoir sampling over the WHOLE reserved stream.
        #
        # This used to be `islice(..., heldout_seq * 20)` and then a choice
        # from that, with a comment claiming it sampled "across the whole
        # reserved set rather than as a prefix". islice IS a prefix:
        # `iter_sequences` with no rng walks the shards in sorted order, so the
        # pool was the first 10,240 sequences of one shard and the comment
        # asserted the opposite of what the code did. A reservoir is O(1)
        # memory, needs no second pass, and is uniform over every sequence in
        # the reserved shards by construction.
        _r = np.random.default_rng(args.heldout_seed)
        _hs: List[str] = []
        for _n, _seq in enumerate(iter_sequences(
                heldout_files(corpora), args.min_len, args.max_len,
                shards=None, include_heldout=True)):
            if len(_hs) < args.heldout_seq:
                _hs.append(_seq)
            else:
                _j = int(_r.integers(0, _n + 1))
                if _j < args.heldout_seq:
                    _hs[_j] = _seq
        # A quarter of the training budget: evaluation runs under no_grad and
        # could afford more, but the batches must stay small enough that a
        # checkpoint evaluation never becomes the step that OOMs the run.
        # `budget_tokens` rather than `args.token_budget` only because the two
        # are the same object by this line and the local says so.
        _groups = _pack_pool(_hs, max(4096, budget_tokens // 4), args.max_batch,
                             np.random.default_rng(args.heldout_seed))
        print(f"[mlm] held-out sample: {len(_hs):,} sequences in "
              f"{len(_groups)} fixed batches, drawn from "
              f"{len(heldout_files(corpora))} RESERVED shards that training "
              f"never reads", flush=True)

        def heldout(mdl, _g=_groups, _bc=batch_chem):
            was = mdl.training
            mdl.eval()
            r = np.random.default_rng(args.heldout_seed)
            tc = tk = 0.0
            nm = 0
            with torch.no_grad():
                for grp in _g:
                    tn, mn, ln = encode_batch(grp)
                    ip, tg, sl = apply_span_mask(tn, ln, r)
                    if not sl.any():
                        continue
                    ip_t = torch.as_tensor(ip, device=device)
                    tg_t = torch.as_tensor(tg, device=device)
                    sl_t = torch.as_tensor(sl, device=device)
                    bm = torch.as_tensor(mn, device=device)
                    cm = _bc(ip_t, bm)
                    nr = bm.sum(1)
                    ft = RouterFeatures(
                        length=nr.float(),
                        chem_summary=(cm.sum(1) / nr.unsqueeze(1).clamp(min=1))[:, :5])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o = mdl(ip_t, torch.zeros_like(ip_t), cm, bm,
                                feats=ft, n_loops=args.n_loops, mlm=True)
                    # genuinely hidden positions only: the 10% of selected
                    # positions BERT leaves unchanged are copied with 99.84%
                    # accuracy and inflate the figure by +0.060
                    hd = sl_t & (ip_t == MASK_ID)
                    if not bool(hd.any()):
                        continue
                    lg = o["mlm_logits"].float()[hd]
                    y = tg_t[hd]
                    tc += float(F.cross_entropy(lg, y, reduction="sum"))
                    tk += float((lg.argmax(-1) == y).sum())
                    nm += int(hd.sum())
            mdl.train(was)
            ce = tc / max(nm, 1)
            return {"ce_nats": ce, "bits": ce / float(np.log(2)),
                    "perplexity": float(np.exp(ce)),
                    "accuracy": tk / max(nm, 1), "n_masked": nm}
    while not stop:
        # Batches are length-bucketed and budgeted by TOKENS, not by sequence
        # count. A fixed count is a latent OOM -- 64 sequences is 1.3k tokens if
        # they are 20 nt and 65k if they are 1,024, and PHAROS-Small at 65k
        # tokens does not fit in 80 GiB. Bucketing is what makes the budget
        # honest: without it 42.9% of each "24,576-token" step was padding.
        for group in iter_batches(corpora, args.min_len, args.max_len,
                                  budget_tokens, args.max_batch, rng,
                                  args.shards):
          try:
              tok_np, mask_np, lengths = encode_batch(group)
              inp_np, tgt_np, sel_np = apply_span_mask(tok_np, lengths, rng)
              if not sel_np.any():
                  continue
              inp = torch.as_tensor(inp_np, device=device)
              tgt = torch.as_tensor(tgt_np, device=device)
              sel = torch.as_tensor(sel_np, device=device)
              bmask = torch.as_tensor(mask_np, device=device)
              # THE CHEMISTRY MUST BE DERIVED FROM THE MASKED INPUT.
              #
              # `chain_chemistry` one-hot-encodes base identity in dims 0-4, so
              # chemistry built from the ORIGINAL sequence leaves a masked
              # position carrying its own answer, and the model reads it straight
              # off. Measured when it did: masked-token accuracy 0.948 and loss
              # 0.392 nats = 0.57 bits at step 100, against a corpus entropy of
              # 2.0165 bits/nt. A number below the entropy of the data is not a
              # good model, it is a leak. `inp` is the masked stream.
              chem = batch_chem(inp, bmask)
              n_real = bmask.sum(1)
              feats = RouterFeatures(
                  length=n_real.float(),
                  chem_summary=(chem.sum(1)
                                / n_real.unsqueeze(1).clamp(min=1))[:, :5])
              lr_now = lr_at(seen, budget, args.lr)
              for _o in opts:
                  scale = (args.muon_lr / max(args.lr, 1e-12)
                           if isinstance(_o, Muon) else 1.0)
                  for g in _o.param_groups:
                      g["lr"] = lr_now * scale
              with torch.autocast("cuda", dtype=torch.bfloat16):
                  out = model(inp, torch.zeros_like(inp), chem, bmask,
                              feats=feats, n_loops=args.n_loops, mlm=True)
                  logits = out["mlm_logits"].float()
                  ce = F.cross_entropy(logits[sel], tgt[sel])
                  bal = out["aux"]["balance_loss"]
                  loss = ce + bal
              for _o in opts:
                _o.zero_grad(set_to_none=True)
              loss.backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
              opt.step()

              with torch.no_grad():
                  acc = float((logits[sel].argmax(-1) == tgt[sel]).float().mean())
              # CE and the balance term are tracked SEPARATELY. The Switch
              # balance loss is `n_experts * sum(frac * pbar)`, which is 1.0 at
              # perfect uniformity, not 0 -- so at `balance_weight` 0.01 across 16
              # blocks it adds a floor of ~0.16 nats that never goes away.
              # Reporting their sum as "loss" and dividing it by ln 2 inflated
              # bits/token by ~0.23 bits and made a model that had gone BELOW the
              # corpus entropy look as though it were still above it.
              run.append(float(ce.detach()))
              bals.append(float(bal.detach()))
              accs.append(acc)
              seen += int(lengths.sum())
              padded += int(tok_np.size)
              step += 1

              if step % args.log_every == 0:
                  # PER-INTERVAL, not cumulative since t0. Averaging from the
                  # start folds in `torch.compile`'s warm-up -- 8 quantised
                  # widths, forward and backward -- so the reported rate climbs
                  # for hundreds of steps and never reaches the steady state it
                  # is supposed to report. The first interval after a resume or a
                  # compile still carries warm-up; the ones after it do not.
                  now = time.time()
                  el = now - tmark
                  tmark = now
                  # MFU is spent on PADDED tokens, so it is padded throughput
                  # that divides into the card's peak. Reporting only real tokens
                  # hid a 42.9% padding tax behind a number that looked fine.
                  rate_p = (padded - padded_mark) / max(el, 1e-9)
                  rate_r = (seen - seen_mark) / max(el, 1e-9)
                  seen_mark, padded_mark = seen, padded
                  mfu = FLOPS_PER_PARAM_TOKEN(args.n_loops) * pc["active"] \
                      * rate_p / A100_BF16_PEAK
                  print(f"[mlm] step {step} {seen/1e6:.1f}M tok "
                        f"ce {np.mean(run[-args.log_every:]):.4f} "
                        f"({np.mean(run[-args.log_every:])/np.log(2):.3f} bits, "
                        f"ppl {np.exp(np.mean(run[-args.log_every:])):.3f}) "
                        f"bal {np.mean(bals[-args.log_every:]):.3f} "
                        f"lr {lr_now:.2e} "
                        f"acc {np.mean(accs[-args.log_every:]):.4f} "
                        f"{rate_r/1e3:.1f}k tok/s "
                        f"({rate_p/1e3:.1f}k padded, pad {100*(1-seen/max(padded,1)):.1f}%) "
                        f"MFU {100*mfu:.1f}%", flush=True)
                  _ce = float(np.mean(run[-args.log_every:]))
                  _bal = float(np.mean(bals[-args.log_every:]))
                  runlog.log("step", step=step, tokens=seen,
                             padded_tokens=padded,
                             ce_nats=round(_ce, 6),
                             bits_per_token=round(_ce / np.log(2), 6),
                             perplexity=round(float(np.exp(_ce)), 6),
                             balance=round(_bal, 6),
                             total_loss=round(_ce + _bal, 6), lr=lr_now,
                             masked_accuracy=round(
                                 float(np.mean(accs[-args.log_every:])), 6),
                             tok_per_s=round(rate_r, 1),
                             padded_per_s=round(rate_p, 1), mfu=round(mfu, 6),
                             pad_frac=round(1 - seen / max(padded, 1), 6),
                             token_budget=budget_tokens, n_oom=n_oom,
                             peak_gib=round(
                                 torch.cuda.max_memory_allocated() / 2**30, 2))
                  hist.append({"step": step, "tokens": seen, "padded": padded,
                               "lr": lr_now,
                               "balance": float(np.mean(bals[-args.log_every:])),
                               "tok_per_s": rate_r,
                               "padded_per_s": rate_p, "mfu": mfu,
                               "loss": float(np.mean(run[-args.log_every:])),
                               "acc": float(np.mean(accs[-args.log_every:]))})
                # ---- seek the VRAM ceiling -------------------------------
              # GIB_PER_TOKEN was fitted on d_model 512; a wider model costs
              # more per token and a hand-updated constant would be wrong again
              # the next time the shape changes. Measure the peak instead and
              # walk the budget toward the target: up when there is headroom,
              # and the OOM handler already walks it down. Growth is capped at
              # 10% a step so it approaches the ceiling rather than jumping
              # over it, and it stops entirely once an OOM has been seen, since
              # past that point the ceiling is known.
              if (step % args.adapt_every == 0 and n_oom == 0
                      and torch.cuda.is_available()):
                  total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
                  peak = torch.cuda.max_memory_allocated() / 2**30
                  want = args.vram_target * total_gib
                  if peak < 0.85 * want:
                      grown = min(int(budget_tokens * 1.10),
                                  int(budget_tokens * want / max(peak, 1e-6)))
                      grown = max(8192, grown // 1024 * 1024)
                      if grown > budget_tokens:
                          print(f"[mlm] peak {peak:.1f} of {want:.1f} GiB target; "
                                f"token budget {budget_tokens:,} -> {grown:,}",
                                flush=True)
                          runlog.event(f"budget grown to {grown}", step=step,
                                       token_budget=grown, peak_gib=round(peak, 2))
                          budget_tokens = grown
                          torch.cuda.reset_peak_memory_stats()
                          break        # rebuild the stream at the new budget

              if step % args.ckpt_every == 0 or seen >= budget:
                  atomic_save({"cfg": cfg.__dict__, "model": _clean_state(model),
                              "opt": opt.state_dict(),
                              # every optimiser, or a Muon resume silently
                              # drops the AdamW moments for the embeddings
                              "opts": [o.state_dict() for o in opts],
                              "optimizer": args.optimizer, "padded": padded,
                              "tokens": seen, "step": step,
                              "history": hist, "run_id": runlog.run_id}, ck)
                  # Held-out evaluation, at every checkpoint.
                  #
                  # The loss printed above is a rolling mean over whatever
                  # shard is streaming, and shard order is shuffled once and
                  # then read through, so a 100-step window at 228k tokens a
                  # step sits largely inside ONE shard. That number moved from
                  # 1.488 bits to 1.863 and back over 300 steps of this run
                  # with the model improving monotonically the whole time --
                  # it was measuring the data. A fixed sample, fixed masking
                  # and the same batches every time is the only way the run
                  # can tell its own progress from its corpus's variance.
                  if heldout is not None:
                      hv = heldout(model)
                      print(f"[mlm] held-out {hv['bits']:.4f} bits "
                            f"ppl {hv['perplexity']:.3f} acc {hv['accuracy']:.4f} "
                            f"(fixed {hv['n_masked']:,} masked positions)",
                            flush=True)
                      runlog.log(kind="eval", step=step, tokens=seen,
                                 ce_nats=round(hv["ce_nats"], 6),
                                 bits_per_token=round(hv["bits"], 6),
                                 masked_accuracy=round(hv["accuracy"], 6),
                                 note="held-out fixed sample")
              if seen >= budget:
                  stop = True
                  break
          except torch.OutOfMemoryError:
            # A run of this length must not die of one batch. Peak memory
            # depends on the batch SHAPE and not only on its token count,
            # so a budget safe for two thousand steps can still meet a pool
            # of long sequences that it is not -- which is exactly how
            # 72.4M tokens of stage 1 ended. Drop the batch, shrink the
            # budget, keep the checkpoint, carry on. The iterator has to be
            # rebuilt because the budget is baked into it at creation.
            n_oom += 1
            for _o in opts:
                _o.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            floor = 1024
            shrunk = max(floor, int(budget_tokens * 0.85) // 1024 * 1024)
            if shrunk == budget_tokens:
                # Already at the floor and still out of memory: shrinking
                # again changes nothing and the loop spins. shared400 did
                # exactly this -- 22 OOMs at step 0, each "recovering" to the
                # same 8,192 it had just failed at, for as long as it was left
                # running. Surviving a bad batch is right; pretending to
                # survive an impossible configuration is not.
                raise RuntimeError(
                    f"out of memory at the minimum token budget ({floor:,}); "
                    f"this configuration does not fit on this card. Reduce the "
                    f"model, enable grad_checkpoint, or free GPU memory.")
            budget_tokens = shrunk
            print(f"[mlm] OOM #{n_oom} at step {step}; token budget "
                  f"-> {budget_tokens:,}, rebuilding the stream and continuing",
                  flush=True)
            runlog.event(f"OOM #{n_oom}: token budget -> {budget_tokens}",
                         step=step, tokens=seen, token_budget=budget_tokens,
                         n_oom=n_oom)
            break

        else:
            print("[mlm] corpus exhausted; looping", flush=True)

    report = {"size": args.size, "params": pc, "tokens": seen, "steps": step,
              "tokens_per_active_param": round(seen / max(pc["active"], 1), 1),
              "final_ce": float(np.mean(run[-200:])),
              "final_balance": float(np.mean(bals[-200:])) if bals else None,
              "final_accuracy": float(np.mean(accs[-200:])),
              # RNA sequence entropy is 2.0165 bits/nt; a model that has learned
              # nothing sits at that, so bits/token below it is the real gain
              # from CE ALONE. The balance term is a regulariser with a
              # non-zero floor and has no business in a bits/token figure.
              "bits_per_token": round(float(np.mean(run[-200:])) / np.log(2), 4),
              "perplexity": round(float(np.exp(np.mean(run[-200:]))), 4),
              # throughput, so the cost model reads a measurement instead of
              # assuming 35% MFU as it did for the whole of v0.1 and v0.2
              "padded_tokens": padded,
              "padding_fraction": round(1 - seen / max(padded, 1), 4),
              "peak_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 1)
                           if device.type == "cuda" else None),
              "compiled": not args.no_compile,
              "best_mfu": max((h.get("mfu", 0.0) for h in hist), default=0.0),
              "best_tok_per_s": max((h.get("tok_per_s", 0.0) for h in hist), default=0.0),
              "history": hist}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pretrain_{args.size}_results.json").write_text(json.dumps(report, indent=1))
    print(f"\nfinal CE {report['final_ce']:.4f} nats = "
          f"{report['bits_per_token']:.4f} bits/token against the corpus's "
          f"2.0165 bits/nt (balance term {report['final_balance']:.3f}, "
          f"excluded -- its floor is 0.01 x 16 blocks = 0.16)")
    print(f"masked-token accuracy {report['final_accuracy']:.4f}")
    print(f"\n[mlm] -> {OUT / f'pretrain_{args.size}_results.json'}")


if __name__ == "__main__":
    main()
