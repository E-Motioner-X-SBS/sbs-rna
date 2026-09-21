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

from pharos.data.chemistry import N_DIMS, chain_chemistry            # noqa: E402
from pharos.data.vocab import PAD_ID, SYM2ID, encode_chain           # noqa: E402
from pharos.model.moe import RouterFeatures                          # noqa: E402
from pharos.model.pharos import Pharos, PharosConfig                 # noqa: E402
from train_block_scorer import gpu_free_gib                          # noqa: E402

#: the pretraining alphabet: elDORS is normalised to these
PRETRAIN_SYMBOLS = ("A", "C", "G", "U", "N")
MASK_ID = SYM2ID["UNK"]          # reuse UNK as the [MASK] symbol
MASK_FRAC = 0.15
#: mean span length; geometric, SpanBERT-style
SPAN_MEAN = 3.0


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


def iter_sequences(corpus: Path, min_len: int, max_len: int,
                   shards: Optional[int] = None) -> Iterator[str]:
    import pyarrow.parquet as pq
    files = sorted(corpus.glob("*.parquet"))
    if shards:
        files = files[:shards]
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=8192, columns=["sequence"]):
            for s in batch.column("sequence").to_pylist():
                if s is None:
                    continue
                n = len(s)
                if min_len <= n <= max_len:
                    yield s.upper().replace("T", "U")


def encode_batch(seqs: List[str], device) -> Dict[str, torch.Tensor]:
    L = max(len(s) for s in seqs)
    B = len(seqs)
    tok = np.full((B, L), PAD_ID, dtype=np.int16)
    chem = np.zeros((B, L, N_DIMS), dtype=np.float32)
    mask = np.zeros((B, L), dtype=bool)
    for i, s in enumerate(seqs):
        comps = list(s)
        t, _ = encode_chain(comps)
        n = len(t)
        tok[i, :n] = t
        chem[i, :n] = chain_chemistry(comps)
        mask[i, :n] = True
    return {"tokens": torch.as_tensor(tok, dtype=torch.long, device=device),
            "chem": torch.as_tensor(chem, device=device),
            "mask": torch.as_tensor(mask, device=device)}


def masked_chemistry(tokens: torch.Tensor, mask: torch.Tensor,
                     device) -> torch.Tensor:
    """Chemistry derived from the MASKED tokens, so it cannot leak the target.

    Every feature in `chain_chemistry` is a function of base identity, and dims
    0-4 are literally a one-hot of it. Computing it from the original sequence
    and masking only the token stream leaves the answer sitting in the chemistry
    channel. This recomputes it from what the model actually sees, so a masked
    position gets the `UNK` row and its stated unknown-residue fallback rather
    than any particular base.
    """
    import numpy as _np
    from pharos.data.chemistry import N_DIMS as _N
    from pharos.data.vocab import SYMBOLS as _SYM
    toks = tokens.detach().cpu().numpy()
    msk = mask.detach().cpu().numpy()
    B, L = toks.shape
    out = _np.zeros((B, L, _N), dtype=_np.float32)
    for i in range(B):
        n = int(msk[i].sum())
        comps = [_SYM[int(t)] if 0 <= int(t) < len(_SYM) else "UNK"
                 for t in toks[i, :n]]
        out[i, :n] = chain_chemistry(comps)
    return torch.as_tensor(out, device=device)


def apply_span_mask(tokens: torch.Tensor, mask: torch.Tensor, rng: np.random.Generator,
                    frac: float = MASK_FRAC, span_mean: float = SPAN_MEAN):
    """BERT corruption with geometric spans. Returns (inputs, targets, selected).

    80/10/10 as usual: replace with [MASK], keep, or substitute a random base.
    The 10% kept and 10% substituted stop the model from learning that a
    prediction is only ever needed where it sees the mask symbol.
    """
    B, L = tokens.shape
    sel = torch.zeros_like(mask)
    p = 1.0 / max(span_mean, 1.0)
    for b in range(B):
        n = int(mask[b].sum())
        if n < 8:
            continue
        want = max(1, int(frac * n))
        got = 0
        guard = 0
        while got < want and guard < 4 * want:
            guard += 1
            ln = min(int(rng.geometric(p)), 10, max(want - got, 1))
            st = int(rng.integers(0, max(n - ln, 1)))
            if bool(sel[b, st:st + ln].any()):
                continue
            sel[b, st:st + ln] = True
            got += ln
    sel &= mask
    targets = tokens.clone()
    targets[~sel] = -100
    inputs = tokens.clone()
    r = torch.rand(tokens.shape, device=tokens.device)
    inputs[sel & (r < 0.8)] = MASK_ID
    rand_pos = sel & (r >= 0.9)
    if bool(rand_pos.any()):
        inputs[rand_pos] = torch.randint(0, 4, (int(rand_pos.sum()),),
                                         device=tokens.device)
    return inputs, targets, sel


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
    ap.add_argument("--size", default="small", choices=("mini", "small"))
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--token-budget", type=int, default=24576,
                    help="padded tokens per step; PHAROS-Small peaks near 43 GiB "
                         "at 32k, so this leaves headroom on an 80 GiB card")
    ap.add_argument("--max-batch", type=int, default=512,
                    help="guard against a batch of thousands of 20-nt sequences")
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--n-loops", type=int, default=2,
                    help="loops give structural refinement depth; MLM needs few")
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--restart", action="store_true",
                    help="ignore an existing checkpoint and start from scratch")
    args = ap.parse_args()

    device = require_gpu(args)
    enable_gpu_fast_paths()
    cfg = PharosConfig.small() if args.size == "small" else PharosConfig.mini()
    cfg.max_length = max(cfg.max_length, args.max_len)
    model = Pharos(cfg).to(device)
    pc = model.param_counts()
    budget = int(args.tokens)
    print(f"[mlm] {args.size}: {pc['total']:,} total / {pc['active']:,} active")
    print(f"[mlm] budget {budget/1e9:.1f}B tokens = "
          f"{budget/max(pc['active'],1):.0f} tokens per active parameter "
          f"(~{budget/max(pc['active'],1)/20:.0f}x Chinchilla)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    rng = np.random.default_rng(0)
    CKPT.mkdir(parents=True, exist_ok=True)

    seen = step = 0
    # Resume, because this is the job that runs for days on a shared card and
    # will be interrupted. Without it every interruption discards everything and
    # the run can never finish -- an unattended pretraining script that restarts
    # from zero is not unattended, it is a loop that makes no progress.
    ck = CKPT / f"pretrain_{args.size}.pt"
    if ck.exists() and not args.restart:
        sd = torch.load(ck, map_location=device)
        model.load_state_dict(sd["model"])
        if "opt" in sd:
            opt.load_state_dict(sd["opt"])
        seen, step = int(sd.get("tokens", 0)), int(sd.get("step", 0))
        # advance the stream so a resumed run does not re-read the same
        # sequences it has already been trained on
        rng = np.random.default_rng(step)
        print(f"[mlm] resumed from {ck.name}: {seen/1e9:.3f}B tokens, "
              f"step {step:,}", flush=True)
        if seen >= budget:
            print(f"[mlm] budget already met ({seen/1e9:.2f}B >= "
                  f"{budget/1e9:.2f}B); nothing to do")
            return
    t0 = time.time()
    run: List[float] = []
    accs: List[float] = []
    hist: List[Dict] = []
    buf: List[str] = []
    stop = False
    while not stop:
        for s in iter_sequences(CORPUS, args.min_len, args.max_len, args.shards):
            buf.append(s)
            # Batch by TOKENS, not by sequence count. A fixed count is a latent
            # OOM: 64 sequences is 1.3k tokens if they are 20 nt and 65k if they
            # are 1,024, and PHAROS-Small at 65k tokens does not fit in 80 GiB
            # (stage 5 peaks at 43 GiB on a 32k budget). The budget also keeps
            # memory flat across a run instead of spiking whenever a batch of
            # long transcripts comes up.
            if (len(buf) * max(len(x) for x in buf) < args.token_budget
                    and len(buf) < args.max_batch):
                continue
            b = encode_batch(buf, device)
            buf = []
            inp, tgt, sel = apply_span_mask(b["tokens"], b["mask"], rng)
            # THE CHEMISTRY MUST BE RECOMPUTED FROM THE MASKED INPUT.
            #
            # `chain_chemistry` one-hot-encodes base identity in dims 0-4, and
            # `encode_batch` derives it from the ORIGINAL sequence -- so a
            # masked position still carried its own answer in the chemistry
            # channel and the model read it straight off. Measured: masked-token
            # accuracy 0.948 and loss 0.392 nats = 0.57 bits at step 100,
            # against a corpus entropy of 2.0165 bits/nt. A number below the
            # entropy of the data is not a good model, it is a leak.
            chem = masked_chemistry(inp, b["mask"], device)
            if not bool(sel.any()):
                continue
            feats = RouterFeatures(
                length=b["mask"].sum(1).float(),
                chem_summary=(chem.sum(1)
                              / b["mask"].sum(1, keepdim=True).clamp(min=1))[:, :5])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(inp, torch.zeros_like(inp), chem, b["mask"],
                            feats=feats, n_loops=args.n_loops, mlm=True)
                logits = out["mlm_logits"].float()
                loss = F.cross_entropy(logits[sel], tgt[sel])
                loss = loss + out["aux"]["balance_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                acc = float((logits[sel].argmax(-1) == tgt[sel]).float().mean())
            run.append(float(loss.detach()))
            accs.append(acc)
            seen += int(b["mask"].sum())
            step += 1

            if step % args.log_every == 0:
                el = time.time() - t0
                print(f"[mlm] step {step} {seen/1e6:.1f}M tok "
                      f"loss {np.mean(run[-args.log_every:]):.4f} "
                      f"acc {np.mean(accs[-args.log_every:]):.4f} "
                      f"{seen/max(el,1e-9)/1e3:.1f}k tok/s", flush=True)
                hist.append({"step": step, "tokens": seen,
                             "loss": float(np.mean(run[-args.log_every:])),
                             "acc": float(np.mean(accs[-args.log_every:]))})
            if step % args.ckpt_every == 0 or seen >= budget:
                torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "tokens": seen, "step": step}, ck)
            if seen >= budget:
                stop = True
                break
        else:
            print("[mlm] corpus exhausted; looping", flush=True)

    report = {"size": args.size, "params": pc, "tokens": seen, "steps": step,
              "tokens_per_active_param": round(seen / max(pc["active"], 1), 1),
              "final_loss": float(np.mean(run[-200:])),
              "final_accuracy": float(np.mean(accs[-200:])),
              # RNA sequence entropy is 2.0165 bits/nt; a model that has learned
              # nothing sits at that, so bits/token below it is the real gain
              "bits_per_token": round(float(np.mean(run[-200:])) / np.log(2), 4),
              "history": hist}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pretrain_{args.size}_results.json").write_text(json.dumps(report, indent=1))
    print(f"\nfinal loss {report['final_loss']:.4f} = "
          f"{report['bits_per_token']:.4f} bits/token against the corpus's "
          f"2.0165 bits/nt")
    print(f"masked-token accuracy {report['final_accuracy']:.4f}")
    print(f"\n[mlm] -> {OUT / f'pretrain_{args.size}_results.json'}")


if __name__ == "__main__":
    main()
