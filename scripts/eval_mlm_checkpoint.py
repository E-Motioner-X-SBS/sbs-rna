#!/usr/bin/env python3
"""Score an MLM checkpoint on a FIXED held-out sample. The run had no such thing.

`pretrain_mlm.py` reports a rolling mean of the training loss over whatever
data happens to be streaming. Shard order is shuffled once at stream creation
and then read through, so a 100-step window at 228k tokens a step -- 22.8M
tokens -- sits largely inside a single shard. When that number moves, it is not
possible to say whether the model changed or the data did.

That ambiguity is not hypothetical. This run went 1.488 bits at step 1700 to
1.863 at step 1900, a swing back to where it had been 1,000 steps earlier, and
the training loss alone cannot distinguish "the model collapsed" from "the
stream reached harder shards".

So: a fixed set of sequences, a fixed masking seed, and the same batch for
every checkpoint. Differences are then the model.

    python scripts/eval_mlm_checkpoint.py --ckpt A.pt --ckpt B.pt

The sample is drawn with a fixed seed from shards the trainer reads LAST, so
for a run at 1% of its corpus it is effectively held out -- stated that way
rather than called a validation set, because the corpus has no declared split
and a shard the run will eventually reach is not held out forever.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pharos.model.pharos import Pharos, PharosConfig          # noqa: E402


def _trainer():
    """Import the trainer's own encoders so eval cannot drift from training."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_pm", ROOT / "scripts/pretrain_mlm.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["_pm"] = m
    spec.loader.exec_module(m)
    return m


def build_fixed_batches(pm, corpus: List[Path], n_seq: int, budget: int,
                        seed: int, min_len: int = 20, max_len: int = 1024
                        ) -> List:
    """The same sequences, the same masking, every time.

    **Sequences must be normalised exactly as `iter_sequences` normalises
    them**: uppercased, with T mapped to U. The corpus stores DNA-alphabet
    sequences -- a sample of elDORS is 43,642 T and zero U -- and the trainer
    converts on read. Passing the raw string instead encodes every T as an
    unknown symbol, so the model is scored on a sequence it cannot read, and
    this reported 3.42 bits against a 1.9964-bit unigram baseline: a number
    that looks like a catastrophically broken model and is entirely an
    artefact of the evaluator. The same checkpoint on correctly normalised
    batches scores 1.82 bits.

    The length filter must match the trainer's for the same reason.
    """
    import pyarrow.parquet as pq
    files: List[Path] = []
    for d in corpus:
        files += sorted(Path(d).glob("*.parquet"))
    if not files:
        raise SystemExit("no parquet shards found")
    # take from the END of the sorted list: the trainer shuffles shard order
    # with its own rng, but at ~1% of the corpus most shards are still unread
    rng = np.random.default_rng(seed)
    seqs: List[str] = []
    for f in reversed(files):
        t = pq.read_table(f, columns=["sequence"])
        col = t["sequence"].to_pylist()
        # exactly what iter_sequences does, and for the reason in the docstring
        col = [s.upper().replace("T", "U") for s in col if s]
        col = [s for s in col if min_len <= len(s) <= max_len]
        if not col:
            continue
        take = min(len(col), n_seq - len(seqs))
        idx = rng.choice(len(col), take, replace=False)
        seqs += [col[int(i)] for i in idx]
        if len(seqs) >= n_seq:
            break
    return seqs


@torch.no_grad()
def score(model, pm, seqs, device, budget: int, n_loops: int, seed: int):
    """Mean masked cross-entropy and accuracy over the fixed sample.

    This mirrors the training step exactly, including the rule that chemistry
    is derived from the MASKED input: `chain_chemistry` one-hot-encodes base
    identity, so chemistry built from the original sequence hands a masked
    position its own answer. The trainer records measuring 0.57 bits against a
    2.0165 bits/nt corpus when that leaked. An evaluator with the leak would
    report a number far better than training and look like a fixed bug.
    """
    import torch.nn.functional as F
    from pharos.model.moe import RouterFeatures

    rng = np.random.default_rng(seed)
    batch_chem = pm.BatchChemistry(pm.SYMBOLS, device)
    tot_ce = tot_correct = 0.0
    tot_masked = 0

    # deterministic length-bucketed packing, the trainer's own quantum
    groups, cur, cur_len = [], [], 0
    for s in sorted(seqs, key=len):
        L = pm._quantised(len(s))
        if cur and (len(cur) + 1) * max(cur_len, L) > budget:
            groups.append(cur); cur, cur_len = [], 0
        cur.append(s); cur_len = max(cur_len, L)
    if cur:
        groups.append(cur)

    for group in groups:
        tok_np, mask_np, lengths = pm.encode_batch(group)
        inp_np, tgt_np, sel_np = pm.apply_span_mask(tok_np, lengths, rng)
        if not sel_np.any():
            continue
        inp = torch.as_tensor(inp_np, device=device)
        tgt = torch.as_tensor(tgt_np, device=device)
        sel = torch.as_tensor(sel_np, device=device)
        bmask = torch.as_tensor(mask_np, device=device)
        chem = batch_chem(inp, bmask)
        n_real = bmask.sum(1)
        feats = RouterFeatures(
            length=n_real.float(),
            chem_summary=(chem.sum(1) / n_real.unsqueeze(1).clamp(min=1))[:, :5])
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            out = model(inp, torch.zeros_like(inp), chem, bmask,
                        feats=feats, n_loops=n_loops, mlm=True)
        lg = out["mlm_logits"].float()[sel]
        y = tgt[sel]
        tot_ce += float(F.cross_entropy(lg, y, reduction="sum"))
        tot_correct += float((lg.argmax(-1) == y).sum())
        tot_masked += int(sel.sum())

    ce = tot_ce / max(tot_masked, 1)
    return {"ce_nats": ce, "bits": ce / float(np.log(2)),
            "perplexity": float(np.exp(ce)),
            "accuracy": tot_correct / max(tot_masked, 1),
            "n_masked": tot_masked, "n_batches": len(groups)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, action="append", required=True)
    ap.add_argument("--size", default="shared400")
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[ROOT / "data/derived/parquet_starter",
                             ROOT / "data/derived/parquet_mars"])
    ap.add_argument("--n-seq", type=int, default=2048)
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--n-loops", type=int, default=2)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=1024,
                    help="must match the trainer, or the model is scored "
                         "outside the distribution it was trained on")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--append-csv", type=Path,
                    default=ROOT / "data/samples/analysis/runs/heldout_mlm.csv",
                    help="accumulate results so the run has a validation CURVE "
                         "rather than isolated readings; the training loss "
                         "cannot provide one because it moves with the shard")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    pm = _trainer()
    device = torch.device(args.device)
    seqs = build_fixed_batches(pm, args.corpus, args.n_seq,
                               args.token_budget, args.seed,
                               args.min_len, args.max_len)
    print(f"[eval] fixed sample: {len(seqs):,} sequences, "
          f"{sum(len(s) for s in seqs):,} nt, "
          f"len {args.min_len}-{args.max_len}, seed {args.seed}")
    print("[eval] corpus entropy reference: 2.0165 bits/nt")

    cfg = getattr(PharosConfig, args.size)()
    model = Pharos(cfg).to(device).eval()
    print(f"\n{'checkpoint':40s} {'step':>7s} {'bits':>7s} {'ppl':>7s} {'acc':>7s}")
    for ck in args.ckpt:
        st = torch.load(ck, map_location=device, weights_only=False)
        sd = {k.replace("_orig_mod.", ""): v for k, v in st.get("model", st).items()}
        model.load_state_dict(sd, strict=False)
        r = score(model, pm, seqs, device, args.token_budget, args.n_loops,
                  args.seed)
        print(f"  {ck.name:38s} {st.get('step','?'):>7} {r['bits']:7.4f} "
              f"{r['perplexity']:7.4f} {r['accuracy']:7.4f}")
        if not args.no_csv:
            import csv as _csv
            import datetime as _dt
            args.append_csv.parent.mkdir(parents=True, exist_ok=True)
            new = not args.append_csv.exists()
            with args.append_csv.open("a", newline="") as fh:
                w = _csv.writer(fh)
                if new:
                    w.writerow(["timestamp", "checkpoint", "step", "tokens",
                                "bits", "perplexity", "accuracy", "n_masked",
                                "n_seq", "seed", "min_len", "max_len"])
                w.writerow([_dt.datetime.now().isoformat(timespec="seconds"),
                            ck.name, st.get("step", ""), st.get("tokens", ""),
                            round(r["bits"], 5), round(r["perplexity"], 5),
                            round(r["accuracy"], 5), r["n_masked"],
                            len(seqs), args.seed, args.min_len, args.max_len])
    if not args.no_csv:
        print(f"\n[eval] appended to {args.append_csv.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
