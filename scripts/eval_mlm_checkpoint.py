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
                        seed: int, min_len: int = 20, max_len: int = 1024,
                        split: str = "legacy") -> List:
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
    if split == "stratified":
        # The corpus's own length histogram, over the reserved shards of both
        # corpora. Measured band frequencies (`pool_composition.json`); the
        # sample is filled per band so one shard's length bias cannot set the
        # score. Interleaved so the bands are actually reachable -- a
        # sequential read of the reserved set would exhaust c020 first.
        files = list(pm.heldout_files(corpus))
        want = {(20, 80): 0.065, (80, 160): 0.305, (160, 320): 0.241,
                (320, 640): 0.278, (640, max_len + 1): 0.112}
        got: dict = {b: [] for b in want}
        need = {b: max(1, int(round(n_seq * f))) for b, f in want.items()}
        for seq in pm.iter_sequences(files, min_len, max_len, None,
                                     np.random.default_rng(seed),
                                     n_interleave=8):
            for (lo, hi) in want:
                if lo <= len(seq) < hi and len(got[(lo, hi)]) < need[(lo, hi)]:
                    got[(lo, hi)].append(seq)
                    break
            if all(len(got[b]) >= need[b] for b in want):
                break
        out: List[str] = []
        for b in want:
            out += got[b]
        short = {f"{b[0]}-{b[1]-1}": (len(got[b]), need[b])
                 for b in want if len(got[b]) < need[b]}
        if short:
            print(f"[eval] stratified sample under-filled: {short}")
        return out
    if split == "reserved":
        # The shards the trainer's own `heldout_files` removes from the
        # training stream -- the guarantee stated rather than inferred, and
        # covering BOTH corpora, so the number speaks for the mixture the
        # model is actually trained on.
        files = list(pm.heldout_files(corpus))
    else:
        # LEGACY, kept so the accumulated curve stays one measurement.
        # It walks the globally sorted shard list backwards and fills from the
        # first shard with enough sequences, which on this corpus is
        # `eldors_c020_shard0004` -- a shard the trainer does reserve, so the
        # sample is in fact held out. It is held out by an accident of sort
        # order rather than by construction: it is elDORS only, it is the
        # SHORTEST chunk of a length-sorted corpus (mean 356 nt against 1,272
        # in c001), and nothing here would have noticed if the ordering had put
        # a trained-on shard last instead.
        files = []
        for d in corpus:
            files += sorted(Path(d).glob("*.parquet"))
        files = list(reversed(files))
    if not files:
        raise SystemExit("no parquet shards found")
    rng = np.random.default_rng(seed)
    seqs: List[str] = []
    for f in files:
        t = pq.read_table(f, columns=["sequence"])
        col = t["sequence"].to_pylist()
        # exactly what iter_sequences does, and for the reason in the docstring
        col = [s.upper().replace("T", "U") for s in col if s]
        col = [s for s in col if min_len <= len(s) <= max_len]
        if not col:
            continue
        # On the reserved split, an equal share from EACH reserved shard.
        # Filling greedily from the first shard is what made the legacy sample
        # one shard of one corpus: `heldout_files` returns MARS before elDORS,
        # so a greedy fill would have swung the sample from all-elDORS to
        # all-MARS and called that an improvement.
        want = (-(-n_seq // len(files))) if split == "reserved" else n_seq
        take = min(len(col), want, n_seq - len(seqs))
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
        inp_t = torch.as_tensor(inp_np, device=device)
        tgt_t = torch.as_tensor(tgt_np, device=device)
        sel_t = torch.as_tensor(sel_np, device=device)
        inp, sel = inp_np, sel_np
        bmask = torch.as_tensor(mask_np, device=device)
        chem = batch_chem(inp_t, bmask)
        n_real = bmask.sum(1)
        feats = RouterFeatures(
            length=n_real.float(),
            chem_summary=(chem.sum(1) / n_real.unsqueeze(1).clamp(min=1))[:, :5])
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            out = model(inp_t, torch.zeros_like(inp_t), chem, bmask,
                        feats=feats, n_loops=n_loops, mlm=True)
        # Score ONLY positions the model could not see.
        #
        # BERT's 80/10/10 leaves 10% of selected positions unchanged and
        # replaces 10% with a random base; both are scored by default, and the
        # model copies the unchanged ones with 99.84% accuracy. Measured on
        # this corpus that inflates the headline by +0.060 -- 0.3829 pooled
        # against 0.3229 on genuinely hidden positions. An evaluator whose job
        # is to say whether the model is learning must not count the positions
        # where it is only reading.
        hidden = sel & (inp == pm.MASK_ID)
        if not hidden.any():
            continue
        h = torch.as_tensor(hidden, device=device)
        lg = out["mlm_logits"].float()[h]
        y = tgt_t[h]
        tot_ce += float(F.cross_entropy(lg, y, reduction="sum"))
        tot_correct += float((lg.argmax(-1) == y).sum())
        tot_masked += int(h.sum())

    ce = tot_ce / max(tot_masked, 1)
    return {"ce_nats": ce, "bits": ce / float(np.log(2)),
            "perplexity": float(np.exp(ce)),
            "accuracy": tot_correct / max(tot_masked, 1),
            "n_masked": tot_masked, "n_batches": len(groups)}



#: Tensors legitimately absent from a checkpoint written before they existed.
#:
#: Two kinds, and both have to be named explicitly rather than waved through by
#: a blanket rule, because the whole point of the check is that a MISSING TRUNK
#: WEIGHT must stop the evaluation:
#:
#:  - heads added after a run started, which the MLM path does not touch;
#:  - running-statistic BUFFERS that are defined to start uninitialised. The
#:    motif bank's query mean is one: it is guarded by `query_mean_n == 0`, so
#:    an absent buffer means "no estimate yet", which is exactly its value at
#:    the start of any run.
#:
#: This list is the maintenance cost of the check and it is worth paying. It
#: caught its own author: adding `query_mean`/`query_mean_n` to the bank made
#: the evaluator refuse every checkpoint from step 8,250 on, and the held-out
#: curve stopped until the two names were added here.
_MAY_BE_MISSING = ("heads.pair.geometry.", "heads.residue.motif.",
                   "motifs.query_mean")


def _check_load(model, sd, ck) -> None:
    """`load_state_dict(strict=False)` and then LOOK at what it dropped.

    The evaluator used to call `strict=False` and ignore the result. That is
    the failure mode this file exists to catch, applied to itself: a load that
    cannot fail, scoring a model whose weights may be partly random, reporting
    a number with no marker that anything went wrong. A renamed trunk
    parameter, a config mismatch, a half-written checkpoint -- each one gives a
    plausible-looking bits figure that is simply wrong, and the curve absorbs
    it.

    So: tolerate exactly the heads that post-date the run, name anything else,
    and refuse to score a model that is missing trunk weight.
    """
    r = model.load_state_dict(sd, strict=False)
    full = model.state_dict()
    unexpected = list(r.unexpected_keys)
    bad = [k for k in r.missing_keys if not k.startswith(_MAY_BE_MISSING)]
    ok = [k for k in r.missing_keys if k.startswith(_MAY_BE_MISSING)]
    n_bad = sum(full[k].numel() for k in bad if k in full)
    if ok:
        print(f"[eval] {Path(ck).name}: {len(ok)} post-run head tensors absent "
              f"(heads 9/10), not used by the MLM path")
    if unexpected:
        print(f"[eval] WARNING {len(unexpected)} unexpected keys, e.g. "
              f"{unexpected[:3]}")
    if bad:
        raise SystemExit(
            f"[eval] REFUSING to score {ck}: {len(bad)} tensors "
            f"({n_bad:,} parameters) are missing from the checkpoint and would "
            f"be scored at random initialisation -- {bad[:5]}")


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
    ap.add_argument("--split", choices=("legacy", "reserved", "stratified"),
                    default="legacy",
                    help="which shards the fixed sample comes from. 'reserved' "
                         "is the trainer's own held-out split across BOTH "
                         "corpora; 'legacy' reproduces the sample the existing "
                         "curve was measured on, and is the default so that "
                         "curve stays one series. They are different samples "
                         "and their numbers must not be plotted together. "
                         "'stratified' is the one to use: it draws from the "
                         "reserved shards of both corpora with the LENGTH "
                         "HISTOGRAM MATCHED TO THE CORPUS, because 'legacy' "
                         "is one shard of one band (mean 185 nt) and measured "
                         "the model 0.19 bits better than a corpus-weighted "
                         "sample does -- 18.2% better than unigram against a "
                         "true 8.8% at step 9,000.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--append-csv", type=Path,
                    default=ROOT / "data/samples/analysis/runs/heldout_mlm.csv",
                    help="accumulate results so the run has a validation CURVE "
                         "rather than isolated readings; the training loss "
                         "cannot provide one because it moves with the shard")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--complementarity", action="store_true", default=True,
                    help="also measure whether a VISIBLE Watson-Crick partner "
                         "improves the prediction. Accuracy can rise from "
                         "base composition or from copying; this cannot -- the "
                         "two conditions differ only in whether the partner is "
                         "readable, so the gap is pairing being used.")
    ap.add_argument("--no-complementarity", dest="complementarity",
                    action="store_false")
    args = ap.parse_args()

    pm = _trainer()
    device = torch.device(args.device)
    seqs = build_fixed_batches(pm, args.corpus, args.n_seq,
                               args.token_budget, args.seed,
                               args.min_len, args.max_len, split=args.split)
    print(f"[eval] fixed sample: {len(seqs):,} sequences, "
          f"{sum(len(s) for s in seqs):,} nt, "
          f"len {args.min_len}-{args.max_len}, seed {args.seed}")
    print("[eval] corpus entropy reference: 2.0165 bits/nt")
    print("[eval] scored on GENUINELY MASKED positions only (see score())")

    cfg = getattr(PharosConfig, args.size)()
    model = Pharos(cfg).to(device).eval()
    print(f"\n{'checkpoint':40s} {'step':>7s} {'bits':>7s} {'ppl':>7s} {'acc':>7s}")
    for ck in args.ckpt:
        st = torch.load(ck, map_location=device, weights_only=False)
        sd = {k.replace("_orig_mod.", ""): v for k, v in st.get("model", st).items()}
        _check_load(model, sd, ck)
        r = score(model, pm, seqs, device, args.token_budget, args.n_loops,
                  args.seed)
        r["comp_gap"] = float("nan")
        if args.complementarity:
            try:
                from probe_rna_understanding import complementarity, real_base_pairs
                # n=350 pairs gives the gap a standard error of +/-0.033, at
                # which every change measured across this run -- including one
                # reported as "almost quadrupled" -- sits under 2 sigma. The
                # gap being consistently POSITIVE is evidence; its movements at
                # that sample size were not. More chains, longer chains.
                pd = real_base_pairs(limit_chains=60, max_len=600)
                if pd:
                    cr = complementarity(model, pm, pd, device, args.n_loops)
                    vh, vt = cr["partner_visible"]
                    mh, mt = cr["partner_masked"]
                    r["comp_visible"] = vh / max(vt, 1)
                    r["comp_masked"] = mh / max(mt, 1)
                    r["comp_gap"] = r["comp_visible"] - r["comp_masked"]
                    r["comp_n"] = vt
                    # the error bar travels with the number, so a reader
                    # cannot mistake a 0.6-sigma wobble for a trend
                    r["comp_se"] = float(
                        np.sqrt(2 * 0.25 * 0.75 / max(vt, 1)))
            except Exception as e:                           # noqa: BLE001
                print(f"  (complementarity unavailable: {e})")
        print(f"  {ck.name:38s} {st.get('step','?'):>7} {r['bits']:7.4f} "
              f"{r['perplexity']:7.4f} {r['accuracy']:7.4f}"
              + (f"   WC-gap {r['comp_gap']:+.4f}+/-{r.get('comp_se',0):.4f} "
                 f"({r.get('comp_visible',0):.3f} vs {r.get('comp_masked',0):.3f} "
                 f"on {r.get('comp_n',0)} pairs)"
                 if r["comp_gap"] == r["comp_gap"] else ""))
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
                                "n_seq", "seed", "min_len", "max_len",
                                "wc_gap", "wc_visible", "wc_masked", "wc_n",
                                # bf16 autocast on cuda, fp32 on cpu: the two
                                # are not the same numeric path and a series
                                # that mixes them silently is not one series
                                "device", "split"])
                # All SIXTEEN columns the header declares.
                #
                # The row wrote twelve. The four Watson-Crick complementarity
                # columns were declared in the header, computed in full, and
                # printed to stdout -- and then dropped on the way to the file,
                # so every row was ragged against its own header and
                # `csv.DictReader` returned None for exactly the measurement
                # that distinguishes "the model learned base composition" from
                # "the model learned pairing". The legacy file predates those
                # columns and is self-consistent at twelve; the mismatch
                # appeared the moment a fresh csv was created.
                _g = r.get("comp_gap", float("nan"))
                _ok = _g == _g                      # NaN when unavailable
                w.writerow([_dt.datetime.now().isoformat(timespec="seconds"),
                            ck.name, st.get("step", ""), st.get("tokens", ""),
                            round(r["bits"], 5), round(r["perplexity"], 5),
                            round(r["accuracy"], 5), r["n_masked"],
                            len(seqs), args.seed, args.min_len, args.max_len,
                            round(_g, 5) if _ok else "",
                            round(r.get("comp_visible", 0.0), 5) if _ok else "",
                            round(r.get("comp_masked", 0.0), 5) if _ok else "",
                            r.get("comp_n", "") if _ok else "",
                            device.type, args.split])
    if not args.no_csv:
        print(f"\n[eval] appended to {args.append_csv.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
