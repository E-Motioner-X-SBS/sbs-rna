#!/usr/bin/env python3
"""Zero-shot fitness scoring on the RNAGym ncRNA leaderboard, PHAROS's MLM head.

This is the number that is comparable to a published table. RNAGym v0.1.1
scores every masked language model the same way -- the masked-marginal
log-likelihood ratio at the mutated positions -- and ranks them by the SIGNED
Spearman correlation against the measured DMS score, macro-averaged over its
three ncRNA categories (ribozyme 26 assays, tRNA 3, aptamer 2) with equal
weight. Its twelve published entries are printed beside this run's, read from
the csv RNAGym ships; no number from that table is written into this module,
for the reason given at LEADERBOARD_CSV.

PHAROS is a masked language model, so this needs no fitness head and no
training: it is stage 1's own masked marginals, read at the mutated positions.

The scoring convention, taken from RNAGym's `fitness/baselines/*/compute_fitness.py`
rather than from its prose, because the two differ:

    context  the WILD TYPE with every mutated position of that variant masked
             at once -- not the variant's own sequence, and not one position
             at a time for a multi-mutant.
    score    sum over the variant's mutations of
             log p(mutant base | masked wild type) - log p(wild-type base | ...)
    index    1-based into the assay's RAW_CONSTRUCT_SEQ.

Two things that would silently produce a number anyway
------------------------------------------------------
1. THE CHEMISTRY MUST COME FROM THE MASKED TOKENS. `chain_chemistry`
   one-hot-encodes base identity in dims 0-4, so chemistry built from the
   unmasked wild type hands the model the very base it is being asked to
   predict. Stage 1 found this the hard way: masked accuracy 0.948 and 0.57
   bits against a corpus entropy of 2.0165. Here the leak would not look like
   an impossible score -- it would look like every variant scoring slightly
   negative, because the wild-type base is the one being leaked, and the
   Spearman would be plausible and wrong. `_logprobs` therefore derives
   chemistry from the tokens it is handed and is never given the unmasked ones.
2. A HEAD THAT RETURNS A CONSTANT SCORES nan, AND nan DOES NOT LOWER A MEAN.
   `scipy.stats.spearmanr` of a constant vector is nan; `np.nanmean` over the
   assays then reports the average of whatever survived, so a model that
   scored nothing at all and a model that scored three assays well look alike.
   Every assay's distinct-score count is checked, counted and printed, and a
   category that lost an assay says so.

Strategies
----------
    masked-marginals  RNAGym's convention exactly: one forward per distinct
                      SET of mutated positions, 472,358 of them over the 31
                      assays. ~47M tokens, so this wants a GPU.
    masked-single     Meier et al.'s masked-marginals: one position masked at
                      a time in the wild type, L forwards per assay, 3,520 in
                      total, and a multi-mutant scored as the sum of its
                      per-position ratios. The two agree exactly on single
                      mutants and differ on the rest -- and 99.4% of these
                      variants are multi-mutants, so they are different
                      numbers and are labelled as such.
    wt-marginals      one unmasked forward per assay, 31 in total, scoring
                      every variant off the same wild-type distribution.
                      RNAGym's own baseline scripts offer this one too.

Every output file records which was used, and the printed comparison names
the strategy in its header, because the published column is
`masked-marginals` and a run of the others is not the same measurement.

Usage:
    python scripts/eval_fitness_zeroshot.py --device cuda
    python scripts/eval_fitness_zeroshot.py --device cpu --strategy wt-marginals
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pharos.data.chemistry_torch import BatchChemistry          # noqa: E402
from pharos.data.vocab import PAD_ID, SYM2ID, SYMBOLS           # noqa: E402
from pharos.model.moe import RouterFeatures                     # noqa: E402
from pharos.model.pharos import Pharos, PharosConfig            # noqa: E402

DATA = ROOT / "data/derived/fitness_v1"
OUT = ROOT / "data/samples/analysis"

#: The same symbol stage 1 masks with: `MASK_ID = SYM2ID["UNK"]`. Reusing UNK
#: rather than adding a symbol is pretrain_mlm.py's choice, and the scorer has
#: to make the same one or the model is shown a token it has never been
#: trained to fill in.
MASK_ID = SYM2ID["UNK"]
BASE_ID = {b: SYM2ID[b] for b in "ACGU"}

#: RNAGym v0.1.1's published table, READ FROM THE FILE IT SHIPS IN rather than
#: transcribed into this module. A hand-copied leaderboard is a second copy of
#: somebody else's numbers that drifts the moment they update it, and the run
#: that prints "rank 5" against a stale table is wrong in the one direction
#: nobody checks. If the file is absent, no comparison is printed -- which is
#: visibly different from printing a comparison against remembered values.
LEADERBOARD_CSV = (ROOT / "data/benchmarks/fitness/rnagym_repo/leaderboard/"
                   "fitness/leaderboard_signed_3ncRNA.csv")


def load_leaderboard() -> List[Tuple[str, float, float, float, float]]:
    if not LEADERBOARD_CSV.exists():
        return []
    d = pd.read_csv(LEADERBOARD_CSV)
    return [(str(r.model), float(r.Ribozyme), float(r.tRNA), float(r.Aptamer),
             float(r.macro_3ncRNA)) for r in d.itertuples()]

CATEGORIES = ("Ribozyme", "tRNA", "Aptamer")


def depth_baseline(d: pd.DataFrame, trips: Sequence[Sequence]) -> float:
    """Spearman of MINUS THE MUTATION COUNT against the measured fitness.

    The floor every number on this leaderboard has to clear, and it was not
    stated anywhere. This predictor reads no sequence, runs no model and knows
    only how many positions changed -- and on these 31 assays it scores a
    macro of 0.187, which is second of the thirteen published entries and
    ahead of every one of them on BOTH the ribozyme and the tRNA column.

    Every other head in this project is reported against its floor: secondary
    structure against the majority class, the MLM against the corpus unigram,
    accuracy against chance. A fitness Spearman quoted without this one is the
    same omission, and it is the difference between "the model ranks RNA
    variants" and "the model has noticed that more mutations are worse".

    NaN where the assay has one depth -- `Pitt_2010_ribozyme` is a pure single
    mutant scan, so counting mutations cannot rank it at all. Returned as NaN
    and counted, not silently dropped into a mean.
    """
    from scipy.stats import spearmanr
    depth = np.array([len(t) for t in trips], dtype=float)
    if len(set(depth)) < 2:
        return float("nan")
    r = spearmanr(-depth, d["dms_score"].to_numpy()).correlation
    return float(r) if np.isfinite(r) else float("nan")


def encode_wt(wt: str) -> np.ndarray:
    """Wild-type bases to token ids. ACGUN only; the builder guarantees it."""
    return np.array([SYM2ID.get(c, SYM2ID["UNK"]) for c in wt], dtype=np.int64)


#: bf16 autocast on CUDA, because that is how `pretrain_mlm.py` scores its own
#: held-out set and the point of this number is that it is comparable. On CPU
#: there is no autocast and the model runs in fp32; the summary records which,
#: since the two differ in the last couple of decimals.
AMP = {"cuda": torch.bfloat16}


@torch.no_grad()
def _logprobs(model, tok: torch.Tensor, n_loops: int,
              chem_fn) -> torch.Tensor:
    """`log_softmax` of the MLM head over a batch of equal-length rows.

    `tok` is the only sequence information that enters: the chemistry and the
    router features are both derived from it here, inside this function, so
    there is no call site that can pass masked tokens and unmasked chemistry.
    """
    mask = torch.ones_like(tok, dtype=torch.bool)
    chem = chem_fn(tok, mask)
    n = mask.sum(1)
    feats = RouterFeatures(
        length=n.float(),
        chem_summary=(chem.sum(1) / n.unsqueeze(1).clamp(min=1))[:, :5])
    dt = AMP.get(tok.device.type)
    if dt is None:
        out = model(tok, torch.zeros_like(tok), chem, mask,
                    feats=feats, n_loops=n_loops, mlm=True)
    else:
        with torch.autocast(tok.device.type, dtype=dt):
            out = model(tok, torch.zeros_like(tok), chem, mask,
                        feats=feats, n_loops=n_loops, mlm=True)
    return torch.log_softmax(out["mlm_logits"].float(), dim=-1)


def score_assay(model, wt: str, muts: Sequence[Sequence[Tuple[str, int, str]]],
                device, n_loops: int, strategy: str, batch_rows: int,
                chem_fn) -> np.ndarray:
    """One score per variant, in the order given. NaN where it could not score."""
    L = len(wt)
    base = encode_wt(wt)
    scores = np.full(len(muts), np.nan, dtype=np.float64)

    if strategy == "masked-single":
        # L forwards, one per position, each with that position alone masked.
        # The context is the same for every variant that mutates a given
        # position, which is what makes it cheap; it is also what makes it a
        # different number from RNAGym's, where a double mutant sees both of
        # its positions masked at once.
        keys = list(range(L))
        table = {}
        for s0 in range(0, L, batch_rows):
            chunk = keys[s0:s0 + batch_rows]
            arr = np.repeat(base[None, :], len(chunk), axis=0)
            for r, i in enumerate(chunk):
                arr[r, i] = MASK_ID
            tok = torch.as_tensor(arr, device=device)
            lp = _logprobs(model, tok, n_loops, chem_fn).cpu().numpy()
            for r, i in enumerate(chunk):
                table[i] = lp[r, i]
        groups = {None: list(range(len(muts)))}
        table = {None: table}
    elif strategy == "wt-marginals":
        tok = torch.as_tensor(base[None, :], device=device)
        lp = _logprobs(model, tok, n_loops, chem_fn)[0].cpu().numpy()
        table = {None: lp}
        groups = {None: list(range(len(muts)))}
    else:
        # one forward per DISTINCT set of mutated positions, which is what the
        # context depends on. 856,629 variants collapse to 472,358 contexts.
        groups: Dict[object, List[int]] = defaultdict(list)
        for k, trip in enumerate(muts):
            groups[frozenset(i for _, i, _ in trip)].append(k)
        table = {}
        keys = list(groups)
        for s in range(0, len(keys), batch_rows):
            chunk = keys[s:s + batch_rows]
            arr = np.repeat(base[None, :], len(chunk), axis=0)
            for r, pat in enumerate(chunk):
                arr[r, list(pat)] = MASK_ID
            tok = torch.as_tensor(arr, device=device)
            lp = _logprobs(model, tok, n_loops, chem_fn).cpu().numpy()
            for r, pat in enumerate(chunk):
                # Keep only the rows that are read -- the masked positions --
                # and COPY them. A numpy slice is a view that keeps its whole
                # batch array alive, and the largest assay has 85,567 patterns
                # over 424 batches, so holding views pins 360 MB to retain
                # 27 MB of it.
                table[pat] = {int(i): lp[r, int(i)].copy() for i in pat}

    for key, idxs in groups.items():
        lp = table[key]
        for k in idxs:
            tot = 0.0
            ok = True
            for w, i, mt in muts[k]:
                if w not in BASE_ID or mt not in BASE_ID:
                    ok = False      # an N in the mutant string; RNAGym maps it
                    break           # to '-' and drops the term. Drop the row.
                # `lp` is indexed by position either way: an (L, V) array for
                # wt-marginals, a {position: V-vector} dict for the masked
                # strategy, which keeps only the rows it will actually read.
                v = lp[i]
                tot += float(v[BASE_ID[mt]] - v[BASE_ID[w]])
            if ok:
                scores[k] = tot
    return scores


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else float("nan")


def auc_mcc(y: np.ndarray, s: np.ndarray) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score, matthews_corrcoef
    yb = (y > np.median(y)).astype(int)
    if yb.min() == yb.max():
        return float("nan"), float("nan")
    a = roc_auc_score(yb, s)
    m = matthews_corrcoef(yb, (s > np.median(s)).astype(int))
    return float(max(a, 1 - a)), float(abs(m))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "data/derived/checkpoints/pretrain_shared400.pt")
    ap.add_argument("--size", default="shared400",
                    choices=("mini", "small", "base400", "shared400"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-loops", type=int, default=3,
                    help="stage 1 trains with --n-loops 3 --sample-loops, so "
                         "the model has seen 1, 2 and 3")
    ap.add_argument("--strategy", default="masked-marginals",
                    choices=("masked-marginals", "masked-single",
                             "wt-marginals"))
    ap.add_argument("--token-budget", type=int, default=65536,
                    help="rows per forward are derived from this and the "
                         "assay's length, so a 425 nt assay does not OOM on "
                         "the batch size a 45 nt one wants")
    ap.add_argument("--max-rows", type=int, default=512)
    ap.add_argument("--limit-per-assay", type=int, default=None,
                    help="score only the first N variants of each assay. For "
                         "a smoke run; the output is marked partial.")
    ap.add_argument("--assays", nargs="*", default=None)
    ap.add_argument("--tag", default=None, help="suffix for the output files")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    bench = pd.read_parquet(DATA / "benchmark.parquet")
    wildtypes = json.loads((DATA / "wildtypes.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    cat_of = dict(zip(bench.assay, bench.category))

    device = torch.device(args.device)
    if device.type == "cuda":
        # Same fast paths every other CUDA path in this project turns on:
        # refusing TF32 on the fp32 residue of a model that trains under bf16
        # autocast costs throughput and buys nothing.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    cfg = getattr(PharosConfig, args.size)()
    model = Pharos(cfg).to(device).eval()
    ck = torch.load(args.checkpoint, map_location=device)
    sd = ck["model"] if "model" in ck else ck
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    # REFUSE a partial load rather than print one. This used to call
    # `load_state_dict(strict=False)`, print the count of missing keys, and
    # score the model anyway -- so a `--size` that did not match the
    # checkpoint, a renamed trunk parameter or a half-written file would have
    # scored a partly random model and printed it a leaderboard position.
    # That is the defect this whole scorer is built to avoid, committed in
    # its own loader. `_check_load` is `eval_mlm_checkpoint.py`'s, reused
    # rather than reimplemented so the two cannot disagree about what counts
    # as a tolerable absence.
    from eval_mlm_checkpoint import _check_load
    _check_load(model, sd, args.checkpoint)
    tokens_seen = float(ck.get("tokens", 0))
    print(f"[fz] {args.checkpoint.name}: {tokens_seen/1e9:.2f}B tokens, "
          f"loaded and checked")
    print(f"[fz] strategy {args.strategy}, n_loops {args.n_loops}, "
          f"device {device}")
    chem_fn = BatchChemistry(SYMBOLS, device)

    from build_fitness_dataset import parse_mutations

    assays = args.assays or sorted(bench.assay.unique())
    rows: List[Dict] = []
    t0 = time.time()
    for a in assays:
        d = bench[bench.assay == a]
        # RNAGym's own `performance_fitness.py` drops the wild-type row before
        # correlating; a variant with no mutations scores exactly 0 and would
        # be a tie against nothing.
        d = d[d.mutant_str != ""]
        if args.limit_per_assay:
            d = d.head(args.limit_per_assay)
        wt = wildtypes[a]
        trips = [parse_mutations(m) for m in d.mutant_str]
        keep = [i for i, t in enumerate(trips) if t is not None
                and all(0 <= j < len(wt) for _, j, _ in t)]
        d = d.iloc[keep]
        trips = [trips[i] for i in keep]
        rows_per = max(1, min(args.max_rows, args.token_budget // max(len(wt), 1)))
        t1 = time.time()
        s = score_assay(model, wt, trips, device, args.n_loops, args.strategy,
                        rows_per, chem_fn)
        y = d.dms_score.to_numpy()
        fin = np.isfinite(s)
        n_distinct = int(len(np.unique(s[fin]))) if fin.any() else 0
        rho = spearman(y[fin], s[fin]) if fin.sum() > 2 and n_distinct > 1 else float("nan")
        auc, mcc = (auc_mcc(y[fin], s[fin]) if fin.sum() > 2 and n_distinct > 1
                    else (float("nan"), float("nan")))
        rows.append({"assay": a, "category": cat_of[a], "n": int(len(d)),
                     "n_scored": int(fin.sum()), "n_distinct_scores": n_distinct,
                     "length": len(wt), "spearman": rho,
                     "depth_baseline": depth_baseline(d, trips),
                     "abs_spearman": abs(rho) if np.isfinite(rho) else float("nan"),
                     "auc": auc, "mcc": mcc,
                     "seconds": round(time.time() - t1, 1)})
        print(f"[fz] {a:34s} {cat_of[a]:8s} n={len(d):7,} scored={int(fin.sum()):7,} "
              f"distinct={n_distinct:7,} rho={rho:+.4f} ({time.time()-t1:.0f}s)",
              flush=True)

    res = pd.DataFrame(rows)

    # ---- the checks that stop a nan from passing for a result -------------
    problems: List[str] = []
    dead = res[res.n_distinct_scores <= 1]
    if len(dead):
        problems.append(f"{len(dead)} assays produced a constant score: "
                        f"{', '.join(dead.assay)}")
    short = res[res.n_scored < res.n]
    for _, r_ in short.iterrows():
        problems.append(f"{r_.assay}: {r_.n - r_.n_scored:,} of {r_.n:,} "
                        f"variants could not be scored")
    miss = res[~np.isfinite(res.spearman)]
    if len(miss):
        problems.append(f"{len(miss)} assays have no Spearman: "
                        f"{', '.join(miss.assay)}")

    per_cat = {}
    for c in CATEGORIES:
        sub = res[res.category == c]
        want = int(manifest["benchmark"]["categories"].get(c, 0))
        got = int(np.isfinite(sub.spearman).sum())
        # The baseline is averaged over the SAME assays, with an assay it
        # cannot rank counted as 0 rather than skipped -- `np.nanmean` would
        # quietly average the baseline over 25 ribozymes and the model over
        # 26 and print them in the same row.
        base = sub.depth_baseline.fillna(0.0)
        per_cat[c] = {"n_assays_expected": want, "n_assays_scored": got,
                      "signed": float(np.nanmean(sub.spearman)) if got else float("nan"),
                      "abs": float(np.nanmean(sub.abs_spearman)) if got else float("nan"),
                      "auc": float(np.nanmean(sub.auc)) if got else float("nan"),
                      "depth_baseline": float(base.mean()) if len(sub) else float("nan"),
                      "depth_baseline_undefined": int(sub.depth_baseline.isna().sum())}
        if got != want and not args.assays:
            problems.append(f"{c}: {got} of {want} assays contributed to the "
                            f"category mean")
    macro = float(np.mean([per_cat[c]["signed"] for c in CATEGORIES]))
    macro_abs = float(np.mean([per_cat[c]["abs"] for c in CATEGORIES]))
    macro_base = float(np.mean([per_cat[c]["depth_baseline"] for c in CATEGORIES]))

    print(f"\n[fz] ==== RNAGym ncRNA leaderboard, {args.strategy} ====")
    hdr = f"{'model':26s} {'ribozyme':>9s} {'tRNA':>8s} {'aptamer':>8s} {'macro':>8s}"
    print("  " + hdr)
    me = ("PHAROS " + args.size, per_cat["Ribozyme"]["signed"],
          per_cat["tRNA"]["signed"], per_cat["Aptamer"]["signed"], macro)
    floor = ("-- mutation count only", per_cat["Ribozyme"]["depth_baseline"],
             per_cat["tRNA"]["depth_baseline"],
             per_cat["Aptamer"]["depth_baseline"], macro_base)
    published = load_leaderboard()
    if not published:
        print(f"  (no published table at {LEADERBOARD_CSV.name}; "
              f"showing this run only)")
    table = published + [me, floor]
    table.sort(key=lambda t: -(t[4] if np.isfinite(t[4]) else -9))
    for i, (nm, rz, tr, ap_, mc) in enumerate(table, 1):
        star = ("  <<<" if nm.startswith("PHAROS")
                else ("  <-- reads no sequence" if nm.startswith("--") else ""))
        print(f"  {i:2d} {nm:26s} {rz:9.4f} {tr:8.4f} {ap_:8.4f} {mc:8.4f}{star}")
    above = sum(1 for t in published if t[4] < macro_base)
    print(f"\n  The mutation-count baseline -- no sequence, no model -- "
          f"outranks {above} of the {len(published)} published entries. "
          f"A fitness Spearman on this benchmark is substantially a "
          f"measurement of how well a score tracks mutation count, and this "
          f"row is the floor to read every other row against.")

    # The banner fires for EITHER kind of subset. It used to fire only for
    # `--limit-per-assay`, so a run with `--assays` printed a rank against the
    # full published table with nothing saying it had scored three of the 31.
    # The json knew -- `"partial": bool(limit) or bool(assays)` -- and the
    # thing a person reads did not.
    _why = []
    if args.limit_per_assay:
        _why.append(f"only the first {args.limit_per_assay:,} variants of "
                    f"each assay")
    if args.assays:
        _why.append(f"only {len(args.assays)} of the 31 assays")
    if _why:
        print(f"\n[fz] PARTIAL: {' and '.join(_why)} were scored. The table "
              f"above compares this against full-benchmark numbers and is "
              f"NOT a ranking.")
    if problems:
        print(f"\n[fz] {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"\n[fz] all {len(res)} assays scored, every one with a finite "
              f"Spearman over more than one distinct prediction")

    tag = args.tag or args.strategy
    out = args.out or OUT
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"fitness_zeroshot_{tag}.csv", index=False)
    summary = {
        "checkpoint": str(args.checkpoint), "tokens": tokens_seen,
        "size": args.size, "n_loops": args.n_loops, "strategy": args.strategy,
        "device": str(device),
        "amp": str(AMP.get(device.type, "fp32")),
        "benchmark_digest": manifest["digest"],
        "limit_per_assay": args.limit_per_assay,
        "partial": bool(args.limit_per_assay) or bool(args.assays),
        "published_leaderboard": [
            {"model": m, "Ribozyme": a, "tRNA": b, "Aptamer": c, "macro": d}
            for m, a, b, c, d in published],
        "rank_of_this_run": 1 + sum(1 for t in published if t[4] > macro),
        "per_category": per_cat,
        "macro_signed": macro, "macro_abs": macro_abs,
        "macro_depth_baseline": macro_base,
        "beats_depth_baseline": bool(macro > macro_base),
        "problems": problems,
        "seconds": round(time.time() - t0, 1),
    }
    (out / f"fitness_zeroshot_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"[fz] -> {out}/fitness_zeroshot_{tag}.json  ({time.time()-t0:.0f}s)")
    return 1 if problems and not args.limit_per_assay and not args.assays else 0


if __name__ == "__main__":
    sys.exit(main())
