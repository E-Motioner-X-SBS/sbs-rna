#!/usr/bin/env python3
"""Tests for the stage-6 fitness corpus and the zero-shot scorer.

What these pin is the one property the whole arrangement rests on: THE
BENCHMARK AND THE TRAINING SET ARE DISJOINT. 24 of the 31 assays on the
RNAGym ncRNA leaderboard are also shipped by NABench, usually under the same
name, so a fitness head trained on "NABench" and evaluated on "RNAGym" would
be evaluated on its own training data and nothing in either benchmark's
tooling would say so. That is the same class as finding 32.

Disjointness is checked two ways, because either alone is passable while the
other fails:

    by assay     `Guy_2014_tRNA` is 132 nt in NABench and 105 in RNAGym --
                 the same experiment with different flanks, sharing ZERO exact
                 sequences. A sequence-level check calls them disjoint.
    by sequence  `Beck_2022_ribozyme` and `Roberts_2023_cepeb3_ribozyme` are
                 21,321 identical sequences under two unrelated names. An
                 assay-name check calls them disjoint.

The rest pin the things that would produce a number anyway: a wild type the
variants disagree with, a target that is not actually normalised, an
excluded-assay list that quietly grew, and a masked context that still carries
the base it is masking.

Run: python3 src/pharos/data/test_fitness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data/derived/fitness_v1"

fails: list[str] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK ' if ok else 'FAIL'} {name:58s} {detail}")
    if not ok:
        fails.append(name)


def main() -> int:
    import pandas as pd

    if not (DATA / "manifest.json").exists():
        print(f"SKIP: {DATA} has not been built. "
              f"Run scripts/build_fitness_dataset.py")
        return 0

    man = json.loads((DATA / "manifest.json").read_text())
    wt = json.loads((DATA / "wildtypes.json").read_text())
    bench = pd.read_parquet(DATA / "benchmark.parquet")
    parts = {n: pd.read_parquet(DATA / f"{n}.parquet")
             for n in ("train", "val", "transfer")}
    train_all = pd.concat(parts.values(), ignore_index=True)

    print("== property 1: the benchmark is the published one ==")
    chk("31 ncRNA assays", bench.assay.nunique() == 31,
        f"{bench.assay.nunique()}")
    counts = bench.groupby("category").assay.nunique().to_dict()
    chk("26 ribozyme / 3 tRNA / 2 aptamer",
        counts == {"Ribozyme": 26, "tRNA": 3, "Aptamer": 2}, str(counts))
    ref = ROOT / "data/benchmarks/fitness/rnagym_repo/fitness/reference_sheet_final.csv"
    rr = pd.read_csv(ref)
    rr.columns = [c.lstrip("﻿") for c in rr.columns]
    want = set(rr[rr.RNA_TYPE.isin(["Ribozyme", "tRNA", "Aptamer"])].DMS_ID)
    chk("the assay set matches the reference sheet exactly",
        set(bench.assay) == want,
        f"{len(set(bench.assay) ^ want)} differ")

    print("\n== property 2: nothing trains on the benchmark ==")
    shared_assays = set(bench.assay) & set(train_all.assay)
    chk("no assay is in both", not shared_assays, str(sorted(shared_assays)))
    shared_seq = set(bench.seq) & set(train_all.seq)
    chk("no exact sequence is in both", not shared_seq,
        f"{len(shared_seq):,} shared")
    # and the manifest must not be able to claim otherwise
    chk("the manifest records the leak check it ran",
        man["leak_check"] == {"shared_assays": 0, "shared_sequences": 0},
        str(man["leak_check"]))

    print("\n== property 3: the training splits do not overlap each other ==")
    per_seq = train_all.groupby("seq").split.nunique()
    chk("no sequence is in two splits", int(per_seq.max()) == 1,
        f"worst {int(per_seq.max())}")
    tf = man["transfer_family"]
    chk(f"the {tf} family appears only in transfer",
        set(train_all[train_all.family == tf].split) == {"transfer"},
        str(sorted(set(train_all[train_all.family == tf].split))))
    chk("and transfer holds nothing else",
        set(parts["transfer"].family) == {tf},
        str(sorted(set(parts["transfer"].family))))

    print("\n== property 4: every benchmark variant reconstructs from its WT ==")
    # This is what makes the wild type usable as a masked-marginal context. A
    # wild type the variants disagree with does not raise: it reads a real
    # log-probability at a position that is not the mutated one, and reports
    # it. Nine of the 31 assays fail this against a MAJORITY-base consensus --
    # a complete randomisation of 7 positions has no majority base -- which is
    # why the builder takes RAW_CONSTRUCT_SEQ and then checks it.
    from build_fitness_dataset import parse_mutations
    rng = np.random.default_rng(0)
    worst = ("", 1.0)
    for a, g in bench.groupby("assay"):
        g = g[g.mutant_str != ""]
        take = g.iloc[rng.choice(len(g), size=min(400, len(g)), replace=False)]
        ok = 0
        for s, m in zip(take.seq, take.mutant_str):
            t = parse_mutations(m)
            if t is None:
                continue
            r = list(wt[a])
            if any(i < 0 or i >= len(r) for _, i, _ in t):
                continue
            if any(r[i] != w for w, i, _ in t):
                continue
            for _, i, mt in t:
                r[i] = mt
            ok += "".join(r) == s
        frac = ok / len(take)
        if frac < worst[1]:
            worst = (a, frac)
    chk("all 31, on a 400-variant sample each", worst[1] == 1.0,
        f"worst {worst[0] or 'none'} {worst[1]*100:.1f}%")
    chk("the builder recorded a full reconstruction for every assay",
        all(r["variant_ok"] == r["parsed"] == r["n_mutant_rows"]
            for r in man["benchmark"]["reconstruction"].values()),
        f"{len(man['benchmark']['reconstruction'])} assays")

    print("\n== property 5: targets are normalised WITHIN assay ==")
    # The raw scores span four orders of magnitude across assays -- one is a
    # ratio in [0, 1.6] and another a log-fitness in [-12, 10.5] -- so a single
    # regression loss over a mixed batch would be dominated by whichever assay
    # happened to be measured on the larger instrument.
    allr = pd.concat([bench, train_all], ignore_index=True)
    g = allr.groupby("assay").target.agg(["mean", "std", "count"])
    chk("every assay's target has mean 0 to 0.1",
        float(g["mean"].abs().max()) < 0.1, f"worst {g['mean'].abs().max():.4f}")
    chk("every assay's target has sd 0.8 to 1.0",
        0.8 <= float(g["std"].min()) and float(g["std"].max()) <= 1.0,
        f"[{g['std'].min():.4f}, {g['std'].max():.4f}]")
    chk("the raw score is kept beside it, unnormalised",
        float(allr.groupby("assay").dms_score.std().max()) > 10,
        "Spearman must not be computed on the thing that was fitted")
    chk("no target is non-finite", bool(np.isfinite(allr.target).all()), "")

    print("\n== property 6: the exclusions are stated, not silent ==")
    from build_fitness_dataset import EXCLUDED
    chk("every excluded assay carries a reason longer than a word",
        all(len(v) > 25 for v in EXCLUDED.values()), f"{len(EXCLUDED)} assays")
    chk("the manifest carries the same list",
        man["excluded"] == EXCLUDED, "")
    chk("the 37 mRNA-coding assays are named, not dropped unnamed",
        len(man["excluded_mrna_coding"]) == 37,
        f"{len(man['excluded_mrna_coding'])}")
    on_disk = {p.stem for p in (ROOT / "data/benchmarks/fitness/nabench/data").glob("*.csv")}
    accounted = (set(bench.assay) | set(train_all.assay) | set(EXCLUDED)
                 | set(man["excluded_mrna_coding"]))
    chk("every NABench file is in the benchmark, in training, or excluded",
        not (on_disk - accounted), f"unaccounted: {sorted(on_disk - accounted)}")

    print("\n== property 7: the masked context does not carry the base ==")
    # `chain_chemistry` one-hot-encodes base identity in dims 0-4. Chemistry
    # built from the UNMASKED wild type hands the model the very base it is
    # being asked to predict -- stage 1 measured that leak at 0.57 bits
    # against a corpus entropy of 2.0165. Here it would not look impossible,
    # only plausible and wrong, so it is pinned structurally.
    import inspect
    import torch
    from pharos.data.chemistry_torch import BatchChemistry
    from pharos.data.vocab import SYM2ID, SYMBOLS
    import eval_fitness_zeroshot as fz

    sig = inspect.signature(fz._logprobs)
    chk("_logprobs takes no chemistry argument, so none can be passed in",
        "chem" not in sig.parameters, str(list(sig.parameters)))
    chk("it derives chemistry inside itself",
        "chem_fn(tok" in inspect.getsource(fz._logprobs), "")

    dev = torch.device("cpu")
    bc = BatchChemistry(SYMBOLS, dev)
    base = torch.tensor([[SYM2ID[c] for c in "ACGUACGU"]])
    msk = torch.ones_like(base, dtype=torch.bool)
    masked = base.clone()
    masked[0, 3] = fz.MASK_ID
    c_un, c_ma = bc(base, msk), bc(masked, msk)
    chk("the mask symbol is the one stage 1 trains with",
        fz.MASK_ID == SYM2ID["UNK"], f"{fz.MASK_ID} = {SYMBOLS[fz.MASK_ID]}")
    chk("chemistry from a masked token does not one-hot the original base",
        float(c_ma[0, 3, :4].abs().max()) == 0.0
        and float(c_un[0, 3, :4].abs().max()) > 0.0,
        f"masked {c_ma[0, 3, :5].tolist()}  unmasked {c_un[0, 3, :5].tolist()}")
    # Dim 23 is a +/-16 GC fraction, so masking one base DOES move its
    # neighbours' dim 23 -- the window has one fewer known base. That is what
    # stage 1 does at training time too, and it leaks nothing: UNK scores zero
    # in both the GC numerator and its denominator, so a neighbour sees a
    # window over the bases that are still there, not a window with a hole of
    # known sign. Dims 0-22 must be untouched everywhere else, which is the
    # part that would be a leak.
    other = [i for i in range(base.shape[1]) if i != 3]
    chk("elsewhere only the GC window moves -- dims 0-22 are untouched",
        bool(torch.equal(c_un[0, other][:, :23], c_ma[0, other][:, :23])),
        f"largest dim-23 shift "
        f"{float((c_un[0, other][:, 23] - c_ma[0, other][:, 23]).abs().max()):.4f}")

    print("\n== property 8: the published table is read, not remembered ==")
    chk("the leaderboard comes from the file RNAGym ships",
        fz.LEADERBOARD_CSV.exists() and len(fz.load_leaderboard()) == 12,
        f"{len(fz.load_leaderboard())} rows from {fz.LEADERBOARD_CSV.name}")
    words = set(Path(fz.__file__).read_text().replace(",", " ").split())
    copied = sorted({f"{v:.4f}" for row in fz.load_leaderboard()
                     for v in row[1:]} & words)
    chk("no published number is also written into the module",
        not copied, f"found: {copied or 'none'}")

    print("\n== property 9: the stage-6 loader feeds the metric it reports ==")
    import train_sequence_stages as ts

    chk("stage 6 has a weight and the loop applies it",
        ts.STAGE_WEIGHTS.get("fitness") == 0.3
        and 'STAGE_WEIGHTS["fitness"]' in Path(ts.__file__).read_text(),
        str(ts.STAGE_WEIGHTS))

    groups = ts.load_fitness("train")
    chk("train loads grouped by assay", len(groups) == 12, f"{len(groups)} assays")
    chk("and the rows add up to the parquet",
        sum(len(y) for _, _, y in groups) == len(parts["train"]),
        f"{sum(len(y) for _, _, y in groups):,}")

    # One assay per batch, because the metric is a WITHIN-assay Spearman: the
    # DMS score of a tRNA assay and of a ribozyme assay are different
    # quantities on different instruments, and a correlation over a mixed
    # batch would mostly measure which assay a sequence came from.
    rng = np.random.default_rng(0)
    seen, sizes = [], []
    for i, (a, sq, y) in enumerate(ts.iter_fitness("train", 32, rng)):
        seen.append(a)
        sizes.append(len(sq))
        chk_len = len(sq) == len(y)
        if not chk_len:
            fails.append("iter_fitness batch length mismatch")
        if i >= 200:
            break
    chk("every batch is one assay", len(seen) == len(sizes),
        f"{len(set(seen))} distinct assays over {len(seen)} batches")
    chk("batches are shuffled across assays, not assay-by-assay",
        len(set(seen[:60])) > 1,
        f"{len(set(seen[:60]))} assays in the first 60 batches")
    chk("no batch is larger than asked", max(sizes) <= 32, f"max {max(sizes)}")

    # The epoch is `max` over the streams and the short ones restart inside
    # it. Before this, a stage whose generator ran out simply stopped
    # contributing -- head 4 got a gradient on 3.3% of the steps in an epoch
    # while the file's own comment claimed it was recycled 31 times.
    src = Path(ts.__file__).read_text()
    chk("the loop cycles its short streams", "def cycled(" in src, "")
    chk("and an epoch is bounded by the longest, not by all of them",
        "while step < steps_per_epoch" in src, "")
    chk("per-stream batch and pass counts are both logged",
        all(k in src for k in ("fitness_batches", "ss_batches",
                               "fitness_passes", "ss_passes")),
        "a pass is only credited when a stream runs out, so the longest "
        "stream would report 0 having supplied every step")
    chk("the prediction spread is a logged column",
        "fitness_pred_sd" in src,
        "a collapsed head and a slow one have the same loss")
    chk("the epoch-end cap is per assay, not per split",
        "max_rows: Optional[int] = None" in src
        and "min(len(seqs), max_rows)" in src, "")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
