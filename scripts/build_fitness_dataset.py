#!/usr/bin/env python3
"""Build the stage-6 fitness corpus, and the benchmark it must never touch.

Two datasets come out of this script and the difference between them is the
whole point:

    benchmark   the 31 ncRNA assays of the RNAGym fitness leaderboard.
                NOTHING trains on these. They are what
                `eval_fitness_zeroshot.py` scores, and that score is directly
                comparable to a published table.
    train/val   the RNA fitness assays that are left once the benchmark is
                removed. Stage 6 trains the `fitness` head on these.
    transfer    one whole construct family held out of training, so the
                supervised head has an unseen-construct number as well as an
                unseen-variant one.

Why the separation has to be structural
---------------------------------------
24 of the 31 leaderboard assays are ALSO shipped by NABench, usually under the
same name. "Train on NABench, evaluate on RNAGym" sounds like two datasets and
is one. A fitness head trained that way would post a leaderboard-beating number
by having been shown the answers, and nothing in either benchmark's own tooling
would have said so. So the benchmark set is defined first, and every training
row is checked against it twice: once by assay identity and once by exact
sequence.

What the sources disagree about, and who wins
---------------------------------------------
Where both ship an assay, RNAGym's copy is used. The reasons are measured, not
assumed:

  * `Roberts_2023_HDV_ribozyme` — the two copies hold the SAME 33,930
    sequences and their scores correlate at Spearman 0.0056. NABench's column
    is integers from 4 to 438,152 with a median of 45: a read count, not a
    fitness. RNAGym's is a normalised activity in [0, 1.6]. A column named
    `DMS_score` that is finite, numeric and complete can still not be the
    quantity it is named after.
  * NABench keeps the full construct (T7 promoter, primer sites) and RNAGym
    trims to the core, so `Guy_2014_tRNA` is 132 nt in one and 105 in the
    other with ZERO exact sequence overlap. Two files that share no sequence
    can still be one experiment; a sequence-level leak check would call them
    disjoint. Assay identity, not sequence identity, is the unit of the split.

Assays that are dropped, each for a stated reason
-------------------------------------------------
  * 37 RNAGym `mRNA-coding` assays: the readout is the protein's function, and
    the sequences are coding DNA up to 5,592 nt. RNAGym excludes them from its
    own headline metric for the first reason. Kept out, not lost -- the files
    are on disk and the manifest names them.
  * 4 NABench DNA-regulatory assays (2 Eilon promoters, Shai promoter, Martin
    enhancer): the readout is transcription driven by protein binding to
    double-stranded DNA. This is an RNA model.
  * `Gregory_2018_mRNA`: all 287 sequences contain a literal `X`. Not a
    nucleotide, and 100% of the rows carry it.
  * `Rachapun_2022_f1u_R1_R2_ribozyme`: byte-identical to
    `Rachapun_2022_L1_R1_R2_ribozyme` (10,000 of 10,000 sequences shared, same
    scores). One experiment under two names, counted once.

Targets
-------
DMS scores are per-assay and incomparable: the range across assays spans four
orders of magnitude and the sign convention is not shared. Every assay's score
is rank-transformed to a standard normal within that assay, so a single
regression loss over a mixed batch is meaningful and no assay dominates by
having larger numbers. The raw score is kept beside it, because Spearman is
computed on ranks and must not be computed on the thing the model fitted.

Usage:
    /store/shuvam/.venv/bin/python scripts/build_fitness_dataset.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmarks/fitness"
NABENCH = BENCH / "nabench/data"
NABENCH_META = BENCH / "nabench/metadata.csv"
RNAGYM_ZIP = BENCH / "rnagym/fitness_processed_assays.zip"
RNAGYM_REF = BENCH / "rnagym_repo/fitness/reference_sheet_final.csv"
OUT = ROOT / "data/derived/fitness_v1"

#: The leaderboard's three ncRNA categories. RNAGym restricts to assays whose
#: readout measures the RNA molecule's own function, and macro-averages the
#: three with equal weight -- so tRNA (3 assays) and aptamer (2) carry two
#: thirds of the metric between them.
NCRNA_TYPES = ("Ribozyme", "tRNA", "Aptamer")

#: Assays dropped, and why. Written as data so the manifest can carry it and a
#: test can assert the list has not quietly grown.
EXCLUDED: Dict[str, str] = {
    "Gregory_2018_mRNA":
        "all 287 sequences contain a literal X, which is not a nucleotide",
    "Eilon_2012_dna_R1_promoter":
        "DNA regulatory: transcription from double-stranded DNA, not RNA function",
    "Eilon_2012_dna_R2_promoter":
        "DNA regulatory: transcription from double-stranded DNA, not RNA function",
    "Shai_2015_promoter":
        "DNA regulatory: transcription from double-stranded DNA, not RNA function",
    "Martin_2018_myc_enhancer":
        "DNA regulatory: a 600 nt enhancer read out as expression, not RNA function",
    "Rachapun_2022_f1u_R1_R2_ribozyme":
        "byte-identical to Rachapun_2022_L1_R1_R2_ribozyme: one experiment, two names",
}

#: Construct families among the training assays. The unit a transfer split has
#: to respect: five Townshend aptamers are one aptamer ladder, not five
#: independent experiments, and ten Rachapun files are one 35 nt library.
FAMILY_RULES: Tuple[Tuple[str, str], ...] = (
    ("Townshend_2015_", "townshend_aptamer"),
    ("Rachapun_2022_", "rachapun_ribozyme"),
    ("Julien_2016_", "splicing"),
    ("Ke_2017_", "splicing"),
)

#: The family held out of training entirely, to give the supervised head an
#: unseen-CONSTRUCT number beside its unseen-variant one. Aptamer, because it
#: is a leaderboard category and because a ribozyme-trained head generalising
#: to an aptamer is the claim worth testing.
TRANSFER_FAMILY = "townshend_aptamer"

#: Fraction of each training assay's variants held out for validation. The
#: split is by SEQUENCE, not by row: the Rachapun files share a few hundred
#: sequences with each other, and a row-wise split would put the same sequence
#: on both sides.
VAL_FRAC = 0.2
SPLIT_SALT = "pharos-fitness-v1"

MUT_RE = re.compile(r"^([ACGUN])(\d+)([ACGUN])$")
VALID_BASES = set("ACGUN")


def norm_seq(s: object) -> str:
    """Upper case, DNA to RNA. The two sources disagree on the alphabet."""
    return str(s).strip().upper().replace("T", "U")


def read_rnagym_assays() -> Dict[str, pd.DataFrame]:
    """The processed assays, straight out of the shipped zip.

    Read from the archive rather than an unpacked copy: an unpacked copy is a
    second version of the data that can drift from the one the manifest
    records a checksum for.
    """
    out: Dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(RNAGYM_ZIP) as z:
        for info in z.infolist():
            if not info.filename.endswith(".csv") or info.is_dir():
                continue
            name = Path(info.filename).stem
            with z.open(info) as fh:
                out[name] = pd.read_csv(io.BytesIO(fh.read()))
    return out


def read_nabench_assays() -> Dict[str, pd.DataFrame]:
    return {p.stem: pd.read_csv(p) for p in sorted(NABENCH.glob("*.csv"))}


def family_of(assay: str) -> str:
    for prefix, fam in FAMILY_RULES:
        if assay.startswith(prefix):
            return fam
    return "other"


def rank_normal(x: np.ndarray) -> np.ndarray:
    """Within-assay rank -> standard normal quantile.

    Ties take their average rank, so a zero-inflated assay (Rachapun L1 has a
    median of 0.0129 against a maximum of 1.44) maps its tied floor to one
    value instead of to an arbitrary ordering the model would then be asked to
    reproduce.
    """
    from scipy.stats import norm as _norm
    n = len(x)
    if n == 0:
        return x.astype(np.float32)
    r = pd.Series(x).rank(method="average").to_numpy()
    return _norm.ppf((r - 0.5) / n).astype(np.float32)


def parse_mutations(m: str) -> Optional[List[Tuple[str, int, str]]]:
    """`"C15G,A78U"` -> `[("C", 14, "G"), ("A", 77, "U")]`, or None.

    Indices in the mutant strings are 1-based into the assay's
    `RAW_CONSTRUCT_SEQ`; this returns 0-based. Returns None rather than raising
    so the caller can COUNT what it could not parse -- a scorer that skips a
    malformed row silently is how an assay ends up contributing nothing and
    still appearing in the average.
    """
    if not m:
        return None
    out = []
    for part in m.split(","):
        g = MUT_RE.match(part.strip())
        if g is None:
            return None
        out.append((g.group(1), int(g.group(2)) - 1, g.group(3)))
    return out


def check_reconstruction(df: pd.DataFrame, wt: str) -> Dict[str, int]:
    """Does every variant equal the wild type with its listed mutations applied?

    This is the check that makes the wild type usable as a masked-marginal
    context. It is run on every assay, and the benchmark refuses an assay that
    fails it, because the failure mode is silent: a scorer that looks up
    position 15 of the wrong reference reads a real log-probability at a
    position that is not the mutated one, and reports a number.
    """
    n = parsed = wt_ok = var_ok = 0
    for s, m in zip(df["seq"], df["mutant_str"]):
        if not m:
            continue
        n += 1
        trip = parse_mutations(m)
        if trip is None:
            continue
        if any(i < 0 or i >= len(wt) for _, i, _ in trip):
            continue
        parsed += 1
        if all(wt[i] == a for a, i, _ in trip):
            wt_ok += 1
        rebuilt = list(wt)
        for _, i, mt in trip:
            rebuilt[i] = mt
        if "".join(rebuilt) == s:
            var_ok += 1
    return {"n_mutant_rows": n, "parsed": parsed, "wt_base_ok": wt_ok,
            "variant_ok": var_ok}


def tidy(df: pd.DataFrame, assay: str, source: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "assay": assay,
        "source": source,
        "mutant_str": df["mutant"].astype("string").fillna("").str.upper()
                        .str.replace("T", "U", regex=False).str.strip(),
        "seq": df["sequence"].map(norm_seq),
        "dms_score": pd.to_numeric(df["DMS_score"], errors="coerce"),
    })
    out = out[np.isfinite(out["dms_score"])]
    bad = ~out["seq"].map(lambda s: set(s) <= VALID_BASES)
    if bad.any():
        out = out[~bad]
    return out.reset_index(drop=True)


def split_of(seq: str, assay_family: str) -> str:
    """train or val, decided by the sequence so a shared sequence cannot split.

    Hashing rather than shuffling: the assignment has to be reproducible from
    the sequence alone, including for a sequence that appears in two assays,
    and it must not change when an assay is added or a row order changes.
    """
    h = hashlib.sha1(f"{SPLIT_SALT}:{seq}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / float(1 << 64)
    return "val" if u < VAL_FRAC else "train"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    for p in (NABENCH, RNAGYM_ZIP, RNAGYM_REF):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    ref = pd.read_csv(RNAGYM_REF)
    ref.columns = [c.lstrip("﻿") for c in ref.columns]
    rg = read_rnagym_assays()
    nb = read_nabench_assays()
    print(f"[fit] RNAGym {len(rg)} assays, NABench {len(nb)} assays, "
          f"reference sheet {len(ref)} rows")

    # ---- the benchmark, defined first ------------------------------------
    bench_ref = ref[ref.RNA_TYPE.isin(NCRNA_TYPES)].copy()
    missing = [d for d in bench_ref.DMS_ID if d not in rg]
    if missing:
        raise SystemExit(f"benchmark assays absent from the zip: {missing}")
    print(f"[fit] benchmark: {len(bench_ref)} ncRNA assays "
          f"({bench_ref.RNA_TYPE.value_counts().to_dict()})")

    wildtypes: Dict[str, str] = {}
    bench_rows: List[pd.DataFrame] = []
    recon: Dict[str, Dict[str, int]] = {}
    for _, r in bench_ref.iterrows():
        a = r.DMS_ID
        d = tidy(rg[a], a, "rnagym")
        wt = norm_seq(r.RAW_CONSTRUCT_SEQ)
        if set(wt) - VALID_BASES:
            raise SystemExit(f"{a}: wild type holds non-nucleotides")
        st = check_reconstruction(d, wt)
        recon[a] = st
        # Every benchmark assay must reconstruct completely. It is not a
        # preference: `eval_fitness_zeroshot.py` masks positions in THIS wild
        # type, and a wild type the variants do not agree with makes every
        # score in the assay a number computed at the wrong place.
        if st["parsed"] != st["n_mutant_rows"] or st["variant_ok"] != st["parsed"]:
            raise SystemExit(
                f"{a}: {st['variant_ok']:,} of {st['n_mutant_rows']:,} variants "
                f"reconstruct from RAW_CONSTRUCT_SEQ -- refusing to benchmark "
                f"against a wild type the data disagrees with")
        wildtypes[a] = wt
        d["category"] = r.RNA_TYPE
        d["split"] = "benchmark"
        d["family"] = f"benchmark/{r.RNA_TYPE.lower()}"
        d["target"] = rank_normal(d["dms_score"].to_numpy())
        bench_rows.append(d)
    benchmark = pd.concat(bench_rows, ignore_index=True)
    print(f"[fit] benchmark rows {len(benchmark):,}; all 31 assays reconstruct "
          f"100% from RAW_CONSTRUCT_SEQ")

    bench_assays = set(bench_ref.DMS_ID)
    bench_seqs = set(benchmark["seq"])

    # ---- what is left to train on ----------------------------------------
    mrna_coding = set(ref[ref.RNA_TYPE == "mRNA-coding"].DMS_ID)
    candidates: Dict[str, Tuple[str, pd.DataFrame]] = {}
    for a, d in nb.items():
        if a in bench_assays or a in EXCLUDED or a in mrna_coding:
            continue
        candidates[a] = ("nabench", d)
    for a, d in rg.items():
        if a in bench_assays or a in EXCLUDED or a in mrna_coding:
            continue
        if a in candidates:
            continue              # NABench's copy already taken; see below
        candidates[a] = ("rnagym", d)

    # For the two assays both sources ship (the splicing pair), NABench keeps
    # the full construct and many more rows -- Julien 16,917 against 189 -- so
    # NABench's copy is the one taken above. Recorded rather than implied.
    train_rows: List[pd.DataFrame] = []
    for a, (src, d) in sorted(candidates.items()):
        t = tidy(d, a, src)
        if t.empty:
            continue
        fam = family_of(a)
        t["family"] = fam
        t["category"] = "splicing" if fam == "splicing" else (
            "aptamer" if "aptamer" in a.lower() else "ribozyme")
        t["target"] = rank_normal(t["dms_score"].to_numpy())
        if fam == TRANSFER_FAMILY:
            t["split"] = "transfer"
        else:
            t["split"] = [split_of(s, fam) for s in t["seq"]]
        train_rows.append(t)
    train = pd.concat(train_rows, ignore_index=True)

    # ---- the guard the whole file exists for ------------------------------
    leaked_assays = sorted(set(train["assay"]) & bench_assays)
    if leaked_assays:
        raise SystemExit(f"benchmark assays in the training set: {leaked_assays}")
    leaked_seqs = bench_seqs & set(train["seq"])
    if leaked_seqs:
        raise SystemExit(
            f"{len(leaked_seqs):,} sequences appear in BOTH the benchmark and "
            f"the training set; example {sorted(leaked_seqs)[0][:60]}")
    print(f"[fit] leak check: 0 shared assays, 0 shared sequences between "
          f"{len(bench_assays)} benchmark and {train['assay'].nunique()} "
          f"training assays")

    # a sequence must not be in two splits of the training set either
    per_seq = train.groupby("seq")["split"].nunique()
    if int(per_seq.max()) > 1:
        bad = per_seq[per_seq > 1]
        raise SystemExit(f"{len(bad):,} sequences land in more than one split")

    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["assay", "family", "category", "source", "split", "mutant_str",
            "seq", "dms_score", "target"]
    for name, part in (("benchmark", benchmark),
                       ("train", train[train.split == "train"]),
                       ("val", train[train.split == "val"]),
                       ("transfer", train[train.split == "transfer"])):
        part[cols].to_parquet(args.out / f"{name}.parquet", index=False)
        print(f"[fit] {name:10s} {len(part):8,} rows  "
              f"{part['assay'].nunique():3d} assays  "
              f"{part['family'].nunique():2d} families")

    (args.out / "wildtypes.json").write_text(json.dumps(wildtypes, indent=1))

    # ---- the digest, in the shape `split_digest` already uses --------------
    pairs = sorted(f"{a}:{s}" for a, s in
                   set(zip(benchmark.assay, benchmark.split))
                   | set(zip(train.assay, train.split)))
    digest = hashlib.sha1("\n".join(pairs).encode()).hexdigest()[:8]

    def per_assay(df: pd.DataFrame) -> List[Dict]:
        g = df.groupby("assay")
        return [{"assay": a, "n": int(len(x)), "family": x.family.iloc[0],
                 "category": x.category.iloc[0], "source": x.source.iloc[0],
                 "length": int(x.seq.str.len().iloc[0]),
                 "score_min": round(float(x.dms_score.min()), 6),
                 "score_median": round(float(x.dms_score.median()), 6),
                 "score_max": round(float(x.dms_score.max()), 6)}
                for a, x in g]

    manifest = {
        "digest": digest,
        "val_frac": VAL_FRAC,
        "split_salt": SPLIT_SALT,
        "transfer_family": TRANSFER_FAMILY,
        "benchmark": {
            "name": "RNAGym fitness leaderboard, ncRNA subset",
            "metric": "signed Spearman per assay, mean per category, "
                      "macro-mean over the 3 categories",
            "n_assays": int(benchmark.assay.nunique()),
            "n_rows": int(len(benchmark)),
            "categories": {k: int(v) for k, v in
                           bench_ref.RNA_TYPE.value_counts().items()},
            "reconstruction": recon,
            "assays": per_assay(benchmark),
        },
        "training": {
            "n_assays": int(train.assay.nunique()),
            "n_rows": int(len(train)),
            "splits": {k: int(v) for k, v in train.split.value_counts().items()},
            "families": {k: int(v) for k, v in train.family.value_counts().items()},
            "assays": per_assay(train),
        },
        "excluded": EXCLUDED,
        "excluded_mrna_coding": sorted(mrna_coding),
        "leak_check": {"shared_assays": 0, "shared_sequences": 0},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[fit] digest {digest}")
    print(f"[fit] -> {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
