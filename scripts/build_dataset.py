#!/usr/bin/env python3
"""Build the PHAROS 3D training set from every RNA-bearing PDB entry.

Reads raw entries -- all 10,520 of them, including the 2,581 that neither
RNA3DB nor gRNAde/RNASolo contains (ARCHITECTURE v0.2 §11.4) -- and writes
sharded `.npz` plus a manifest carrying per-chain metadata and the split
assignment.

Splits are **family-disjoint** (D17), not random. The corpus is rRNA-dominated:
G3 puts 85.94% of residues in ribosome-like entries, so a random split puts
homologues of the test set in training and reports a number that means nothing.
Assignment is by Rfam family where `pdb_hunter` knows one (6,316 of 10,424
entries), and by entry id otherwise -- so an entry with no family annotation is
never split across its own chains, and families never straddle the boundary.

Quality travels with the example rather than filtering it (D16): a resolution
cutoff would discard the cryo-EM majority, so `train_weight` is carried and the
sampler decides.

Usage:
    /store/shuvam/.venv/bin/python scripts/build_dataset.py --workers 8
    /store/shuvam/.venv/bin/python scripts/build_dataset.py --limit 200   # smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/structures/raw_pdb_entries"
OUT = ROOT / "data/derived/pharos3d"
HUNTER = ROOT / "data/catalog/pdb_hunter_index.json"
COMPOSITION = ROOT / "data/samples/analysis/entry_composition_rawpdb_table.json"
sys.path.insert(0, str(ROOT / "src"))

CHAINS_PER_SHARD = 512
#: Fractions **of the splittable remainder**, not of the corpus.
#:
#: Four rRNA families hold 84.4% of all structural residues and each exceeds any
#: plausible held-out quota, so they are forced to train (see `assign_splits`).
#: What is left to split is 15.6% of residues, and taking 5% of *that* gave
#: val/test of 170 and 319 chains -- technically balanced and too thin to
#: measure with. Held-out **folds** are the scarce resource here, not residues:
#: spending 20% of the remainder on each of val and test costs train 5% of its
#: residues and roughly quadruples the evaluation set.
VAL_FRAC, TEST_FRAC = 0.20, 0.20
#: Separately: the largest share of the WHOLE corpus a single family may hold
#: and still be eligible for a held-out set. These are two different quantities
#: and conflating them is a live trap -- deriving the eligibility threshold from
#: VAL_FRAC meant that raising the held-out fraction to 20% also raised the
#: threshold to 2.63M residues, which let SSU_rRNA_eukarya (1.58M) become the
#: validation set on its own.
MAX_HELDOUT_GROUP_FRAC = 0.05


def entry_metadata() -> Dict[str, Dict]:
    """Per-entry resolution, method, clashscore, weight, Rfam, composition."""
    meta: Dict[str, Dict] = {}
    if HUNTER.exists():
        for e in json.loads(HUNTER.read_text())["entries"]:
            meta[e["pdb_id"].lower()] = {
                "resolution": e.get("resolution"), "method": e.get("method", "?"),
                "clashscore": e.get("clashscore"),
                "train_weight": e.get("train_weight", 1.0),
                "rfam_family": e.get("rfam_family"),
            }
    if COMPOSITION.exists():
        for e in json.loads(COMPOSITION.read_text()):
            m = meta.setdefault(e["pdb"].lower(), {})
            m["has_protein"] = e.get("has_protein", False)
            m["ribosome_like"] = e.get("ribosome_like", False)
            m["n_rna_res"] = e.get("n_rna_res", 0)
    return meta


_META: Dict[str, Dict] = {}


def _init(meta: Dict[str, Dict]) -> None:
    global _META
    _META = meta


def _one(path_str: str):
    from pharos.data.dataset import build_entry
    p = Path(path_str)
    pdb = p.stem.replace(".cif", "").lower()
    return build_entry(p, _META.get(pdb, {}))


def assign_splits(chains: List[Dict], seed: str = "pharos-v02") -> Dict[str, str]:
    """`group -> split`, where a group is an Rfam family or a lone entry.

    **A 90/5/5 family-disjoint split of this corpus is arithmetically
    impossible, and pretending otherwise produces a meaningless number.**
    Measured on the built set: four families hold **84.4%** of all structural
    RNA residues --

        SSU_rRNA_bacteria  27.9%      LSU_rRNA_eukarya  17.3%
        LSU_rRNA_bacteria  27.2%      SSU_rRNA_eukarya  12.0%

    -- and a 5% quota is 658,650 residues, so even the smallest of the four is
    2.4x an entire held-out bucket. Any assignment that keeps families whole
    must put each of them somewhere, and wherever it puts them the target
    fractions are gone. A first attempt at balanced greedy packing returned
    55/27/17 for exactly this reason.

    So the split is made in two parts, each answering a question it can
    actually answer:

    **Primary, family-disjoint.** Families larger than the held-out quota go to
    `train` by construction; `val` and `test` are packed from the remainder,
    which is 1,286 groups and 15.6% of residues, where balance *is* achievable.
    What this measures is generalisation **to folds the model has not seen** --
    which is the question a family-disjoint split exists to ask, and it is
    better asked on a set that contains no ribosomes than on one whose answer
    is dominated by them.

    **Secondary, entry-disjoint within the giant families** (`test_ribosomal`).
    Held-out *entries* from the four rRNA families, so the bulk of the corpus is
    still evaluated -- but homologues of these are in training by necessity, so
    it is reported under its own name and never averaged with the primary
    number. Labelling it honestly is the whole point: a single blended figure
    would be 84% a homolog-leaking measurement wearing a family-disjoint name.
    """
    size: Dict[str, int] = defaultdict(int)
    members: Dict[str, List[Dict]] = defaultdict(list)
    for c in chains:
        g = c["rfam"] or f"entry:{c['pdb']}"
        size[g] += int(c["length"])
        members[g].append(c)
    total = sum(size.values()) or 1
    # eligibility is judged against the WHOLE corpus, packing against the
    # remainder -- see the note on MAX_HELDOUT_GROUP_FRAC
    heldout_quota = MAX_HELDOUT_GROUP_FRAC * total

    oversized = {g for g in size if size[g] > heldout_quota}
    rest = [g for g in size if g not in oversized]
    rest_total = sum(size[g] for g in rest) or 1

    # pack the remainder to the target ratios, largest-first, most-underfilled
    quota = {"train": (1.0 - VAL_FRAC - TEST_FRAC) * rest_total,
             "val": VAL_FRAC * rest_total,
             "test": TEST_FRAC * rest_total}
    filled = {k: 0.0 for k in quota}

    def order_key(g: str):
        h = int(hashlib.sha256(f"{seed}/{g}".encode()).hexdigest()[:8], 16)
        return (-size[g], h, g)

    out: Dict[str, str] = {g: "train" for g in oversized}
    for g in sorted(rest, key=order_key):
        k = min(quota, key=lambda s: (filled[s] / quota[s]) if quota[s] else 1.0)
        out[g] = k
        filled[k] += size[g]

    # secondary: hold out whole ENTRIES from the giant families, so the corpus
    # bulk is still measured, under a name that says what it is
    for g in sorted(oversized):
        entries = sorted({c["pdb"] for c in members[g]})
        n_hold = max(1, int(round(TEST_FRAC * len(entries))))
        hold = set()
        for e in sorted(entries, key=lambda e: hashlib.sha256(
                f"{seed}/ribo/{e}".encode()).hexdigest()):
            if len(hold) >= n_hold:
                break
            hold.add(e)
        for c in members[g]:
            if c["pdb"] in hold:
                c["_force_split"] = "test_ribosomal"
    return out


def summarise_split(all_meta: List[Dict]) -> Dict:
    by_chain: Counter = Counter()
    by_res: Counter = Counter()
    for m in all_meta:
        by_chain[m["split"]] += 1
        by_res[m["split"]] += m["length"]
    return {"by_chain": dict(by_chain), "by_residue": dict(by_res)}


def resplit(out: Path) -> None:
    """Recompute the split in place, from the manifest alone."""
    man = json.loads((out / "manifest.json").read_text())
    all_meta = [c for sh in man["shards"] for c in sh["chains"]]
    for m in all_meta:
        m.pop("_force_split", None)
        m.pop("split", None)
    split_of = assign_splits(all_meta)
    for m in all_meta:
        m["split"] = (m.pop("_force_split", None)
                      or split_of[m["rfam"] or f"entry:{m['pdb']}"])
    man["split"].update(summarise_split(all_meta))
    man["split"]["n_groups"] = len(split_of)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"resplit {len(all_meta):,} chains")
    print(f"  by chain  : {man['split']['by_chain']}")
    print(f"  by residue: {man['split']['by_residue']}")


#: What a correctly built corpus must satisfy. These are not style checks:
#: every one of them was passing silently in a corpus that had a real defect,
#: because nothing read the coordinates back after writing them.
#:
#: `MAX_ABS_COORD` is the centring check. float16 spacing is relative, so a
#: coordinate's precision depends entirely on how far from the origin it sits:
#: 0.031 A at 32, 0.125 at 128, 0.25 at 256, 1.0 at 1024. A correctly centred
#: chain has |coord| bounded by its own radius; the largest RNA chain in the
#: PDB is under 200 A across, so anything past 256 means a deposition frame
#: leaked through the cast.
MAX_ABS_COORD = 256.0
#: C4'-N is the glycosidic bond: 3.38 A, essentially rigid. Its median is the
#: cheapest honest readout that coordinates survived the pipeline, and its
#: SPREAD is the readout of storage noise -- float32 gives 0.051 A, uncentred
#: float16 gave 0.104.
BOND_C4_N_A = 3.38
BOND_C4_N_TOL = 0.05
BOND_C4_N_MAX_SD = 0.075


def _validate_geometry(out_dir: Path) -> Dict:
    """Read the shards back and check the coordinates are physically real.

    This exists because the corpus it was written for passed every other
    check. Coverage was 99.9%, bond medians matched reference values to two
    decimal places, and training ran -- while 22% of coordinates were quantised
    at 0.25 A or worse, because they were stored in the deposition's frame
    hundreds of Angstroms from the origin and float16's resolution scales with
    magnitude. Every existing check was translation invariant, so none of them
    could see it.

    Returns the measured numbers for the manifest. Raises if they are wrong:
    a corpus that fails here must not be shipped, because the next thing that
    reads it will be a training run that cannot tell.
    """
    import numpy as np

    shards = sorted(out_dir.glob("shard-*.npz"))
    if not shards:
        raise RuntimeError(f"no shards written to {out_dir}")
    max_abs = 0.0
    n_res = 0
    cn: List = []
    n_center = 0
    center_mag: List = []
    for f in shards:
        with np.load(f) as z:
            if "coord_center" not in z.files:
                raise RuntimeError(
                    f"{f.name} has no coord_center: the translation removed "
                    f"from coords was not recorded, so the deposition frame "
                    f"is unrecoverable")
            x = z["coords"].astype(np.float32)
            m = z["coord_mask"]
            c = z["coord_center"]
            n_center += len(c)
            center_mag.append(np.abs(c).max(-1))
            if m.any():
                max_abs = max(max_abs, float(np.abs(x[m]).max()))
            n_res += len(x)
            k = m[:, 1] & m[:, 2]
            if k.any():
                cn.append(np.linalg.norm(x[k, 1] - x[k, 2], axis=-1))
    d = np.concatenate(cn)
    q = np.percentile(d, [25, 75])
    med, sd = float(np.median(d)), float((q[1] - q[0]) / 1.349)
    cm = np.concatenate(center_mag)
    out = {
        "max_abs_coord_A": round(max_abs, 2),
        "c4p_n_median_A": round(med, 4),
        "c4p_n_robust_sd_A": round(sd, 4),
        "n_coord_residues": n_res,
        "n_coord_center": n_center,
        "center_median_abs_A": round(float(np.median(cm)), 1),
        "center_max_abs_A": round(float(cm.max()), 1),
    }
    fail = []
    if max_abs > MAX_ABS_COORD:
        fail.append(f"max |coord| {max_abs:.1f} A > {MAX_ABS_COORD} -- "
                    f"coordinates are NOT centred, so float16 is quantising "
                    f"them at {np.spacing(np.float16(max_abs)):.3f} A")
    if abs(med - BOND_C4_N_A) > BOND_C4_N_TOL:
        fail.append(f"C4'-N median {med:.4f} A, expected "
                    f"{BOND_C4_N_A} +/- {BOND_C4_N_TOL}")
    if sd > BOND_C4_N_MAX_SD:
        fail.append(f"C4'-N robust sd {sd:.4f} A > {BOND_C4_N_MAX_SD} -- "
                    f"the glycosidic bond is rigid, so this is storage noise")
    print(f"[ds] geometry: max |coord| {max_abs:.1f} A  "
          f"C4'-N {med:.4f} +/- {sd:.4f} A  "
          f"centroids {n_center:,} (median |c| {out['center_median_abs_A']} A)",
          flush=True)
    if fail:
        raise RuntimeError("corpus geometry check FAILED:\n  - "
                           + "\n  - ".join(fail))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--resplit", action="store_true",
                    help="recompute splits from the existing manifest; the "
                         "assignment depends only on per-chain metadata, so it "
                         "needs no re-read of the 16 GB of mmCIF")
    args = ap.parse_args()

    if args.resplit:
        resplit(args.out)
        return

    from pharos.data.dataset import write_shard

    files = sorted(str(p) for p in RAW.glob("*.cif.gz"))
    if args.limit:
        files = files[::max(1, len(files) // args.limit)][:args.limit]
    meta = entry_metadata()
    print(f"[ds] {len(files):,} entries, {args.workers} workers, "
          f"metadata for {len(meta):,}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    shards: List[Dict] = []
    buf: List = []
    all_meta: List[Dict] = []
    n_chains = 0

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        info = write_shard(args.out / f"shard-{len(shards):04d}.npz", buf)
        shards.append(info)
        buf = []

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init, initargs=(meta,)) as ex:
        for n, exs in enumerate(ex.map(_one, files, chunksize=4), 1):
            for e in exs:
                buf.append(e)
                all_meta.append(e.meta())
                n_chains += 1
                if len(buf) >= CHAINS_PER_SHARD:
                    flush()
            if n % 1000 == 0:
                print(f"[ds]   {n}/{len(files)} chains={n_chains:,} "
                      f"shards={len(shards)}", flush=True)
    flush()

    split_of = assign_splits(all_meta)
    by_split: Counter = Counter()
    res_by_split: Counter = Counter()
    for m in all_meta:
        s = m.pop("_force_split", None) or split_of[m["rfam"] or f"entry:{m['pdb']}"]
        m["split"] = s
        by_split[s] += 1
        res_by_split[s] += m["length"]

    # re-attach the split to the per-shard chain lists
    it = iter(all_meta)
    for sh in shards:
        sh["chains"] = [next(it) for _ in range(sh["n_chains"])]

    fam = Counter(m["rfam"] for m in all_meta if m["rfam"])
    lengths = sorted(m["length"] for m in all_meta)
    manifest = {
        "n_entries_scanned": len(files),
        "n_chains": n_chains,
        "n_residues": sum(s["n_residues"] for s in shards),
        "n_contacts": sum(s["n_contacts"] for s in shards),
        "n_shards": len(shards),
        "chains_per_shard": CHAINS_PER_SHARD,
        "contact_def": {"cutoff_A": 8.0, "min_separation": 4},
        "split": {
            "by_chain": dict(by_split), "by_residue": dict(res_by_split),
            "n_groups": len(split_of),
            "policy": ("family-disjoint (D17) on families small enough to hold "
                       "out; four rRNA families are 84.4% of residues and each "
                       "exceeds the quota, so they are forced to train and a "
                       "separate entry-disjoint 'test_ribosomal' measures them"),
            "primary": "val/test are family-disjoint: no Rfam family is shared with train",
            "secondary": ("test_ribosomal is ENTRY-disjoint only -- homologues "
                          "are in train by necessity. Never average it with test."),
        },
        "length": {"min": lengths[0] if lengths else 0,
                   "median": lengths[len(lengths) // 2] if lengths else 0,
                   "max": lengths[-1] if lengths else 0},
        "rfam_families_present": len(fam),
        "chains_with_rfam": sum(fam.values()),
        "top_families": fam.most_common(10),
        #: read back from the written shards, not from what we believed we
        #: wrote -- see `_validate_geometry`
        "geometry": _validate_geometry(args.out),
        "shards": shards,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"\nchains {n_chains:,}  residues {manifest['n_residues']:,}  "
          f"contacts {manifest['n_contacts']:,}  shards {len(shards)}")
    print(f"length min/median/max: {manifest['length']['min']}/"
          f"{manifest['length']['median']}/{manifest['length']['max']}")
    print(f"split by chain  : {dict(by_split)}")
    print(f"split by residue: {dict(res_by_split)}")
    print(f"Rfam families   : {len(fam)} covering {sum(fam.values()):,} chains")
    print(f"\n[ds] -> {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
