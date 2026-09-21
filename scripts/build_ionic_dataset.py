#!/usr/bin/env python3
"""Build the ionic-response training channel from the RMDB titrations.

The claim this channel exists to make trainable, from §6.4: PHAROS takes ionic
condition as an input and its prediction should *change* with it. The physics
terms do that by construction; the learned part needs examples where the
concentration was varied, and the PDB structurally cannot provide them --
deposited Mg2+ spans 5-15 mM because unfolded RNA is not crystallised.

One example per (construct, concentration), never averaged: the quantity of
interest is how reactivity at each position changes with [Mg2+], and averaging a
ladder destroys exactly that.

The build also **checks that the physics is in the data** rather than assuming
it. As Mg2+ rises, RNA folds and becomes less reactive to SHAPE and DMS
chemistry, so mean reactivity should fall monotonically along each ladder. A
ladder where it does not is either a construct that does not fold, a normalisation
artefact, or a parsing error, and the build reports the split rather than
averaging over it.

Usage:
    /store/shuvam/.venv/bin/python scripts/build_ionic_dataset.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RDAT = ROOT / "data/benchmarks/rmdb/rdat"
OUT = ROOT / "data/derived/ionic"
sys.path.insert(0, str(ROOT / "src"))

from pharos.data.rdat import titration_examples                   # noqa: E402
from pharos.data.chemistry import chain_chemistry                 # noqa: E402
from pharos.data.vocab import encode_chain, is_deoxy              # noqa: E402


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without a scipy import for two short vectors."""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ion", default="MgCl2")
    ap.add_argument("--min-levels", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    files = sorted(RDAT.glob("*.rdat"))
    print(f"[ionic] scanning {len(files):,} RDATs for {args.ion} ladders")

    ladders: List[Dict] = []
    examples: List[Dict] = []
    for f in files:
        ex = titration_examples(f, ion=args.ion, min_levels=args.min_levels)
        if not ex:
            continue
        conc = np.array([e["conc_mM"] for e in ex], dtype=float)
        mean_r = np.array([float(np.nanmean(e["reactivity"][e["valid"]]))
                           if e["valid"].any() else np.nan for e in ex])
        ok = np.isfinite(mean_r)
        rho = spearman(conc[ok], mean_r[ok]) if ok.sum() >= 3 else float("nan")
        ladders.append({
            "rmdb_id": ex[0]["rmdb_id"], "name": ex[0]["name"],
            "probe": ex[0]["probe"], "length": len(ex[0]["sequence"]),
            "n_levels": int(len(set(np.round(conc, 6)))),
            "conc_min": float(conc.min()), "conc_max": float(conc.max()),
            # negative rho = reactivity falls as Mg rises = the RNA folds
            "spearman_conc_vs_reactivity": round(rho, 4) if np.isfinite(rho) else None,
            "mean_reactivity_low": float(mean_r[ok][0]) if ok.any() else None,
            "mean_reactivity_high": float(mean_r[ok][-1]) if ok.any() else None,
        })
        examples.extend(ex)

    if not examples:
        raise SystemExit("no titration examples found")

    # pack: sequence tokens, chemistry, reactivity, concentration
    args.out.mkdir(parents=True, exist_ok=True)
    toks, mods, chems, reacts, valids, offs = [], [], [], [], [], [0]
    for e in examples:
        comps = list(e["sequence"].replace("T", "U").upper())
        t, m = encode_chain(comps)
        toks.append(t)
        mods.append(m)
        chems.append(chain_chemistry(comps, deoxy_mask=is_deoxy(t)).astype(np.float16))
        reacts.append(np.nan_to_num(e["reactivity"], nan=0.0).astype(np.float16))
        valids.append(e["valid"])
        offs.append(offs[-1] + len(t))
    np.savez_compressed(
        args.out / "ionic.npz",
        tokens=np.concatenate(toks), mod_ids=np.concatenate(mods),
        chem=np.concatenate(chems), reactivity=np.concatenate(reacts),
        valid=np.concatenate(valids), offsets=np.asarray(offs, dtype=np.int64),
        conc_mM=np.asarray([e["conc_mM"] for e in examples], dtype=np.float32))

    rhos = [l["spearman_conc_vs_reactivity"] for l in ladders
            if l["spearman_conc_vs_reactivity"] is not None]
    folds = [r for r in rhos if r < -0.3]
    meta = {
        "ion": args.ion,
        "n_ladders": len(ladders),
        "n_examples": len(examples),
        "n_residues": int(offs[-1]),
        "concentration_range_mM": [min(l["conc_min"] for l in ladders),
                                   max(l["conc_max"] for l in ladders)],
        "ladders_crossing_sub_mM": sum(1 for l in ladders if l["conc_min"] < 1.0),
        "folding_check": {
            "description": ("reactivity should FALL as Mg2+ rises, because the "
                            "RNA folds; Spearman(conc, mean reactivity) < -0.3 "
                            "is counted as showing the transition"),
            "n_with_rho": len(rhos),
            "n_showing_folding": len(folds),
            "median_rho": round(float(np.median(rhos)), 4) if rhos else None,
        },
        "ladders": ladders,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))

    print(f"[ionic] {len(ladders)} ladders -> {len(examples):,} examples, "
          f"{offs[-1]:,} residues")
    print(f"[ionic] concentration range {meta['concentration_range_mM'][0]}"
          f"-{meta['concentration_range_mM'][1]} mM; "
          f"{meta['ladders_crossing_sub_mM']} cross sub-mM")
    fc = meta["folding_check"]
    print(f"[ionic] folding transition visible in {fc['n_showing_folding']}"
          f"/{fc['n_with_rho']} ladders (median rho {fc['median_rho']})")
    for l in sorted(ladders, key=lambda x: (x["spearman_conc_vs_reactivity"]
                                            if x["spearman_conc_vs_reactivity"]
                                            is not None else 0))[:6]:
        print(f"    {l['rmdb_id']:20s} L={l['length']:4d} "
              f"{l['n_levels']:3d} levels  rho {l['spearman_conc_vs_reactivity']}  "
              f"{l['mean_reactivity_low']:.3f} -> {l['mean_reactivity_high']:.3f}")
    print(f"\n[ionic] -> {args.out / 'ionic.npz'}")


if __name__ == "__main__":
    main()
