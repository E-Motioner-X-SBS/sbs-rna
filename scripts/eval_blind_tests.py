#!/usr/bin/env python3
"""Score predictions against RNA-Puzzles / CASP, and say where they would rank.

Run with no model, this establishes the field: for every RNA-Puzzles target it
scores all 885 submitted competitor models and reports the distribution. That
is the number PHAROS has to beat, and knowing it before training finishes is
worth more than knowing it after -- "0.52 TM" means nothing until you know the
best submission on that target scored 0.61 and the median scored 0.34.

Run with `--pred DIR`, it scores PHAROS's own predictions (one `<target>.pdb`
or `.cif` per target) and inserts them into that field, reporting a rank rather
than a bare score.

    python scripts/eval_blind_tests.py                     # the field
    python scripts/eval_blind_tests.py --pred out/preds    # and our place in it

Nothing here trains or loads a model. Scoring is separated from prediction on
purpose: an evaluation harness that can only run inside the training script is
one nobody checks independently.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharos.eval.base_pairs import full_atom_pairs, geometric_pairs  # noqa: E402
from pharos.eval.blind_tests import (Target, all_targets,  # noqa: E402
                                     competitor_label)
from pharos.eval.metrics import (clash_score, gdt_ts, geometry_validity,  # noqa: E402
                                 inf, lddt, rmsd, tm_score)
from pharos.eval.structure import align_by_resnum, read_structure  # noqa: E402

OUT = ROOT / "data/samples/analysis/blind_tests.json"


def score_one(ref_path: Path, pred_path: Path,
              ref_pairs: Optional[set] = None) -> Optional[Dict[str, float]]:
    """Every metric for one (reference, prediction) pair, or None if unusable."""
    try:
        ref = read_structure(ref_path)
        pred = read_structure(pred_path)
    except Exception:
        return None
    a, b = align_by_resnum(ref, pred)
    keep = a.residue_mask & b.residue_mask
    n = int(keep.sum())
    if n < 10:
        return None
    A = a.coords[keep].reshape(-1, 3)
    B = b.coords[keep].reshape(-1, 3)
    # TM-score and GDT-TS are defined PER RESIDUE: the sum runs over residues
    # and is normalised by the reference's residue count. Feeding them all
    # three atoms sums 3L terms against a divisor of L and returns scores above
    # 1, which is how this was caught. C4' is the representative atom, the
    # direct analogue of the C3' that RNA-align uses.
    Ar, Br = a.coords[keep][:, 1], b.coords[keep][:, 1]
    # normalise by the FULL reference length, so a prediction covering half the
    # target cannot score as though the other half did not exist
    out = {
        "n_scored": n,
        "coverage": n / max(len(ref), 1),
        "rmsd": rmsd(B, A),
        "tm": tm_score(Br, Ar, n_norm=len(ref)),
        "lddt": lddt(B, A),
        "gdt_ts": gdt_ts(Br, Ar),
        "clash": clash_score(b.coords[keep]),
        # Is it a bonded chain at all? Every other metric here compares shapes
        # after superposition and would rank a cloud of unconnected points the
        # same way it ranks a merely inaccurate fold. The first predictions off
        # this pipeline had bonds six to eleven times too long with a clash
        # score of 0.000 and nothing objected.
        **geometry_validity(b.coords[keep]),
    }
    if ref_pairs is not None:
        got = geometric_pairs(pred.coords, pred.seq, mask=pred.residue_mask,
                              chain_ids=pred.chain_ids)
        out["inf"] = inf(got, ref_pairs)
    return out


def _pct(values: List[float], q: float) -> float:
    v = [x for x in values if x == x]
    return float(np.percentile(v, q)) if v else float("nan")


def evaluate(targets: List[Target], pred_dir: Optional[Path],
             limit_competitors: Optional[int]) -> Dict:
    results = []
    for t in targets:
        try:
            ref_pairs = full_atom_pairs(t.reference)
        except Exception:
            ref_pairs = None

        field = []
        comps = t.competitors[:limit_competitors] if limit_competitors else t.competitors
        for f in comps:
            s = score_one(t.reference, f, ref_pairs)
            if s:
                g, mdl = competitor_label(f)
                field.append({"group": g, "model": mdl, **s})

        ours = None
        if pred_dir is not None:
            for ext in (".pdb", ".cif"):
                p = pred_dir / f"{t.name}{ext}"
                if p.exists():
                    ours = score_one(t.reference, p, ref_pairs)
                    break

        rec = {"target": t.name, "source": t.source, "description": t.description,
               "n_competitors": len(field), "field": field, "pharos": ours}
        if field:
            for k in ("tm", "lddt", "rmsd", "gdt_ts", "inf"):
                vals = [f[k] for f in field if k in f and f[k] == f[k]]
                if not vals:
                    continue
                best = min(vals) if k == "rmsd" else max(vals)
                rec[f"{k}_best"] = best
                rec[f"{k}_median"] = _pct(vals, 50)
            if ours:
                better = sum(1 for f in field if f.get("tm", -1) > ours.get("tm", -1))
                rec["tm_rank"] = better + 1
                rec["tm_of"] = len(field) + 1
        results.append(rec)
    return {"targets": results}


def report(data: Dict, show_field: bool) -> None:
    rows = data["targets"]
    rp = [r for r in rows if r["source"] == "rna_puzzles" and r["n_competitors"]]
    print(f"\n{'target':12s} {'n':>4s} {'TM best':>8s} {'TM med':>7s} "
          f"{'RMSD best':>10s} {'lDDT best':>10s} {'INF best':>9s}")
    print("  " + "-" * 74)
    for r in rp:
        print(f"  {r['target']:10s} {r['n_competitors']:4d} "
              f"{r.get('tm_best', float('nan')):8.3f} {r.get('tm_median', float('nan')):7.3f} "
              f"{r.get('rmsd_best', float('nan')):10.2f} "
              f"{r.get('lddt_best', float('nan')):10.3f} "
              f"{r.get('inf_best', float('nan')):9.3f}")
    if rp:
        print("  " + "-" * 74)
        for label, key in (("best submission", "_best"), ("median submission", "_median")):
            tm = [r[f"tm{key}"] for r in rp if f"tm{key}" in r]
            ld = [r[f"lddt{key}"] for r in rp if f"lddt{key}" in r]
            print(f"  {label:18s} mean TM {np.mean(tm):.3f}   mean lDDT {np.mean(ld):.3f}")
        print("\n  A TM-score of 0.45 is roughly where the field calls a fold correct.")
        n_ok = sum(1 for r in rp if r.get("tm_best", 0) >= 0.45)
        print(f"  The BEST submission clears it on {n_ok} of {len(rp)} targets.")

    ours = [r for r in rows if r.get("pharos")]
    if ours:
        print(f"\n{'target':12s} {'our TM':>7s} {'rank':>10s} {'our lDDT':>9s} "
              f"{'our RMSD':>9s} {'bonds ok':>9s}")
        print("  " + "-" * 64)
        for r in ours:
            p = r["pharos"]
            rk = f"{r.get('tm_rank','?')}/{r.get('tm_of','?')}"
            bo = 0.5 * (p.get("c4_n_ok", 0.0) + p.get("p_p_ok", 0.0))
            print(f"  {r['target']:10s} {p['tm']:7.3f} {rk:>10s} "
                  f"{p['lddt']:9.3f} {p['rmsd']:9.2f} {bo:9.3f}")
        bo_all = [0.5 * (r["pharos"].get("c4_n_ok", 0.0)
                         + r["pharos"].get("p_p_ok", 0.0)) for r in ours]
        if bo_all and np.nanmean(bo_all) < 0.5:
            print(f"\n  WARNING: mean bond validity {np.nanmean(bo_all):.3f}. These are not")
            print("  bonded chains, so the TM and lDDT above measure the shape of a point")
            print("  cloud. Fix the geometry before reading anything into the ranking.")
    if show_field:
        for r in rp:
            print(f"\n  {r['target']}:")
            for f in sorted(r["field"], key=lambda x: -x.get("tm", 0))[:10]:
                print(f"    {f['group']:22s} m{f['model']:2s} TM {f.get('tm',0):.3f} "
                      f"lDDT {f.get('lddt',0):.3f} RMSD {f.get('rmsd',0):6.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, default=None,
                    help="directory of <target>.pdb predictions to score")
    ap.add_argument("--source", default=None,
                    choices=["rna_puzzles", "casp15", "casp16"])
    ap.add_argument("--target", default=None, help="score one target only")
    ap.add_argument("--max-competitors", type=int, default=None,
                    help="cap submissions per target; the full set is 885 models "
                         "and scoring them all takes a few minutes")
    ap.add_argument("--show-field", action="store_true",
                    help="list the top submissions per target")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    targets = all_targets()
    if args.source:
        targets = [t for t in targets if t.source == args.source]
    if args.target:
        targets = [t for t in targets if t.name == args.target]
    if not targets:
        print("no matching targets", file=sys.stderr)
        return 1

    print(f"[eval] {len(targets)} targets, "
          f"{sum(t.n_competitors for t in targets)} competitor models")
    data = evaluate(targets, args.pred, args.max_competitors)
    report(data, args.show_field)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1))
    print(f"\n[eval] -> {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
