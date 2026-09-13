#!/usr/bin/env python3
"""Close OQ-1: does the >=30-RNA-residue guard bias the Mg2+/rigidity gradient?

`analyze_rigidity.py` drops structures with fewer than 30 RNA residues. Small
RNAs are exactly where the inner-sphere Mg2+ fraction was measured LOWEST
(0.296 for chains <500 nt vs 0.511 overall), so excluding them could inflate
the headline 1.76 sigma gradient. The rigor audit left the direction UNKNOWN.

Method: recompute the gradient at a range of guard thresholds. The guard exists
because z-normalising B-factors inside a structure of very few residues is
unstable, so this is a bias/variance trade-off and both sides are reported.
"""
from __future__ import annotations
import gzip, json, math, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMP = ROOT / "data" / "samples" / "structures"
OUT = ROOT / "data" / "samples" / "analysis"
RNA = {"A", "C", "G", "U"}
BINS = [(0, 4), (4, 6), (6, 8), (8, 12), (12, 20), (20, 10**9)]
GUARDS = [30, 20, 15, 10, 5, 1]


def parse(path: Path):
    op = gzip.open if path.suffix == ".gz" else open
    cols, in_loop, header, method = [], False, False, None
    res, mg = {}, []
    with op(path, "rt", errors="ignore") as fh:
        for line in fh:
            s = line.rstrip("\n")
            tok = s.strip().split(None, 1)
            if tok and tok[0] == "_exptl.method" and len(tok) > 1:
                method = tok[1].strip().strip("'\"")
            if s.startswith("_atom_site."):
                if not in_loop:
                    in_loop, cols = True, []
                cols.append(s.split(".", 1)[1].strip()); header = True; continue
            if header and in_loop:
                if s.startswith(("#", "_", "loop_")):
                    in_loop = header = False; continue
                if "label_comp_id" not in cols:
                    raise SystemExit(f"_atom_site loop missing label_comp_id in {path.name}; "
                                     f"got {cols[:4]}")
                p = s.split()
                if len(p) < len(cols):
                    continue
                r = dict(zip(cols, p))
                comp = r["label_comp_id"].strip('"')
                atom = r["label_atom_id"].strip('"')
                try:
                    x, y, z = float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])
                except (ValueError, KeyError):
                    continue
                if comp == "MG":
                    mg.append((x, y, z)); continue
                if comp not in RNA:
                    continue
                try:
                    b = float(r["B_iso_or_equiv"])
                except (ValueError, KeyError):
                    continue
                key = (r.get("label_asym_id", "?"), r.get("label_seq_id", "?"))
                d = res.setdefault(key, {"bs": [], "c1": None})
                d["bs"].append(b)
                if atom == "C1'":
                    d["c1"] = (x, y, z)
    return method, {k: v for k, v in res.items() if v["c1"] and v["bs"]}, mg


def main():
    # parse once, re-bin per guard
    parsed = []
    for f in sorted(SAMP.glob("*.cif.gz")):
        try:
            method, res, mg = parse(f)
        except Exception:
            continue
        if not method or "X-RAY" not in method.upper() or not mg or not res:
            continue
        parsed.append((f.name.split(".")[0], res, mg))
    print(f"X-ray structures containing Mg2+ (any size): {len(parsed)}\n")

    results = []
    for guard in GUARDS:
        rows, n_struct, small = [], 0, 0
        for pdb, res, mg in parsed:
            if len(res) < guard:
                continue
            n_struct += 1
            if len(res) < 30:
                small += 1
            keys = list(res)
            bs = [st.mean(res[k]["bs"]) for k in keys]
            mu, sd = st.mean(bs), (st.pstdev(bs) or 1.0)
            for k, b in zip(keys, bs):
                d = min(math.dist(res[k]["c1"], m) for m in mg)
                rows.append(((b - mu) / sd, d))
        out = {"guard": guard, "structures": n_struct, "added_small": small,
               "nt": len(rows), "bins": []}
        for lo, hi in BINS:
            sel = [z for z, d in rows if lo <= d < hi]
            out["bins"].append({"range": f"{lo}-{hi if hi < 10**9 else '+'}", "n": len(sel),
                                "mean_zB": round(st.mean(sel), 3) if sel else None})
        vals = [b["mean_zB"] for b in out["bins"] if b["mean_zB"] is not None]
        out["span_sigma"] = round(vals[-1] - vals[0], 3) if len(vals) >= 2 else None
        out["monotonic"] = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        results.append(out)

    if not parsed:
        raise SystemExit("no X-ray structures with Mg2+ parsed - parser is broken, not the data")
    print(f"{'guard':>6}{'structs':>9}{'+small':>8}{'nt':>8}{'span':>8}{'mono':>7}   per-bin mean zB")
    for r in results:
        bins = " ".join(f"{b['mean_zB']:+.2f}" if b["mean_zB"] is not None else "   --"
                        for b in r["bins"])
        print(f"{r['guard']:>6}{r['structures']:>9}{r['added_small']:>8}{r['nt']:>8}"
              f"{r['span_sigma']:>8}{str(r['monotonic']):>7}   {bins}")

    base = results[0]["span_sigma"]
    verdict = {
        "question": "Does the >=30-RNA-residue guard bias the Mg2+/rigidity gradient?",
        "baseline_guard_30_span_sigma": base,
        "max_abs_change_vs_baseline": round(
            max(abs((r["span_sigma"] or 0) - base) for r in results), 3),
        "all_monotonic": all(r["monotonic"] for r in results),
        "results": results,
    }
    (OUT / "residue_guard_bias.json").write_text(json.dumps(verdict, indent=2))
    print(f"\nbaseline span (guard=30): {base} sigma")
    print(f"max |change| across all guards: {verdict['max_abs_change_vs_baseline']} sigma")
    print(f"monotonic at every guard: {verdict['all_monotonic']}")


if __name__ == "__main__":
    main()
