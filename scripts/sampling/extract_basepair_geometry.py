#!/usr/bin/env python3
"""Mine the mmCIF categories the earlier analyses ignored.

Discovered during the rigor audit: every RNA mmCIF already ships curated
annotations that we were reconstructing badly (or not at all) from coordinates.

  _ndb_struct_na_base_pair       base-pair graph + 6 intra-pair parameters
                                 (shear, stretch, stagger, buckle, propeller,
                                 opening) + Saenger (hbond_type_28) and
                                 Leontis-Westhof (hbond_type_12) classes
  _ndb_struct_na_base_pair_step  6 inter-pair step parameters
                                 (shift, slide, rise, tilt, roll, twist)
  _struct_conn                   curated connectivity: hydrog / metalc / covale
  _pdbx_unobs_or_zero_occ_residues   unmodelled residues = maximal flexibility

The step parameters are the deformation coordinates of nucleic-acid elasticity.
Their covariance across instances of a given context yields a STIFFNESS MATRIX
(F = kT * C^-1, Olson-style), i.e. measured force constants per deformation mode.
That converts "some motifs are stiff" from an assertion into numbers the model
can be trained against.
"""
from __future__ import annotations
import gzip, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "data" / "samples" / "structures"
OUT = ROOT / "data" / "samples" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
STEP_COORDS = ["shift", "slide", "rise", "tilt", "roll", "twist"]
PAIR_COORDS = ["shear", "stretch", "stagger", "buckle", "propeller", "opening"]


def _tokens(line: str):
    """Split an mmCIF data line, honouring single/double quoted values."""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c.isspace():
            i += 1; continue
        if c in "'\"":
            j = i + 1
            while j < n and not (line[j] == c and (j + 1 >= n or line[j + 1].isspace())):
                j += 1
            out.append(line[i + 1:j]); i = j + 1
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            out.append(line[i:j]); i = j
    return out


def loops(path: Path, wanted: set[str]):
    """Yield (category, cols, rows) for each requested mmCIF loop category.

    mmCIF WRAPS long rows across several physical lines, so tokens are
    accumulated until a full row's worth is available rather than assuming one
    row per line. Getting this wrong silently yields ZERO rows for any category
    with many columns -- which is exactly what happened on the first attempt.
    """
    op = gzip.open if path.suffix == ".gz" else open
    cur, cols, rows, buf = None, [], [], []

    def flush():
        nonlocal cur, cols, rows, buf
        if cur and rows:
            yield_val = (cur, cols, rows)
        else:
            yield_val = None
        cur, cols, rows, buf = None, [], [], []
        return yield_val

    with op(path, "rt", errors="ignore") as fh:
        for line in fh:
            t = line.strip()
            if t.startswith("_"):
                cat = t.split(".", 1)[0]
                if cat in wanted:
                    if cur != cat:
                        if cur and rows:
                            yield cur, cols, rows
                        cur, cols, rows, buf = cat, [], [], []
                    cols.append(t.split(".", 1)[1].split()[0])
                    continue
                if cur:
                    if rows:
                        yield cur, cols, rows
                    cur, cols, rows, buf = None, [], [], []
                continue
            if cur:
                if not t or t.startswith(("#", "loop_", ";")):
                    if rows:
                        yield cur, cols, rows
                    cur, cols, rows, buf = None, [], [], []
                    continue
                buf.extend(_tokens(t))
                while len(buf) >= len(cols):          # rows may span lines
                    rows.append(dict(zip(cols, buf[:len(cols)])))
                    buf = buf[len(cols):]
    if cur and rows:
        yield cur, cols, rows


def fnum(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def main():
    WANT = {"_ndb_struct_na_base_pair", "_ndb_struct_na_base_pair_step",
            "_struct_conn", "_pdbx_unobs_or_zero_occ_residues"}
    steps_by_ctx = defaultdict(list)
    pair_by_class = defaultdict(list)
    saenger = Counter(); lw = Counter(); conn = Counter()
    metal_partners = Counter(); metal_by_ion = Counter()
    n_unobs = 0; n_struct = 0; n_with_bp = 0

    files = sorted(S.glob("*.cif.gz"))
    for k, f in enumerate(files, 1):
        n_struct += 1
        got_bp = False
        for cat, cols, rows in loops(f, WANT):
            if cat == "_ndb_struct_na_base_pair_step":
                for r in rows:
                    vec = [fnum(r.get(c)) for c in STEP_COORDS]
                    if any(v is None for v in vec):
                        continue
                    ctx = (f"{r.get('i_label_comp_id_1','?')}{r.get('i_label_comp_id_2','?')}"
                           f"/{r.get('j_label_comp_id_2','?')}{r.get('j_label_comp_id_1','?')}")
                    steps_by_ctx[ctx].append(vec)
            elif cat == "_ndb_struct_na_base_pair":
                got_bp = True
                for r in rows:
                    s28 = r.get("hbond_type_28", "?"); s12 = r.get("hbond_type_12", "?")
                    saenger[s28] += 1; lw[s12] += 1
                    vec = [fnum(r.get(c)) for c in PAIR_COORDS]
                    if not any(v is None for v in vec):
                        pair_by_class[s28].append(vec)
            elif cat == "_struct_conn":
                for r in rows:
                    ct = r.get("conn_type_id", "?")
                    conn[ct] += 1
                    if ct == "metalc":
                        a1 = r.get("ptnr1_label_comp_id", "?"); a2 = r.get("ptnr2_label_comp_id", "?")
                        at1 = r.get("ptnr1_label_atom_id", "?"); at2 = r.get("ptnr2_label_atom_id", "?")
                        METALS = {"MG","K","NA","ZN","CA","MN","SR","BA","CD",
                                  "CO","NI","CU","FE","HG","PB","LI","CS","RB","OS","IR","PT"}
                        if a1 in METALS:
                            ion, atom = a1, at2
                        elif a2 in METALS:
                            ion, atom = a2, at1
                        else:
                            continue          # not a metal-to-ligand record
                        metal_by_ion[ion] += 1
                        metal_partners[atom.strip('"')] += 1
            elif cat == "_pdbx_unobs_or_zero_occ_residues":
                n_unobs += len(rows)
        if got_bp:
            n_with_bp += 1
        if k % 40 == 0:
            print(f"  {k}/{len(files)}", flush=True)

    # ---- stiffness matrices, F = kT * C^-1 (kT = 1 => units of kT/deformation^2)
    stiff = {}
    for ctx, vecs in sorted(steps_by_ctx.items(), key=lambda x: -len(x[1])):
        if len(vecs) < 200:
            continue
        A = np.array(vecs, float)
        C = np.cov(A.T)
        try:
            F = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            continue
        stiff[ctx] = {
            "n": len(vecs),
            "mean": {c: round(float(m), 3) for c, m in zip(STEP_COORDS, A.mean(0))},
            "sd": {c: round(float(v), 3) for c, v in zip(STEP_COORDS, A.std(0))},
            "force_constants_diag": {c: round(float(F[i, i]), 4) for i, c in enumerate(STEP_COORDS)},
        }

    summary = {
        "structures_scanned": n_struct,
        "structures_with_base_pair_annotations": n_with_bp,
        "total_annotated_steps": sum(len(v) for v in steps_by_ctx.values()),
        "distinct_step_contexts": len(steps_by_ctx),
        "connectivity_records": dict(conn.most_common()),
        "metal_coordination_by_ion": dict(metal_by_ion.most_common(8)),
        "metal_coordinating_atoms": dict(metal_partners.most_common(10)),
        "unobserved_residue_records": n_unobs,
        "saenger_classes_present": len([k for k in saenger if k not in "?."]),
        "top_saenger": dict(saenger.most_common(8)),
        "top_leontis_westhof": dict(lw.most_common(8)),
        "stiffness_by_step_context": stiff,
    }
    (OUT / "basepair_geometry.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: v for k, v in summary.items() if k != "stiffness_by_step_context"}, indent=2))
    print(f"\n=== stiffness matrices derived for {len(stiff)} step contexts (n>=200) ===")
    print(f"{'context':>10} {'n':>6}  " + "  ".join(f"{c[:5]:>7}" for c in STEP_COORDS))
    print(f"{'':>10} {'':>6}  " + "  ".join(f"{'sd':>7}" for _ in STEP_COORDS))
    for ctx, d in list(stiff.items())[:10]:
        print(f"{ctx:>10} {d['n']:>6}  " + "  ".join(f"{d['sd'][c]:>7.2f}" for c in STEP_COORDS))


if __name__ == "__main__":
    main()
