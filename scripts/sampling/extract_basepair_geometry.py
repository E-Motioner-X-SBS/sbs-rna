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
    """Yield (category, cols, rows) for each requested mmCIF category.

    mmCIF WRAPS long rows across several physical lines, so tokens are
    accumulated until a full row's worth is available rather than assuming one
    row per line. Getting this wrong silently yields ZERO rows for any category
    with many columns -- which is exactly what happened on the first attempt.

    Defect #26 (cycle 8): mmCIF also has TWO serialisations, and this handled
    only one. The `loop_` form puts tags in a header and data on following
    lines; the KEY-VALUE form (`_cat.tag  value`, used when a category has
    exactly one row) puts the value on the tag's own line. This function split
    on whitespace and kept only the tag, discarding the value -- so a key-value
    category yielded zero rows, silently, exactly as `_entity_poly` did in
    defect #25. Measured over the 180-file sample: 9 rows across 9 structures
    (`_ndb_struct_na_base_pair_step` 8B6Z, `_ndb_struct_na_base_pair` 7OEA,
    `_struct_conn` 7VKI/8G90/8OIV, `_pdbx_unobs_or_zero_occ_residues`
    8FMW/8JY0/8V1I, `_em_buffer_component` 8IYQ). Negligible against 103,965
    steps here; unbounded on a corpus we have not sampled.
    """
    op = gzip.open if path.suffix == ".gz" else open
    cur, cols, rows, buf = None, [], [], []
    kv, kv_pending = {}, None

    def emit():
        """Close the current category, converting a key-value block to one row."""
        nonlocal cur, cols, rows, buf, kv, kv_pending
        out = None
        if cur:
            if not rows and kv:
                out = (cur, list(kv.keys()), [dict(kv)])
            elif rows:
                out = (cur, cols, rows)
        cur, cols, rows, buf = None, [], [], []
        kv, kv_pending = {}, None
        return out

    with op(path, "rt", errors="ignore") as fh:
        for line in fh:
            t = line.strip()
            # a `;`-delimited value belonging to a bare key-value tag
            if kv_pending is not None:
                if t.startswith(";"):
                    kv[kv_pending] = t[1:].strip()
                else:
                    kv[kv_pending] = t.strip("'\"")
                kv_pending = None
                continue
            if t.startswith("_"):
                cat = t.split(".", 1)[0]
                if cat in wanted:
                    if cur != cat:
                        got = emit()
                        if got:
                            yield got
                        cur = cat
                    tag, _, rest = t.split(".", 1)[1].partition(" ")
                    rest = rest.strip()
                    if rest:
                        kv[tag] = rest.strip("'\"")     # KEY-VALUE form
                    else:
                        cols.append(tag)                # loop header, or `;` next
                        if kv or not cols[:-1]:
                            kv_pending = None
                    continue
                got = emit()
                if got:
                    yield got
                continue
            if cur:
                if t.startswith(";") and not rows and cols:
                    # value for the last bare tag, in key-value form
                    kv_pending = cols.pop()
                    kv[kv_pending] = t[1:].strip()
                    kv_pending = None
                    continue
                if not t or t.startswith(("#", "loop_", ";")):
                    got = emit()
                    if got:
                        yield got
                    continue
                buf.extend(_tokens(t))
                while cols and len(buf) >= len(cols):  # rows may span lines
                    rows.append(dict(zip(cols, buf[:len(cols)])))
                    buf = buf[len(cols):]
    got = emit()
    if got:
        yield got


def fnum(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


RNA_RES = {"A", "C", "G", "U"}
DNA_RES = {"DA", "DC", "DG", "DT", "DU"}


def main():
    WANT = {"_ndb_struct_na_base_pair", "_ndb_struct_na_base_pair_step",
            "_struct_conn", "_pdbx_unobs_or_zero_occ_residues"}
    steps_by_ctx = defaultdict(list)
    pair_by_class = defaultdict(list)
    saenger = Counter(); lw = Counter(); conn = Counter()
    metal_partners = Counter(); metal_by_ion = Counter()
    n_unobs = 0; n_unobs_rna = 0; n_unobs_dna = 0; n_unobs_other = 0
    n_struct = 0; n_with_bp = 0

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
                # This category covers EVERY polymer in the entry. The sample
                # contains ribosomes whose protein chains dominate the count, so
                # an unfiltered total is ~3x the usable RNA figure. Split by
                # residue type and report the RNA count separately.
                n_unobs += len(rows)
                key = ("auth_comp_id" if "auth_comp_id" in cols
                       else "label_comp_id" if "label_comp_id" in cols else None)
                for r in rows:
                    comp = (r.get(key, "?") if key else "?").strip()
                    if comp in RNA_RES:
                        n_unobs_rna += 1
                    elif comp in DNA_RES:
                        n_unobs_dna += 1
                    else:
                        n_unobs_other += 1
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
            # Significant-figure rounding, not fixed decimals. Twist/roll/tilt force
            # constants run to ~1e-4, so round(x, 4) left them with ONE significant
            # figure -- and a ratio quoted from them (e.g. "134x") then carried far
            # more precision than the data supported.
            "force_constants_diag": {c: float(f"{float(F[i, i]):.6g}") for i, c in enumerate(STEP_COORDS)},
            "condition_number": float(f"{float(np.linalg.cond(C)):.4g}"),
        }

    summary = {
        "structures_scanned": n_struct,
        "structures_with_base_pair_annotations": n_with_bp,
        "total_annotated_steps": sum(len(v) for v in steps_by_ctx.values()),
        "distinct_step_contexts": len(steps_by_ctx),
        "connectivity_records": dict(conn.most_common()),
        "metal_coordination_by_ion": dict(metal_by_ion.most_common(8)),
        "metal_coordinating_atoms": dict(metal_partners.most_common(10)),
        "unobserved_residue_records_ALL_POLYMERS": n_unobs,
        "unobserved_residue_records_RNA": n_unobs_rna,
        "unobserved_residue_records_DNA": n_unobs_dna,
        "unobserved_residue_records_protein_other": n_unobs_other,
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
