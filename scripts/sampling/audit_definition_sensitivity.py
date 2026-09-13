#!/usr/bin/env python3
"""Do the residue-weighted measurements depend on the RNA definition? (C14)

Cycles 6-7 fixed the definition of "an RNA residue" and re-derived the
G-findings under it. Every OTHER residue-weighted number -- contact scaling,
block occupancy, contact separation, rigidity -- still comes from a script that
defines RNA as the four letters `{A, C, G, U}`.

That is not merely a different label set. The canonical measurement says 1.05%
of RNA polymer residues are modified, so a strict ACGU filter DELETES those
residues from the chain. The residues either side then become adjacent, which
changes:

  * chain length L
  * sequence separation |i - j| for every pair spanning a deleted residue
  * which pairs clear the SEQ_SEP guard at all
  * block indices i//b, and therefore block occupancy

So the question is not "are a few residues missing" but "does deleting 1% of
positions from the middle of chains move the numbers the sparse-track budget
was sized on". This measures it directly: the same chain, under both
definitions, for every structure in the sample.
"""
from __future__ import annotations
import gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/pharos/data"))
from mmcif_entities import entity_poly_types

SAMP = ROOT / "data" / "samples" / "structures"
OUT = ROOT / "data" / "samples" / "analysis"
ACGU = {"A", "C", "G", "U"}
CUT, SEQ_SEP, B = 8.0, 4, 4
MAX_L = 3000


def load_chain(path: Path, canonical: bool):
    """Longest RNA chain as a list of per-residue atom lists.

    canonical=False reproduces the published rule (label_comp_id in ACGU).
    canonical=True  uses the declared polymer entity, keeping modified residues.
    """
    rna_chains = None
    if canonical:
        types = entity_poly_types(path)
        rna_chains = {c for c, t in types.items()
                      if t == "polyribonucleotide"
                      or t.startswith("polydeoxyribonucleotide/polyribonucleotide")}
        if not rna_chains:
            return None
    op = gzip.open if path.suffix == ".gz" else open
    cols, in_loop, header = [], False, False
    res = defaultdict(list)
    has_o2 = set()
    with op(path, "rt", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("_atom_site."):
                if not in_loop:
                    in_loop, cols = True, []
                cols.append(s.split(".", 1)[1]); header = True; continue
            if header and in_loop:
                if s.startswith(("#", "_", "loop_")):
                    in_loop = header = False; continue
                p = s.split()
                if len(p) < len(cols):
                    continue
                r = dict(zip(cols, p))
                if r.get("type_symbol") == "H":
                    continue
                if canonical:
                    if r.get("auth_asym_id", "?") not in rna_chains:
                        continue
                    if r.get("label_seq_id", ".") in (".", "?"):
                        continue
                else:
                    if r["label_comp_id"].strip('"') not in ACGU:
                        continue
                try:
                    xyz = (float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"]))
                    sq = int(r.get("label_seq_id", "0"))
                except (ValueError, KeyError):
                    continue
                key = (r.get("label_asym_id", "?"), sq)
                res[key].append(xyz)
                if r.get("label_atom_id", "").strip('"') == "O2'":
                    has_o2.add(key)
    if canonical:
        # hybrid chains: keep only the ribonucleotides (the O2' test)
        hyb = {c for c, t in entity_poly_types(path).items()
               if t.startswith("polydeoxyribonucleotide/polyribonucleotide")}
        if hyb:
            res = {k: v for k, v in res.items() if k[0] not in hyb or k in has_o2}
    by = defaultdict(list)
    for k in res:
        by[k[0]].append(k)
    if not by:
        return None
    ch = max(by, key=lambda c: len(by[c]))
    return [res[k] for k in sorted(by[ch], key=lambda k: k[1])]


def stats(atoms):
    """contacts/nt, block occupancy at b=4, effective c, mean |i-j|.

    Residue pairs are in contact if any heavy atom is within CUT. Done with a
    KD-tree over all atoms at once: the obvious nested loop over candidate pairs
    is O(pairs x atoms^2) in Python and does not finish on ribosomal chains.
    """
    L = len(atoms)
    if L < 32:
        return None
    flat = np.array([xyz for at in atoms for xyz in at], float)
    owner = np.concatenate([np.full(len(at), i, int) for i, at in enumerate(atoms)])
    tree = cKDTree(flat)
    hits = tree.query_pairs(CUT, output_type="ndarray")
    if len(hits) == 0:
        return None
    ri, rj = owner[hits[:, 0]], owner[hits[:, 1]]
    lo = np.minimum(ri, rj); hi = np.maximum(ri, rj)
    keep = (hi - lo) >= SEQ_SEP
    lo, hi = lo[keep], hi[keep]
    if lo.size == 0:
        return None
    pairs = np.unique(np.stack([lo, hi], 1), axis=0)
    nb = (L + B - 1) // B
    occ = np.unique(pairs // B, axis=0)
    seps = pairs[:, 1] - pairs[:, 0]
    return {"L": L, "contacts": int(len(pairs)),
            "contacts_per_nt": float(len(pairs) / L),
            "block_occ": float(len(occ) / (nb * (nb + 1) / 2)),
            "effective_c": float(len(occ) * B * B / L),
            "mean_sep": float(seps.mean())}


def main():
    rows = []
    files = sorted(SAMP.glob("*.cif.gz"))
    for k, f in enumerate(files, 1):
        try:
            a = load_chain(f, canonical=False)
            b = load_chain(f, canonical=True)
        except Exception:
            continue
        if not a or not b:
            continue
        if len(a) > MAX_L or len(b) > MAX_L:
            continue
        sa, sb = stats(a), stats(b)
        if not sa or not sb:
            continue
        rows.append({"pdb": f.name.split(".")[0], "published": sa, "canonical": sb})
        if k % 40 == 0:
            print(f"  {k}/{len(files)} ({len(rows)} comparable)", flush=True)

    changed = [r for r in rows if r["published"]["L"] != r["canonical"]["L"]]
    def agg(key, which):
        return float(np.median([r[which][key] for r in rows]))
    summary = {
        "chains_compared": len(rows),
        "chains_whose_length_changes": len(changed),
        "frac_changed": round(len(changed) / max(len(rows), 1), 4),
        "residues_added_total": sum(r["canonical"]["L"] - r["published"]["L"] for r in rows),
        "median": {k: {"published": round(agg(k, "published"), 4),
                       "canonical": round(agg(k, "canonical"), 4)}
                   for k in ("L", "contacts_per_nt", "block_occ", "effective_c", "mean_sep")},
        "worst_by_length_delta": sorted(
            ({"pdb": r["pdb"], "L_pub": r["published"]["L"], "L_can": r["canonical"]["L"],
              "occ_pub": round(r["published"]["block_occ"], 5),
              "occ_can": round(r["canonical"]["block_occ"], 5),
              "c_pub": round(r["published"]["effective_c"], 2),
              "c_can": round(r["canonical"]["effective_c"], 2)}
             for r in changed),
            key=lambda d: -(d["L_can"] - d["L_pub"]))[:10],
    }
    (OUT / "definition_sensitivity.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
