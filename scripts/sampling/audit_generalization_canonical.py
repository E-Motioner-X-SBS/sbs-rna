#!/usr/bin/env python3
"""Re-derive the G-findings with ONE canonical RNA definition (defect #22/#23).

`audit_generalization.py` defined an RNA residue as "any component <= 3 chars,
not water, group=ATOM". That swept in DNA from hybrid duplexes and UNK records,
inflating G7 ~8.6x. The canonical test is mmCIF's own declaration:
a residue is RNA iff its chain's `_entity_poly.type` is `polyribonucleotide`
AND it occupies a polymer position (`label_seq_id` assigned).
"""
from __future__ import annotations
import gzip, json, statistics as st, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/pharos/data"))
from mmcif_entities import rna_residues, RNA_STD

SAMP = ROOT / "data" / "samples" / "structures"
OUT = ROOT / "data" / "samples" / "analysis"
AA = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
      "MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE"}


def protein_residues(path: Path, types: dict) -> tuple[int, int]:
    prot_chains = {c for c, t in types.items() if t.startswith("polypeptide")}
    if not prot_chains:
        return 0, 0
    cols, in_loop, header = [], False, False
    seen = set()
    with gzip.open(path, "rt", errors="ignore") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if s.startswith("_atom_site."):
                if not in_loop:
                    in_loop, cols = True, []
                cols.append(s.split(".", 1)[1].strip()); header = True; continue
            if header and in_loop:
                if s.startswith(("#", "_", "loop_")):
                    in_loop = header = False; continue
                p = s.split()
                if len(p) < len(cols):
                    continue
                r = dict(zip(cols, p))
                ch = r.get("auth_asym_id", "?")
                if ch in prot_chains and r.get("label_seq_id", ".") not in (".", "?"):
                    seen.add((ch, r.get("auth_seq_id", "?")))
    return len(seen), len(prot_chains)


def main():
    rows, mod_tot = [], Counter()
    for i, f in enumerate(sorted(SAMP.glob("*.cif.gz")), 1):
        try:
            chains, comp, types = rna_residues(f)
        except Exception:
            continue
        n_rna = sum(len(v) for v in chains.values())
        if not n_rna:
            continue
        n_prot, n_pchain = protein_residues(f, types)
        lens = sorted((len(v) for v in chains.values()), reverse=True)
        mod = {k: v for k, v in comp.items() if k not in RNA_STD}
        mod_tot.update(mod)
        rows.append({"pdb": f.name.split(".")[0], "n_rna_chains": len(lens),
                     "longest_chain": lens[0], "total_rna_res": n_rna,
                     "n_modified": sum(mod.values()),
                     "protein_residues": n_prot, "protein_chains": n_pchain,
                     "has_dna": any(t.startswith("polydeoxy") for t in types.values())})
        if i % 40 == 0:
            print(f"  {i}/180 ({len(rows)} with RNA)", flush=True)

    n = len(rows)
    tot_rna = sum(r["total_rna_res"] for r in rows)
    longest = [r["longest_chain"] for r in rows]
    withprot = [r for r in rows if r["protein_residues"] > 0]
    multi = [r for r in rows if r["n_rna_chains"] > 1]
    ribo = [r for r in rows if r["total_rna_res"] > 2000 and r["protein_residues"] > 500]
    tot_mod = sum(mod_tot.values())

    S = {
      "canonical_definition": "chain _entity_poly.type == polyribonucleotide AND label_seq_id assigned",
      "structures_with_RNA": n, "structures_scanned": 180,
      "G1": {"longest_chain_median": st.median(longest), "longest_chain_max": max(longest),
             "chains_over_2048": sum(1 for x in longest if x > 2048),
             "chains_over_4096": sum(1 for x in longest if x > 4096),
             "total_rna_res_max": max(r["total_rna_res"] for r in rows),
             "entries_over_4096_total": sum(1 for r in rows if r["total_rna_res"] > 4096)},
      "G2": {"structures_with_protein": len(withprot),
             "frac_structures_with_protein": round(len(withprot)/n, 4),
             "multi_rna_chain": len(multi), "frac_multi_chain": round(len(multi)/n, 4),
             "median_protein_chains": st.median([r["protein_chains"] for r in withprot]) if withprot else 0,
             "rna_residues_total": tot_rna,
             "rna_residues_in_complexes": sum(r["total_rna_res"] for r in withprot),
             "frac_rna_residues_in_complexes": round(
                 sum(r["total_rna_res"] for r in withprot)/tot_rna, 4)},
      "G3": {"ribosome_like": len(ribo), "frac_structures": round(len(ribo)/n, 4),
             "frac_rna_residues": round(sum(r["total_rna_res"] for r in ribo)/tot_rna, 4)},
      "G7": {"modified_instances": tot_mod,
             "frac_of_rna_residues": round(tot_mod/tot_rna, 5),
             "distinct_types": len(mod_tot),
             "top": dict(mod_tot.most_common(10))},
    }
    (OUT / "generalization_canonical.json").write_text(
        json.dumps({"summary": S, "structures": rows}, indent=2))
    print(json.dumps(S, indent=2))


if __name__ == "__main__":
    main()
