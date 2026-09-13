#!/usr/bin/env python3
"""Cycle-2 architecture audit: can PHAROS actually handle *any* RNA?

The design claims contact maps, structure and other properties for RNA in
general. This tests the generalization assumptions against the sampled
structures rather than asserting them.

G1  length coverage      - the design caps at L <= 4096. What exceeds it?
G2  multi-chain / RNP    - the design is single-chain. How much RNA is not?
G3  composition bias     - is the evidence base ribosome-dominated?
G7  modified nucleotides - vocab is 5 symbols. Real RNA has >170 modifications.
"""
from __future__ import annotations
import gzip, json, re, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMP = ROOT / "data" / "samples" / "structures"
OUT = ROOT / "data" / "samples" / "analysis"
RNA = {"A", "C", "G", "U"}
AA = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
      "MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE","SEC","PYL"}


def scan(path: Path):
    """One pass over _atom_site: RNA chains+lengths, modified residues, protein presence."""
    cols, in_loop, header = [], False, False
    chain_res = defaultdict(set)          # rna chain -> residue ids
    mod = Counter()
    n_prot_res = 0
    prot_chains = set()
    seen_prot = set()
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
                comp = r.get("label_comp_id", "?").strip('"')
                ch = r.get("label_asym_id", "?")
                seq = r.get("label_seq_id", "?")
                grp = r.get("group_PDB", "")
                if comp in AA:
                    prot_chains.add(ch)
                    if (ch, seq) not in seen_prot:
                        seen_prot.add((ch, seq)); n_prot_res += 1
                    continue
                if grp != "ATOM":
                    continue
                # polymer residue that is not protein: nucleic
                if comp in RNA:
                    chain_res[ch].add(seq)
                elif len(comp) <= 3 and comp not in {"HOH"}:
                    # modified / non-standard nucleotide inside a polymer chain
                    chain_res[ch].add(seq); mod[comp] += 1
    return chain_res, mod, n_prot_res, len(prot_chains)


def main():
    files = sorted(SAMP.glob("*.cif.gz"))
    rows, mod_total = [], Counter()
    for i, f in enumerate(files, 1):
        try:
            chain_res, mod, n_prot, n_pchain = scan(f)
        except Exception:
            continue
        if not chain_res:
            continue
        lens = sorted((len(v) for v in chain_res.values()), reverse=True)
        mod_total.update(mod)
        rows.append({"pdb": f.name.split(".")[0], "n_rna_chains": len(lens),
                     "longest_chain": lens[0], "total_rna_res": sum(lens),
                     "n_modified": sum(mod.values()),
                     "protein_residues": n_prot, "protein_chains": n_pchain})
        if i % 40 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    n = len(rows)
    longest = [r["longest_chain"] for r in rows]
    # G1 length
    g1 = {"n_structures": n,
          "longest_chain_median": st.median(longest),
          "longest_chain_max": max(longest),
          "chains_over_1024": sum(1 for r in rows if r["longest_chain"] > 1024),
          "chains_over_2048": sum(1 for r in rows if r["longest_chain"] > 2048),
          "chains_over_4096": sum(1 for r in rows if r["longest_chain"] > 4096),
          "total_rna_res_over_4096": sum(1 for r in rows if r["total_rna_res"] > 4096),
          "total_rna_res_max": max(r["total_rna_res"] for r in rows)}
    # G2 multi-chain / RNP
    multi = [r for r in rows if r["n_rna_chains"] > 1]
    withprot = [r for r in rows if r["protein_residues"] > 0]
    g2 = {"single_rna_chain": n - len(multi),
          "multi_rna_chain": len(multi),
          "frac_multi_chain": round(len(multi) / n, 4),
          "structures_with_protein": len(withprot),
          "frac_with_protein": round(len(withprot) / n, 4),
          "median_protein_chains_when_present": st.median([r["protein_chains"] for r in withprot]) if withprot else 0,
          "rna_residues_in_protein_complexes": sum(r["total_rna_res"] for r in withprot),
          "rna_residues_total": sum(r["total_rna_res"] for r in rows)}
    g2["frac_rna_residues_in_complexes"] = round(
        g2["rna_residues_in_protein_complexes"] / max(g2["rna_residues_total"], 1), 4)
    # G3 composition proxy: big multi-chain + protein = ribosome-like
    ribo = [r for r in rows if r["total_rna_res"] > 2000 and r["protein_residues"] > 500]
    g3 = {"ribosome_like_structures": len(ribo),
          "frac_structures_ribosome_like": round(len(ribo) / n, 4),
          "rna_residues_in_ribosome_like": sum(r["total_rna_res"] for r in ribo),
          "frac_rna_residues_ribosome_like": round(
              sum(r["total_rna_res"] for r in ribo) / max(g2["rna_residues_total"], 1), 4)}
    # G7 modifications
    tot_mod = sum(mod_total.values())
    g7 = {"distinct_modified_residue_types": len(mod_total),
          "modified_residue_instances": tot_mod,
          "frac_of_all_rna_residues": round(tot_mod / max(g2["rna_residues_total"], 1), 5),
          "structures_with_any_modification": sum(1 for r in rows if r["n_modified"] > 0),
          "top_modifications": dict(mod_total.most_common(12))}

    summary = {"G1_length_coverage": g1, "G2_multichain_and_RNP": g2,
               "G3_composition_bias": g3, "G7_modified_nucleotides": g7}
    (OUT / "generalization_audit.json").write_text(
        json.dumps({"summary": summary, "structures": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
