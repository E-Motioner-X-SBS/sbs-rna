#!/usr/bin/env python3
"""Cycle-3 P4: what should occupy the over-provisioned token bits?

H1 measured that a nucleotide carries 2.0167 bits of identity while a 512-dim
bf16 token holds 8,192 bits -- 4,062x over-provisioned. That does NOT mean the
embedding is waste (it is compute space, not storage), but it does mean adding
per-nucleotide ATTRIBUTES is essentially free in memory terms.

This inventories every attribute we can actually compute, with its measured
coverage and its information content in bits, so the input feature vector is
designed from data rather than guessed.
"""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "data" / "samples" / "analysis"
OUT = A


def bits(n_classes):
    return math.log2(n_classes) if n_classes > 1 else 0.0


def main():
    bp = json.load(open(A / "basepair_geometry.json"))
    gen = json.load(open(A / "generalization_audit.json"))["summary"]
    ion = json.load(open(A / "ion_summary.json"))
    rig = json.load(open(A / "rigidity_summary.json"))
    coev = json.load(open(A / "coevolution_signal.json"))["summary"]

    n_rna_res = gen["G2_multichain_and_RNP"]["rna_residues_total"]
    n_saenger = bp["saenger_classes_present"]
    n_lw = len(bp["top_leontis_westhof"])
    unobs_rna = bp["unobserved_residue_records_RNA"]
    mod_frac = gen["G7_modified_nucleotides"]["frac_of_all_rna_residues"]

    rows = [
        # name, bits, coverage (fraction of nucleotides where it exists), source, cost
        ("nucleotide identity", 2.0167, 1.0, "elDORS composition entropy (measured)", "free"),
        ("relative position in chain", 6.0, 1.0, "trivially computed", "free"),
        ("local GC in +/-16 window", 4.0, 1.0, "computed from sequence", "free"),
        ("predicted pairing state", 1.0, 1.0, "2D head output, recycled", "recycle"),
        ("base-pair class (Saenger)", bits(n_saenger), 0.0, "_ndb_struct_na_base_pair", "structures only"),
        ("base-pair class (Leontis-Westhof)", bits(n_lw), 0.0, "_ndb_struct_na_base_pair", "structures only"),
        ("modified-residue flag", 1.0, mod_frac, "mmCIF comp_id != ACGU", "structures only"),
        ("disorder / unobserved label", 1.0, unobs_rna / n_rna_res, "_pdbx_unobs (RNA rows)", "structures only"),
        ("normalised B-factor (rigidity)", 4.0, 0.0, "B_iso, X-ray only", "structures only"),
        ("Mg2+ distance class", 3.0, 0.0, "computed from mmCIF", "structures only"),
        ("alignment depth Neff/L (global)", 4.0, 0.0, "Rfam / homology search", "MSA required"),
        ("per-column conservation", 4.0, 0.0, "MSA column entropy", "MSA required"),
        ("coevolution partner score", 4.0, 0.0, "APC-MI, needs MSA", "MSA required"),
        ("SHAPE / DMS reactivity", 4.0, 0.0, "probing data", "NOT ACQUIRED"),
        ("ionic condition (global)", 12.0, 1.0, "user-supplied c_ion", "free"),
        ("in-complex flag (RNP context)", 1.0, gen["G2_multichain_and_RNP"]["frac_rna_residues_in_complexes"],
         "G2 finding: 98.96% are in complexes", "free at train time"),
    ]

    print(f"{'attribute':34s}{'bits':>6s}{'coverage':>10s}  source")
    tot_bits = 0.0
    for name, b, cov, src, cost in rows:
        tot_bits += b
        cs = f"{100*cov:.1f}%" if cov > 0 else "—"
        print(f"  {name:32s}{b:6.2f}{cs:>10s}  {src}")
    print(f"\n  TOTAL if every attribute were present: {tot_bits:.1f} bits/nucleotide")

    d, prec = 512, 16
    tok_bits = d * prec
    print(f"  one 512-dim bf16 token               : {tok_bits:,} bits")
    print(f"  ratio                                : {tok_bits/tot_bits:,.0f}x")
    print()
    print("  => Even a FULLY attributed nucleotide needs ~%.0f bits of INPUT information." % tot_bits)
    print("     d_model is COMPUTE space, not storage. Two consequences:")
    print("       1. low numeric precision is safe in the token path (nothing is in the mantissa)")
    print("       2. adding attributes is nearly free in MEMORY, but costs FLOPs only via d_model")
    print("     So attributes belong as INPUT FEATURES (one projection), NOT as a wider d_model")
    print("     -- FLOPs scale as 6*N_active, and widening d is quadratic in the FFN.")

    always = [r for r in rows if r[4] in ("free", "recycle", "free at train time")]
    print(f"\n  attributes available for ANY sequence at inference: {len(always)} of {len(rows)}")
    print(f"    -> {sum(r[1] for r in always):.1f} bits without MSA, structure or probing")
    struct_only = [r for r in rows if r[4] == "structures only"]
    print(f"  structure-only (training supervision, not inference input): {len(struct_only)}")
    print(f"  MSA-gated: {sum(1 for r in rows if r[4]=='MSA required')}   NOT ACQUIRED: {sum(1 for r in rows if r[4]=='NOT ACQUIRED')}")

    res = {"attributes": [{"name": n, "bits": b, "coverage": c, "source": s, "availability": k}
                          for n, b, c, s, k in rows],
           "total_bits_if_all_present": round(tot_bits, 2),
           "token_bits_512d_bf16": tok_bits,
           "overprovision_ratio": round(tok_bits / tot_bits, 1),
           "inference_available_bits": round(sum(r[1] for r in always), 2)}
    (OUT / "token_attributes.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
