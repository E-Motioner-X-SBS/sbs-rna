#!/usr/bin/env python3
"""Tests for the canonical RNA resolver.

The resolver exists because defect #22 showed every script had its own RNA
definition. A shared definition that is itself untested just relocates the
problem, so these tests pin the three properties the rest of the pipeline
depends on:

  1. SOLVENT EXCLUSION   -- no water, ion or ligand may appear as a residue.
                            This is what defect #23 was: 404,205 residues,
                            24.87% "non-ACGU", because HOH and MG sitting in
                            RNA auth chains were being counted.
  2. DNA EXCLUSION       -- deoxyribonucleotide chains contribute nothing.
                            This is what inflated the published G7 by 8.6x.
  3. AGGREGATE STABILITY -- the whole-sample totals the deliverables quote.

Run: python3 src/pharos/data/test_mmcif_entities.py
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmcif_entities import RNA_STD, entity_poly_types, rna_residues

S = Path(__file__).resolve().parents[3] / "data" / "samples" / "structures"

SOLVENT = {"HOH", "DOD", "WAT"}
IONS = {"MG", "NA", "K", "ZN", "MN", "CA", "CL", "SR", "CD", "NI", "CO", "FE",
        "CU", "BA", "CS", "RB", "IR", "OS", "PT", "AU", "HG", "PB", "SM", "EU",
        "TB", "YB", "LU", "SO4", "PO4", "GOL", "EDO", "SPM", "SPD", "PUT"}
DNA = {"DA", "DC", "DG", "DT", "DU", "DI"}

fails: list[str] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:52s} {detail}")
    if not ok:
        fails.append(name)


def main() -> int:
    files = sorted(S.glob("*.cif.gz"))
    if not files:
        print(f"no sample structures under {S}")
        return 1

    print(f"== scanning {len(files)} sampled structures ==")
    tot = 0
    n_with_rna = 0
    comp_all: Counter[str] = Counter()
    solvent_hits: list[str] = []
    dna_hits: list[str] = []
    seq_id_missing: list[str] = []
    types_seen: Counter[str] = Counter()

    for f in files:
        chains, comp, types = rna_residues(f)
        types_seen.update(types.values())
        n = sum(len(v) for v in chains.values())
        if n:
            n_with_rna += 1
        tot += n
        comp_all.update(comp)
        pdb = f.name.split(".")[0]
        for k in comp:
            if k in SOLVENT or k in IONS:
                solvent_hits.append(f"{pdb}:{k}")
            if k in DNA:
                dna_hits.append(f"{pdb}:{k}")
        # every returned residue must carry a real auth_seq_id
        for ch, keys in chains.items():
            for _, seq, _ins in keys:
                if seq in (".", "?"):
                    seq_id_missing.append(f"{pdb}:{ch}")
                    break

    print("\n== property 1: solvent and ions are excluded (defect #23) ==")
    chk("no water/ion/ligand counted as an RNA residue", not solvent_hits,
        "clean" if not solvent_hits else f"{len(solvent_hits)} hits: {solvent_hits[:6]}")

    print("\n== property 2: DNA is excluded (defect #23, the 8.6x) ==")
    chk("no deoxyribonucleotide counted as an RNA residue", not dna_hits,
        "clean" if not dna_hits else f"{len(dna_hits)} hits: {dna_hits[:6]}")

    print("\n== property 3: every residue occupies a polymer position ==")
    chk("all residues carry a resolved auth_seq_id", not seq_id_missing,
        "clean" if not seq_id_missing else f"{seq_id_missing[:6]}")

    print("\n== property 4: aggregate totals the deliverables quote ==")
    chk("structures with a declared RNA entity", n_with_rna == 162, f"{n_with_rna} (want 162)")
    chk("canonical RNA residues", tot == 306857, f"{tot:,} (want 306,857)")
    nonstd = sum(c for k, c in comp_all.items() if k not in RNA_STD)
    chk("residues outside A/C/G/U", nonstd == 3189, f"{nonstd:,} (want 3,189)")
    chk("distinct modification types",
        len({k for k in comp_all if k not in RNA_STD}) == 76,
        f"{len({k for k in comp_all if k not in RNA_STD})} (want 76)")
    frac = 100 * nonstd / tot
    chk("non-ACGU fraction below 2%", frac < 2.0, f"{frac:.2f}%")
    chk("pseudouridine is the top modification",
        max((k for k in comp_all if k not in RNA_STD), key=lambda k: comp_all[k]) == "PSU",
        f"PSU={comp_all['PSU']}")

    print("\n== property 5: hybrid chains are excluded, and that is immaterial ==")
    HYB = "polydeoxyribonucleotide/polyribonucleotide hybrid"
    chk("hybrid chains present in the sample", types_seen[HYB] > 0,
        f"{types_seen[HYB]} chain declarations")
    chk("polyribonucleotide is the declared RNA type",
        types_seen["polyribonucleotide"] > 0, f"{types_seen['polyribonucleotide']} chains")

    print("\n== property 6: determinism ==")
    a = rna_residues(files[0])[1]
    b = rna_residues(files[0])[1]
    chk("repeated calls agree", a == b, files[0].name)

    print("\n== property 7: entity_poly parsing survives multi-line sequences ==")
    # `_entity_poly.pdbx_seq_one_letter_code` is a `;`-delimited block running to
    # thousands of characters in ribosome entries; a naive line-oriented parser
    # desynchronises on it and returns {} or misassigns chains.
    big = max(files, key=lambda f: f.stat().st_size)
    t = entity_poly_types(big)
    chk("largest entry parses to a chain->type map", bool(t),
        f"{big.name}: {len(t)} chains, {len(set(t.values()))} distinct types")
    chk("its chain ids are plausible (no stray sequence text)",
        all(len(c) <= 4 for c in t), f"max id len {max((len(c) for c in t), default=0)}")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
