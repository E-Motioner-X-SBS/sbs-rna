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
  4. BOTH mmCIF SERIALISATIONS -- defect #25: only the `loop_` form was handled,
                            so all 16 single-polymer-entity structures parsed to
                            zero RNA. They are the small ISOLATED RNAs, and
                            losing them inflated G2 from 98.95% to 99.70%.
  5. HYBRID RESOLUTION   -- DNA/RNA hybrid chains split per residue by the
                            presence of an O2' atom.

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
    chk("structures with a declared RNA entity", n_with_rna == 179, f"{n_with_rna} (want 179)")
    chk("canonical RNA residues", tot == 309197, f"{tot:,} (want 309,197)")
    nonstd = sum(c for k, c in comp_all.items() if k not in RNA_STD)
    chk("residues outside A/C/G/U", nonstd == 3237, f"{nonstd:,} (want 3,237)")
    chk("distinct modification types",
        len({k for k in comp_all if k not in RNA_STD}) == 82,
        f"{len({k for k in comp_all if k not in RNA_STD})} (want 82)")
    frac = 100 * nonstd / tot
    chk("non-ACGU fraction below 2%", frac < 2.0, f"{frac:.2f}%")
    chk("pseudouridine is the top modification",
        max((k for k in comp_all if k not in RNA_STD), key=lambda k: comp_all[k]) == "PSU",
        f"PSU={comp_all['PSU']}")

    print("\n== property 5: both mmCIF serialisations parse (defect #25) ==")
    # 7LNE and 8FEQ write _entity_poly in the KEY-VALUE form (one polymer
    # entity, so no `loop_`). Before the fix both returned {} and reported zero
    # RNA. They are single small RNAs -- 7LNE has no protein at all.
    for pdb, want_chains in (("7LNE", 2), ("8FEQ", 1)):
        ty = entity_poly_types(S / f"{pdb}.cif.gz")
        got = sum(1 for v in ty.values() if v == "polyribonucleotide")
        chk(f"key-value form parses ({pdb})", got == want_chains,
            f"{got} RNA chains (want {want_chains})")
    chk("no structure silently parses to an empty declaration",
        n_with_rna >= 179, f"{n_with_rna}/180 carry countable RNA")

    print("\n== property 6: hybrid chains resolved per residue by O2' ==")
    HYB = "polydeoxyribonucleotide/polyribonucleotide hybrid"
    chk("hybrid chains present in the sample", types_seen[HYB] > 0,
        f"{types_seen[HYB]} chain declarations")
    # 7S3B chain B: 6 ribo (U,C,G) + 2 deoxy (DU, BRU). 7PU7 and 8DFA hybrid
    # chains are entirely deoxy in the modelled coordinates.
    ch7, comp7, _ = rna_residues(S / "7S3B.cif.gz")
    chk("7S3B hybrid contributes its 6 ribonucleotides",
        sum(comp7[k] for k in ("U", "C", "G")) == 6, f"{dict(comp7)}")
    chk("7S3B hybrid excludes DU and BRU (no O2')",
        comp7["DU"] == 0 and comp7["BRU"] == 0, "deoxy excluded")
    ch9, comp9, ty9 = rna_residues(S / "7PU7.cif.gz")
    chk("7PU7 hybrid is all-deoxy, so contributes nothing",
        sum(len(v) for v in ch9.values()) == 0,
        "its only nucleic entity has no RNA in the coordinates")
    chk("polyribonucleotide is the declared RNA type",
        types_seen["polyribonucleotide"] > 0, f"{types_seen['polyribonucleotide']} chains")

    print("\n== property 7: determinism ==")
    a = rna_residues(files[0])[1]
    b = rna_residues(files[0])[1]
    chk("repeated calls agree", a == b, files[0].name)

    print("\n== property 8: entity_poly parsing survives multi-line sequences ==")
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
