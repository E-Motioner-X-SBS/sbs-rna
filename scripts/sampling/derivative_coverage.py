#!/usr/bin/env python3
"""How much of the PDB the curated derivatives actually contain -- and where the
extremes live.

Every structural parameter in this project before the raw pass was fitted on
RNA3DB or gRNAde/RNASolo. This matches both corpora's entry identifiers against
every RNA-bearing entry in the archive and splits the chain-level statistics by
coverage, which is what showed that `target_c`'s tail is a coverage artefact
rather than the modified-residue artefact D9 assumed:

    chains from covered entries    12,573   max effective_c 21.14
    chains from uncovered entries   1,533   max effective_c 23.30

21.14 is exactly the figure published from the derivatives. They were right
about the population they held.

Inputs (produced by the two raw-corpus scans):
    data/samples/analysis/entry_composition_rawpdb_table.json
    data/samples/analysis/targetc_g2g3_rawpdb_chains.json

Usage:
    /store/shuvam/.venv/bin/python scripts/sampling/derivative_coverage.py
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/samples/analysis"
RNA3DB = ROOT / "data/structures/databases/rna3db/rna3db_extracted"
GRNADE = ROOT / "data/structures/databases/grnade_rnasolo/extracted"
#: the chain-length window the contact scans use
MIN_L, MAX_L = 64, 3000


def derivative_entries() -> tuple[set[str], set[str]]:
    """Entry ids held by each derivative corpus.

    RNA3DB names its extracts `<pdb>_<chain>.cif`; gRNAde/RNASolo files start
    with the 4-character id. Both are matched case-insensitively.
    """
    r3 = {p.stem.split("_")[0].lower() for p in RNA3DB.rglob("*.cif")}
    gr = set()
    for p in GRNADE.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdb", ".cif", ".ent"}:
            m = re.match(r"([0-9a-zA-Z]{4})", p.stem)
            if m:
                gr.add(m.group(1).lower())
    return r3, gr


def main() -> None:
    r3, gr = derivative_entries()
    cov = r3 | gr

    table = json.loads((OUT / "entry_composition_rawpdb_table.json").read_text())
    with_rna = [e for e in table if e["n_rna_res"] > 0]
    raw = {e["pdb"].lower() for e in with_rna}
    by = {e["pdb"].lower(): e for e in with_rna}
    missing = sorted(raw - cov)
    m = [by[x] for x in missing]

    chains = json.loads((OUT / "targetc_g2g3_rawpdb_chains.json").read_text())
    inside = [c for c in chains if c["pdb"].lower() in cov]
    outside = [c for c in chains if c["pdb"].lower() not in cov]
    top = sorted(chains, key=lambda c: -c["effective_c"])[:34]
    band = [c for c in chains if 500 <= c["L"] <= 700]

    def mod_frac(cs):
        return sum(c["n_modified"] for c in cs) / max(sum(c["L"] for c in cs), 1)

    # rank of the first chain outside the dominant deposition series
    series = top[0]["pdb"][:2]
    rank_other = next((i + 1 for i, c in enumerate(top)
                       if not c["pdb"].startswith(series)), None)

    res = {
        "entries": {
            "rna3db": len(r3), "grnade_rnasolo": len(gr), "union": len(cov),
            "raw_with_rna": len(raw),
            "uncovered": len(missing),
            "frac_uncovered": round(len(missing) / max(len(raw), 1), 4),
            "in_derivative_not_in_raw": len(cov - raw),
        },
        "uncovered_character": {
            "rna_residues": sum(e["n_rna_res"] for e in m),
            "frac_of_all_rna_residues": round(
                sum(e["n_rna_res"] for e in m)
                / max(sum(e["n_rna_res"] for e in with_rna), 1), 4),
            "median_longest_chain": st.median([e["longest_rna_chain"] for e in m]),
            "entries_with_chain_ge_64": sum(1 for e in m if e["longest_rna_chain"] >= 64),
            "entries_chain_in_window": sum(
                1 for e in m if MIN_L <= e["longest_rna_chain"] <= MAX_L),
            "protein_free": sum(1 for e in m if not e["has_protein"]),
            "protein_free_with_chain_ge_64": sum(
                1 for e in m if not e["has_protein"] and e["longest_rna_chain"] >= 64),
            "ribosome_sized": sum(1 for e in m if e["n_rna_res"] > 2000),
        },
        # The finding: the derivative maximum is right for the derivative's
        # population, and every chain above it is one the derivatives lack.
        "effective_c_by_coverage": {
            "chains_covered": len(inside),
            "max_covered": round(max((c["effective_c"] for c in inside), default=0), 2),
            "chains_uncovered": len(outside),
            "max_uncovered": round(max((c["effective_c"] for c in outside), default=0), 2),
        },
        "tail": {
            "n_over_20": sum(1 for c in chains if c["effective_c"] > 20),
            "dominant_series_prefix": series,
            "n_top34_in_series": sum(1 for c in top if c["pdb"].startswith(series)),
            "rank_of_first_other_structure": rank_other,
            "top34_all_uncovered": all(c["pdb"].lower() not in cov for c in top[:30]),
            "modified_frac_top34": round(mod_frac(top), 4),
            "modified_frac_all": round(mod_frac(chains), 4),
            "modified_frac_length_matched_500_700": round(mod_frac(band), 4),
            "enrichment_vs_length_matched": round(mod_frac(top) / max(mod_frac(band), 1e-9), 1),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "derivative_coverage.json").write_text(json.dumps(res, indent=1))
    (OUT / "uncovered_entries.json").write_text(json.dumps(missing, separators=(",", ":")))

    e, u, c, t = (res["entries"], res["uncovered_character"],
                  res["effective_c_by_coverage"], res["tail"])
    print(f"RNA3DB {e['rna3db']:,} | gRNAde {e['grnade_rnasolo']:,} | "
          f"union {e['union']:,} of {e['raw_with_rna']:,} RNA-bearing entries")
    print(f"UNCOVERED: {e['uncovered']:,} ({100*e['frac_uncovered']:.1f}%)  "
          f"-> {u['entries_chain_in_window']:,} with a chain in {MIN_L}-{MAX_L}, "
          f"{u['protein_free']:,} protein-free")
    print(f"\neffective_c  covered {c['chains_covered']:,} chains -> max {c['max_covered']}"
          f"   uncovered {c['chains_uncovered']:,} -> max {c['max_uncovered']}")
    print(f"tail: {t['n_over_20']} chains over 20; top-34 holds "
          f"{t['n_top34_in_series']} from the '{t['dominant_series_prefix']}' series; "
          f"first other structure at rank {t['rank_of_first_other_structure']}")
    print(f"      top-30 all outside both derivatives: {t['top34_all_uncovered']}")
    print(f"      modified fraction {t['modified_frac_top34']} vs "
          f"{t['modified_frac_length_matched_500_700']} length-matched "
          f"= {t['enrichment_vs_length_matched']}x")
    print(f"\n-> {OUT / 'derivative_coverage.json'}")


if __name__ == "__main__":
    main()
