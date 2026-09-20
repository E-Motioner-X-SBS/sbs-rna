#!/usr/bin/env python3
"""Whole-entry composition over the raw PDB corpus -> G1, G2, G3.

`recheck_targetc_g2g3_rawpdb.py` computes composition too, but it only keeps
the summary; the per-entry table is thrown away, so every later question about
entry composition would mean another 16 GB decompression pass. This script
does the scan once and **persists the table**, then derives G1/G2/G3 from it.

The three claims it settles, all of them entry-level and therefore never
computable on RNA3DB or RNASolo (which distribute per-chain extracts with the
protein stripped out), all of them standing on the same 180 BGSU structures:

* **G1** total RNA residues per entry: "max 11,478", "44 of 179 entries
  exceed 4,096".  Sets the sequence-length budget.
* **G2** "98.95% of RNA residues sit in entries containing protein".
  ARCHITECTURE v0.2 §11.2 calls this the single largest generalization hazard.
* **G3** "92.65% of residues come from ribosome-like entries".

Usage
-----
    PY=/store/shuvam/.venv/bin/python
    $PY scripts/sampling/entry_composition_rawpdb.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/structures/raw_pdb_entries"
OUT = ROOT / "data/samples/analysis"
sys.path.insert(0, str(ROOT / "src"))

#: Identical to recheck_targetc_g2g3_rawpdb.py so the two are comparable.
RIBO_RNA, RIBO_PROT = 2000, 500
#: G1's published length budget.
LEN_BUDGET = 4096


def composition(path_str: str) -> Optional[Dict]:
    """One entry's polymer composition, plus the two derived G3/length flags.

    The counting itself lives in `pharos.data.mmcif_entities.entry_composition`
    -- the canonical resolver -- and not here. C15: the private copy this
    script started with keyed `_atom_site` rows by `label_asym_id` while
    `entity_poly_types` returns **auth** chain ids, so every entry whose two
    labellings differ reported zero polymer residues. That was 3,254 of 10,527
    raw entries, and it deflated both G2 and G3.
    """
    from pharos.data.mmcif_entities import entry_composition
    try:
        rec = entry_composition(Path(path_str))
    except Exception:                                        # noqa: BLE001
        return None
    rec["ribosome_like"] = (rec["n_rna_res"] > RIBO_RNA
                            and rec["n_protein_res"] > RIBO_PROT)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="entry_composition_rawpdb.json")
    args = ap.parse_args()

    files = sorted(str(p) for p in RAW.glob("*.cif.gz"))
    if args.limit:
        files = files[::max(1, len(files) // args.limit)][:args.limit]
    print(f"[comp] {len(files):,} raw entries, {args.workers} workers", flush=True)

    entries: List[Dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, r in enumerate(ex.map(composition, files, chunksize=8), 1):
            if r:
                entries.append(r)
            if n % 1000 == 0:
                print(f"[comp]   {n}/{len(files)} parsed={len(entries)}", flush=True)

    with_rna = [e for e in entries if e["n_rna_res"] > 0]
    tot = sum(e["n_rna_res"] for e in with_rna)
    in_complex = sum(e["n_rna_res"] for e in with_rna if e["has_protein"])
    ribo = [e for e in with_rna if e["ribosome_like"]]
    over = [e for e in with_rna if e["n_rna_res"] > LEN_BUDGET]
    lens = sorted(e["n_rna_res"] for e in with_rna)

    def pct(p: float) -> int:
        return lens[min(len(lens) - 1, int(round(p / 100 * (len(lens) - 1))))] if lens else 0

    res = {
        "n_entries_scanned": len(files),
        "n_entries_parsed": len(entries),
        "n_entries_with_rna": len(with_rna),
        "published_on_180_bgsu": {
            "G1_max_rna_per_entry": 11478, "G1_entries_over_4096": "44 of 179",
            "G2_frac_residues_in_complex": 0.9895, "G3_frac_residues_ribosomal": 0.9265,
        },
        "G1_length": {
            "max_rna_residues_per_entry": lens[-1] if lens else 0,
            "p50": pct(50), "p90": pct(90), "p99": pct(99), "p999": pct(99.9),
            "entries_over_4096": len(over),
            "frac_entries_over_4096": round(len(over) / max(len(with_rna), 1), 4),
            "max_single_rna_chain": max((e["longest_rna_chain"] for e in with_rna), default=0),
            "largest": [(e["n_rna_res"], e["pdb"]) for e in
                        sorted(with_rna, key=lambda e: -e["n_rna_res"])[:6]],
        },
        "G2_complex": {
            "entries_with_protein": sum(1 for e in with_rna if e["has_protein"]),
            "frac_entries": round(sum(1 for e in with_rna if e["has_protein"])
                                  / max(len(with_rna), 1), 4),
            "rna_residues_total": tot,
            "rna_residues_in_complex": in_complex,
            "frac_residues": round(in_complex / max(tot, 1), 4),
            "median_protein_chains_when_present": (
                st.median([e["n_protein_chains"] for e in with_rna if e["has_protein"]])
                if any(e["has_protein"] for e in with_rna) else None),
            "entries_rna_only": sum(1 for e in with_rna
                                    if not e["has_protein"] and not e["has_dna"]),
            "frac_residues_rna_only": round(
                sum(e["n_rna_res"] for e in with_rna
                    if not e["has_protein"] and not e["has_dna"]) / max(tot, 1), 4),
        },
        "G3_ribosomal": {
            "n_ribosome_like": len(ribo),
            "frac_entries": round(len(ribo) / max(len(with_rna), 1), 4),
            "rna_residues": sum(e["n_rna_res"] for e in ribo),
            "frac_residues": round(sum(e["n_rna_res"] for e in ribo) / max(tot, 1), 4),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(res, indent=1))
    (OUT / "entry_composition_rawpdb_table.json").write_text(
        json.dumps(entries, separators=(",", ":")))

    g1, g2, g3 = res["G1_length"], res["G2_complex"], res["G3_ribosomal"]
    print(f"\nentries with RNA: {len(with_rna):,} of {len(entries):,} parsed")
    print(f"G1  published max 11,478 | RAW {g1['max_rna_residues_per_entry']:,}   "
          f"p99 {g1['p99']:,}  p50 {g1['p50']:,}")
    print(f"    over 4096: {g1['entries_over_4096']:,} "
          f"({100*g1['frac_entries_over_4096']:.1f}%) vs published 44/179 = 24.6%")
    print(f"    largest single RNA chain: {g1['max_single_rna_chain']:,}")
    print(f"G2  published 0.9895 of residues in complex | RAW {g2['frac_residues']} "
          f"({g2['rna_residues_in_complex']:,}/{tot:,});  entries {g2['frac_entries']}")
    print(f"    RNA-only entries: {g2['entries_rna_only']:,} carrying "
          f"{100*g2['frac_residues_rna_only']:.1f}% of residues")
    print(f"G3  published 0.9265 ribosomal | RAW {g3['frac_residues']} "
          f"({g3['n_ribosome_like']:,} entries)")
    print(f"\n[comp] -> {OUT / args.out}")


if __name__ == "__main__":
    main()
