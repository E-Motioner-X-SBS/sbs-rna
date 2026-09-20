#!/usr/bin/env python3
"""Close D9 (`target_c`) and re-measure G2/G3 on RAW PDB entries.

Two open actions, both blocked until the raw corpus existed, both now possible.

**D9 — `target_c` is interim.** It was raised 20 -> 24 on the strength of a
21.14 maximum measured over 20,266 chains of RNA3DB and RNASolo. Those are
pre-normalised derivatives carrying **0.025%** modified residues against raw
PDB's **1.005%** — a 40x under-representation of precisely the quantity the
original warning named as the driver. The number that sets the pair-track
budget has therefore never been measured on data that contains the thing it
depends on. This script measures it.

**G2 / G3 are whole-entry properties.** G2 (98.95% of RNA residues sit in
entries containing protein) and G3 (92.65% of residues come from ribosome-like
entries) describe *entries*, not chains. Both were measured on 180 BGSU
structures, and neither is computable on RNA3DB or RNASolo at all, because
those distribute per-chain extracts with the protein stripped out. G2 is
flagged in ARCHITECTURE v0.2 §11.2 as "the single largest generalization
hazard", so it should not rest on 180 structures.

Method
------
Geometry comes from `pharos.data.mmcif_entities.rna_chain_coords`, the canonical
resolver: a residue is RNA iff its chain declares
`_entity_poly.type = polyribonucleotide` *and* it occupies a polymer position,
with hybrid chains split per residue by the presence of an `O2'` atom. Modified
residues are RETAINED, which is the whole point.

Contacts, blocks and filters are identical to
`recheck_block_sparsity_fullcorpus.py` so the numbers are directly comparable:
8.0 A any-heavy-atom, |i-j| >= 4, 64 <= L <= 3000, >= 20 contacts,
effective_c = occupied_blocks * b^2 / L.

Usage
-----
    PY=/store/shuvam/.venv/bin/python
    $PY scripts/sampling/recheck_targetc_g2g3_rawpdb.py --workers 10
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics as st
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/structures/raw_pdb_entries"
OUT = ROOT / "data/samples/analysis"
sys.path.insert(0, str(ROOT / "src"))

from pharos.data.mmcif_entities import rna_chain_coords   # noqa: E402

ACGU = {"A", "C", "G", "U"}
CUT, SEQ_SEP = 8.0, 4
BLOCKS = [4, 8, 16, 32]
MIN_L, MAX_L, MIN_CONTACTS = 64, 3000, 20
#: G3's definition of ribosome-like: a big RNA plus a lot of protein.
RIBO_RNA, RIBO_PROT = 2000, 500


def contact_pairs(atoms: List[List[tuple]]) -> np.ndarray:
    """Residue pairs in contact: any heavy atom within `CUT`, |i-j| >= SEQ_SEP.

    Returns an (n, 2) int array of residue indices, i < j, deduplicated.

    Vectorised, and it has to be. The obvious version -- iterate the atom pairs
    and add `(i, j)` tuples to a Python set -- is O(atom pairs) in interpreted
    code with ~100 bytes of object overhead per entry retained. On a 1,758-nt
    rRNA chain (37,455 heavy atoms, several million atom pairs inside 8 A) one
    worker of the raw-corpus scan reached **28 GB RSS and a 52 GB peak** and
    stopped making progress. Mapping atom pairs to residue pairs in numpy and
    deduplicating with a single `np.unique` over a packed key does the same work
    in tens of MB.
    """
    counts = [len(a) for a in atoms]
    n_at = sum(counts)
    if n_at == 0:
        return np.empty((0, 2), dtype=np.int64)
    flat = np.empty((n_at, 3), dtype=np.float64)
    owner = np.empty(n_at, dtype=np.int64)
    k = 0
    for i, at in enumerate(atoms):
        if not at:
            continue
        flat[k:k + len(at)] = at
        owner[k:k + len(at)] = i
        k += len(at)

    pr = cKDTree(flat).query_pairs(CUT, output_type="ndarray")
    if pr.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    a, b = owner[pr[:, 0]], owner[pr[:, 1]]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    keep = (hi - lo) >= SEQ_SEP
    if not keep.any():
        return np.empty((0, 2), dtype=np.int64)
    lo, hi = lo[keep], hi[keep]
    # pack (lo, hi) into one int64 so the dedup is a single sort
    n = len(atoms)
    uniq = np.unique(lo * n + hi)
    return np.stack((uniq // n, uniq % n), axis=1)


def entry_composition(path: Path) -> Dict:
    """Protein/RNA residue counts per entry, for G2 and G3.

    Delegates to `pharos.data.mmcif_entities.entry_composition`. C15: the
    private implementation this script carried keyed `_atom_site` rows by
    `label_asym_id` while `entity_poly_types` returns **auth** chain ids, so
    every entry whose two labellings differ -- 3,254 of 10,527, mostly older
    depositions such as 1ARJ (`label A` / `auth N`) -- reported zero polymer
    residues and silently deflated G2 and G3. That is defect #22 recurring, so
    the counting now lives once, beside the resolver whose keys it must match.
    """
    from pharos.data.mmcif_entities import entry_composition as _comp
    try:
        rec = _comp(path)
    except Exception:                                        # noqa: BLE001
        return {}
    rec["ribosome_like"] = (rec["n_rna_res"] > RIBO_RNA
                            and rec["n_protein_res"] > RIBO_PROT)
    return rec


def analyse(path_str: str) -> Optional[Dict]:
    path = Path(path_str)
    comp = entry_composition(path)
    if not comp:
        return None
    rec: Dict = {"pdb": path.stem.replace(".cif", ""), **comp, "chains": []}
    try:
        chains = rna_chain_coords(path)
    except Exception:                                        # noqa: BLE001
        return rec
    for ch, residues in chains.items():
        atoms = [a for _, _, a in residues]
        comps = [c for _, c, _ in residues]
        L = len(atoms)
        if not (MIN_L <= L <= MAX_L):
            continue
        pairs = contact_pairs(atoms)
        if len(pairs) < MIN_CONTACTS:
            continue
        blocks = {}
        for b in BLOCKS:
            bi, bj = pairs[:, 0] // b, pairs[:, 1] // b
            nb = (L + b - 1) // b
            n_occ = int(np.unique(bi * (nb + 1) + bj).size)
            blocks[str(b)] = {"effective_c": round(n_occ * b * b / L, 2),
                              "occupancy": round(n_occ / max(nb * (nb + 1) // 2, 1), 5)}
        rec["chains"].append({
            "chain": ch, "L": L, "n_contacts": len(pairs),
            "contacts_per_nt": round(len(pairs) / L, 4),
            "n_modified": sum(1 for c in comps if c not in ACGU),
            "blocks": blocks,
        })
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="targetc_g2g3_rawpdb.json")
    args = ap.parse_args()

    files = sorted(str(p) for p in RAW.glob("*.cif.gz"))
    if args.limit:
        files = files[::max(1, len(files) // args.limit)][:args.limit]
    print(f"[rawc] {len(files):,} raw entries, {args.workers} workers")

    entries: List[Dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, r in enumerate(ex.map(analyse, files, chunksize=4), 1):
            if r:
                entries.append(r)
            if n % 500 == 0:
                nch = sum(len(e["chains"]) for e in entries)
                print(f"[rawc]   {n}/{len(files)} entries={len(entries)} chains={nch}",
                      flush=True)

    chains = [c for e in entries for c in e["chains"]]
    cs = np.array([c["blocks"]["4"]["effective_c"] for c in chains]) if chains else np.zeros(0)
    cpn = np.array([c["contacts_per_nt"] for c in chains]) if chains else np.zeros(0)
    nmod = sum(c["n_modified"] for c in chains)
    nres = sum(c["L"] for c in chains)

    # G2 / G3 over whole entries
    with_rna = [e for e in entries if e["n_rna_res"] > 0]
    tot_rna = sum(e["n_rna_res"] for e in with_rna)
    rna_in_complex = sum(e["n_rna_res"] for e in with_rna if e["has_protein"])
    ribo = [e for e in with_rna if e["ribosome_like"]]
    rna_in_ribo = sum(e["n_rna_res"] for e in ribo)

    res = {
        "n_entries_scanned": len(files),
        "n_entries_with_rna": len(with_rna),
        "n_chains": len(chains),
        "published": {"max_effective_c_derivatives": 21.14, "target_c_v02": 24,
                      "G2_frac_in_complex": 0.9895, "G3_frac_ribosomal": 0.9265,
                      "max_contacts_per_nt_derivatives": 7.66},
        "target_c_raw": {
            "max_effective_c": round(float(cs.max()), 2) if cs.size else None,
            "p99": round(float(np.percentile(cs, 99)), 2) if cs.size else None,
            "p999": round(float(np.percentile(cs, 99.9)), 2) if cs.size else None,
            "mean": round(float(cs.mean()), 2) if cs.size else None,
            "n_over_20": int((cs > 20).sum()), "n_over_24": int((cs > 24).sum()),
            "frac_over_24": round(float((cs > 24).mean()), 6) if cs.size else None,
            "worst": sorted(((c["blocks"]["4"]["effective_c"], e["pdb"], c["chain"], c["L"])
                             for e in entries for c in e["chains"]), reverse=True)[:6],
        },
        "contacts_per_nt_raw": {
            "max": round(float(cpn.max()), 2) if cpn.size else None,
            "p999": round(float(np.percentile(cpn, 99.9)), 2) if cpn.size else None,
            "median": round(float(np.median(cpn)), 2) if cpn.size else None,
        },
        "modified_residues": {
            "residues": nres, "modified": nmod,
            "frac": round(nmod / max(nres, 1), 5),
        },
        "G2_in_complex": {
            "entries_with_protein": sum(1 for e in with_rna if e["has_protein"]),
            "frac_entries": round(sum(1 for e in with_rna if e["has_protein"])
                                  / max(len(with_rna), 1), 4),
            "rna_residues_total": tot_rna,
            "rna_residues_in_complex": rna_in_complex,
            "frac_residues": round(rna_in_complex / max(tot_rna, 1), 4),
            "median_protein_chains_when_present": (
                st.median([e["n_protein_chains"] for e in with_rna if e["has_protein"]])
                if any(e["has_protein"] for e in with_rna) else None),
        },
        # G1 whole-entry claims. Like G2/G3 these describe ENTRIES, so they
        # were never computable on the per-chain derivatives either: "total RNA
        # residues per entry, max 11,478" and "44 of 179 entries exceed 4,096"
        # have rested on 180 structures since they were written.
        "G1_whole_entry": {
            "published_max_rna_per_entry": 11478,
            "published_entries_over_4096": "44 of 179",
            "max_rna_residues_per_entry": max((e["n_rna_res"] for e in with_rna),
                                              default=0),
            "entries_over_4096": sum(1 for e in with_rna if e["n_rna_res"] > 4096),
            "frac_entries_over_4096": round(
                sum(1 for e in with_rna if e["n_rna_res"] > 4096) / max(len(with_rna), 1), 4),
            "median_rna_residues_per_entry": (
                st.median([e["n_rna_res"] for e in with_rna]) if with_rna else None),
        },
        "G3_ribosomal": {
            "n_ribosome_like": len(ribo),
            "frac_entries": round(len(ribo) / max(len(with_rna), 1), 4),
            "rna_residues": rna_in_ribo,
            "frac_residues": round(rna_in_ribo / max(tot_rna, 1), 4),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(res, indent=1))
    # Persist the per-chain table, not just the summary. Twice now a question
    # about the shape of the tail -- how concentrated is it, is it one molecule
    # deposited six times -- has cost a second 16 GB pass over the archive
    # because only the summary survived.
    (OUT / args.out.replace(".json", "_chains.json")).write_text(json.dumps(
        [{"pdb": e["pdb"], "chain": c["chain"], "L": c["L"],
          "n_contacts": c["n_contacts"], "contacts_per_nt": c["contacts_per_nt"],
          "n_modified": c["n_modified"],
          "effective_c": c["blocks"]["4"]["effective_c"],
          "ribosome_like": e.get("ribosome_like", False),
          "has_protein": e.get("has_protein", False)}
         for e in entries for c in e["chains"]], separators=(",", ":")))

    p, t = res["published"], res["target_c_raw"]
    print(f"\nentries with RNA: {len(with_rna):,}   chains analysed: {len(chains):,}")
    print(f"modified residues: {nmod:,}/{nres:,} = {100*res['modified_residues']['frac']:.3f}% "
          f"(derivatives 0.025%)")
    print(f"\nD9 target_c   derivatives max {p['max_effective_c_derivatives']} | "
          f"RAW max {t['max_effective_c']}  p99.9 {t['p999']}  mean {t['mean']}")
    print(f"              chains over 20: {t['n_over_20']:,}   over 24: {t['n_over_24']:,} "
          f"({100*(t['frac_over_24'] or 0):.4f}%)")
    print(f"              worst: {[(a,b,c,d) for a,b,c,d in t['worst'][:3]]}")
    print(f"\nG2  published {p['G2_frac_in_complex']} of residues in complex | "
          f"RAW {res['G2_in_complex']['frac_residues']} "
          f"({res['G2_in_complex']['rna_residues_in_complex']:,}/{tot_rna:,})")
    g1 = res["G1_whole_entry"]
    print(f"G1  published max RNA/entry {g1['published_max_rna_per_entry']:,} | "
          f"RAW {g1['max_rna_residues_per_entry']:,}   "
          f"entries over 4096: {g1['entries_over_4096']:,} "
          f"({100*g1['frac_entries_over_4096']:.1f}%)")
    print(f"G3  published {p['G3_frac_ribosomal']} ribosomal | "
          f"RAW {res['G3_ribosomal']['frac_residues']} "
          f"({res['G3_ribosomal']['n_ribosome_like']:,} entries)")
    print(f"\n[rawc] -> {OUT / args.out}")


if __name__ == "__main__":
    main()
