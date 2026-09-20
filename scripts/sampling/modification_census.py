#!/usr/bin/env python3
"""Census of modified RNA residues across the raw PDB archive.

`CHEMISTRY.md` dims 20-22 classify a residue as methylation / pseudouridylation
/ other, and `VOCAB.md` specifies a `MOD` embedding keyed by mmCIF `comp_id`
"falling back to the parent base". Both were written against five hand-picked
examples. This measures the real distribution so the classifier is fitted to
what the archive contains rather than to what came to mind.

What it collects: every RNA polymer residue's `comp_id`, counted, plus the
`_chem_comp.type` and `name` the entry declares for it.

**What it deliberately does not collect: the parent base.** The obvious source
is `_chem_comp.mon_nstd_parent_comp_id` -- mmCIF's own declaration of which
standard base a non-standard residue derives from -- and a first version of
this script parsed it and returned a parent for **zero of 55** species. That is
not a parser bug: entry files carry only
`id / type / mon_nstd_flag / name / pdbx_synonyms / formula / formula_weight`,
and the parent field lives in the **PDB Chemical Component Dictionary**, not in
the entries. Resolving parents therefore needs the CCD and is a separate step;
guessing them from the residue name would be exactly the curated-list mistake
defect #22 was about.

Usage:
    /store/shuvam/.venv/bin/python scripts/sampling/modification_census.py --workers 8
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/structures/raw_pdb_entries"
OUT = ROOT / "data/samples/analysis"
sys.path.insert(0, str(ROOT / "src"))

ACGU = {"A", "C", "G", "U"}


def chem_comp_types(body: str) -> Dict[str, str]:
    """`comp_id -> _chem_comp.type`, e.g. `RNA linking` / `DNA linking`.

    Handles both mmCIF serialisations and, crucially, the `;`-delimited
    multi-line value: `_chem_comp.name` routinely uses it (1B23 writes 5MU's
    name that way), and a tokeniser that does not know about `;` blocks
    desynchronises the whole loop from that row onward and mis-assigns every
    subsequent field.
    """
    out: Dict[str, str] = {}
    tok = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("_chem_comp."):
            i += 1
            continue
        cols, kv = [], {}
        while i < len(lines) and lines[i].strip().startswith("_chem_comp."):
            tag, _, rest = lines[i].strip().partition(" ")
            tag = tag.split(".", 1)[1]
            rest = rest.strip()
            (kv.__setitem__(tag, rest.strip("'\"")) if rest else cols.append(tag))
            i += 1
        if kv and not cols:
            if kv.get("id"):
                out[kv["id"]] = kv.get("type", "?")
            continue
        if not cols or "id" not in cols:
            continue
        ii = cols.index("id")
        it = cols.index("type") if "type" in cols else None
        buf: list = []
        semi = False
        semi_val: list = []
        while i < len(lines):
            t = lines[i]
            st = t.strip()
            if semi:
                if st.startswith(";"):
                    semi = False
                    buf.append("".join(semi_val))
                    semi_val = []
                else:
                    semi_val.append(st)
                i += 1
            elif st.startswith(";"):
                semi = True
                semi_val = [st[1:]]
                i += 1
            elif st.startswith(("#", "_", "loop_")):
                break
            else:
                if st:
                    buf.extend(x.strip("'\"") for x in tok.findall(st))
                i += 1
            while len(buf) >= len(cols):
                row, buf = buf[:len(cols)], buf[len(cols):]
                out[row[ii]] = row[it] if it is not None else "?"
        break
    return out


def census(path_str: str) -> Optional[Tuple[Counter, Dict[str, str], int]]:
    from pharos.data.mmcif_entities import rna_chain_coords
    path = Path(path_str)
    try:
        chains = rna_chain_coords(path)
    except Exception:                                        # noqa: BLE001
        return None
    comps: Counter = Counter()
    for res in chains.values():
        for _, comp, _ in res:
            comps[comp] += 1
    if not comps:
        return Counter(), {}, 0
    try:
        with gzip.open(path, "rt", errors="ignore") as fh:
            body = fh.read()
        types = {k: v for k, v in chem_comp_types(body).items() if k in comps}
    except OSError:
        types = {}
    return comps, types, sum(comps.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="modification_census.json")
    args = ap.parse_args()

    files = sorted(str(p) for p in RAW.glob("*.cif.gz"))
    if args.limit:
        files = files[::max(1, len(files) // args.limit)][:args.limit]
    print(f"[mod] {len(files):,} entries, {args.workers} workers", flush=True)

    total: Counter = Counter()
    types: Dict[str, str] = {}
    n_entries = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, r in enumerate(ex.map(census, files, chunksize=8), 1):
            if r is None:
                continue
            comps, ty, k = r
            if k:
                n_entries += 1
            total.update(comps)
            for cid, t in ty.items():
                if t not in (".", "?", ""):
                    types.setdefault(cid, t)
            if n % 1000 == 0:
                print(f"[mod]   {n}/{len(files)} species={len(total)}", flush=True)

    n_res = sum(total.values())
    mods = Counter({k: v for k, v in total.items() if k not in ACGU})
    n_mod = sum(mods.values())
    res = {
        "n_entries_with_rna": n_entries,
        "n_residues": n_res,
        "n_modified": n_mod,
        "frac_modified": round(n_mod / max(n_res, 1), 5),
        "n_distinct_species": len(mods),
        "standard": {k: total[k] for k in sorted(ACGU)},
        "top_modifications": mods.most_common(40),
        # every species, not just the top 40: the class totals in
        # resolve_ccd_parents.py are summed from these, and truncating the list
        # silently zeroed 330 species holding 3,835 residues.
        "counts": dict(mods.most_common()),
        "declared_type": dict(Counter(
            types.get(k, "?") for k, v in mods.items() for _ in range(v)).most_common()),
        "types": {k: types.get(k, "?") for k, _ in mods.most_common()},
        "note_parents": ("parent base is NOT in entry files -- _chem_comp there "
                         "carries only id/type/mon_nstd_flag/name/synonyms/"
                         "formula/formula_weight. Resolve via the PDB Chemical "
                         "Component Dictionary; see resolve_ccd_parents.py."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(res, indent=1))

    print(f"\nresidues {n_res:,}   modified {n_mod:,} = {100*res['frac_modified']:.3f}%"
          f"   distinct species {len(mods):,}")
    print(f"declared types among modified residues: {res['declared_type']}")
    print("\ntop 20 modifications:")
    for k, v in mods.most_common(20):
        print(f"  {k:6s} {v:9,d}  {100*v/max(n_mod,1):5.2f}%  type={types.get(k,'?')}")
    print(f"\n[mod] -> {OUT / args.out}")


if __name__ == "__main__":
    main()
