#!/usr/bin/env python3
"""Canonical RNA-chain resolution from mmCIF. ONE definition, shared.

Defect #22 (cycle 6): every analysis script rolled its own notion of "an RNA
residue" and they disagreed on **72 of 180 structures**, in both directions:

  * `analyze_ions_motifs.py`  -- a hardcoded list of 16 residue names. Misses any
    modification not on the list (7PKT: 767 residues short).
  * `audit_generalization.py` -- "any component <= 3 chars that is not water,
    group=ATOM". Over-inclusive, and drops HETATM-coded RNA (8JDJ: 156 short the
    other way).

Neither is authoritative. mmCIF *declares* polymer type in `_entity_poly.type`:
`polyribonucleotide` / `polydeoxyribonucleotide` / `polypeptide(L)`, with
`pdbx_strand_id` naming the chains. That declaration is the ground truth, so a
residue is RNA iff it sits in a chain whose entity is a polyribonucleotide --
regardless of how exotic its modification is.

Hybrid chains (`polydeoxyribonucleotide/polyribonucleotide hybrid`) are
excluded, because the declaration does not say which of their residues are the
RNA ones. That exclusion is a choice, so it was measured rather than assumed:
across the 180-structure sample only 3 entries carry hybrid chains (7PU7 P/T,
7S3B B, 8DFA N), holding 59 residues of which **6** are A/C/G/U -- 0.002% of the
306,857 canonical total. The choice is immaterial at this sample size; if a
future corpus carries many hybrids, resolve them per-residue instead.
"""
from __future__ import annotations
import gzip
import re
from pathlib import Path

RNA_STD = {"A", "C", "G", "U"}


def _open(path: Path):
    return gzip.open(path, "rt", errors="ignore") if str(path).endswith(".gz") else open(path, errors="ignore")


def entity_poly_types(path: Path) -> dict[str, str]:
    """auth chain id -> entity_poly.type, from the declaration itself.

    Handles the `;`-delimited multi-line sequence fields that make this loop
    awkward: tokens are accumulated across physical lines until a row is full.
    """
    cols: list[str] = []
    rows: list[list[str]] = []
    in_hdr = in_loop = False
    buf: list[str] = []
    semi = False
    semi_val: list[str] = []
    with _open(path) as fh:
        for line in fh:
            s = line.rstrip("\n")
            if s.startswith("_entity_poly."):
                cols.append(s.strip().split(".", 1)[1].split()[0])
                in_hdr = True
                continue
            if in_hdr and not in_loop:
                if not s.strip() or s.startswith("#"):
                    continue
                in_loop = True
            if in_loop:
                if s.startswith("#") or (s.startswith("_") and not semi):
                    break
                if semi:
                    if s.startswith(";"):
                        semi = False
                        buf.append("".join(semi_val))
                        semi_val = []
                    else:
                        semi_val.append(s)
                    if not semi and len(buf) >= len(cols):
                        rows.append(buf[:len(cols)]); buf = buf[len(cols):]
                    continue
                if s.startswith(";"):
                    semi = True
                    semi_val = [s[1:]]
                    continue
                buf.extend(re.findall(r"'[^']*'|\"[^\"]*\"|\S+", s))
                while len(buf) >= len(cols):
                    rows.append(buf[:len(cols)]); buf = buf[len(cols):]
    if not cols or not rows:
        return {}
    try:
        i_type = cols.index("type")
        i_strand = cols.index("pdbx_strand_id")
    except ValueError:
        return {}
    out: dict[str, str] = {}
    for r in rows:
        t = r[i_type].strip("'\"")
        for ch in r[i_strand].strip("'\"").split(","):
            ch = ch.strip()
            if ch and ch != "?":
                out[ch] = t
    return out


def rna_residues(path: Path):
    """Return (rna_chain -> set of residue ids, per-chain composition counter).

    A residue counts iff its AUTH chain is declared `polyribonucleotide`.
    """
    from collections import Counter, defaultdict
    types = entity_poly_types(path)
    rna_chains = {c for c, t in types.items() if t == "polyribonucleotide"}
    if not rna_chains:
        return {}, Counter(), types
    cols: list[str] = []
    in_loop = header = False
    chains: dict[str, set] = defaultdict(set)
    comp = Counter()
    with _open(path) as fh:
        for line in fh:
            s = line.rstrip("\n")
            if s.startswith("_atom_site."):
                if not in_loop:
                    in_loop, cols = True, []
                cols.append(s.split(".", 1)[1].strip())
                header = True
                continue
            if header and in_loop:
                if s.startswith(("#", "_", "loop_")):
                    in_loop = header = False
                    continue
                p = s.split()
                if len(p) < len(cols):
                    continue
                r = dict(zip(cols, p))
                ch = r.get("auth_asym_id", r.get("label_asym_id", "?"))
                if ch not in rna_chains:
                    continue
                # POLYMER test: mmCIF assigns label_seq_id only to polymer
                # positions. Water, ions and ligands sitting in the same auth
                # chain carry '.', and counting them was inflating the total by
                # ~24% (HOH 37,857 and MG 9,997 in the first 60 files alone).
                if r.get("label_seq_id", ".") in (".", "?"):
                    continue
                key = (ch, r.get("auth_seq_id", r.get("label_seq_id", "?")),
                       r.get("pdbx_PDB_ins_code", "?"))
                if key not in chains[ch]:
                    chains[ch].add(key)
                    comp[r["label_comp_id"].strip('"')] += 1
    return chains, comp, types


if __name__ == "__main__":
    import sys
    from collections import Counter
    root = Path(__file__).resolve().parents[3] / "data/samples/structures"
    files = sorted(root.glob("*.cif.gz"))
    tot = nonstd = 0
    per_type = Counter()
    n_rna_struct = 0
    for f in files:
        ch, comp, types = rna_residues(f)
        per_type.update(types.values())
        n = sum(len(v) for v in ch.values())
        if n:
            n_rna_struct += 1
        tot += n
        nonstd += sum(c for k, c in comp.items() if k not in RNA_STD)
    print(f"structures with a declared RNA entity : {n_rna_struct}/{len(files)}")
    print(f"RNA residues (canonical)              : {tot:,}")
    print(f"  of which outside A/C/G/U            : {nonstd:,} = {100*nonstd/max(tot,1):.2f}%")
    print(f"declared entity types seen            : {dict(per_type.most_common())}")
