#!/usr/bin/env python3
"""Compile the BGSU motif atlas into the frozen bank of ARCHITECTURE v0.2 §8.

RNA reuses a small vocabulary of local structure -- GNRA and UNCG tetraloops,
kink-turns, A-minor motifs, sarcin-ricin loops. §8 calls these "the things that
do not change" and specifies **retrieval, not memorisation**: a frozen KV bank
of 667 motif classes, queried by the pair track after pairing is estimated.

Release 4.12 holds exactly that: 413 internal-loop and 254 hairpin-loop classes,
**667** together.

What a motif class becomes here
-------------------------------
Each class carries a base-pair signature (`cWW-cWW-L-cWW-L` -- the Leontis-
Westhof interaction at each position) and an alignment of its instances, each
instance naming its residues. From those:

* a **base-pair signature histogram**, over the interaction families the atlas
  uses, which is what makes a motif a motif -- a kink-turn is a pattern of
  non-canonical pairs, not a sequence;
* a **position-specific base profile**, computed across instances, which is what
  captures GNRA-ness where it exists;
* size, loop type, and instance count, which say how much to trust the class.

Why the profile is a *value* and not a query key
------------------------------------------------
§8's critical negative result: GNRA k-mer context alone predicts rigidity at
**0.073 sigma** -- negligible. Motif identity needs the interaction graph, not
sequence n-grams. So the sequence profile is carried in the bank as something to
retrieve, and the retrieval key is built from the pair track's estimated pairing.
Querying this bank from raw k-mers would reproduce the measurement that already
failed.

Usage:
    /store/shuvam/.venv/bin/python scripts/build_motif_bank.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/structures/indices/bgsu_motifs"
OUT = ROOT / "data/derived/motif_bank"

BASES = ("A", "C", "G", "U")
#: The interaction-family vocabulary is READ FROM THE ATLAS, not written here.
#: A hand-written Leontis-Westhof list missed `R` (529 occurrences), `tSH` and
#: `cSH` -- cWH and cHW are different annotations, not the same pair seen from
#: two sides, and a curated list silently dropped them into nothing. Same
#: lesson as defect #22: if the data declares a vocabulary, read it.
BP_FAMILIES: tuple = ()
#: profile positions kept; motifs longer than this are truncated and the length
#: is recorded separately so nothing is silently lost
MAX_POS = 24

_UNIT = re.compile(r"^[^|]+\|[^|]*\|[^|]*\|([ACGU])\|", re.I)


def signature_vocabulary(motifs: List[Dict]) -> tuple:
    """Every interaction family the atlas actually uses, ordered by frequency."""
    c: Counter = Counter()
    for m in motifs:
        for t in (m.get("bp_signature") or "").split("-"):
            if t.strip():
                c[t.strip()] += 1
    return tuple(t for t, _ in c.most_common())


def parse_signature(sig: str, vocab: tuple) -> np.ndarray:
    """Histogram of interaction families in a motif's base-pair signature."""
    v = np.zeros(len(vocab), dtype=np.float32)
    for tok in (sig or "").split("-"):
        t = tok.strip()
        if t in vocab:
            v[vocab.index(t)] += 1.0
    return v


def base_profile(alignment: Dict[str, Dict[str, str]]) -> tuple:
    """(MAX_POS, 4) position-specific base frequencies, and the coverage count.

    Positions are the atlas's own alignment columns, so a profile column means
    the same structural position across every instance -- which is what makes
    averaging them meaningful rather than a smear.
    """
    counts = np.zeros((MAX_POS, len(BASES)), dtype=np.float32)
    seen = np.zeros(MAX_POS, dtype=np.float32)
    for inst in alignment.values():
        for pos, unit in inst.items():
            try:
                p = int(pos) - 1
            except ValueError:
                continue
            if not (0 <= p < MAX_POS):
                continue
            m = _UNIT.match(str(unit))
            if not m:
                continue
            counts[p, BASES.index(m.group(1).upper())] += 1.0
            seen[p] += 1.0
    prof = counts / np.maximum(seen[:, None], 1.0)
    return prof, seen


def main() -> None:
    motifs: List[Dict] = []
    for kind, fname in (("IL", "motifs_il_4.12.json"), ("HL", "motifs_hl_4.12.json")):
        path = SRC / fname
        if not path.exists():
            raise SystemExit(f"missing {path}")
        for m in json.loads(path.read_text()):
            motifs.append({**m, "_kind": kind})

    n = len(motifs)
    vocab = signature_vocabulary(motifs)
    sig = np.zeros((n, len(vocab)), dtype=np.float32)
    prof = np.zeros((n, MAX_POS, len(BASES)), dtype=np.float32)
    cover = np.zeros((n, MAX_POS), dtype=np.float32)
    scalar = np.zeros((n, 4), dtype=np.float32)   # kind, n_nt, log instances, chainbreak
    meta: List[Dict] = []

    for i, m in enumerate(motifs):
        sig[i] = parse_signature(m.get("bp_signature", ""), vocab)
        p, seen = base_profile(m.get("alignment", {}) or {})
        prof[i], cover[i] = p, seen
        n_nt = float(m.get("num_nucleotides") or 0)
        n_inst = float(m.get("num_instances") or 0)
        scalar[i] = (1.0 if m["_kind"] == "IL" else 0.0,
                     n_nt / MAX_POS,
                     np.log1p(n_inst) / np.log(1000.0),
                     1.0 if m.get("chainbreak") else 0.0)
        meta.append({"motif_id": m.get("motif_id"), "kind": m["_kind"],
                     "bp_signature": m.get("bp_signature", ""),
                     "common_name": m.get("common_name") or "",
                     "num_instances": int(n_inst), "num_nucleotides": int(n_nt)})

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "bank.npz", signature=sig, profile=prof,
                        coverage=cover, scalar=scalar)
    (OUT / "meta.json").write_text(json.dumps({
        "n_motifs": n,
        "n_internal_loop": sum(1 for m in meta if m["kind"] == "IL"),
        "n_hairpin_loop": sum(1 for m in meta if m["kind"] == "HL"),
        "release": "BGSU motif atlas 4.12",
        "bp_families": list(vocab),
        "max_positions": MAX_POS,
        "n_instances_total": int(sum(m["num_instances"] for m in meta)),
        "signature_dim": len(vocab),
        "profile_dim": [MAX_POS, len(BASES)],
        "motifs": meta,
    }, indent=1))

    fam = Counter()
    for i, m in enumerate(meta):
        for t in m["bp_signature"].split("-"):
            if t.strip():
                fam[t.strip()] += 1
    print(f"motif classes: {n}  (IL {sum(1 for m in meta if m['kind']=='IL')}, "
          f"HL {sum(1 for m in meta if m['kind']=='HL')})")
    print(f"instances     : {sum(m['num_instances'] for m in meta):,}")
    print(f"classes with >= 5 instances: {sum(1 for m in meta if m['num_instances']>=5)}")
    unseen = [t for t in fam if t not in vocab]
    print(f"signature vocabulary: {len(vocab)} families, "
          f"top {dict(fam.most_common(8))}")
    print(f"families dropped by the encoding: {unseen or 'none'}")
    covered = float((cover > 0).any(1).mean())
    print(f"classes with a usable base profile: {100*covered:.1f}%")
    print(f"\n-> {OUT/'bank.npz'}  and  {OUT/'meta.json'}")


if __name__ == "__main__":
    main()
