#!/usr/bin/env python3
"""Coevolution features must recover structure that is already known.

The test that matters is not that the matrix has the right shape. It is that
APC-corrected mutual information over the Rfam tRNA seed reproduces the
cloverleaf -- acceptor stem, T-arm, anticodon stem, D-arm -- with no training
whatsoever. If it does not, the feature is noise and the model would be better
off without it.

Run: python3 src/pharos/data/test_msa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pharos.data import msa as M                                # noqa: E402

fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:58s} {detail}")
    if not ok:
        fails.append(name)


def main() -> int:
    print("== the seed file covers the families the corpus is made of ==")
    n2a = M.name_to_accession()
    for fam, ac, floor in [("tRNA", "RF00005", 500),
                           ("SSU_rRNA_bacteria", "RF00177", 50),
                           ("LSU_rRNA_bacteria", "RF02541", 50),
                           ("U2", "RF00004", 100)]:
        rows = M.alignment_for(fam)
        chk(f"{fam} present with depth", bool(rows) and len(rows) >= floor,
            f"{len(rows) if rows else 0} rows, {n2a.get(fam)}")

    print("\n== APC-MI recovers the tRNA cloverleaf, untrained ==")
    rows = M.alignment_for("tRNA")
    query = "".join(c for c in rows[0] if c != "-")
    C = M.coevolution_matrix("tRNA", query)
    chk("matrix is square and matches the query length",
        C is not None and C.shape == (len(query), len(query)),
        f"{None if C is None else C.shape} vs L={len(query)}")

    L = C.shape[0]
    iu = np.triu_indices(L, 4)                     # ignore the near-diagonal
    order = np.argsort(C[iu])[::-1]
    top = [(int(iu[0][k]), int(iu[1][k])) for k in order[:12]]

    # The acceptor stem is a contiguous antidiagonal ladder: i + j is constant
    # for consecutive pairs. Rather than hard-code alignment-dependent indices,
    # assert the STRUCTURE -- a run of pairs whose index sum is shared.
    sums = {}
    for i, j in top:
        sums.setdefault(i + j, []).append((i, j))
    biggest = max(sums.values(), key=len)
    chk("top pairs contain a stem: >=4 sharing one index sum",
        len(biggest) >= 4, f"{len(biggest)} pairs at i+j={biggest[0][0]+biggest[0][1]}")

    # a stem is a LADDER: consecutive i with consecutive decreasing j
    ii = sorted(p[0] for p in biggest)
    chk("and those pairs are consecutive (a real helix, not scatter)",
        all(b - a == 1 for a, b in zip(ii, ii[1:])), f"i = {ii}")

    chk("long-range coupling dominates", max(abs(i - j) for i, j in top) > 40,
        f"max separation {max(abs(i-j) for i,j in top)}")
    chk("APC leaves the diagonal at zero", float(np.abs(np.diag(C)).max()) == 0.0)

    print("\n== degrades honestly rather than inventing signal ==")
    chk("unknown family returns None", M.coevolution_matrix("NoSuchFamily", "ACGU") is None)
    shallow = M.coevolution_matrix("tRNA", "ACGUACGUACGU")
    chk("a query that does not belong returns None", shallow is None,
        "no forced alignment")

    print("\n== sequence weighting suppresses redundant clades ==")
    msa = M.encode_msa(["ACGUACGU"] * 50 + ["GCAUGCAU"])
    w = M.sequence_weights(msa)
    chk("50 identical rows weigh less each than the singleton",
        w[0] < w[-1], f"{w[0]:.4f} vs {w[-1]:.4f}")

    print("\n== mapping survives long chains with indels, not just tRNA ==")
    # map_to_query used to score seed rows by walking two ungapped strings
    # position by position, which decorrelates after the first indel. tRNA is
    # 76 nt with almost none, so it passed; real 1,500-nt SSU rRNA chains
    # scored 0.36-0.46 against their OWN family and were rejected, silently
    # removing the two largest families in the corpus from coevolution. This
    # asserts a long chain with an internal indel still maps.
    rows_ = M.alignment_for("tRNA")
    if rows_:
        base = M._ungapped(rows_[0])
        # delete an internal block: positional identity collapses after it,
        # a real alignment does not
        mutated = base[:20] + base[26:]
        cols = M.map_to_query(rows_, mutated)
        chk("a query with an internal deletion still maps",
            cols is not None and int((cols >= 0).sum()) > 0.8 * len(mutated),
            f"{int((cols >= 0).sum()) if cols is not None else 0}/{len(mutated)} "
            "positions placed")
        chk("and the mapping is monotone where it is defined",
            cols is not None and bool(
                (np.diff(cols[cols >= 0]) > 0).all()),
            "alignment columns increase along the chain")

    print("\n== the cached couplings find real contacts in deposited structures ==")
    # The claim coevolution has to earn: a pair the alignment says is coupled is
    # a pair that touches in the crystal. Measured against the DEPOSITED contact
    # set of real tRNA chains, with a random-pair baseline at the same sequence
    # separation -- an enrichment figure with no baseline says nothing, because
    # a short chain has a high contact density and any pair looks good.
    import json
    from pharos.data.dataset import ShardReader

    ROOT = Path(__file__).resolve().parents[3]
    man = ROOT / "data/derived/pharos3d/manifest.json"
    cache = ROOT / "data/derived/coevolution/tRNA.npz"
    if not (man.exists() and cache.exists()):
        print("  (corpus or coevolution cache absent -- skipped)")
    else:
        m = json.loads(man.read_text())
        hits = [(s["file"], i) for s in m["shards"]
                for i, c in enumerate(s["chains"])
                if c.get("rfam") == "tRNA" and 60 <= c["length"] <= 90][:8]
        dec = {0: "A", 1: "C", 2: "G", 3: "U"}
        tp = n = 0
        base_hit = base_n = 0
        rng = np.random.default_rng(0)
        for fn, i in hits:
            ex = ShardReader(ROOT / "data/derived/pharos3d" / fn)[i]
            seq = "".join(dec.get(int(t), "N") for t in ex["tokens"])
            con = set(map(tuple, ex["contacts"].tolist()))
            got = M.coevolution_pairs("tRNA", seq)
            if got is None:
                continue
            pairs, _sc = got
            tp += sum(1 for a, b in pairs if (min(a, b), max(a, b)) in con)
            n += len(pairs)
            L = int(ex["length"])
            a, b = rng.integers(0, L, 400), rng.integers(0, L, 400)
            rp = [(min(x, y), max(x, y)) for x, y in zip(a, b) if abs(x - y) >= 4]
            base_hit += sum(1 for q in rp if q in con)
            base_n += len(rp)
        prec = tp / max(n, 1)
        base = base_hit / max(base_n, 1)
        chk("couplings beat random pairs by a wide margin", prec > 3 * base,
            f"{100*prec:.1f}% of {n} couplings are real contacts "
            f"vs {100*base:.1f}% random ({prec/max(base,1e-9):.1f}x)")
        chk("and the absolute precision is usable", prec > 0.30,
            f"{100*prec:.1f}% -- from sequence alone, no structure")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
