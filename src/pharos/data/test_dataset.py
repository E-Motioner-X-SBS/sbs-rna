#!/usr/bin/env python3
"""Tests for the data pipeline: tokenizer, contacts, shards, batching, splits.

This is what every head trains on, and its failures are the quiet kind. A
tokenizer that maps a modified G to `N` still trains. A shard whose offsets are
off by one still loads. A split that leaks a family still reports a number. So
the properties pinned here are the ones whose violation produces a *plausible*
result rather than an error:

  1. MODIFIED RESIDUES KEEP THEIR PARENT -- token = CCD parent, mod_id = species.
     Collapsing them to `N` is what VOCAB.md exists to stop.
  2. CONTACTS MATCH A REFERENCE -- the vectorised set against a naive one.
  3. SHARDS ROUND-TRIP EXACTLY -- an off-by-one in the offset table silently
     shifts every label of every chain after the first.
  4. PADDING CANNOT LEAK -- and each head's validity mask means what it says,
     in particular D12's X-ray-only rigidity.
  5. THE SPLIT IS FAMILY-DISJOINT -- D25's guarantee, asserted rather than
     assumed: no Rfam family may appear in both train and a held-out split.
  6. RDAT PARSES BOTH VERSIONS -- 0.34 is tab-separated and 0.24 space-
     separated, and the space-separated form parsed to nothing.

Run: python3 src/pharos/data/test_dataset.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pharos.data.dataset import (ChainExample, ShardReader,        # noqa: E402
                                 contact_set, disorder_is_meaningful,
                                 pad_batch, write_shard)
from pharos.data.vocab import (SYM2ID, SYMBOLS, encode_chain,      # noqa: E402
                               encode_residue, is_deoxy, mod_vocab_size)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/derived/pharos3d"
fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} {detail}")
    if not ok:
        fails.append(name)


def naive_contacts(atoms, cutoff=8.0, min_sep=4):
    out = set()
    for i, ai in enumerate(atoms):
        for j, aj in enumerate(atoms):
            if j - i < min_sep:
                continue
            for p in ai:
                if any(sum((p[k] - q[k]) ** 2 for k in range(3)) <= cutoff ** 2
                       for q in aj):
                    out.add((i, j))
                    break
    return sorted(out)


def main() -> int:
    rng = np.random.default_rng(0)

    print("== property 1: the tokenizer keeps parents and species ==")
    chk("standard bases are themselves",
        [encode_residue(b)[0] for b in "ACGU"] == [0, 1, 2, 3], "")
    chk("DNA keeps its own symbols, not folded onto RNA",
        encode_residue("DT")[0] == SYM2ID["DT"] != SYM2ID["U"],
        "hybrid duplexes are real; calling a DT a U is a chemistry error")
    if mod_vocab_size():
        t, m = encode_residue("PSU")
        chk("PSU -> parent U with a species id", (SYMBOLS[t], m > 0) == ("U", True),
            f"{SYMBOLS[t]}, mod {m}")
        t2, m2 = encode_residue("A2M")
        chk("A2M -> parent A (only the CCD knows this)",
            SYMBOLS[t2] == "A" and m2 > 0, f"{SYMBOLS[t2]}, mod {m2}")
        chk("distinct modifications get distinct ids", m != m2, f"{m} vs {m2}")
    chk("an unknown component is N_struct, not a crash",
        SYMBOLS[encode_residue("ZZZ")[0]] == "N", "")
    tok, _ = encode_chain(["A", "DT", "G"])
    chk("is_deoxy flags only the DNA position",
        list(is_deoxy(tok)) == [False, True, False], "")

    print("\n== property 2: contacts match a naive reference ==")
    for n_res, seed in ((12, 1), (30, 2), (60, 3)):
        r = np.random.default_rng(seed)
        atoms = [[tuple(r.normal(0, 6, 3)) for _ in range(r.integers(3, 8))]
                 for _ in range(n_res)]
        got = contact_set(atoms)
        want = naive_contacts(atoms)
        same = len(got) == len(want) and all(tuple(g) == w
                                             for g, w in zip(got.tolist(), want))
        chk(f"n={n_res:<3} vectorised == naive", same,
            f"{len(got)} vs {len(want)} pairs")
    atoms = [[(0.0, 0.0, float(i))] for i in range(20)]
    c = contact_set(atoms)
    chk("minimum separation respected",
        bool(len(c) == 0 or (c[:, 1] - c[:, 0]).min() >= 4), f"{len(c)} pairs")
    chk("strictly upper triangular and unique",
        bool(len(c) == 0 or ((c[:, 1] > c[:, 0]).all()
                             and len(np.unique(c, axis=0)) == len(c))), "")

    print("\n== property 3: shards round-trip exactly ==")
    ex = []
    for i, L in enumerate((7, 13, 5)):
        ex.append(ChainExample(
            pdb=f"t{i}", chain="A", length=L,
            tokens=rng.integers(0, 4, L).astype(np.int8),
            mod_ids=rng.integers(0, 5, L).astype(np.int16),
            chem=rng.random((L, 24)).astype(np.float16),
            contacts=np.array([[0, L - 1]], dtype=np.int32),
            mg_site=rng.integers(0, 2, L).astype(np.uint8),
            b_factor_z=rng.random(L).astype(np.float16),
            unknown_base=np.zeros(L, np.uint8),
            unobserved_seq_id=np.array([1, 2] if i == 1 else [], dtype=np.int32),
            n_polymer=L + (2 if i == 1 else 0), rigidity_valid=(i == 0)))
    with tempfile.TemporaryDirectory() as d:
        info = write_shard(Path(d) / "s.npz", ex)
        rd = ShardReader(Path(d) / "s.npz", [e.meta() for e in ex])
        chk("chain count survives", len(rd) == 3, str(len(rd)))
        ok = True
        for i, e in enumerate(ex):
            g = rd[i]
            ok &= (np.array_equal(g["tokens"], e.tokens)
                   and np.array_equal(g["mod_ids"], e.mod_ids)
                   and np.array_equal(g["mg_site"], e.mg_site)
                   and np.array_equal(g["contacts"], e.contacts)
                   and int(g["length"]) == e.length)
        chk("every array comes back identical", ok,
            "offsets are the thing that silently shifts labels")
        chk("per-chain unobserved lists stay with their chain",
            list(rd[1]["unobserved_seq_id"]) == [1, 2]
            and list(rd[0]["unobserved_seq_id"]) == [], "")

        print("\n== property 4: batching, masks and D12 ==")
        items = [rd[i] for i in range(3)]
        for it, e in zip(items, ex):
            it["meta"] = e.meta()
        b = pad_batch(items)
        L = max(e.length for e in ex)
        chk("padded to the longest chain", b["tokens"].shape == (3, L),
            str(b["tokens"].shape))
        chk("mask marks exactly the real residues",
            bool((b["mask"].sum(1) == [e.length for e in ex]).all()), "")
        chk("padding is not marked valid",
            all(not b["mask"][i, e.length:].any() for i, e in enumerate(ex)), "")
        chk("D12: rigidity_mask covers only the X-ray chain",
            int(b["rigidity_mask"].sum()) == ex[0].length,
            f"{int(b['rigidity_mask'].sum())} of {int(b['mask'].sum())} residues")
        chk("rigidity_mask never exceeds the residue mask",
            bool((b["rigidity_mask"] <= b["mask"]).all()), "")

    print("\n== property 4b: the shard is decompressed once, not per access ==")
    # np.load on an .npz returns a lazy ZIP handle whose __getitem__
    # decompresses the WHOLE named array every call. Indexing it per chain made
    # block-scorer training run at 4 s/step with the GPU at 0% -- 13 hours for
    # 8 epochs, all of it in zlib. This is the guard against it coming back.
    import time as _t
    with tempfile.TemporaryDirectory() as d:
        big = []
        r2 = np.random.default_rng(7)
        for i in range(64):
            L = 200
            big.append(ChainExample(
                pdb=f"b{i}", chain="A", length=L,
                tokens=r2.integers(0, 4, L).astype(np.int8),
                mod_ids=np.zeros(L, np.int16),
                chem=r2.random((L, 24)).astype(np.float16),
                contacts=np.array([[0, L - 1]], dtype=np.int32)))
        write_shard(Path(d) / "big.npz", big)
        rd2 = ShardReader(Path(d) / "big.npz")
        _ = rd2[0]                                  # warm
        t0 = _t.perf_counter()
        for i in range(len(rd2)):
            _ = rd2[i]
        per = (_t.perf_counter() - t0) / len(rd2) * 1e6
        chk("a warm chain read is microseconds, not milliseconds", per < 500,
            f"{per:.0f} us per chain over {len(rd2)} chains")
        chk("arrays are materialised, not a lazy NpzFile",
            hasattr(rd2, "_a") and isinstance(rd2._a.get("tokens"), np.ndarray),
            "ShardReader._a holds real arrays")

    print("\n== property 5: disorder validity ==")
    chk("a fully-modelled chain is a valid disorder observation",
        disorder_is_meaningful(500, 520), "")
    chk("a fragment of a long construct is not",
        not disorder_is_meaningful(604, 18998),
        "7ANE chain 2: 604 modelled of a declared 18,998")
    chk("short chains are always valid",
        disorder_is_meaningful(40, 150), "below the polymer floor")

    print("\n== property 6: the split is family-disjoint (D25) ==")
    mf = DATA / "manifest.json"
    if mf.exists():
        man = json.loads(mf.read_text())
        rows = [c for sh in man["shards"] for c in sh["chains"]]
        fam = {}
        for c in rows:
            if not c.get("rfam"):
                continue
            fam.setdefault(c["rfam"], set()).add(c["split"])
        leaks = {f: s for f, s in fam.items()
                 if "train" in s and (s & {"val", "test"})}
        chk("no Rfam family spans train and val/test", not leaks,
            f"{len(fam)} families; leaks: {list(leaks)[:3] or 'none'}")
        rib = {f for f, s in fam.items() if "test_ribosomal" in s}
        chk("test_ribosomal is entry-disjoint only, and says so",
            all("train" in fam[f] for f in rib) if rib else True,
            f"{len(rib)} families appear in it, all also in train -- by design")
        sp = man["split"]["by_residue"]
        # Not exact equality. The splitter balances by assigning whole CHAINS,
        # so a perfect tie is only ever available when the chain lengths happen
        # to sum that way -- the v3 corpus lands on 531,547 against 531,546 and
        # an equality test calls that a failure. The property worth holding is
        # that neither side is materially larger; 1% of the smaller side is far
        # tighter than any imbalance that could move a val/test comparison, and
        # is not a coin flip on the last chain.
        _lo = min(sp["val"], sp["test"])
        chk("val and test are balanced by residue (within 1%)",
            abs(sp["val"] - sp["test"]) <= max(1, 0.01 * _lo),
            f"{sp['val']:,} vs {sp['test']:,} "
            f"(delta {abs(sp['val'] - sp['test']):,} = "
            f"{abs(sp['val'] - sp['test']) / max(_lo, 1):.4%})")
    else:
        chk("dataset manifest present", 0, "run scripts/build_dataset.py")

    print("\n== property 7: RDAT parses both format versions ==")
    from pharos.data.rdat import parse_rdat, titration_examples
    rd_dir = ROOT / "data/benchmarks/rmdb/rdat"
    cases = [("MTTR9_MGTI_0001", "0.34, tab-separated"),
             ("SRPDIV_DMS_0001", "0.24, SPACE-separated")]
    for stem, label in cases:
        f = rd_dir / f"{stem}.rdat"
        if not f.exists():
            chk(f"{stem} available", 0, "run scripts/acquire_rmdb_titrations.py")
            continue
        r = parse_rdat(f)
        chk(f"{label}: parses to rows", r is not None and r.n_rows > 0,
            f"{r.n_rows if r else 0} rows, L={len(r.sequence) if r else 0}")
        ex = titration_examples(f)
        chk(f"{label}: yields titration examples", len(ex) > 0,
            f"{len(ex)} examples")
        if ex:
            chk(f"{label}: concentrations vary across them",
                len({e['conc_mM'] for e in ex}) >= 3,
                f"{len({e['conc_mM'] for e in ex})} distinct")
            chk(f"{label}: reactivity aligns to the sequence",
                all(len(e["reactivity"]) == len(e["sequence"]) for e in ex), "")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
