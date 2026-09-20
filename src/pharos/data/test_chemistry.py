#!/usr/bin/env python3
"""Tests for the 24-dim chemistry vector.

The properties worth pinning are not "does it return an array". They are the
ones whose violation would be invisible in training and fatal in evaluation:

  1. THE LEAK BOUNDARY   -- dim 13 (shifted pKa) is set from structural context
                            and must default to 0. Everything else must be
                            derivable from sequence alone. Feeding a structural
                            observable in as a feature leaks the answer, and the
                            failure mode is a model that evaluates well and
                            predicts nothing.
  2. PARENT RESOLUTION   -- comes from the PDB Chemical Component Dictionary,
                            not from a hand-written list of comp_ids. A2M -> A
                            is the test case: nothing in the string says so.
  3. MODIFIED != N       -- a modified G must still carry G's chemistry and
                            count as G in the GC window. Mapping modifications
                            to N is what VOCAB.md says the design must stop
                            doing, and it is 0.565% of the archive.
  4. HONEST DEGRADATION  -- an unknown component returns MOD / parent N /
                            class other, not a crash and not a guess.
  5. AGREEMENT WITH THE SPEC -- the static table reproduces CHEMISTRY.md's
                            published values exactly, so a silent edit to one
                            fails the build.

Run: python3 src/pharos/data/test_chemistry.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chemistry import (BASES, GC_WINDOW, N_DIMS, chain_chemistry,  # noqa: E402
                       residue_chemistry, resolve, _ccd_table)

ANALYSIS = Path(__file__).resolve().parents[3] / "data/samples/analysis"
fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:52s} {detail}")
    if not ok:
        fails.append(name)


def main() -> int:
    tbl = _ccd_table()

    print("== property 1: shape, dtype, finiteness ==")
    v = chain_chemistry(["A", "C", "G", "U"])
    chk("chain_chemistry returns (L, 24)", v.shape == (4, N_DIMS), str(v.shape))
    chk("dtype is float32", v.dtype == np.float32, str(v.dtype))
    chk("no NaN or inf", bool(np.isfinite(v).all()), "all finite")
    chk("empty chain returns (0, 24)", chain_chemistry([]).shape == (0, N_DIMS), "")

    print("\n== property 2: the leak boundary (dim 13) ==")
    chk("shifted-pKa defaults to 0 for every base",
        all(residue_chemistry(b)[13] == 0.0 for b in BASES),
        "structural context may only raise it on recycles >= 1")
    chk("chain_chemistry never sets dim 13",
        float(chain_chemistry(["A", "PSU", "G", "OMG", "U"])[:, 13].max()) == 0.0,
        "the chain-level entry point has no way to set it")
    chk("it can be set explicitly, per residue",
        residue_chemistry("A", shifted_pka=True)[13] == 1.0,
        "recycle >= 1 path exists")

    print("\n== property 3: the static table matches CHEMISTRY.md ==")
    # Values quoted directly from the spec tables; a silent edit to either side
    # breaks this.
    spec = {
        #      purine, wcD, wcA, hgD, hgA, sgD, sgA, pKa
        "A": (1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 3.5),
        "C": (0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 4.2),
        "G": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.2),
        "U": (0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 9.2),
    }
    for b, want in spec.items():
        got = residue_chemistry(b)[5:13]
        # float32, so compare with a tolerance: 9.2 stores as 9.19999980926...
        chk(f"{b}: purine/H-bond edges/pKa", np.allclose(got, want, atol=1e-6),
            ", ".join(f"{x:g}" for x in got))
    stack = {"A": (0.82, 0.85), "C": (0.61, 0.63), "G": (1.00, 1.00), "U": (0.58, 0.55)}
    chk("stacking priors ordered G > A > C > U",
        all(abs(residue_chemistry(b)[16] - s[0]) < 1e-6
            and abs(residue_chemistry(b)[17] - s[1]) < 1e-6
            for b, s in stack.items()),
        "polarisability and stacking-energy, normalised to G = 1")
    chk("pucker propensities sum to 1",
        all(abs(residue_chemistry(b)[14] + residue_chemistry(b)[15] - 1.0) < 1e-6
            for b in BASES), "C3'-endo + C2'-endo")
    chk("phosphate charge is -1 mid-chain, 0 at the 5' terminus",
        residue_chemistry("A")[18] == -1.0
        and residue_chemistry("A", five_prime_terminus=True)[18] == 0.0,
        "dim 18 is what Manning condensation acts on")

    print("\n== property 4: one-hot discipline ==")
    for b in BASES:
        r = residue_chemistry(b)
        chk(f"{b} sets exactly one identity bit and not MOD",
            float(r[:5].sum()) == 1.0 and r[4] == 0.0, f"{r[:5].astype(int)}")
    chk("2'-OH present on ribo, absent on deoxy",
        residue_chemistry("A")[19] == 1.0 and residue_chemistry("A", deoxy=True)[19] == 0.0,
        "dim 19 distinguishes hybrid-duplex DNA")
    chk("deoxy flips the pucker south",
        residue_chemistry("A", deoxy=True)[15] > residue_chemistry("A", deoxy=True)[14],
        "deoxyribose is C2'-endo")

    print("\n== property 5: modified residues keep their parent's chemistry ==")
    if tbl:
        # A2M is 2'-O-methyladenosine. Nothing in the string says "adenosine";
        # only the CCD does. This is the test that the resolution is not a
        # name-pattern guess.
        p, k, mod = resolve("A2M")
        chk("A2M resolves to parent A via the CCD", p == "A" and mod,
            f"parent={p} class={k}")
        chk("A2M carries A's H-bond chemistry",
            np.allclose(residue_chemistry("A2M")[5:13], residue_chemistry("A")[5:13]),
            "a modified A is still an A to the physics")
        chk("A2M sets both the A bit and MOD",
            residue_chemistry("A2M")[0] == 1.0 and residue_chemistry("A2M")[4] == 1.0,
            "identity is not discarded")
        chk("PSU resolves to U, class pseudouridine",
            resolve("PSU")[:2] == ("U", "pseudouridine"), "the commonest modification")
        chk("PSU gains the N1-H Hoogsteen donor",
            residue_chemistry("PSU")[8] == residue_chemistry("U")[8] + 1.0,
            "it changes base-pairing capacity, which is why it has its own dim")
        chk("OMG resolves to G, class methylation",
            resolve("OMG")[:2] == ("G", "methylation"), "")
        for cls, dim in (("methylation", 20), ("pseudouridine", 21)):
            cid = next((c for c, r in tbl.items() if r.get("class") == cls), None)
            if cid:
                chk(f"class {cls} sets dim {dim}",
                    residue_chemistry(cid)[dim] == 1.0, cid)
        chk("exactly one class dim fires per modification",
            all(float(residue_chemistry(c)[20:23].sum()) == 1.0
                for c in list(tbl)[:50]), "first 50 species")
    else:
        chk("CCD table present", 0, "run scripts/sampling/resolve_ccd_parents.py")

    print("\n== property 6: honest degradation ==")
    p, k, mod = resolve("ZZZ")
    chk("an unknown component does not crash", True, f"parent={p} class={k}")
    chk("it reports MOD, parent N, class other", p == "N" and k == "other" and mod,
        "not a guess and not a silent ACGU")
    chk("its chemistry is the base mean, not zeros",
        float(residue_chemistry("ZZZ")[5:20].sum()) != 0.0, "stated fallback")

    print("\n== property 7: local GC counts parents, not literals ==")
    # A run of modified G must read as GC-rich. If modifications were mapped to
    # N this window would read 0.
    v = chain_chemistry(["OMG"] * 8) if tbl else None
    if v is not None:
        chk("a run of modified G reads GC = 1.0", abs(float(v[0, 23]) - 1.0) < 1e-6,
            f"{v[0, 23]:.3f}")
    v = chain_chemistry(["A"] * 8)
    chk("a run of A reads GC = 0.0", float(v[:, 23].max()) == 0.0, "")
    seq = ["G"] * 10 + ["A"] * 10
    v = chain_chemistry(seq, gc_window=2)
    chk("the window is local, not global",
        v[0, 23] == 1.0 and v[-1, 23] == 0.0, f"ends {v[0,23]:.2f} / {v[-1,23]:.2f}")
    chk("the window narrows at chain ends rather than padding",
        abs(float(chain_chemistry(["G", "A"], gc_window=GC_WINDOW)[0, 23]) - 0.5) < 1e-6,
        "a 2-nt chain sees both residues, not 33 slots")
    chk("unknown parents are excluded from the GC denominator",
        abs(float(chain_chemistry(["G", "ZZZ"], gc_window=8)[0, 23]) - 1.0) < 1e-6,
        "1 of 1 known, not 1 of 2")

    print("\n== property 8: deoxy_mask is validated, not silently ignored ==")
    try:
        chain_chemistry(["A", "C", "G"], deoxy_mask=[True])
        chk("a mismatched deoxy_mask raises", 0, "it was accepted")
    except ValueError:
        chk("a mismatched deoxy_mask raises", 1, "ValueError")

    print("\n== property 9: the class totals match the measured census ==")
    f = ANALYSIS / "ccd_parents.json"
    if f.exists():
        r = json.loads(f.read_text())
        by = r["by_class_residues"]
        chk("no residue is left unclassified", "unknown" not in by,
            f"classes: {sorted(by)}")
        chk("pseudouridine is the single commonest modification",
            by.get("pseudouridine", 0) > 0.2 * sum(by.values()),
            f"{by.get('pseudouridine', 0):,} of {sum(by.values()):,}")
        chk("most modified residues resolve to a standard parent",
            r["frac_residues_with_parent"] > 0.85,
            f"{100*r['frac_residues_with_parent']:.1f}%")
    else:
        chk("ccd_parents.json present", 0, "run resolve_ccd_parents.py")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
