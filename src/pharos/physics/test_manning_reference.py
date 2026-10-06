#!/usr/bin/env python3
"""§6.2's closed form, checked against an INDEPENDENT implementation.

`test_manning.py` checks our Manning constants against numbers we derived
ourselves. That catches arithmetic slips and nothing else: a wrong formula
reproduces its own wrong value perfectly. This module checks them against
`md_rnaions` (Ryan Hayes, Rice/Michigan) -- the reference code behind the
generalized Manning condensation work -- which was written independently,
in C, from a different empirical fit for the permittivity of water.

    Hayes, src/parms.c:
        eps = exp(5.71455988 - 0.004540698*T)
        l_B = 16710.7 / (eps * T)                      nm
        kappa = sqrt(4*pi*(c_K + c_Cl)*l_B)            nm^-1

    pharos.physics.manning:
        eps = Malmberg-Maryott cubic in (T - 273.15)
        l_B = e^2 / (4*pi*eps0*eps_r*kB*T)
        kappa^2 = 2*N_A*e^2*I/(eps_r*eps0*kB*T),  I = 1/2 sum c_i z_i^2

Nothing is shared between them but the physics, so agreement is evidence and
disagreement is a bug in one of them. The archive is kept at
`research/prior-art/md_rnaions/` and is gitignored; this module hardcodes the
two formulas above so the test stands on its own.

Run: python3 src/pharos/physics/test_manning_reference.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pharos.physics.manning import (IonicCondition,  # noqa: E402
                                    bjerrum_length, condensed_fraction,
                                    debye_length_A, effective_charge,
                                    manning_xi, water_permittivity)

fails: list[str] = []
MOLAR = 0.6022140857          # particles per nm^3 in a 1 M solution


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:52s} {detail}")
    if not ok:
        fails.append(name)


def hayes_eps(T: float) -> float:
    return math.exp(5.71455988 - 0.004540698 * T)


def hayes_lB_nm(T: float) -> float:
    return 16710.7 / (hayes_eps(T) * T)


def hayes_debye_A(c_K_M: float, c_Cl_M: float, T: float) -> float:
    kappa = math.sqrt(4 * math.pi * (c_K_M + c_Cl_M) * MOLAR * hayes_lB_nm(T))
    return 10.0 / kappa


def main() -> int:
    T = 298.15
    print("== the permittivity of water, two different empirical fits ==")
    he, pe = hayes_eps(T), water_permittivity(T)
    chk("they agree to better than 1e-3 relative",
        abs(he - pe) / pe < 1e-3, f"{he:.5f} vs {pe:.5f}  "
        f"rel {abs(he - pe) / pe:.3e}")

    print("\n== the Bjerrum length ==")
    h, p = hayes_lB_nm(T) * 10, bjerrum_length(T) * 1e10
    chk("l_B agrees to better than 1e-3 relative",
        abs(h - p) / p < 1e-3, f"{h:.5f} A vs {p:.5f} A  "
        f"rel {abs(h - p) / p:.3e}")

    print("\n== the Debye length, MONOVALENT salt ==")
    # Hayes's kappa sums the NUMBER densities of the two monovalent species;
    # ours sums 1/2 c z^2 and doubles it. For 1:1 salt these are the same
    # quantity written two ways, and if they were not, one of the two
    # screening lengths would be wrong by a constant factor at every
    # concentration -- which is exactly the kind of error that survives
    # because the shape of the curve still looks right.
    for c_mM in (25.0, 50.0, 100.0, 150.0, 300.0):
        c = c_mM / 1000.0
        hd = hayes_debye_A(c, c, T)
        pd = debye_length_A({"K+": c_mM})
        chk(f"lambda_D at {c_mM:5.0f} mM KCl", abs(hd - pd) / pd < 1e-3,
            f"{hd:8.4f} A vs {pd:8.4f} A  rel {abs(hd - pd) / pd:.3e}")

    print("\n== the Debye length with Mg2+: the two models DIVERGE, by design ==")
    # Hayes runs Mg as EXPLICIT particles, so they must not also appear in the
    # implicit screening -- his c_Cl rises by 2*c_Mg for electroneutrality and
    # the Mg itself is left out of kappa. PHAROS has no explicit ions, so Mg
    # belongs in the ionic strength with z^2 = 4.
    #
    # This is a difference of model class, not an error in either, and it is
    # asserted rather than left to be rediscovered: anyone comparing PHAROS's
    # ionic response against a number from that paper is comparing two
    # different definitions of the screening length.
    c_K, c_Mg = 0.050, 0.015
    hd = hayes_debye_A(c_K, c_K + 2 * c_Mg, T)
    pd = debye_length_A({"K+": c_K * 1000, "Mg2+": c_Mg * 1000})
    chk("they differ by more than 15% at 15 mM Mg",
        abs(hd - pd) / pd > 0.15,
        f"explicit-Mg {hd:.4f} A vs implicit-Mg {pd:.4f} A  "
        f"rel {abs(hd - pd) / pd:.1%}")
    chk("and PHAROS's is the SHORTER, as implicit divalents must be",
        pd < hd, "z^2 = 4 screens harder than the K+/Cl- alone")

    print("\n== Mg2+ moves screening but NOT condensation ==")
    # The gap the reference exposes. `IonicCondition.screening()` calls
    # `condensed_fraction(1, T)` and `effective_charge(1, T)` with the valence
    # PINNED AT ONE, whatever is in the solution. So the whole response to
    # Mg2+ runs through the Debye length, and the channel that is left out is
    # the larger of the two: B_elec goes as q_eff^2.
    s0 = IonicCondition(mg_mM=0.0, k_mM=100.0).screening()
    s1 = IonicCondition(mg_mM=15.0, k_mM=100.0).screening()
    chk("theta is bit-identical at 0 and 15 mM Mg",
        s0["theta"] == s1["theta"], f"{s0['theta']:.6f} == {s1['theta']:.6f}")
    chk("q_eff is bit-identical too",
        s0["q_eff"] == s1["q_eff"], f"{s0['q_eff']:+.6f}")
    chk("while the screening length does move",
        abs(s1["kappa_inv_A"] - s0["kappa_inv_A"]) > 1.0,
        f"{s0['kappa_inv_A']:.4f} -> {s1['kappa_inv_A']:.4f} A")
    # and the size of what is being left out
    th1, th2 = condensed_fraction(1), condensed_fraction(2)
    q1, q2 = effective_charge(1), effective_charge(2)
    chk("a divalent counterion condenses MORE, per Manning",
        th2 > th1, f"theta {th1:.4f} (z=1) -> {th2:.4f} (z=2)")
    ratio = (q2 / q1) ** 2
    chk("and B_elec goes as q_eff^2, so the omitted channel is ~4x",
        ratio < 0.3, f"q_eff {q1:+.4f} -> {q2:+.4f}, "
        f"B_elec ratio {ratio:.3f}x against the {s0['kappa_inv_A']/s1['kappa_inv_A']:.3f}x "
        f"the screening term alone gives at fixed distance")

    print("\n== Manning's bound is the range any learned theta must stay in ==")
    xi = manning_xi()
    chk("theta(z=1) = 1 - 1/xi", abs(th1 - (1 - 1 / xi)) < 1e-12,
        f"{th1:.8f} vs {1 - 1 / xi:.8f}")
    chk("theta(z=2) = 1 - 1/(2 xi)", abs(th2 - (1 - 1 / (2 * xi))) < 1e-12,
        f"{th2:.8f} vs {1 - 1 / (2 * xi):.8f}")
    chk("so [theta(z=1), theta(z=2)] is the physical interval",
        0.0 < th1 < th2 < 1.0, f"[{th1:.5f}, {th2:.5f}]")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
