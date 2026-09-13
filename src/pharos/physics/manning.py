#!/usr/bin/env python3
"""Manning counterion condensation and Debye screening — closed form, no learned parameters.

Every quantity here follows from physical constants and the ionic condition, so
none of it needs training data. That is why the B_elec term survives the
ionic-metadata audit (only the *learned response* to varying conditions does not).

Cycle-1 defect #16 is encoded here: Manning's `b` is the AXIAL charge spacing,
not the P-P contour distance. For A-form RNA, 2.8 A per base pair carrying 2
charges gives b = 1.40 A -- NOT the 5.9-7.0 A contour figure. The widely quoted
theta ~ 0.76 is the B-DNA value (b = 1.70 A).
"""
from __future__ import annotations
import math
from dataclasses import dataclass

# physical constants (SI)
E_CHARGE = 1.602176634e-19      # C
EPS0 = 8.8541878128e-12         # F/m
KB = 1.380649e-23               # J/K
N_A = 6.02214076e23             # 1/mol

# axial charge spacing, metres. A-form RNA: 2.8 A rise per base pair, 2 charges.
B_AXIAL_A_RNA = 1.40e-10
B_AXIAL_B_DNA = 1.70e-10


def water_permittivity(T: float = 298.15) -> float:
    """Relative permittivity of water. Empirical fit, valid ~273-373 K."""
    t = T - 273.15
    return 87.740 - 0.40008 * t + 9.398e-4 * t**2 - 1.410e-6 * t**3


def bjerrum_length(T: float = 298.15, eps_r: float | None = None) -> float:
    """l_B = e^2 / (4 pi eps0 eps_r kB T), metres. ~7.15 A in water at 298 K."""
    if eps_r is None:
        eps_r = water_permittivity(T)
    return E_CHARGE**2 / (4 * math.pi * EPS0 * eps_r * KB * T)


def manning_xi(T: float = 298.15, b: float = B_AXIAL_A_RNA) -> float:
    """Manning parameter xi = l_B / b."""
    return bjerrum_length(T) / b


def condensed_fraction(z: int = 1, T: float = 298.15, b: float = B_AXIAL_A_RNA) -> float:
    """theta = 1 - 1/(z xi); 0 below the condensation threshold (xi < 1/z)."""
    xi = manning_xi(T, b)
    return max(0.0, 1.0 - 1.0 / (z * xi))


def effective_charge(z: int = 1, T: float = 298.15, b: float = B_AXIAL_A_RNA) -> float:
    """Renormalised phosphate charge, q_eff = -(1 - theta)."""
    return -(1.0 - condensed_fraction(z, T, b))


def ionic_strength(conc_mM: dict[str, float]) -> float:
    """I = 1/2 sum c_i z_i^2, mol/m^3. Keys like 'Mg2+', 'K+', 'Na+', 'Cl-'."""
    Z = {"Mg2+": 2, "Ca2+": 2, "Mn2+": 2, "K+": 1, "Na+": 1, "Li+": 1, "Cl-": -1}
    tot = 0.0
    for ion, c in conc_mM.items():
        z = Z.get(ion)
        if z is None:
            raise KeyError(f"unknown ion {ion!r}; add its charge to Z")
        tot += (c * 1e-3 * 1e3) * z * z          # mM -> mol/m^3
    # counter-anions for the salts, so the solution is electroneutral
    cations = sum((c * 1e-3 * 1e3) * Z[i] for i, c in conc_mM.items() if Z[i] > 0)
    tot += cations * 1.0
    return 0.5 * tot


def debye_kappa(conc_mM: dict[str, float], T: float = 298.15) -> float:
    """Inverse Debye length, 1/m. kappa = sqrt(2 N_A e^2 I / (eps eps0 kB T))."""
    I = ionic_strength(conc_mM)
    eps_r = water_permittivity(T)
    return math.sqrt(2 * N_A * E_CHARGE**2 * I / (eps_r * EPS0 * KB * T))


def debye_length_A(conc_mM: dict[str, float], T: float = 298.15) -> float:
    """Debye screening length in angstrom."""
    return 1e10 / debye_kappa(conc_mM, T)


@dataclass
class IonicCondition:
    """The c_ion input vector. Defaults are physiological-ish."""
    mg_mM: float = 5.0
    k_mM: float = 100.0
    na_mM: float = 0.0
    T: float = 298.15
    pH: float = 7.0

    def as_dict(self) -> dict[str, float]:
        d = {}
        if self.mg_mM: d["Mg2+"] = self.mg_mM
        if self.k_mM:  d["K+"] = self.k_mM
        if self.na_mM: d["Na+"] = self.na_mM
        return d

    def screening(self) -> dict[str, float]:
        """Everything B_elec needs, computed once per example."""
        d = self.as_dict()
        return {"l_B_A": bjerrum_length(self.T) * 1e10,
                "xi": manning_xi(self.T),
                "theta": condensed_fraction(1, self.T),
                "q_eff": effective_charge(1, self.T),
                "kappa_inv_A": debye_length_A(d, self.T) if d else float("inf")}


def b_elec(d_ij_A, cond: IonicCondition, lam: float = 1.0):
    """Screened-Coulomb attention bias.

    B = -lam * q_eff^2 * l_B * exp(-kappa d) / d, with d in angstrom.
    NOTE: needs a distance estimate, so it is disabled at recycle 0.
    """
    s = cond.screening()
    kappa_inv = s["kappa_inv_A"]
    try:                      # accept torch tensors or floats
        import torch
        if torch.is_tensor(d_ij_A):
            d = d_ij_A.clamp(min=1e-3)
            return -lam * s["q_eff"]**2 * s["l_B_A"] * torch.exp(-d / kappa_inv) / d
    except ImportError:
        pass
    d = max(float(d_ij_A), 1e-3)
    return -lam * s["q_eff"]**2 * s["l_B_A"] * math.exp(-d / kappa_inv) / d


if __name__ == "__main__":
    print(f"{'quantity':28s}{'A-RNA':>12s}{'B-DNA':>12s}")
    print(f"{'axial charge spacing b (A)':28s}{B_AXIAL_A_RNA*1e10:12.2f}{B_AXIAL_B_DNA*1e10:12.2f}")
    print(f"{'Bjerrum length l_B (A)':28s}{bjerrum_length()*1e10:12.2f}{bjerrum_length()*1e10:12.2f}")
    print(f"{'Manning xi':28s}{manning_xi():12.2f}{manning_xi(b=B_AXIAL_B_DNA):12.2f}")
    print(f"{'condensed fraction theta':28s}{condensed_fraction():12.3f}"
          f"{condensed_fraction(b=B_AXIAL_B_DNA):12.3f}")
    print(f"{'q_eff per phosphate':28s}{effective_charge():12.3f}"
          f"{effective_charge(b=B_AXIAL_B_DNA):12.3f}")
    print()
    for name, c in [("0 mM Mg, 100 mM K", IonicCondition(mg_mM=0, k_mM=100)),
                    ("5 mM Mg, 100 mM K", IonicCondition(mg_mM=5, k_mM=100)),
                    ("15 mM Mg, 150 mM K", IonicCondition(mg_mM=15, k_mM=150))]:
        s = c.screening()
        print(f"  {name:20s} Debye length {s['kappa_inv_A']:6.2f} A   "
              f"B_elec at 10 A = {b_elec(10.0, c):8.4f}")
