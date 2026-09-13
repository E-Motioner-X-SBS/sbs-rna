#!/usr/bin/env python3
"""Lock the physics constants so defect #16 cannot reappear."""
import sys, math
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from manning import (bjerrum_length, manning_xi, condensed_fraction, effective_charge,
                     debye_length_A, IonicCondition, b_elec,
                     B_AXIAL_A_RNA, B_AXIAL_B_DNA)

F = []
def chk(name, got, want, tol=5e-3):
    ok = abs(got - want) <= tol * max(abs(want), 1e-9)
    print(f"  {'OK  ' if ok else 'FAIL'} {name:38s} {got:10.4f}  want {want}")
    if not ok: F.append(name)

chk("Bjerrum length, A, 298 K", bjerrum_length() * 1e10, 7.15, 2e-2)
chk("A-RNA Manning xi", manning_xi(), 5.11, 1e-2)
chk("A-RNA theta", condensed_fraction(), 0.804, 5e-3)
chk("A-RNA q_eff", effective_charge(), -0.196, 2e-2)
chk("B-DNA theta (the 0.76 figure)", condensed_fraction(b=B_AXIAL_B_DNA), 0.762, 5e-3)
# the whole point of the ionic input: more salt -> shorter screening
d0 = debye_length_A(IonicCondition(mg_mM=0, k_mM=100).as_dict())
d1 = debye_length_A(IonicCondition(mg_mM=15, k_mM=150).as_dict())
print(f"  {'OK  ' if d1 < d0 else 'FAIL'} screening shortens as salt rises      "
      f"{d0:.2f} -> {d1:.2f} A")
if d1 >= d0: F.append("screening monotonic")
b0 = b_elec(10.0, IonicCondition(mg_mM=0, k_mM=100))
b1 = b_elec(10.0, IonicCondition(mg_mM=15, k_mM=150))
print(f"  {'OK  ' if abs(b1) < abs(b0) else 'FAIL'} B_elec weakens as salt rises          "
      f"{b0:.4f} -> {b1:.4f}")
if abs(b1) >= abs(b0): F.append("B_elec monotonic")
print("\n" + ("ALL TESTS PASS" if not F else f"{len(F)} FAILURES: {F}"))
sys.exit(1 if F else 0)
