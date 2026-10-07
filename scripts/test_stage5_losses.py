#!/usr/bin/env python3
"""`step_losses` and `evaluate`, exercised on synthetic data that is real geometry.

Stage 5's two central functions had no test. Everything around them did --
the heads, the dataset, the metric helpers, the checkpoint -- but the
functions that assemble eleven losses and reduce a split to a dict were
reached only by running the stage on a GPU for hours, which is why two
defects in head 11's wiring survived review and were found by running it:

* `tv = tvalid & mask.unsqueeze(-1)` named a variable that does not exist in
  `step_losses` (it is `m`). A `NameError` on the first batch.
* the evaluate half had never executed at all.

A synthetic batch is enough for both, provided the geometry is REAL: an
ideal A-form helix built from its published parameters (32.7 deg twist,
2.81 A rise), so the torsion targets are a tight cluster rather than noise
and a wrong sign or a wrong atom triple shows up as a wrong angle instead of
as plausible garbage.

Run: python3 scripts/test_stage5_losses.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pharos.data.torsions import (TORSION_NAMES,  # noqa: E402
                                  pseudotorsions_torch)
from pharos.model.pharos import Pharos, PharosConfig          # noqa: E402
import train_pharos as TP                                      # noqa: E402

fails: list[str] = []
B, L = 2, 36


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} {detail}")
    if not ok:
        fails.append(name)


def helix(offset: float = 0.0) -> np.ndarray:
    """Ideal A-form RNA: 32.7 deg of twist and 2.81 A of rise per residue."""
    tw, rise = np.radians(32.7), 2.81
    co = np.zeros((L, 3, 3))
    for i in range(L):
        t = i * tw + offset
        for k, (r, dz, dt) in enumerate([(8.8, 0.0, 0.0),      # P
                                         (9.5, 1.0, 0.33),     # C4'
                                         (4.5, 1.6, 0.75)]):   # N
            co[i, k] = [r * np.cos(t + dt), r * np.sin(t + dt), i * rise + dz]
    return co


def make_batch(gap: bool = False) -> dict:
    co = np.stack([helix(0.0), helix(0.7)])
    cm = torch.ones(B, L, dtype=torch.bool)
    if gap:
        cm[1, 17] = False
    return {
        "tokens": torch.randint(1, 5, (B, L)),
        "mod_ids": torch.zeros(B, L, dtype=torch.long),
        "chem": torch.zeros(B, L, 24),
        "mask": torch.ones(B, L, dtype=torch.bool),
        "lengths": torch.full((B,), L),
        "coords": torch.tensor(co, dtype=torch.float32),
        "coord_residue_mask": cm,
        "coord_mask": torch.ones(B, L, 3, dtype=torch.bool),
        "contacts": [torch.tensor([[2, 9], [5, 20]]) for _ in range(B)],
        "mg_site": torch.zeros(B, L), "b_factor_z": torch.zeros(B, L),
        "rigidity_mask": torch.zeros(B, L, dtype=torch.bool),
        "base_mask": torch.zeros(B, L, dtype=torch.bool),
        "weights": torch.ones(B), "in_complex": torch.zeros(B),
        "meta": [{"has_protein": False} for _ in range(B)],
    }


class _FakeDS:
    """Three identical batches. `evaluate` only needs `iter_batches`."""

    def iter_batches(self, token_budget: int = 8192, shuffle: bool = False):
        for _ in range(3):
            yield make_batch()


def main() -> int:
    cfg = PharosConfig.mini()
    cfg.n_blocks, cfg.n_experts, cfg.d_expert, cfg.n_loops = 2, 8, 64, 2
    model = Pharos(cfg).eval()
    t = make_batch(gap=True)

    print("== the torsion targets come out of real geometry ==")
    ang, valid = pseudotorsions_torch(t["coords"], t["coord_residue_mask"])
    chk("targets exist for the batch", bool(valid.any()),
        f"{int(valid.sum())} of {valid.numel()} defined")
    chk("an unresolved residue removes eta at i-1, i and i+1",
        not valid[1, 16, 0] and not valid[1, 17, 0] and not valid[1, 18, 0])
    e = ang[0, valid[0, :, 0], 0].rad2deg() % 360
    chk("eta along an ideal helix is one tight cluster", float(e.std()) < 1.0,
        f"mean {float(e.mean()):.2f} sd {float(e.std()):.4f} deg -- a wrong "
        f"atom triple or a wrong sign would not cluster")

    print("\n== step_losses: every head this batch can supervise ==")
    total, parts, out = TP.step_losses(model, t, cfg, n_neg=8, n_loops=2)
    chk("it returns a finite total", torch.isfinite(total).all(),
        f"{float(total):.5f}")
    chk("`torsion` is among the reported parts", "torsion" in parts,
        f"{parts.get('torsion', float('nan')):.5f}")
    for nm in TORSION_NAMES:
        chk(f"{nm}: error, FLOOR and lift all reported",
            all(f"tors_{nm}_{x}" in parts for x in ("mae", "base", "lift")),
            f"mae {parts.get(f'tors_{nm}_mae', float('nan')):7.2f}  "
            f"floor {parts.get(f'tors_{nm}_base', float('nan')):7.2f} deg")
    # `mg_auroc` is NaN here and that is CORRECT: this batch has no positive
    # Mg label, and an AUROC over one class is undefined. `_binary_metrics`
    # returns NaN rather than fabricating 0.5, which is the honest answer and
    # the one worth pinning -- a metric that invents a plausible number for a
    # degenerate input is exactly what this register keeps finding. The run
    # log writes it as an EMPTY FIELD, never the string "nan", which is why
    # the training watch's non-finite check does not fire on it in
    # production; checked against 56 real rows of stage5_3d.v1.csv.
    undefined = {"mg_auroc"}
    nonfinite = {k for k, v in parts.items()
                 if isinstance(v, (int, float)) and not np.isfinite(v)}
    chk("every part is finite except the ones that are undefined here",
        nonfinite <= undefined, f"{len(parts)} parts, non-finite: "
        f"{sorted(nonfinite) or 'none'}")
    chk("and an AUROC with no positive label is NaN, not a fabricated 0.5",
        "mg_auroc" in nonfinite and parts.get("mg_pos_rate") == 0.0,
        "one class present, so the rank statistic has no value")
    chk("the contact head fired too, so this is not a torsion-only batch",
        "contact" in parts, f"contact {parts.get('contact', float('nan')):.4f}")

    print("\n== and the gradient reaches head 11 ==")
    model.train()
    total, parts, out = TP.step_losses(model, t, cfg, n_neg=8, n_loops=2)
    total.backward()
    tg = {n: p for n, p in model.named_parameters() if "residue.torsion" in n}
    nz = {n: float(p.grad.abs().max()) for n, p in tg.items()
          if p.grad is not None}
    chk("every tensor of the torsion head has a gradient",
        len(nz) == len(tg) == 6, f"{len(nz)} of {len(tg)}")
    chk("and none of them is zero", nz and all(v > 0 for v in nz.values()),
        f"min |g| {min(nz.values()):.3e}" if nz else "no grads")
    model.zero_grad(set_to_none=True)
    model.eval()

    print("\n== the angular metric is calibrated, not merely present ==")
    # A metric that returns a plausible number for every input is the thing
    # this register keeps finding. Pin both ends of its range.
    tgt = torch.stack([ang.sin(), ang.cos()], -1)
    d = ((tgt - tgt) ** 2).sum(-1)
    chk("a perfect prediction scores exactly 0", float(d.max()) == 0.0)
    flip = torch.stack([(ang + np.pi).sin(), (ang + np.pi).cos()], -1)
    dot = (flip * tgt).sum(-1).clamp(-1, 1)
    err = dot.arccos().rad2deg()[valid]
    chk("a 180-degree flip scores 180", abs(float(err.mean()) - 180.0) < 1e-3,
        f"{float(err.mean()):.4f} deg")

    print("\n== evaluate(): the half that had never executed ==")
    res = TP.evaluate(model, _FakeDS(), torch.device("cpu"), cfg, max_batches=3)
    chk("it returns without raising", isinstance(res, dict), f"{len(res)} keys")
    for nm in TORSION_NAMES:
        chk(f"{nm}: pooled mae / floor / lift / n",
            all(f"tors_{nm}_{x}" in res for x in ("mae", "base", "lift", "n")),
            f"mae {res.get(f'tors_{nm}_mae')} floor {res.get(f'tors_{nm}_base')} "
            f"n {res.get(f'tors_{nm}_n')}")
    chk("errors are degrees in [0, 180]",
        all(0 <= res[f"tors_{n}_mae"] <= 180 for n in TORSION_NAMES))
    chk("lift is exactly floor - mae",
        all(abs(res[f"tors_{n}_lift"]
                - (res[f"tors_{n}_base"] - res[f"tors_{n}_mae"])) < 1e-3
            for n in TORSION_NAMES))
    # POOLED, not averaged per batch -- the correction `evaluate`'s own
    # docstring records for the correlations, applied to this head.
    chk("n is pooled across all three batches, not averaged",
        res["tors_theta_n"] == 3 * B * (L - 1),
        f"{res['tors_theta_n']} == 3 x {B} x {L - 1}")
    chk("eta is defined on two fewer residues per chain than theta",
        res["tors_eta_n"] == 3 * B * (L - 2),
        f"{res['tors_eta_n']} vs {res['tors_theta_n']}")

    print("\n== the run log declares everything these two produce ==")
    # A metric the loop computes and the csv has no column for is a metric
    # nobody reads. `_stage5_fields` is derived, so this checks the derivation.
    declared = set(TP.STAGE5_FIELDS)
    undeclared = [k for k in parts if f"part_{k}" not in declared]
    chk("every key step_losses returns has a part_ column",
        not undeclared, f"undeclared: {undeclared or 'none'}")
    und_val = [k for k in res if f"val_{k}" not in declared]
    chk("every key evaluate returns has a val_ column",
        not und_val, f"undeclared: {und_val or 'none'}")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
