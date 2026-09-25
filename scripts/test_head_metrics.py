#!/usr/bin/env python3
"""The head metrics have their own floors checked, because they are the floors.

Every head in `step_losses` reported a raw loss and nothing else, and a raw
loss is not a measurement when the target is imbalanced or z-scored: a
weighted BCE on a 4.7%-positive Mg target falls when the head learns the prior,
a smooth-L1 on a z-scored B-factor is already respectable for the constant
zero, and a 13-class cross-entropy where one class is 76.6% of the data looks
excellent for a head that has learned exactly nothing. So each head got a
metric with a floor -- and adding an unchecked metric to fix unchecked metrics
would be the same mistake one level up. These are the checks.

  AUROC        0.5 for any constant or random score, 1.0 for a perfect one,
               0.0 for an inverted one, and NaN rather than a lie when a class
               is absent from the batch.
  correlation  exactly 0 for any constant prediction, however well tuned.
  class lift   exactly 0 for a head that always predicts the majority, and
               macro recall 1/k for the same head.

Run: python3 scripts/test_head_metrics.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("tp", ROOT / "scripts/train_pharos.py")
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)

fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} {detail}")
    if not ok:
        fails.append(name)


def check_help_strings() -> int:
    """`--help` must work on every trainer.

    argparse `%`-formats help strings, so a literal `%` raises
    `TypeError: not enough arguments for format string` -- and only when
    `--help` is actually run, which nobody does on a training script. Six bare
    `%` accumulated in `pretrain_mlm.py` and two in `eval_mlm_checkpoint.py`
    across one day of adding measurements to help text, and `--help` was broken
    the whole time without a single run noticing.
    """
    import subprocess
    root = Path(__file__).resolve().parents[1]
    py = "/store/shuvam/.venv/bin/python"
    bad = []
    for name in ("pretrain_mlm.py", "train_pharos.py",
                 "train_sequence_stages.py", "eval_mlm_checkpoint.py",
                 "train_block_scorer.py", "audit_router.py"):
        f = root / "scripts" / name
        if not f.exists():
            continue
        r = subprocess.run([py, str(f), "--help"], capture_output=True,
                           text=True, timeout=300)
        ok = r.returncode == 0
        print(f"  {'OK  ' if ok else 'FAIL'} {name:34s} --help exits "
              f"{r.returncode}")
        if not ok:
            bad.append(name)
            print("       " + r.stderr.strip().splitlines()[-1][:110])
    return len(bad)


def main() -> int:
    torch.manual_seed(0)

    print("== AUROC has a floor at 0.5 and does not move with imbalance ==")
    y = (torch.rand(20000) < 0.047).float()          # the measured Mg base rate
    rnd = tp._binary_metrics(torch.randn(20000), y, "mg")["mg_auroc"]
    con = tp._binary_metrics(torch.zeros(20000), y, "mg")["mg_auroc"]
    per = tp._binary_metrics(y * 10, y, "mg")["mg_auroc"]
    inv = tp._binary_metrics(-(y * 10), y, "mg")["mg_auroc"]
    chk("random scores sit at chance", abs(rnd - 0.5) < 0.03, f"{rnd:.4f}")
    chk("a CONSTANT score sits at chance", abs(con - 0.5) < 0.01, f"{con:.4f}")
    chk("a perfect score reaches 1", per > 0.999, f"{per:.4f}")
    chk("an inverted score reaches 0", inv < 0.001, f"{inv:.4f}")
    nan = tp._binary_metrics(torch.randn(50), torch.zeros(50), "mg")["mg_auroc"]
    chk("a one-class batch reports NaN, not a number", nan != nan, f"{nan}")
    chk("the positive rate is reported alongside it",
        abs(tp._binary_metrics(torch.randn(20000), y, "mg")["mg_pos_rate"]
            - float(y.mean())) < 1e-6, f"{float(y.mean()):.4f}")

    print("\n== correlation is 0 for any constant prediction ==")
    tgt = torch.randn(5000)
    for c in (0.0, 0.3, -7.5):
        r = tp._regression_metrics(torch.full((5000,), c), tgt, "rg")["rg_r"]
        chk(f"constant {c:+.1f} scores r = 0", abs(r) < 1e-6, f"r {r:.2e}")
    chk("a perfect prediction scores r = 1",
        tp._regression_metrics(tgt, tgt, "rg")["rg_r"] > 0.999, "")
    base = tp._regression_metrics(torch.zeros(5000), tgt, "rg")["rg_base"]
    chk("the constant-mean baseline loss is reported", base > 0.0, f"{base:.4f}")

    print("\n== class lift is 0 for a head that predicts only the prior ==")
    # 76.57% is the corpus's measured Leontis-Westhof majority share
    maj = torch.rand(5000) < 0.7657
    t13 = torch.where(maj, torch.zeros(5000, dtype=torch.long),
                      torch.randint(1, 13, (5000,)))
    lg = torch.zeros(5000, 13)
    lg[:, 0] = 10.0                                  # always the majority class
    m = tp._class_metrics(lg, t13, "lw", 13)
    chk("accuracy equals the majority share", abs(m["lw_acc"] - m["lw_major"]) < 1e-6,
        f"acc {m['lw_acc']:.4f} vs major {m['lw_major']:.4f}")
    chk("so the LIFT is exactly zero", abs(m["lw_lift"]) < 1e-6,
        f"{m['lw_lift']:.2e} -- this is the number that catches a prior-predictor")
    chk("and macro recall collapses to 1/k", abs(m["lw_macro"] - 1 / 13) < 1e-4,
        f"{m['lw_macro']:.4f} vs {1/13:.4f}")
    # a head that actually separates the classes must beat both
    perfect = torch.nn.functional.one_hot(t13, 13).float() * 10.0
    mp = tp._class_metrics(perfect, t13, "lw", 13)
    chk("a perfect head shows positive lift and macro 1",
        mp["lw_lift"] > 0.2 and mp["lw_macro"] > 0.999,
        f"lift {mp['lw_lift']:.4f}, macro {mp['lw_macro']:.4f}")

    print("\n== every trainer's --help runs ==")
    chk("no argparse help string has an unescaped %",
        check_help_strings() == 0,
        "argparse %-formats help text, so a literal % raises only when "
        "--help is run -- which nobody does on a trainer")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
