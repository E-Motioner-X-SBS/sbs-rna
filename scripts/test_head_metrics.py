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



def check_runlog_fields() -> int:
    """A key the RunLog was not told about is dropped without a word.

    `RunLog` builds its `csv.DictWriter` with `extrasaction="ignore"`, which is
    the right call for a writer that must never crash a training run -- and it
    means adding a measurement to a `log()` call does NOTHING unless the field
    is also declared. Today's router telemetry -- dead experts, routing width,
    real batch tokens, Muon's learning rate -- was printed to stdout and absent
    from the csv for exactly that reason, and `measure_batch_effect.py` had to
    parse the log file instead.

    So: prove the drop happens, and then prove stage 1 declares what it logs.
    """
    import csv as _csv
    import importlib.util as _iu
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pharos.train.telemetry import RunLog
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        rl = RunLog(Path(d), "probe", ["declared"])
        rl.log("step", step=1, declared=42, undeclared=99)
        rl.finish()
        row = list(_csv.DictReader(next(Path(d).rglob("probe*.csv")).open()))[0]
        if row.get("declared") != "42":
            print("  FAIL a DECLARED field was not recorded"); bad += 1
        if "undeclared" in row:
            print("  FAIL an undeclared field WAS recorded"); bad += 1
    print(f"  OK   undeclared fields are dropped silently         "
          f"(so they must be declared)")
    # ...and now they are dropped LOUDLY. Silence is what let stage 1 lose four
    # columns for a run and stage 5 declare nine against forty.
    import io as _io
    import contextlib as _ctx
    with tempfile.TemporaryDirectory() as d:
        rl = RunLog(Path(d), "probe", ["declared"])
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            rl.log("step", step=1, declared=1, undeclared=2)
            rl.log("step", step=2, declared=1, undeclared=2)
        rl.finish()
        out = buf.getvalue()
        if "undeclared" not in out or "[telemetry]" not in out:
            print("  FAIL an undeclared field was dropped WITHOUT a warning")
            bad += 1
        elif out.count("[telemetry]") != 1:
            print(f"  FAIL the warning repeated ({out.count('[telemetry]')}x); "
                  f"it must be once per key per run")
            bad += 1
        else:
            print("  OK   and it says so, once                          "
                  f"{out.strip().splitlines()[0][:52]}")

    # Stage 5's field list is BUILT from the metric tags, so the test checks
    # that the build covers what the two `runlog.log` calls actually pass
    # rather than checking a literal. The list was nine names long against
    # roughly forty passed, and four of the nine matched nothing `evaluate()`
    # returns -- both halves of a mismatch nobody could see from the call site.
    spec5 = _iu.spec_from_file_location(
        "tp5", Path(__file__).resolve().parents[1] / "scripts/train_pharos.py")
    tp5 = _iu.module_from_spec(spec5)
    spec5.loader.exec_module(tp5)
    f5 = set(tp5.STAGE5_FIELDS)
    for need in ("part_contact", "part_lw_lift", "part_motif_macro",
                 "part_mg_auroc", "part_rigidity_r", "part_coev_frac",
                 "val_contact_ap", "val_structure_loss", "val_motif_lift",
                 "val_lw_macro", "val_mg_average_precision"):
        ok = need in f5
        print(f"  {'OK  ' if ok else 'FAIL'} stage 5 declares {need:22s}")
        if not ok:
            bad += 1
    if "val_val_motif_acc" in f5:
        print("  FAIL stage 5 declares a double-prefixed val_val_ column")
        bad += 1

    # Stages 2/3: the declared list must cover every key `hist.append({...})`
    # builds, because the epoch call logs that dict. It used to hand-pick four
    # of thirteen, so the validation accuracy, the generalisation gap, the
    # majority rate, the macro recall and the val lift -- the two of which say
    # whether stage 2 is learning or memorising -- were computed, printed,
    # written to json and dropped on the way to the csv.
    import re as _re
    src23 = (Path(__file__).resolve().parents[1]
             / "scripts/train_sequence_stages.py").read_text()
    decl23 = set(_re.findall(
        r'"([a-z_0-9]+)"',
        src23.split('RunLog(ROOT, "stage23_seq"', 1)[1].split("], manifest", 1)[0]))
    h23 = src23.split("hist.append({", 1)[1]
    _d, _i = 1, 0
    while _d:
        _d += (h23[_i] == "{") - (h23[_i] == "}")
        _i += 1
    hist_keys = set(_re.findall(r'"([a-z_0-9]+)":', h23[:_i - 1])) - {"epoch", "steps"}
    dropped = sorted(hist_keys - decl23)
    print(f"  {'OK  ' if not dropped else 'FAIL'} "
          f"stage 2/3 declares its whole history  "
          f"{len(hist_keys)} keys"
          f"{'' if not dropped else ', DROPPED: ' + ', '.join(dropped)}")
    if dropped:
        bad += 1

    src = (Path(__file__).resolve().parents[1]
           / "scripts/pretrain_mlm.py").read_text()
    decl = src.split('RunLog(ROOT, "stage1_mlm"', 1)[1].split("])", 1)[0]
    for f in ("muon_lr", "batch_tokens", "dead_expert_frac",
              "route_width_mean", "route_width_max"):
        ok = f'"{f}"' in decl
        print(f"  {'OK  ' if ok else 'FAIL'} stage 1 declares {f:22s}")
        if not ok:
            bad += 1
    return bad


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

    # r = 0 is reported for BOTH a collapsed head and a live uncorrelated one,
    # so the correlation alone cannot tell them apart. The prediction's own
    # spread is the number that can, and it was not reported.
    flat = tp._regression_metrics(torch.full((5000,), 0.3), tgt, "rg")
    live = tp._regression_metrics(torch.randn(5000), tgt, "rg")
    chk("a collapsed head and an uncorrelated one both score r = 0",
        abs(flat["rg_r"]) < 1e-6 and abs(live["rg_r"]) < 0.1,
        f"collapsed {flat['rg_r']:.2e}, live {live['rg_r']:+.4f}")
    # NOT `== 0.0`. `torch.std` of a constant tensor returns 2.98e-08, not
    # zero -- the two-pass variance does not cancel exactly in fp32. Which is
    # the reason a collapsed head is detected with a THRESHOLD and not an
    # equality, here and in stage 6's `DEAD_PRED_SD`.
    chk("but only the collapsed one has a vanishing prediction spread",
        flat["rg_pred_sd"] < 1e-6 and live["rg_pred_sd"] > 0.5,
        f"collapsed sd {flat['rg_pred_sd']:.2e} (not exactly 0), "
        f"live sd {live['rg_pred_sd']:.4f}")

    print("\n== a supervision target is a function of its own residue ==")
    # `fluctuation`'s target was `b_factor_z - b_factor_z.min()` over the
    # CURRENT BATCH, so the same residue's target moved with whichever other
    # chains shared its batch -- measured spread 4.83 over 35 stage-5
    # batches, against a signal of sd 1.0. And `fluct_r` is Pearson, which is
    # invariant to a constant offset, so the metric beside it could not see
    # the defect in the quantity it measured.
    _src_tp = Path(tp.__file__).read_text()
    chk("the fluctuation target does not reduce over the batch",
        ".min())" not in _src_tp.split("fl = out[\"fluctuation\"]")[1][:600]
        and "F.softplus(t[\"b_factor_z\"]" in _src_tp,
        "softplus of the residue's own z-scored B-factor")
    # and the property itself: the same residue, two different batches
    import torch.nn.functional as _F
    _z = torch.randn(64)
    _a = _F.softplus(_z)
    _b = _F.softplus(torch.cat([_z, torch.randn(64) * 5 - 10]))[:64]
    chk("so the same residue gets the same target in any batch",
        float((_a - _b).abs().max()) == 0.0,
        f"max delta {float((_a - _b).abs().max()):.2e} with a far more "
        f"extreme batch-mate")
    chk("and the target is non-negative, which the head's output is",
        bool((_F.softplus(torch.randn(10000) * 3) >= 0).all()), "")

    print("\n== the block scorer's micro average can actually be computed ==")
    # It could not. The commit that added the micro average added two
    # `acc[...].append` calls and not the two dict keys they append to, so
    # `evaluate()` raised KeyError on its first chain -- and the guarded
    # `sum(acc.get(key, []))` three lines below turned that into a plausible
    # `l1_recall_micro: None` in the saved report. The metric the function's
    # own comment argues is the only meaningful one had never run once.
    #
    # Checked by RUNNING it, on the model-free `separation` mode so this needs
    # no GPU and no checkpoint. A source-level check for the missing key would
    # be a regex over code, which is how the last two of these tests got it
    # wrong.
    import torch as _t
    from train_block_scorer import evaluate as _ev
    from pharos.model.block_scorer import ScorerConfig as _SC
    from pharos.data.loader import Pharos3DDataset as _DS
    _corpus = ROOT / "data/derived/pharos3d"
    if not _corpus.exists():
        chk("corpus present to evaluate against", False, str(_corpus))
    else:
        _r = _ev(None, _DS(_corpus, split="test"), _SC(), _t.device("cpu"),
                 "separation", max_batches=2, token_budget=2048)
        chk("the micro average is a number, not None",
            _r["l1_recall_micro"] is not None and _r["l2_recall_micro"] is not None,
            f"L1 {_r['l1_recall_micro']}, L2 {_r['l2_recall_micro']} over "
            f"{_r['l1_positives_total']:,} positive L1 blocks")
        chk("and it differs from the chain-weighted macro",
            _r["l1_recall_micro"] != _r["l1_recall"],
            f"macro {_r['l1_recall']} vs micro {_r['l1_recall_micro']} -- "
            f"93.6% of the test split is under 128 nt, where the budget keeps "
            f"about one block")

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

    print("\n== the run log records what the step line prints ==")
    chk("every telemetry field the trainer logs is declared",
        check_runlog_fields() == 0,
        "extrasaction='ignore' drops undeclared keys without an error")

    print("\n== no undefined name, at the Python this project declares ==")
    # `check_loss(loss, gstep, parts)` in train_pharos.py raised NameError on
    # the FIRST optimiser step of stage 5 -- `gstep` does not exist there, the
    # counter is `step`. The guard added to stop a NaN reaching the optimiser
    # was itself a hard crash, in the one stage that has never run, so nothing
    # reported it for two weeks. `--help` exits before the loop and the unit
    # tests never execute it; a static undefined-name pass costs a second and
    # catches the whole class.
    #
    # Run at the FLOOR of `requires-python`, not at the interpreter running
    # this file. Two modules used a backslash inside an f-string expression,
    # which is a SyntaxError before 3.12 -- and one of them was
    # `verify_claims.py`, the gate that decides whether training may proceed.
    # A project that declares 3.10 and cannot parse its own gate on 3.10 has
    # a declaration, not a floor.
    import re as _re
    import subprocess as _sp
    _floor = _re.search(r'requires-python\s*=\s*"[^0-9]*(\d+)\.(\d+)',
                        (ROOT / "pyproject.toml").read_text())
    _tgt = f"py{_floor.group(1)}{_floor.group(2)}" if _floor else "py310"
    _r = _sp.run(["ruff", "check", "--select", "F821", "--target-version",
                  _tgt, "--no-cache", "--quiet", "scripts", "src", "research"],
                 cwd=ROOT, capture_output=True, text=True)
    if _r.returncode == 127 or "No such file" in _r.stderr:
        chk(f"ruff present to run the parse check at {_tgt}", False, _r.stderr[:80])
    else:
        _bad = [ln for ln in (_r.stdout + _r.stderr).splitlines()
                if "invalid-syntax" in ln or "undefined-name" in ln
                or "F821" in ln]
        chk(f"every module parses and resolves at {_tgt}",
            _r.returncode == 0 and not _bad,
            "clean" if not _bad else f"{len(_bad)}: {_bad[:2]}")

    print("\n== the NaN guard cannot crash the thing it guards ==")
    # `check_loss` exists so a non-finite loss cannot reach the optimiser. In
    # stage 5 it was written `check_loss(loss, gstep, parts)` and stage 5 has
    # no `gstep`, so it raised NameError on the first optimiser step -- the
    # guard against a silent failure was a loud one, in the one stage that
    # had never run. Both halves are checked here: the step argument must be
    # a name the function binds, and it must be bound BEFORE the call, since
    # a counter first assigned by `step += 1` further down is an
    # UnboundLocalError on the first iteration and the static pass above
    # cannot see it.
    import ast as _ast
    _probs = []
    for _f in sorted((ROOT / "scripts").glob("*.py")):
        try:
            _tree = _ast.parse(_f.read_text())
        except SyntaxError as _e:
            _probs.append(f"{_f.name}: {_e}")
            continue
        for _fn in [n for n in _ast.walk(_tree)
                    if isinstance(n, _ast.FunctionDef)]:
            _calls = [n for n in _ast.walk(_fn) if isinstance(n, _ast.Call)
                      and isinstance(n.func, _ast.Name)
                      and n.func.id == "check_loss"]
            if not _calls:
                continue
            # first line each local name is written on. A plain loop: the
            # nested comprehension this replaced referred to a name its own
            # outer scope did not bind, which is precisely the defect below.
            _binds: dict = {a.arg: _fn.lineno for a in _fn.args.args}
            for _nd in _ast.walk(_fn):
                if isinstance(_nd, _ast.Name) and isinstance(_nd.ctx, _ast.Store):
                    _binds[_nd.id] = min(_binds.get(_nd.id, _nd.lineno),
                                         _nd.lineno)
            for _c in _calls:
                if len(_c.args) < 2 or not isinstance(_c.args[1], _ast.Name):
                    continue
                _n = _c.args[1].id
                if _n not in _binds:
                    _probs.append(f"{_f.name}:{_c.lineno} `{_n}` is undefined")
                elif _binds[_n] > _c.lineno:
                    _probs.append(f"{_f.name}:{_c.lineno} `{_n}` is first "
                                  f"bound at line {_binds[_n]}, after the call")
    chk("every check_loss step argument is bound before the call",
        not _probs, "; ".join(_probs) if _probs else
        "4 trainers: pretrain_mlm, train_pharos, train_sequence_stages, "
        "train_block_scorer")

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
