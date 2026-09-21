#!/usr/bin/env python3
"""Does the code implement what ARCHITECTURE v0.2 specifies? Section by section.

Written because "is the architecture done" is a question the repo should be
able to answer mechanically rather than from memory. Each row names a component
the specification requires and the symbol that has to exist for it to be real;
the check is an import and an attribute lookup, so a module that was renamed,
gutted or never written fails rather than passing on the strength of its
docstring.

It deliberately does NOT check quality -- only presence. A component can be
present and untrained, present and wrong, or present and slow. Those are what
`verify_claims.py`, the test suites and the training runs are for.

Usage:
    /store/shuvam/.venv/bin/python scripts/sampling/audit_completeness.py
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

#: (section, component, module, symbol, note)
SPEC: List[Tuple[str, str, str, str, str]] = [
    ("2",  "input embedding",          "pharos.model.pharos",   "InputEmbedding", "tokens + mods + chemistry + position"),
    ("3",  "structural tokenizer",     "pharos.data.vocab",     "encode_residue", "parent base + modification id (D6, D11)"),
    ("3",  "modification vocabulary",  "pharos.data.vocab",     "mod_vocab_size", "370 CCD species"),
    ("4",  "24-dim chemistry",         "pharos.data.chemistry", "chain_chemistry", "CHEMISTRY.md"),
    ("4",  "CCD parent resolution",    "pharos.data.chemistry", "resolve",        "by dictionary, not by name"),
    ("5.1", "Gated DeltaNet",          "pharos.model.attention", "GatedDeltaNet",  "linear, chunked delta rule"),
    ("5.1", "sliding-window attn",     "pharos.model.attention", "SlidingWindowAttention", "w=128"),
    ("5.1", "full attn + physics bias", "pharos.model.attention", "FullAttention", "2 of 16 blocks"),
    ("5.1", "block pattern",           "pharos.model.attention", "BLOCK_PATTERN",  "10 GDN / 4 SWA / 2 FULL"),
    ("5.2", "trunk with loops",        "pharos.model.trunk",     "TokenTrunk",     "one-step gradient, deep supervision"),
    ("5.3", "MoE feed-forward",        "pharos.model.moe",       "MoEFeedForward", "fine-grained + shared expert"),
    ("5.3", "conditioned router",      "pharos.model.moe",       "RouterFeatures", "length, Neff/L, in-complex, chemistry"),
    ("6.1", "Manning condensation",    "pharos.physics.manning", "condensed_fraction", "theta = 0.804 for A-RNA"),
    ("6.2", "Debye screening",         "pharos.physics.manning", "debye_kappa",    "kappa from ionic condition"),
    ("6.2", "electrostatic pair bias", "pharos.model.pharos",    "ElectrostaticBias", "B_elec into FULL attention"),
    ("7",   "hierarchical pair track", "pharos.model.block_scorer", "BlockScorer", "L1/L2 selectors"),
    ("7",   "occupancy loss",          "pharos.model.block_scorer", "occupancy_loss", "recall-weighted BCE"),
    ("7.3", "budgets from target_c",   "pharos.model.block_scorer", "l2_budget",   "K = target_c . L"),
    ("8",   "frozen motif bank",       "pharos.model.motif_bank", "MotifBank",     "667 BGSU classes"),
    ("9",   "ten output heads",        "pharos.model.heads",     "PharosHeads",    "contact..base identity"),
    ("9",   "head specification",      "pharos.model.heads",     "HEAD_SPEC",      "the table, as data"),
    ("10",  "stiffness field",         "pharos.model.dynamics",  "StiffnessField", "per-step 6x6, Cholesky"),
    ("10",  "harmonic ensemble",       "pharos.model.dynamics",  "HarmonicEnsemble", "normal modes -> amplitudes"),
    ("10",  "O(L) selected inversion", "pharos.model.dynamics",  "block_tridiagonal_variance", "not O(L^3)"),
    ("11",  "entry composition (G2/G3)", "pharos.data.mmcif_entities", "entry_composition", "whole-entry properties"),
    ("12",  "3D dataset",              "pharos.data.dataset",    "build_entry",    "contacts + block labels"),
    ("12",  "free supervision",        "pharos.data.mmcif_entities", "residue_labels", "Mg, B-factor, N_struct, disorder"),
    ("12",  "length-bucketed loader",  "pharos.data.loader",     "Pharos3DDataset", "quality weighting (D16)"),
    ("12",  "ionic channel",           "pharos.data.rdat",       "titration_examples", "RMDB Mg ladders"),
    ("12",  "assembled model",         "pharos.model.pharos",    "Pharos",         "everything above, wired"),
]

#: curriculum stage -> the script that runs it
STAGES = [
    ("1 pretrain (MLM + spans)",   "scripts/pretrain_mlm.py"),
    ("2 secondary structure",      "scripts/train_sequence_stages.py"),
    ("3 chemical probing",         "scripts/train_sequence_stages.py"),
    ("4 physics",                  None),      # closed form, no stage
    ("5 3D multi-task",            "scripts/train_pharos.py"),
    ("R1 block selector",          "scripts/train_block_scorer.py"),
]

#: things the specification names that are deliberately NOT built
NOT_BUILT = {
    "decoder.py (frame diffusion)": "§9 head 3 emits coordinates directly; diffusion is a v0.3 item",
    "eval/ package": "splits are enforced in build_dataset.py and reported by the trainers",
    "train/precision.py": "settled as a decision (PRECISION.md, D3), not code",
    "attributes.py": "the supervision-target array; labels live in dataset.py",
}

#: Every trainer must schedule its learning rate, and the audit checks that it
#: DOES rather than that a file exists.
#:
#: `train/schedule.py` used to sit in NOT_BUILT above, justified as "settled as
#: a decision (D2), the token budget is a CLI flag". That conflated two
#: different schedules. D2 settles the TOKEN BUDGET -- 5B/25B/100B/323B -- and
#: is genuinely a CLI flag. It says nothing about the LEARNING RATE, and stage 1
#: and stages 2-3 shipped with none at all: a constant rate from the first step
#: to the last, no warm-up and no decay, while the block scorer and stage 5 both
#: ran OneCycleLR. The audit's own note is what made that absence look
#: deliberate, so the entry is gone and this check replaces it.
#: Every trainer must be able to stop and resume, and the audit checks the
#: behaviour rather than the existence of a file.
#:
#: Only stage 1 could. The other three saved at EPOCH BOUNDARIES with no
#: optimiser state and no path that read it back -- and because the cron runner
#: passes `--init-from <previous stage>` on every fire, an interrupted stage
#: restarted from the previous stage's weights and discarded everything it had
#: done. An epoch of stages 2-3 is about 10,500 steps. That is the same defect
#: stage 1 had and that its own docstring called out: "an unattended
#: pretraining script that restarts from zero is not unattended, it is a loop
#: that makes no progress."
RESUMABLE = {
    "scripts/pretrain_mlm.py":          "token/step counters, optimiser state",
    "scripts/train_sequence_stages.py": "epoch/gstep, optimiser state",
    "scripts/train_pharos.py":          "epoch/step, optimiser + OneCycleLR",
    "scripts/train_block_scorer.py":    "epoch/step, optimiser + OneCycleLR",
}

SCHEDULED = {
    "scripts/pretrain_mlm.py":          ("lr_at", "cosine on token progress"),
    "scripts/train_sequence_stages.py": ("lr_at", "cosine on estimated steps"),
    "scripts/train_pharos.py":          ("OneCycleLR", "torch scheduler"),
    "scripts/train_block_scorer.py":    ("OneCycleLR", "torch scheduler"),
}


def main() -> None:
    rows, missing = [], []
    for sec, name, mod, sym, note in SPEC:
        try:
            m = importlib.import_module(mod)
            ok = hasattr(m, sym)
        except Exception as e:                                   # noqa: BLE001
            ok, note = False, f"import failed: {type(e).__name__}"
        rows.append({"section": sec, "component": name, "module": mod,
                     "symbol": sym, "present": ok, "note": note})
        if not ok:
            missing.append(f"§{sec} {name} ({mod}.{sym})")

    stage_rows = []
    for label, script in STAGES:
        present = True if script is None else (ROOT / script).exists()
        stage_rows.append({"stage": label, "script": script, "present": present})
        if not present:
            missing.append(f"curriculum {label} ({script})")

    print(f"{'§':6s} {'component':28s} {'ok':3s}  note")
    for r in rows:
        print(f"{r['section']:6s} {r['component']:28s} "
              f"{'OK ' if r['present'] else 'MISS'}  {r['note']}")
    print(f"\n{'curriculum stage':30s} {'ok':3s}  script")
    for r in stage_rows:
        print(f"{r['stage']:30s} {'OK ' if r['present'] else 'MISS'}  "
              f"{r['script'] or '(closed form, no stage)'}")
    print("\ndeliberately not built:")
    for k, v in NOT_BUILT.items():
        print(f"  {k:30s} {v}")

    print("\nlearning-rate schedule, per trainer")
    sched_rows = []
    for script, (marker, how) in sorted(SCHEDULED.items()):
        f = ROOT / script
        has = f.exists() and marker in f.read_text()
        print(f"  {'OK  ' if has else 'FAIL'} {script:38s} {how}")
        sched_rows.append({"script": script, "how": how, "scheduled": has})
        if not has:
            missing.append(f"{script}: no learning-rate schedule")

    print("\nstop and resume, per trainer")
    resume_rows = []
    for script, how in sorted(RESUMABLE.items()):
        f = ROOT / script
        src = f.read_text() if f.exists() else ""
        checks = {
            "loads optimiser state": "opt.load_state_dict" in src,
            "saves optimiser state": '"opt": opt.state_dict()' in src,
            "checkpoints mid-epoch": "ckpt_every" in src,
            "survives an OOM": "OutOfMemoryError" in src,
            "--restart is non-destructive": "superseded" in src,
        }
        ok = all(checks.values())
        bad = [k for k, v in checks.items() if not v]
        print(f"  {'OK  ' if ok else 'FAIL'} {script:38s} {how}"
              + ("" if ok else f"  MISSING: {', '.join(bad)}"))
        resume_rows.append({"script": script, "how": how, "resumable": ok,
                            "missing": bad})
        if not ok:
            missing.append(f"{script}: cannot resume ({', '.join(bad)})")

    res = {"n_components": len(rows),
           "n_resumable": sum(1 for r in resume_rows if r["resumable"]),
           "resume": resume_rows,
           "n_scheduled": sum(1 for r in sched_rows if r["scheduled"]),
           "n_trainers": len(sched_rows), "schedules": sched_rows,
           "n_present": sum(1 for r in rows if r["present"]),
           "n_stages": len(stage_rows),
           "n_stages_present": sum(1 for r in stage_rows if r["present"]),
           "missing": missing, "components": rows, "stages": stage_rows,
           "not_built": NOT_BUILT}
    out = ROOT / "data/samples/analysis/completeness.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"\ncomponents present: {res['n_present']}/{res['n_components']}   "
          f"curriculum stages: {res['n_stages_present']}/{res['n_stages']}   "
          f"trainers scheduled: {res['n_scheduled']}/{res['n_trainers']}   "
          f"resumable: {res['n_resumable']}/{res['n_trainers']}")
    print("MISSING: " + (", ".join(missing) if missing else "none"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
