#!/usr/bin/env python3
"""Read the training telemetry and say what is WRONG with it.

The durable half of the training watch. A system cron fires
`scripts/training_watch.sh` every half hour; that calls this; this reads
whatever the runs have written since last time and applies a battery of
checks for the defect class this project keeps finding -- a component that
exists, is measured, and whose output nobody checks for basic validity.

It is deliberately not a monitor. `training_monitor.py` answers "is stage 1
alive and is what it reports meaningful". This answers "given everything on
disk, name the things that are broken", and it writes its answer to a file
so that an assistant session which was asleep, rate-limited or killed can
read the whole backlog when it comes back instead of only seeing now.

Every check is a function returning zero or more findings. A check that
raises is reported as a BROKEN CHECK rather than being allowed to kill the
run -- a watchdog that dies silently is worse than no watchdog, which is the
same lesson as everywhere else in this tree.

Severity:
    CRIT   training is dead, diverged, or producing invalid numbers
    WARN   a head or channel looks broken; worth a human or an assistant
    INFO   state worth recording, no action

Exit code is 0 unless the report could not be written: cron must not treat
"found problems" as "the watcher failed".
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data/samples/analysis/runs"
CKPT = ROOT / "data/derived/checkpoints"
OUT = ROOT / "data/samples/analysis/watch"

#: A metric equal to its own floor is the finding-75 signature, generalised.
#: Head 2 reported `dist_acc` 0.47 for a stage while `dist_major` was 0.47,
#: and nothing said so because nothing compared them. Any `X_acc` with an
#: `X_major` beside it is now compared on every fire.
FLOOR_EPS = 0.005
#: Below this a prediction is constant, whatever its correlation says.
DEAD_SD = 1e-4
#: A gate still exactly zero after this many steps has never had a gradient.
GATE_STEPS = 500


def rows(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open() as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def num(r: Dict[str, str], k: str) -> Optional[float]:
    v = r.get(k, "")
    if v is None or v == "" or v == "None":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def series(rs: List[Dict], k: str) -> List[float]:
    return [v for v in (num(r, k) for r in rs) if v is not None]


def newest_run(stage: str) -> Optional[Path]:
    d = RUNS / stage
    if not d.is_dir():
        return None
    cs = sorted(d.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cs[0] if cs else None


# ---------------------------------------------------------------- checks ---

def check_liveness(F: List[Tuple[str, str, str]]) -> None:
    """Is anything training, and has it moved?"""
    try:
        ps = subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True,
                            text=True, timeout=20).stdout.splitlines()
    except Exception as e:                                   # noqa: BLE001
        F.append(("WARN", "liveness", f"could not list processes: {e}"))
        return
    # Which SCRIPT maps to which telemetry directory. Without this the check
    # reported "a trainer is running but stage1_mlm has not moved for 62
    # hours" while the running trainer was stage 2/3 and stage 1 had been
    # finished for days. A staleness check that does not know which stage the
    # live process belongs to reports every stage that is not running.
    OWNER = {"pretrain_mlm.py": "stage1_mlm",
             "train_sequence_stages.py": "stage23_seq",
             "train_pharos.py": "stage5_3d",
             "train_block_scorer.py": "r1_block_scorer"}
    # SELF-MATCHING. The first version matched any line containing a script
    # name, and matched its own shell -- the `bash -c` that launched this
    # very check had the names in it as string literals, so the watcher
    # reported itself as a trainer "up 0 min". Same trap as `pgrep -f`.
    # A real trainer is an interpreter invoking a script, so require both and
    # exclude shells outright.
    def _is_trainer(line: str) -> bool:
        if " -c " in line or line.split(None, 2)[-1].startswith(("/bin/", "sh ")):
            return False
        return ("python" in line
                and any(f"scripts/{k}" in line or line.rstrip().endswith(k)
                        for k in OWNER))

    alive = [l for l in ps if _is_trainer(l)]
    if not alive:
        F.append(("INFO", "liveness", "no trainer process is running"))
        return
    running: set = set()
    for l in alive:
        pr = l.split(None, 2)
        F.append(("INFO", "liveness",
                  f"pid {pr[0]} up {int(pr[1])//60} min: {pr[2][:90]}"))
        for k, stage in OWNER.items():
            if k in l:
                running.add(stage)
    for stage in sorted(running):
        c = newest_run(stage)
        if c is None:
            F.append(("WARN", "no telemetry",
                      f"{stage} is running and has written no csv at all"))
            continue
        age = (time.time() - c.stat().st_mtime) / 60.0
        if age > 45:
            F.append(("CRIT", "stalled",
                      f"{stage} is RUNNING and its telemetry has not moved "
                      f"for {age:.0f} min ({c.name})"))


def check_finite(F: List[Tuple[str, str, str]], stage: str,
                 rs: List[Dict]) -> None:
    """Any non-finite number anywhere is a stop-everything."""
    for i, r in enumerate(rs):
        for k, v in r.items():
            x = num(r, k)
            if x is not None and not math.isfinite(x):
                F.append(("CRIT", "non-finite",
                          f"{stage} row {i} column {k} is {v!r}"))


def check_floors(F: List[Tuple[str, str, str]], stage: str,
                 rs: List[Dict]) -> None:
    """Every `X_acc` against the `X_major` beside it -- finding 75's shape."""
    if not rs:
        return
    cols = set(rs[-1])
    for acc in sorted(c for c in cols if c.endswith("_acc")):
        base = acc[:-4]
        major = f"{base}_major"
        if major not in cols:
            F.append(("WARN", "no floor",
                      f"{stage} reports {acc} with no {major} beside it -- "
                      f"an accuracy with no floor cannot be read"))
            continue
        a, m = series(rs, acc), series(rs, major)
        n = min(len(a), len(m))
        if n < 2:
            continue
        a, m = a[-n:], m[-n:]
        gap = [abs(x - y) for x, y in zip(a, m)]
        if max(gap) < FLOOR_EPS:
            F.append(("WARN", "at the floor",
                      f"{stage} {acc} tracks {major} to within "
                      f"{max(gap):.4f} over {n} readings "
                      f"({a[-1]:.4f} vs {m[-1]:.4f}) -- the head predicts the "
                      f"majority class and the bare accuracy hides it"))
    for macro in sorted(c for c in cols if c.endswith("_macro")):
        nc = f"{macro[:-6]}_n_class"
        v = series(rs, macro)
        k = series(rs, nc)
        if v and k and k[-1] > 1 and v[-1] <= 1.15 / k[-1]:
            F.append(("WARN", "macro at chance",
                      f"{stage} {macro} = {v[-1]:.4f}, chance for "
                      f"{k[-1]:.0f} classes is {1/k[-1]:.4f}"))


def check_dead_regressions(F: List[Tuple[str, str, str]], stage: str,
                           rs: List[Dict]) -> None:
    """A constant prediction, which a correlation of 0 cannot distinguish."""
    if not rs:
        return
    for sd in sorted(c for c in set(rs[-1]) if c.endswith("_pred_sd")):
        v = series(rs, sd)
        if v and abs(v[-1]) < DEAD_SD:
            F.append(("WARN", "collapsed",
                      f"{stage} {sd} = {v[-1]:.3e} -- the head emits a "
                      f"constant, so any correlation beside it is 0 by "
                      f"construction and not by failure to correlate"))


def check_lifts(F: List[Tuple[str, str, str]], stage: str,
                rs: List[Dict]) -> None:
    """A lift at or below zero means the head is not beating its baseline."""
    if len(rs) < 3:
        return
    for lift in sorted(c for c in set(rs[-1]) if c.endswith("_lift")):
        v = series(rs, lift)
        if len(v) < 3:
            continue
        tail = v[-min(5, len(v)):]
        if max(tail) <= 0.0:
            F.append(("WARN", "no lift",
                      f"{stage} {lift} has been <= 0 for {len(tail)} readings "
                      f"(latest {v[-1]:+.4f})"))


def check_empty_columns(F: List[Tuple[str, str, str]], stage: str,
                        rs: List[Dict]) -> None:
    """A declared metric that is never populated.

    This is the defect class the register is named for: the column exists,
    the csv has it, and the head behind it never fired. `part_torsion` empty
    for a whole stage-5 run would mean head 11 never reached its loss, and
    the run would otherwise look perfect.
    """
    if len(rs) < 3:
        return
    # ONE SCHEMA, SEVERAL ROW KINDS. `kind=step` rows carry the per-step
    # metrics and `kind=epoch` rows carry the validation block, in the same
    # csv with the same header -- so a validation column is legitimately
    # empty on every step row. Checking the step rows alone reported
    # `ss_val_accuracy`, `fitness_val_spearman` and nine more as "never
    # written" in a run that had simply not finished its first epoch.
    #
    # So: a column is only unwritten if NO row of ANY kind carries it, and
    # only once the run has produced an epoch row -- before that, there has
    # been no opportunity.
    kinds = {r.get("kind", "") for r in rs}
    if not (kinds & {"epoch", "eval", "val"}):
        return
    for k in sorted(set(rs[-1])):
        if k in ("note", "run_id", "stage", "kind", "timestamp"):
            continue
        if all(num(r, k) is None for r in rs):
            F.append(("WARN", "never written",
                      f"{stage} declares column {k} and not one of "
                      f"{len(rs)} rows of any kind has a value in it"))


def check_loss_trend(F: List[Tuple[str, str, str]], stage: str,
                     rs: List[Dict]) -> None:
    """Flat or rising loss, and spikes."""
    for k in ("loss", "ss_loss", "probing_loss", "fitness_loss"):
        v = series(rs, k)
        if len(v) < 6:
            continue
        half = len(v) // 2
        a = sum(v[:half]) / half
        b = sum(v[half:]) / (len(v) - half)
        if b > a * 1.05:
            F.append(("WARN", "loss rising",
                      f"{stage} {k} second half {b:.4f} against first half "
                      f"{a:.4f} over {len(v)} readings"))
        mu = sum(v) / len(v)
        sd = (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5
        if sd > 0 and v[-1] > mu + 5 * sd:
            F.append(("CRIT", "loss spike",
                      f"{stage} {k} latest {v[-1]:.4f} is {(v[-1]-mu)/sd:.1f} "
                      f"sd above its own mean {mu:.4f}"))


def check_oom(F: List[Tuple[str, str, str]], stage: str,
              rs: List[Dict]) -> None:
    v = series(rs, "n_oom")
    if not v or v[-1] <= 0:
        return
    # Only while it is still happening. A finished run's OOM count is
    # history, and re-reporting it every half hour forever is noise.
    fresh = len(v) > 1 and v[-1] > v[-2]
    sev = "CRIT" if (fresh and v[-1] > 20) else ("WARN" if fresh else "INFO")
    F.append((sev, "oom", f"{stage} has taken {v[-1]:.0f} OOMs"
                          + (" and is still taking them" if fresh
                             else " (not rising)")))


def check_gates(F: List[Tuple[str, str, str]]) -> None:
    """A wired component whose gate has never left zero.

    §6.2 sat wired-but-frozen for a whole curriculum and nothing said so.
    These are the parameters whose exact zero means "this physics has never
    been trained", and they are cheap to read straight out of the file.
    """
    # (description, the checkpoints whose stage CAN train it).
    #
    # Scoped, because an unscoped version reported `coev_proj` frozen in the
    # stage-1 and stage-2/3 checkpoints -- correctly, and uselessly:
    # `coev_proj` is applied by `train_pharos.py` and by nothing else, so it
    # is SUPPOSED to be zero everywhere but stage 5. A warning that fires on
    # intended behaviour trains the reader to skip the section.
    gates = {
        "elec.log_scale": ("§6.2 screened-Coulomb scale", ("seqstages", "pharos")),
        "elec.site_scale": ("§6.2a per-site condensation gate", ("seqstages", "pharos")),
        "trunk.blocks.7.mixer.bias_scale": ("§6.2 pair-bias gate, block 7",
                                            ("seqstages", "pharos")),
        "trunk.blocks.15.mixer.bias_scale": ("§6.2 pair-bias gate, block 15",
                                             ("seqstages", "pharos")),
        "coev_proj.weight": ("coevolution injection", ("pharos",)),
    }
    import torch                                             # noqa: PLC0415
    # The LIVE checkpoints only, and mmap'd. These files are 4.4 GiB each and
    # a watchdog that reads twenty of them into RAM every half hour would be
    # competing with the trainer it is watching -- which is the mistake
    # finding 76 was withdrawn for. mmap pages in the handful of small gate
    # tensors and leaves the 394M-parameter bulk on disk.
    live = [f for f in sorted(CKPT.glob("*.pt"))
            if not any(x in f.name for x in
                       ("superseded", "pre_", "_step", "_muon", "preMuon",
                        "randinit", "baseline", "postdip", "best"))]
    for f in live:
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
        except Exception as e:                               # noqa: BLE001
            F.append(("CRIT", "unreadable",
                      f"{f.name} will not load: {type(e).__name__}: {e}"))
            continue
        sd = ck.get("model")
        if not isinstance(sd, dict):
            continue
        step = ck.get("gstep") or ck.get("step") or 0
        # A checkpoint written before a parameter existed cannot have trained
        # it, and the ABSENCE of its sibling is how to tell. `vdist.proj.
        # weight` arrived with the §6.2 wiring, so a file without it is from
        # a process that had the pre-wiring code loaded -- which is exactly
        # the mistake made by reading the gate values out of three
        # checkpoints and concluding the physics was inert by design.
        wired = "vdist.proj.weight" in sd
        for g, (why, stages) in gates.items():
            if g not in sd:
                continue
            if not any(t in f.name for t in stages):
                continue            # that stage does not train this gate
            if g.endswith("bias_scale") or g.startswith("elec."):
                if not wired:
                    continue        # predates the wiring; zero means nothing
            if float(sd[g].abs().max()) == 0.0 and step >= GATE_STEPS:
                F.append(("WARN", "frozen gate",
                          f"{f.name} at step {step:,}: {g} is still exactly "
                          f"0.0 -- {why} has never taken a gradient"))
        del ck, sd


def check_disk(F: List[Tuple[str, str, str]]) -> None:
    try:
        du = shutil.disk_usage(str(ROOT))
    except Exception:                                        # noqa: BLE001
        return
    free = du.free / 2**30
    if free < 25:
        F.append(("CRIT", "disk", f"only {free:.1f} GiB free; a checkpoint "
                                  f"is about 4.4 GiB"))
    elif free < 80:
        F.append(("WARN", "disk", f"{free:.1f} GiB free"))
    else:
        F.append(("INFO", "disk", f"{free:.1f} GiB free"))


def check_gpu(F: List[Tuple[str, str, str]]) -> None:
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=25).stdout.strip()
        used, total, util = (int(x) for x in q.split(",")[:3])
        F.append(("INFO", "gpu", f"{used:,} / {total:,} MiB, {util}% util"))
    except Exception as e:                                   # noqa: BLE001
        F.append(("WARN", "gpu", f"nvidia-smi unavailable: {e}"))


# ------------------------------------------------------------------ main ---

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    F: List[Tuple[str, str, str]] = []
    broke: List[str] = []

    def run(fn: Callable, *a) -> None:
        # A watchdog that dies silently is worse than no watchdog.
        try:
            fn(F, *a)
        except Exception as e:                               # noqa: BLE001
            broke.append(f"{fn.__name__}{a[:1]}: {type(e).__name__}: {e}")

    run(check_liveness)
    run(check_disk)
    run(check_gpu)
    run(check_gates)
    for stage in ("stage1_mlm", "stage23_seq", "stage5_3d", "r1_block_scorer"):
        c = newest_run(stage)
        if c is None:
            continue
        rs = rows(c)
        if not rs:
            continue
        tag = f"{stage}/{c.stem}"
        F.append(("INFO", "run", f"{tag}: {len(rs)} rows, last written "
                                 f"{(time.time()-c.stat().st_mtime)/60:.0f} min ago"))
        for fn in (check_finite, check_floors, check_dead_regressions,
                   check_lifts, check_empty_columns, check_loss_trend,
                   check_oom):
            run(fn, tag, rs)

    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    crit = [f for f in F if f[0] == "CRIT"]
    warn = [f for f in F if f[0] == "WARN"]
    info = [f for f in F if f[0] == "INFO"]

    lines = [f"# training watch — {stamp}", ""]
    lines.append(f"**{len(crit)} CRIT, {len(warn)} WARN, {len(info)} INFO**"
                 + (f", {len(broke)} CHECKS BROKEN" if broke else ""))
    lines.append("")
    for name, group in (("CRIT", crit), ("WARN", warn), ("INFO", info)):
        if not group:
            continue
        lines.append(f"## {name}")
        for _, kind, msg in group:
            lines.append(f"- **{kind}** — {msg}")
        lines.append("")
    if broke:
        lines.append("## CHECKS THAT RAISED")
        lines += [f"- {b}" for b in broke]
        lines.append("")
    body = "\n".join(lines)

    (OUT / f"report-{time.strftime('%Y%m%d-%H%M%S')}.md").write_text(body)
    (OUT / "latest.md").write_text(body)
    with (OUT / "history.jsonl").open("a") as fh:
        fh.write(json.dumps({"t": stamp, "crit": len(crit), "warn": len(warn),
                             "broken": broke,
                             "findings": [{"sev": s, "kind": k, "msg": m}
                                          for s, k, m in crit + warn]}) + "\n")
    # The ALERT file is what an assistant session checks first: it exists only
    # while something is actually wrong, so "no file" is a real all-clear and
    # not an absence of information.
    alert = OUT / "ALERT"
    if crit or warn or broke:
        alert.write_text(body)
    else:
        alert.unlink(missing_ok=True)
    print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
