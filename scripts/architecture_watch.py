#!/usr/bin/env python3
"""Every 30 minutes: check the architecture, then do whatever work is possible.

This does not stop. Cron fires it, it checks, it does what it can, and it says
what it is waiting for. When the only work left needs the GPU and the GPU has
not been offered, it says **"waiting for GPU permission"** and exits cleanly --
that is a normal outcome, not a failure.

Runs the two audits, records a snapshot, and appends a dated entry to
`research/architecture/CHANGELOG.md` **only when something actually changed**.
A watcher that writes a paragraph every half hour is a watcher nobody reads, so
an unchanged tick updates the status file and writes nothing else.

What it checks:

  verify_claims.py        every pinned number still reproduces
  audit_completeness.py   components, curriculum stages, schedules, resume,
                          telemetry
  the specification       `research/architecture/ARCHITECTURE.md` describes the
                          FINAL design: it must not drift back into narrating
                          its own corrections, and the history folder it defers
                          to must still be populated
  training                if a run is live or resumable, where it has got to

The changelog entry says what moved and in which direction. It does not record
failures or intermediate states -- those belong in
`history_of_failed_attempts/`, and the specification links there.

## The GPU gate

The loop never takes the card on its own. It runs GPU work only when BOTH hold:

    data/samples/analysis/cron/GPU_PERMITTED exists     (you put it there)
    the card has enough free memory                      (measured, not assumed)

Remove the file and the loop goes back to CPU work and reports what it is
blocked on. This is deliberate: the card is shared, and a maintenance loop that
decides for itself when to seize it is a maintenance loop that will do so at
the worst moment.

    touch data/samples/analysis/cron/GPU_PERMITTED    # allow GPU work
    rm    data/samples/analysis/cron/GPU_PERMITTED    # take it back

## The CPU work queue

Each task names an output, the inputs it derives from, and a maximum age. It
runs when the output is missing, older than any input, or past its age. One
task per tick, most-stale first, so a tick stays short and the loop stays
responsive to being killed.

Usage:
    python3 scripts/architecture_watch.py            # check, work, record
    python3 scripts/architecture_watch.py --quick    # skip verify_claims
    python3 scripts/architecture_watch.py --no-work  # check only
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
STATE = ROOT / "data/samples/analysis/architecture_watch.json"
CHANGELOG = ROOT / "research/architecture/CHANGELOG.md"
SPEC = ROOT / "research/architecture/ARCHITECTURE.md"
HISTORY = ROOT / "history_of_failed_attempts"

#: Phrasings that belong in the history folder, not in a specification. The
#: point is not style: a spec that argues with its own past is a spec whose
#: current claim has to be inferred, and that is how superseded numbers get
#: quoted back as though they were live.
NARRATIVE_MARKERS = [
    "was wrong", "turned out to be wrong", "the old value", "previously claimed",
    "defect #", "this was a bug", "used to be", "had never been measured",
]


A = ROOT / "data/samples/analysis"
GPU_PERMIT = ROOT / "data/samples/analysis/cron/GPU_PERMITTED"
HOUR = 3600

#: CPU-only work, re-run when it goes stale. `out` is what it produces, `deps`
#: the things it derives from, `max_age_h` how long a result stays trustworthy
#: even when nothing it depends on has changed -- the data on disk can move
#: under a measurement without any tracked file changing.
CPU_TASKS = [
    ("source inventory", "scripts/sampling/emit_source_inventory.py",
     "data/catalog/SOURCES.md", ["scripts/acquire_all.py"], 24 * 7),
    ("completeness audit", "scripts/sampling/audit_completeness.py",
     "data/samples/analysis/completeness.json",
     ["scripts/pretrain_mlm.py", "scripts/train_pharos.py",
      "scripts/train_sequence_stages.py", "scripts/train_block_scorer.py"], 12),
    ("inventory gap", "scripts/sampling/audit_inventory_gap.py",
     "data/samples/analysis/inventory_gap.json", [], 24),
    ("data presence", "scripts/sampling/audit_data_presence.py",
     "data/samples/analysis/data_presence.json", [], 24),
    ("data integrity", "scripts/sampling/verify_data_integrity.py",
     "data/samples/analysis/data_integrity.json", [], 24 * 3),
    ("pretrain coverage", "scripts/sampling/audit_pretrain_coverage.py",
     "data/samples/analysis/pretrain_coverage.json",
     ["scripts/eldors_to_parquet.py"], 24 * 7),
    ("packing waste", "scripts/sampling/measure_packing_waste.py",
     "data/samples/analysis/packing_waste.json",
     ["scripts/pretrain_mlm.py"], 24 * 7),
]

#: Work that needs the card, in the order it should be taken. `need_gib` is
#: measured against free memory before anything starts; `detach` marks work
#: that runs for hours and must not be held inside this tick.
#:
#: Nothing here runs unless GPU_PERMITTED exists. The loop reports what it is
#: blocked on instead, which is the useful thing to say.
GPU_TASKS = [
    {"name": "resume stage 1 and the curriculum",
     "cmd": ["scripts/gpu_cron_runner.sh"], "need_gib": 60.0, "detach": True,
     "out": None, "max_age_h": 0.0},
    {"name": "re-probe router specialisation",
     "cmd": [PY, "scripts/sampling/probe_router_specialisation.py"],
     "need_gib": 8.0, "detach": False,
     "out": "data/samples/analysis/router_specialisation.json",
     "max_age_h": 6.0},
    {"name": "re-measure cascade recall",
     "cmd": [PY, "scripts/sampling/measure_cascade_recall.py"],
     "need_gib": 8.0, "detach": False,
     "out": "data/samples/analysis/cascade_recall.json", "max_age_h": 24.0},
]


def gpu_free_gib() -> Optional[float]:
    """Free VRAM without creating a CUDA context."""
    code, out = run(["nvidia-smi", "--query-gpu=memory.free",
                     "--format=csv,noheader,nounits"], timeout=60)
    if code != 0 or not out.strip():
        return None
    try:
        return int(out.strip().splitlines()[0]) / 1024.0
    except (ValueError, IndexError):
        return None


def stale(out: str, deps: list, max_age_h: float) -> Optional[str]:
    """Why this task is due, or None if it is not."""
    o = ROOT / out
    if not o.exists():
        return "never run"
    age_h = (datetime.now().timestamp() - o.stat().st_mtime) / HOUR
    for d in deps:
        dp = ROOT / d
        if dp.exists() and dp.stat().st_mtime > o.stat().st_mtime:
            return f"{d} is newer"
    if age_h > max_age_h:
        return f"{age_h:.0f}h old (max {max_age_h:.0f}h)"
    return None


def do_cpu_work() -> Dict:
    """Run the single most-stale due task. One per tick keeps a tick short."""
    due = []
    for name, script, out, deps, age in CPU_TASKS:
        why = stale(out, deps, age)
        if why:
            o = ROOT / out
            mtime = o.stat().st_mtime if o.exists() else 0
            due.append((mtime, name, script, why))
    if not due:
        return {"ran": None, "due": 0}
    due.sort()
    _, name, script, why = due[0]
    code, out = run([PY, script], timeout=2400)
    return {"ran": name, "script": script, "why": why, "ok": code == 0,
            "due": len(due), "tail": "" if code == 0 else out[-600:]}


def training_alive() -> bool:
    code, out = run(["pgrep", "-f",
                     "gpu_cron_runner.sh|pretrain_mlm.py|train_pharos.py|"
                     "train_sequence_stages.py|train_block_scorer.py"],
                    timeout=60)
    return code == 0 and bool(out.strip())


def gpu_status() -> Dict:
    """Permitted, free, and what is queued behind the gate."""
    free = gpu_free_gib()
    permitted = GPU_PERMIT.exists()
    ready, short = [], []
    for task in GPU_TASKS:
        (ready if (free is not None and free >= task["need_gib"]) else short
         ).append(task["name"])
    return {"permitted": permitted,
            "free_gib": None if free is None else round(free, 1),
            "ready_if_permitted": ready, "short_of_memory": short,
            "training_alive": training_alive(),
            "waiting": (not permitted) or not ready}


def do_gpu_work(g: Dict) -> Dict:
    """Run one GPU task, but only behind the gate.

    Training is launched DETACHED: it runs for hours and holding it inside this
    tick would mean a 30-minute cron killing its own training run. Everything
    else is short and runs inline.
    """
    if not g["permitted"]:
        return {"ran": None, "why": "no GPU permission"}
    if g["training_alive"]:
        return {"ran": None, "why": "training already running"}
    free = g["free_gib"]
    for task in GPU_TASKS:
        if free is None or free < task["need_gib"]:
            continue
        if task["out"] and not stale(task["out"], [], task["max_age_h"]):
            continue
        if task["detach"]:
            # setsid so it outlives this tick and the cron session
            code, out = run(["setsid", "nohup", *[str(c) for c in task["cmd"]]],
                            timeout=20)
            return {"ran": task["name"], "detached": True,
                    "note": "launched in the background"}
        code, out = run([str(c) for c in task["cmd"]], timeout=2400)
        return {"ran": task["name"], "detached": False, "ok": code == 0,
                "tail": "" if code == 0 else out[-600:]}
    return {"ran": None, "why": "permitted, but nothing due"}


def run(cmd: list, timeout: int = 2400) -> tuple:
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.SubprocessError as e:
        return -1, f"{type(e).__name__}: {e}"


def git(*args: str) -> str:
    code, out = run(["git", *args], timeout=60)
    return out.strip() if code == 0 else ""


def check_claims() -> Dict:
    code, out = run([PY, "scripts/sampling/verify_claims.py"])
    n_ok = len(re.findall(r"^\s*OK ", out, re.M))
    n_fail = len(re.findall(r"^\s*FAIL", out, re.M))
    return {"passed": code == 0 and "ALL CLAIMS REPRODUCE" in out,
            "checks_ok": n_ok, "checks_failed": n_fail,
            "detail": "" if code == 0 else out[-1200:]}


def check_completeness() -> Dict:
    code, out = run([PY, "scripts/sampling/audit_completeness.py"], timeout=1800)
    got = {}
    m = re.search(r"components present: (\d+)/(\d+).*?stages: (\d+)/(\d+).*?"
                  r"scheduled: (\d+)/(\d+).*?resumable: (\d+)/(\d+).*?"
                  r"telemetry: (\d+)/(\d+)", out, re.S)
    if m:
        v = [int(x) for x in m.groups()]
        got = {"components": f"{v[0]}/{v[1]}", "stages": f"{v[2]}/{v[3]}",
               "scheduled": f"{v[4]}/{v[5]}", "resumable": f"{v[6]}/{v[7]}",
               "telemetry": f"{v[8]}/{v[9]}"}
    return {"passed": code == 0 and "MISSING: none" in out, **got}


def check_spec() -> Dict:
    if not SPEC.exists():
        return {"passed": False, "why": "specification missing"}
    text = SPEC.read_text()
    low = text.lower()
    found = [m for m in NARRATIVE_MARKERS if m in low]
    hist = sorted(p.name for p in HISTORY.glob("*.md")) if HISTORY.is_dir() else []
    return {"passed": not found and len(hist) >= 3,
            "narrative_markers": found,
            "history_files": hist,
            "spec_lines": len(text.splitlines()),
            "links_history": "history_of_failed_attempts" in text}


def check_training() -> Dict:
    out: Dict = {"state": "unknown"}
    s = ROOT / "data/samples/analysis/cron/status.json"
    if s.exists():
        try:
            out.update({k: v for k, v in json.loads(s.read_text()).items()
                        if k in ("state", "detail", "updated")})
        except (OSError, ValueError):
            pass
    ck = ROOT / "data/derived/checkpoints/pretrain_small.pt"
    if ck.exists():
        code, o = run([PY, "-c",
                       "import torch,sys;d=torch.load(sys.argv[1],map_location='cpu',"
                       "weights_only=False);print(d.get('tokens',0),d.get('step',0))",
                       str(ck)], timeout=600)
        if code == 0 and o.split():
            tok, step = o.split()[:2]
            out["stage1_tokens"], out["stage1_step"] = int(tok), int(step)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="skip verify_claims (which runs 13 test suites)")
    ap.add_argument("--no-work", action="store_true",
                    help="check only; do not run any queued CPU work")
    args = ap.parse_args()

    now = datetime.now().astimezone()
    snap: Dict = {
        "checked": now.isoformat(timespec="seconds"),
        "git_head": git("rev-parse", "--short", "HEAD"),
        "git_subject": git("log", "-1", "--format=%s"),
        # The watcher writes CHANGELOG.md, so counting its own output as
        # dirt makes `git_clean` false forever and meaningless.
        "git_clean": not [
            ln for ln in git("status", "--porcelain").splitlines()
            if ln and not ln.endswith(("architecture_watch.json",
                                       "CHANGELOG.md"))],
        "spec": check_spec(),
        "completeness": check_completeness(),
        "training": check_training(),
    }
    prev: Optional[Dict] = None
    if STATE.exists():
        try:
            prev = json.loads(STATE.read_text())
        except (OSError, ValueError):
            prev = None

    if args.quick:
        # Carry the previous result forward rather than recording "skipped".
        # Writing a null here makes the NEXT full check see None -> 330 and
        # report a change that did not happen -- which it did, twice, in the
        # first two entries of the changelog.
        snap["claims"] = dict((prev or {}).get("claims", {}), stale=True)
    else:
        snap["claims"] = check_claims()

    snap["work"] = {"skipped": True} if args.no_work else do_cpu_work()
    snap["gpu"] = gpu_status()
    snap["gpu_work"] = ({"ran": None, "why": "--no-work"} if args.no_work
                        else do_gpu_work(snap["gpu"]))

    def summary(s: Dict) -> Dict:
        return {"head": s.get("git_head"),
                "claims_ok": s.get("claims", {}).get("checks_ok"),
                **{k: v for k, v in s.get("completeness", {}).items()
                   if k != "passed"},
                "tokens": s.get("training", {}).get("stage1_tokens")}

    changed = prev is None or summary(prev) != summary(snap)
    problems = [k for k in ("spec", "completeness", "claims")
                if snap[k].get("passed") is False]

    STATE.parent.mkdir(parents=True, exist_ok=True)
    snap["changed_since_last"] = changed
    snap["problems"] = problems
    STATE.write_text(json.dumps(snap, indent=1))

    print(f"{now:%Y-%m-%d %H:%M}  head {snap['git_head']}  "
          f"claims {'skipped' if args.quick else snap['claims'].get('checks_ok')}  "
          f"{' '.join(f'{k} {v}' for k, v in snap['completeness'].items() if k != 'passed')}")
    w, g = snap["work"], snap["gpu"]
    if w.get("ran"):
        print(f"  ran: {w['ran']} ({w['why']})"
              f"{'' if w.get('ok') else '  FAILED'}")
        if not w.get("ok"):
            print(f"    {w.get('tail', '')[:300]}")
            problems.append("work")
    elif not w.get("skipped"):
        print("  CPU work queue: nothing due")

    gw = snap["gpu_work"]
    if gw.get("ran"):
        print(f"  GPU: started {gw['ran']}"
              + ("  (detached)" if gw.get("detached") else
                 "" if gw.get("ok", True) else "  FAILED"))
    elif not g["permitted"]:
        print(f"  **I am waiting for GPU permission.** "
              f"{g['free_gib']} GiB free; queued: "
              f"{', '.join(x['name'] for x in GPU_TASKS)}")
        print(f"  (grant it with: touch {GPU_PERMIT.relative_to(ROOT)})")
    elif g["training_alive"]:
        print(f"  GPU: training is already running; leaving it alone")
    else:
        print(f"  GPU permitted, {g['free_gib']} GiB free — {gw.get('why')}")

    if problems:
        print(f"  PROBLEMS: {', '.join(problems)}")
        for k in problems:
            print(f"    {k}: {json.dumps(snap.get(k, {}))[:300]}")
    if not changed:
        print("  unchanged since the last check; nothing written")
        return 1 if problems else 0

    if prev is not None:
        a, b = summary(prev), summary(snap)
        lines = [f"\n## {now:%Y-%m-%d %H:%M %Z}\n"]
        if a["head"] != b["head"]:
            lines.append(f"- **{snap['git_subject']}** (`{b['head']}`)")
        for k in ("claims_ok", "components", "stages", "scheduled", "resumable",
                  "telemetry", "tokens"):
            if a.get(k) != b.get(k) and b.get(k) is not None:
                lines.append(f"- {k}: {a.get(k)} → **{b.get(k)}**")
        if len(lines) > 1:
            head = "" if CHANGELOG.exists() else (
                "# Architecture changelog\n\nWritten by "
                "`scripts/architecture_watch.py`, which checks every 30 minutes "
                "and records only what changed. The architecture itself is in "
                "[`ARCHITECTURE.md`](ARCHITECTURE.md); what did not work is in "
                "[`history_of_failed_attempts/`](../../history_of_failed_attempts/).\n")
            with CHANGELOG.open("a") as fh:
                if head:
                    fh.write(head)
                fh.write("\n".join(lines) + "\n")
            print(f"  recorded {len(lines)-1} change(s) in {CHANGELOG.name}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
