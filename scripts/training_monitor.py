#!/usr/bin/env python3
"""Is stage 1 actually training, and is what it reports still meaningful?

Not a dashboard. Every section below ASSERTS something and says so when the
assertion fails, because this project's recurring defect is not a number that
drifts -- it is a number that is computed, logged, and never checked against
the range it is allowed to take. Seventeen findings in a week came from that,
including a scheduled job that scored a model on chains it had trained on and
a budget that sat at half size for 3,700 steps while a flag said it should
not.

So the report leads with VERDICT, and a verdict is one of:

  OK     the invariant holds
  WARN   worth a human's attention, not worth stopping for
  ALERT  something is wrong NOW; exit status is non-zero

Checks, and why each exists rather than being obvious:

  alive        A process at 100% GPU utilisation can be wedged. The test is
               that the STEP COUNTER moved since the last report, not that
               something is running.
  resumable    A run that stops checkpointing is a run whose last hours are
               gone when the card is taken. Checkpoint mtime must advance.
  throughput   The trainer's own tok/s column is cumulative since process
               start, so it reads low for an hour after a resume and tells
               you nothing. This computes the INTERVAL rate between rows.
  progress     Training loss tracks which shard is streaming and is not a
               progress signal -- it swung 1.488 -> 1.863 while the model
               improved monotonically. The held-out sample is the signal.
  one series   Held-out readings are comparable only within one reserved
               shard set, one device and one sample size. The corpus grows,
               and when it does `heldout_files` silently reserves different
               shards. A spliced curve looks like progress or regression that
               never happened.
  router       `dead` and routing width are the only things that separate a
               specialising MoE from a dense model paying MoE's memory bill;
               balance alone cannot, because the mean is uniform either way.
  budget       After an OOM the budget must WALK BACK, not freeze. It froze
               for 3,700 steps once because the recovery branch was
               unreachable code.
  headroom     Disk, GPU and RAM, because the run dies quietly on all three.

Run:  python3 scripts/training_monitor.py            # report, exit 0/1/2
      python3 scripts/training_monitor.py --quiet    # only WARN/ALERT lines
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data/samples/analysis/runs"
CRON = ROOT / "data/samples/analysis/cron"
STATE = CRON / "monitor_state.json"
CKPT = ROOT / "data/derived/checkpoints/pretrain_shared400.pt"

#: RNA sequence entropy. A model above this has learned nothing a lookup
#: table of base frequencies does not already know.
UNIGRAM_BITS = 2.0165
#: shared400. Width outside [1, N_EXPERTS] is not a width.
N_EXPERTS = 512
#: `balance_weight` 0.01 x 18 blocks, the value at perfect uniformity.
BALANCE_FLOOR = 0.18
#: Peak during an atomic checkpoint save is about 2x the 2.9 GB file.
DISK_MIN_GB = 15.0
#: Minutes that must have passed before "it has not moved" means anything.
#:
#: The trainer logs a step row every `--log-every` 100 steps, which is about
#: 12 minutes. Two reports closer together than that see the same last row
#: and conclude the run is wedged. Caught on this monitor's own first test:
#: two invocations 23 seconds apart produced `step counter has NOT moved ...
#: after 0 min`, an ALERT about a healthy run. A check that does not test its
#: own precondition is the defect this file was written to find, so it is not
#: shipping with one.
MIN_ELAPSED_MIN = 20.0

rows: list = []


def say(verdict: str, name: str, detail: str = "") -> None:
    rows.append((verdict, name, detail))


def load_csv(p: Path) -> list:
    if not p.exists():
        return []
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def f(x, default=None):
    """float(x), or `default` -- and NaN counts as unreadable, not as a number.

    `float("nan")` parses without complaint, and every comparison against it
    is False. So a diverged run writing `nan` into the held-out CSV read as
    "not worse than unigram", skipped the no-progress check (whose deltas are
    also NaN, also False), and was reported OK -- forever. The monitor whose
    whole job is to notice a run going wrong was structurally unable to
    notice the single most obvious way a run goes wrong.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true",
                    help="print only WARN and ALERT lines")
    args = ap.parse_args()

    prev = {}
    if STATE.exists():
        try:
            prev = json.loads(STATE.read_text())
        except (OSError, ValueError):
            prev = {}
    now = time.time()

    # ---- held? -----------------------------------------------------------
    held = (CRON / "HOLD").exists()

    # ---- alive, and MOVING ----------------------------------------------
    steps = [r for r in load_csv(RUNS / "stage1_mlm.csv") if r.get("kind") == "step"]
    cur_step = int(steps[-1]["step"]) if steps else None
    cur_tok = int(steps[-1]["tokens"]) if steps else None
    pid = subprocess.run(["pgrep", "-f", "scripts/pretrain_mlm"],
                         capture_output=True, text=True).stdout.split()
    running = bool(pid)

    # Every "how long since X" check below measures time since TRAINING last
    # did something. While the run is deliberately held they all keep ticking
    # and all eventually fire, so a clean pause reported one ALERT and a WARN
    # about a checkpoint and a token budget that are exactly where they were
    # left. An alarm that goes off because you turned the machine off is an
    # alarm people learn to ignore, which is the only way this monitor fails.
    paused = held and not running
    if paused:
        say("OK", "scheduler", "HOLD is set and nothing is training (deliberate)")
    elif not running:
        say("ALERT", "trainer is NOT running",
            "and HOLD is not set, so the cron should have restarted it")
    else:
        last = prev.get("step")
        mins = (now - prev.get("at", now)) / 60
        if last is None:
            say("OK", "trainer running", f"step {cur_step:,} (first report)")
        elif cur_step is None or cur_step <= last:
            if mins < MIN_ELAPSED_MIN:
                say("OK", "trainer running",
                    f"step {cur_step:,}; only {mins:.0f} min since the last "
                    f"report, under the {MIN_ELAPSED_MIN:.0f} min a step row "
                    f"takes -- too soon to judge")
            else:
                say("ALERT", "step counter has NOT moved",
                    f"still {last:,} after {mins:.0f} min -- the process is up "
                    f"at 100% GPU but may be wedged")
        else:
            d = cur_step - last
            say("OK", "trainer advancing",
                f"step {cur_step:,} (+{d:,} in {mins:.0f} min)")

    # ---- resumable -------------------------------------------------------
    if CKPT.exists():
        age = (now - CKPT.stat().st_mtime) / 60
        if paused:
            say("OK", "checkpoint freshness",
                f"written {age:.0f} min ago, at the pause -- nothing has run "
                f"since, so nothing is at risk")
        else:
            v = "OK" if age < 60 else ("WARN" if age < 180 else "ALERT")
            say(v, "checkpoint freshness",
                f"written {age:.0f} min ago"
                + ("" if v == "OK" else
                   " -- everything since is lost if the card goes"))
    elif not held:
        say("ALERT", "no checkpoint", str(CKPT))

    # ---- throughput, measured over the INTERVAL --------------------------
    if len(steps) >= 2:
        a, b = steps[-2], steps[-1]
        dt = f(b["elapsed_s"], 0) - f(a["elapsed_s"], 0)
        dtok = int(b["tokens"]) - int(a["tokens"])
        if dt > 0 and dtok > 0:
            rate = dtok / dt
            mfu = f(b.get("mfu"), 0) * 100
            v = "OK" if rate >= 10000 else ("WARN" if rate >= 5000 else "ALERT")
            say(v, "throughput",
                f"{rate:,.0f} tok/s interval, MFU {mfu:.1f}% "
                f"(run 2 steady state 15,900)")

    # ---- progress: the held-out sample, not the training loss ------------
    ho = load_csv(RUNS / "heldout_stratified.csv")
    if not ho:
        say("WARN", "no held-out readings", "the watcher may not be running")
    else:
        ids = {r.get("heldout_id") for r in ho}
        cfgs = {(r.get("device"), r.get("n_seq")) for r in ho}
        if len(ids) != 1 or None in ids or "" in ids:
            say("ALERT", "held-out curve splices samples",
                f"heldout_id values: {sorted(str(i) for i in ids)}")
        elif len(cfgs) != 1:
            say("ALERT", "held-out curve mixes configurations",
                f"{sorted(f'{d}/n={n}' for d, n in cfgs)}")
        else:
            say("OK", "held-out is one series",
                f"{len(ho)} readings, {ids.pop()}, "
                f"{'/'.join(str(x) for x in cfgs.pop())}")

        last = ho[-1]
        bits = f(last["bits"])
        if bits is None:
            say("ALERT", "held-out bits unreadable", str(last.get("bits")))
        elif bits >= UNIGRAM_BITS:
            say("ALERT", "model is no better than base frequencies",
                f"{bits:.5f} bits vs {UNIGRAM_BITS} unigram")
        else:
            gain = 100 * (UNIGRAM_BITS - bits) / UNIGRAM_BITS
            say("OK", "held-out",
                f"{bits:.5f} bits, acc {f(last['accuracy']):.5f} at step "
                f"{int(last['step']):,} ({gain:.1f}% better than unigram)")

        if len(ho) >= 4:
            recent = [f(r["bits"]) for r in ho[-4:]]
            if recent[-1] > recent[0]:
                say("WARN", "held-out has risen over the last 4 readings",
                    " -> ".join(f"{b:.5f}" for b in recent))
            else:
                say("OK", "held-out trending down",
                    " -> ".join(f"{b:.5f}" for b in recent))

        # is the watcher keeping up with the trainer?
        if cur_step and ho:
            behind = cur_step - int(ho[-1]["step"])
            if behind > 2500:
                say("WARN", "held-out watcher is behind",
                    f"last reading at step {int(ho[-1]['step']):,}, "
                    f"trainer at {cur_step:,} ({behind:,} steps)")

    # ---- router ----------------------------------------------------------
    if steps:
        s = steps[-1]
        dead = f(s.get("dead_expert_frac"))
        wmean = f(s.get("route_width_mean"))
        wmax = f(s.get("route_width_max"))
        bal = f(s.get("balance"))
        if dead is None:
            say("WARN", "router telemetry absent",
                "this run predates the dead/width columns")
        else:
            if dead > 0.05:
                say("ALERT", "experts are dying", f"{100*dead:.1f}% below "
                    f"10% of a uniform share")
            else:
                say("OK", "no dead experts", f"{100*dead:.1f}%")
            if wmean is not None and not (1.0 <= wmean <= N_EXPERTS):
                say("ALERT", "routing width is not a width",
                    f"mean {wmean} outside [1, {N_EXPERTS}]")
            elif wmean is not None:
                v = "WARN" if wmean > 0.5 * N_EXPERTS else "OK"
                say(v, "routing width",
                    f"{wmean:.1f} / {wmax:.0f} of {N_EXPERTS}"
                    + (" -- this is a dense model paying MoE's bill"
                       if v == "WARN" else ""))
        if bal is not None:
            if bal < BALANCE_FLOOR - 1e-6:
                say("ALERT", "balance below its analytic floor",
                    f"{bal:.4f} < {BALANCE_FLOOR}")
            else:
                say("OK", "load balance",
                    f"{bal:.4f} ({bal/BALANCE_FLOOR:.2f}x the {BALANCE_FLOOR} floor)")

    # ---- budget: after an OOM it must walk back, not freeze --------------
    if steps:
        s = steps[-1]
        budget = int(s["token_budget"]) if s.get("token_budget") else None
        n_oom = int(f(s.get("n_oom"), 0) or 0)
        if n_oom == 0:
            say("OK", "no OOMs this run", f"budget {budget:,}")
        else:
            pb, pn = prev.get("budget"), prev.get("n_oom")
            mins = (now - prev.get("at", now)) / 60
            at_ceiling = budget is not None and budget >= 162201
            if at_ceiling:
                say("OK", f"budget after {n_oom} OOM(s)",
                    f"{budget:,} -- recovered to the ceiling")
            elif paused:
                say("OK", f"budget after {n_oom} OOM(s)",
                    f"{budget:,} at the pause -- it only moves while training "
                    f"runs, so standing still here means nothing")
            elif (pb is not None and pn == n_oom and budget == pb
                    and mins >= MIN_ELAPSED_MIN):
                say("WARN", "budget has not moved since the last report",
                    f"{budget:,} after {n_oom} OOM(s), {mins:.0f} min "
                    f"-- recovery should step it up toward 162,201")
            else:
                say("OK", f"budget after {n_oom} OOM(s)",
                    f"{budget:,}" + (f" (was {pb:,})" if pb else ""))

    # ---- headroom --------------------------------------------------------
    du = shutil.disk_usage(ROOT)
    free_gb = du.free / 1e9
    v = "OK" if free_gb > DISK_MIN_GB * 4 else ("WARN" if free_gb > DISK_MIN_GB else "ALERT")
    say(v, "disk", f"{free_gb:.0f} GB free"
        + ("" if v == "OK" else f" -- a checkpoint needs ~{DISK_MIN_GB:.0f} GB to save"))

    g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    if g.returncode == 0 and g.stdout.strip():
        used, total = (int(x) for x in g.stdout.strip().splitlines()[0].split(","))
        say("OK", "gpu", f"{used/1024:.1f} of {total/1024:.1f} GiB")

    # ---- report ----------------------------------------------------------
    worst = 0
    print(f"=== PHAROS stage 1 === {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    if cur_tok:
        print(f"    {cur_tok/1e9:.3f}B of 8.000B tokens ({100*cur_tok/8e9:.1f}%)"
              f"  step {cur_step:,}")
    for verdict, name, detail in rows:
        worst = max(worst, {"OK": 0, "WARN": 1, "ALERT": 2}[verdict])
        if args.quiet and verdict == "OK":
            continue
        print(f"  {verdict:5s} {name:38s} {detail}")
    n_alert = sum(1 for v, _, _ in rows if v == "ALERT")
    n_warn = sum(1 for v, _, _ in rows if v == "WARN")
    print(f"    {len(rows)} checks: {n_alert} ALERT, {n_warn} WARN")

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(
        {"at": now, "step": cur_step, "tokens": cur_tok,
         "budget": int(steps[-1]["token_budget"]) if steps and steps[-1].get("token_budget") else None,
         "n_oom": int(f(steps[-1].get("n_oom"), 0) or 0) if steps else None,
         "alerts": n_alert, "warns": n_warn}, indent=1))
    return worst


if __name__ == "__main__":
    sys.exit(main())
