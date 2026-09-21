#!/usr/bin/env python3
"""Every 30 minutes: is the architecture still consistent with itself?

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

Usage:
    python3 scripts/architecture_watch.py            # check, record if changed
    python3 scripts/architecture_watch.py --quick    # skip verify_claims
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
    if problems:
        print(f"  PROBLEMS: {', '.join(problems)}")
        for k in problems:
            print(f"    {k}: {json.dumps(snap[k])[:300]}")
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
