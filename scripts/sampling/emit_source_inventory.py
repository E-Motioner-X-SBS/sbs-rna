#!/usr/bin/env python3
"""Emit `data/catalog/SOURCES.md`: every source, its link, size and destination.

The catalog README describes the corpus in prose and contains **not one URL**.
Every link lived inside `scripts/acquire_all.py`, which is executable and
therefore authoritative, and unreadable as documentation. If the acquisition
has to be repeated on another machine -- which is exactly how this corpus got
here -- the answer to "where did this come from" should not be "read a 569-line
downloader".

So the document is GENERATED from `build_jobs()` rather than written beside it.
It cannot drift: add a job and the table grows, change a URL and the table
changes. What it cannot show is a `cmd` job's remote endpoint, because those
delegate to another script; those rows name the script instead.

Usage: python3 scripts/sampling/emit_source_inventory.py
"""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/catalog/SOURCES.md"


def load_jobs():
    spec = importlib.util.spec_from_file_location(
        "acquire_all", ROOT / "scripts/acquire_all.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["acquire_all"] = mod
    spec.loader.exec_module(mod)
    return mod.build_jobs(), mod


def human(n) -> str:
    if not n:
        return "—"
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= div:
            return f"{n/div:.2f} {unit}"
    return f"{n} B"


def main() -> int:
    jobs, mod = load_jobs()
    by_group = defaultdict(list)
    for j in jobs:
        by_group[j.group].append(j)

    total = sum(j.expected or 0 for j in jobs)
    known = sum(1 for j in jobs if j.expected)

    L: list[str] = []
    L.append("# Sources — every link, every program\n")
    L.append("**Generated** by `scripts/sampling/emit_source_inventory.py` from "
             "`build_jobs()` in `scripts/acquire_all.py`. Do not edit by hand: "
             "regenerate it, so the documentation cannot drift from the "
             "downloader that is actually used.\n")
    L.append(f"{len(jobs)} jobs in {len(by_group)} groups. "
             f"{known} carry a verified byte size, totalling "
             f"**{human(total)}**; the rest are directory or repository "
             f"fetches whose size is not known before the transfer.\n")
    L.append("## How to fetch\n")
    L.append("```bash\n"
             "uv run python scripts/acquire_all.py --list            # the plan\n"
             "uv run python scripts/acquire_all.py --group sequence  # ~1.3B seqs\n"
             "uv run python scripts/acquire_all.py --group catalog   # everything else\n"
             "uv run python scripts/acquire_all.py --only mars       # the 427 GB D18 skips\n"
             "```\n")
    L.append("> **On this machine `--list` under-reports.** The corpus was "
             "downloaded on another host and only `data/` was copied across, "
             "not `data/acquisition/`, so jobs whose completion is recorded by "
             "a state marker read as pending while their output is on disk. "
             "`scripts/sampling/audit_inventory_gap.py` measures the tree as it "
             "is and is the authority: 17 of 18 sources present, 0 missing, 1 "
             "absent by decision.\n")

    for group in ("sequence", "catalog", "long"):
        js = by_group.get(group, [])
        if not js:
            continue
        gtot = sum(j.expected or 0 for j in js)
        L.append(f"## `{group}` — {len(js)} jobs, {human(gtot)} known\n")
        L.append("| job | host | size | source | lands in |")
        L.append("|---|---|---|---|---|")
        for j in sorted(js, key=lambda x: x.name):
            if j.kind == "cmd":
                src = "`" + " ".join(
                    str(c) for c in j.cmd if "python" not in str(c)
                    and "uv" not in str(c)) + "`"
            elif j.url:
                src = f"<{j.url}>"
            else:
                src = "—"
            dest = (str(j.dest.relative_to(ROOT)) if j.dest
                    and str(j.dest).startswith(str(ROOT)) else str(j.dest or "—"))
            L.append(f"| `{j.name}` | {j.server or '—'} | {human(j.expected)} "
                     f"| {src} | `{dest}` |")
        L.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"{len(jobs)} jobs, {known} with verified sizes, {human(total)} total")
    for g, js in sorted(by_group.items()):
        print(f"  {g:10s} {len(js):4d} jobs  "
              f"{human(sum(j.expected or 0 for j in js))}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
