#!/usr/bin/env python3
"""Is every catalogued dataset actually on disk, at the size it was recorded at?

`data/catalog/MANIFEST.json` lists 25 sources and every file in them with a
byte count taken at acquisition. That is a record of what *was* downloaded; this
checks what is *there* now. The two can diverge quietly -- a move, a partial
re-download, a cleanup that took more than it meant to -- and a corpus that is
missing a source fails at training time rather than at audit time.

Three states per file, and the distinction matters:

    present   size matches the manifest exactly
    RESIZED   present but a different size -- worse than missing, because it
              will load and be wrong
    MISSING   not on disk

Usage:
    /store/shuvam/.venv/bin/python scripts/sampling/audit_data_presence.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/catalog/MANIFEST.json"
OUT = ROOT / "data/samples/analysis/data_presence.json"


def main() -> None:
    man = json.loads(MANIFEST.read_text())
    base = ROOT / "data"
    rows: List[Dict] = []
    for src in man["sources"]:
        n_ok = n_miss = n_resize = 0
        bytes_ok = bytes_missing = 0
        missing: List[str] = []
        resized: List[str] = []
        for f in src.get("files", []):
            p = base / f["path"]
            want = int(f.get("size") or 0)
            if not p.exists():
                n_miss += 1
                bytes_missing += want
                if len(missing) < 4:
                    missing.append(f["path"])
                continue
            got = p.stat().st_size
            if want and got != want:
                n_resize += 1
                if len(resized) < 4:
                    resized.append(f"{f['path']} ({got:,} vs {want:,})")
            else:
                n_ok += 1
                bytes_ok += got
        rows.append({
            "source": src["name"], "n_files": len(src.get("files", [])),
            "present": n_ok, "resized": n_resize, "missing": n_miss,
            "gib_present": round(bytes_ok / 2**30, 2),
            "gib_missing": round(bytes_missing / 2**30, 2),
            "missing_examples": missing, "resized_examples": resized,
            "complete": n_miss == 0 and n_resize == 0,
        })

    rows.sort(key=lambda r: (-r["missing"], -r["resized"], r["source"]))
    print(f"{'source':26s} {'files':>6s} {'ok':>6s} {'resized':>8s} {'missing':>8s} {'GiB':>9s}")
    for r in rows:
        flag = "" if r["complete"] else "   <--"
        print(f"{r['source']:26s} {r['n_files']:6d} {r['present']:6d} "
              f"{r['resized']:8d} {r['missing']:8d} {r['gib_present']:9.2f}{flag}")

    n_src = len(rows)
    ok_src = sum(1 for r in rows if r["complete"])
    res = {
        "n_sources": n_src, "n_sources_complete": ok_src,
        "n_files": sum(r["n_files"] for r in rows),
        "n_present": sum(r["present"] for r in rows),
        "n_resized": sum(r["resized"] for r in rows),
        "n_missing": sum(r["missing"] for r in rows),
        "gib_present": round(sum(r["gib_present"] for r in rows), 2),
        "gib_missing": round(sum(r["gib_missing"] for r in rows), 2),
        "sources": rows,
    }
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nsources complete: {ok_src}/{n_src}   "
          f"files present {res['n_present']:,}/{res['n_files']:,}   "
          f"{res['gib_present']:.1f} GiB")
    if res["n_missing"] or res["n_resized"]:
        print(f"MISSING {res['n_missing']} files ({res['gib_missing']:.1f} GiB), "
              f"RESIZED {res['n_resized']}")
        for r in rows:
            if not r["complete"]:
                print(f"  {r['source']}: missing {r['missing_examples']} "
                      f"resized {r['resized_examples']}")
    else:
        print("every catalogued file is present at its recorded size")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
