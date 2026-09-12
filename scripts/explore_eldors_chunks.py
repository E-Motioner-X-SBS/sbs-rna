#!/usr/bin/env python3
"""Per-chunk composition profiling of elDORS: length populations + GC + N.

Outputs:
  figures/08_eldors_chunk_populations.png  (stacked length-bin bars)
  reports/eldors_chunk_profiles.csv
"""

from __future__ import annotations

import csv
import gzip
import statistics
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/store/shuvam/E-motioner-X-SBS/sbs-rna/data")
FIG = ROOT / "exploration" / "figures"
REP = ROOT / "exploration" / "reports"
N = 100_000  # sequences per chunk

BINS = [
    ("<200", 0, 200),
    ("200-500", 200, 500),
    ("500-1500", 500, 1500),
    ("1500+", 1500, 10**9),
]


def main() -> None:
    rows = []
    for cpath in sorted(
        (ROOT / "sequences" / "elDORS_v1").glob("elDORS_v1_*.fasta.gz")
    ):
        lens, ccomp, n, cur = [], Counter(), 0, 0
        with gzip.open(cpath, "rt", errors="ignore") as fh:
            for line in fh:
                if line.startswith(">"):
                    if n >= N:
                        break
                    if n > 0:
                        lens.append(cur)
                    n, cur = n + 1, 0
                else:
                    s = line.strip().upper()
                    ccomp.update(s)
                    cur += len(s)
        if n > 0:
            lens.append(cur)
        total = max(sum(ccomp.values()), 1)
        bins = {
            name: sum(1 for l in lens if lo <= l < hi) / len(lens) * 100
            for name, lo, hi in BINS
        }
        gc = (ccomp["G"] + ccomp["C"]) / total * 100
        rows.append(
            {
                "chunk": cpath.name.replace("elDORS_v1_", "").replace(".fasta.gz", ""),
                "n_sampled": len(lens),
                "median": int(statistics.median(lens)),
                "mean": round(statistics.mean(lens), 1),
                "gc_pct": round(gc, 2),
                "n_pct": round(ccomp["N"] / total * 100, 3),
                **{f"pct_{k}": round(v, 1) for k, v in bins.items()},
            }
        )
        print(
            f"{rows[-1]['chunk']}  median={rows[-1]['median']:>5}  "
            f"GC={gc:5.1f}%  <200nt={bins['<200']:5.1f}%  1500+={bins['1500+']:5.1f}%",
            flush=True,
        )

    with open(REP / "eldors_chunk_profiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # stacked bar figure
    labels = [r["chunk"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    bottom = [0.0] * len(rows)
    colors = ["#fc8181", "#f6ad55", "#68d391", "#4299e1"]
    for (name, lo, hi), col in zip(BINS, colors):
        vals = [r[f"pct_{name}"] for r in rows]
        ax1.bar(labels, vals, bottom=bottom, label=name + " nt", color=col, width=0.8)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax1.set_ylabel("% of sequences")
    ax1.set_title(
        f"elDORS chunk composition by sequence length ({N:,} sampled per chunk)"
    )
    ax1.legend(ncol=4, fontsize=8, loc="upper right")
    ax1.set_ylim(0, 105)
    ax2.plot(labels, [r["gc_pct"] for r in rows], "o-", color="#553c9a", ms=4)
    ax2.set_ylabel("GC %")
    ax2.set_xlabel("chunk")
    ax2.set_ylim(35, 70)
    fig.tight_layout()
    fig.savefig(FIG / "08_eldors_chunk_populations.png")
    print("wrote figure 08 + CSV")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.0f}s")
