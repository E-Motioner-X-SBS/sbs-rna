#!/usr/bin/env python3
"""RNA data exploration: statistics, figures, and reports.

Outputs:
  data/rna/exploration/figures/*.png
  data/rna/exploration/reports/*.md

Run:  python scripts/explore_rna_data.py [--sample-per-chunk 50000]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
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
FIG.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────── 1. SEQUENCES ───────────────────────────
def sample_eldors(per_chunk_n: int):
    """Sample sequences from each elDORS chunk: (lengths, composition, per-chunk stats)."""
    chunks = sorted((ROOT / "sequences" / "elDORS_v1").glob("elDORS_v1_*.fasta.gz"))
    all_lens: list[int] = []
    comp = Counter()
    chunk_stats = []
    for ci, cpath in enumerate(chunks, 1):
        lens: list[int] = []
        ccomp = Counter()
        n = 0
        cur = 0
        with gzip.open(cpath, "rt", errors="ignore") as fh:
            for line in fh:
                if line.startswith(">"):
                    if n >= per_chunk_n:
                        break
                    if n > 0:
                        lens.append(cur)
                    n += 1
                    cur = 0
                else:
                    s = line.strip().upper()
                    ccomp.update(s)
                    cur += len(s)
            if n > 0 and (not lens or len(lens) < n):
                lens.append(cur)
        gc = (ccomp["G"] + ccomp["C"]) / max(sum(ccomp.values()), 1) * 100
        nfrac = ccomp["N"] / max(sum(ccomp.values()), 1) * 100
        chunk_stats.append(
            {
                "chunk": cpath.name,
                "sampled": n,
                "median_len": statistics.median(lens),
                "mean_len": round(statistics.mean(lens), 1),
                "max_len": max(lens),
                "gc_pct": round(gc, 2),
                "n_pct": round(nfrac, 3),
            }
        )
        all_lens.extend(lens)
        comp.update(ccomp)
        log(f"  chunk {ci}/20 sampled ({n:,} seqs)")
    return all_lens, comp, chunk_stats


def fig_sequence_stats(lens, comp, per_chunk):
    # length distribution
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(lens, bins=120, color="#2b6cb0", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length (nt)")
    ax.set_ylabel("count (log)")
    ax.set_title(f"elDORS sequence length distribution (n={len(lens):,} sampled)")
    med = statistics.median(lens)
    ax.axvline(med, color="crimson", ls="--", lw=1, label=f"median {med:.0f} nt")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "01_eldors_length_distribution.png")
    plt.close(fig)

    # composition
    total = sum(comp.values())
    labels = ["A", "C", "G", "T", "N"]
    vals = [comp.get(x, 0) / total * 100 for x in labels]
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(labels, vals, color=["#38a169", "#2b6cb0", "#d69e2e", "#e53e3e", "#718096"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.4, f"{v:.2f}%", ha="center", fontsize=8)
    ax.set_ylabel("% of nucleotides")
    ax.set_title("elDORS nucleotide composition")
    ax.set_ylim(0, max(vals) * 1.2)
    fig.tight_layout()
    fig.savefig(FIG / "02_eldors_composition.png")
    plt.close(fig)

    # per-chunk GC + median length
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    xs = range(1, len(per_chunk) + 1)
    ax1.plot(xs, [c["gc_pct"] for c in per_chunk], "o-", color="#2b6cb0", ms=3)
    ax1.set_xlabel("chunk #")
    ax1.set_ylabel("GC %")
    ax1.set_title("GC content per chunk")
    ax2.plot(xs, [c["median_len"] for c in per_chunk], "o-", color="#d69e2e", ms=3)
    ax2.set_xlabel("chunk #")
    ax2.set_ylabel("median length (nt)")
    ax2.set_title("Median length per chunk")
    fig.tight_layout()
    fig.savefig(FIG / "03_eldors_per_chunk.png")
    plt.close(fig)


# ─────────────────────────── 2. 3D STRUCTURES ───────────────────────────
def explore_structures():
    filt = json.load(
        open(
            ROOT
            / "structures/databases/rna3db/rna3db_extracted/rna3db-jsons/filter.json"
        )
    )
    res = [
        v["resolution"]
        for v in filt.values()
        if isinstance(v.get("resolution"), (int, float))
    ]
    lens = [v["length"] for v in filt.values()]
    methods = Counter(
        (
            "cryo-EM"
            if "electron" in v.get("structure_method", "")
            else "X-ray"
            if "x-ray" in v.get("structure_method", "")
            else "other"
        )
        for v in filt.values()
    )

    # resolution histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
    ax1.hist([r for r in res if r <= 10], bins=60, color="#805ad5", alpha=0.85)
    ax1.set_xlabel("resolution (Å)")
    ax1.set_ylabel("chains")
    ax1.set_title(f"RNA3DB resolution (n={len(res):,} with resolution)")
    ax2.pie(
        methods.values(),
        labels=[f"{k}\n{v:,}" for k, v in methods.items()],
        colors=["#4299e1", "#ed8936", "#a0aec0"],
        autopct="%1.0f%%",
        textprops={"fontsize": 8},
    )
    ax2.set_title("Structure determination method")
    fig.tight_layout()
    fig.savefig(FIG / "04_structures_resolution_method.png")
    plt.close(fig)

    # length distribution (log x)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(lens, bins=120, color="#2f855a", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("chain length (nt)")
    ax.set_ylabel("chains")
    ax.set_title(f"RNA3DB chain lengths (n={len(lens):,})")
    med = statistics.median(lens)
    ax.axvline(med, color="crimson", ls="--", lw=1, label=f"median {med:.0f} nt")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "05_structures_lengths.png")
    plt.close(fig)

    # gRNAde Rfam coverage
    fams = Counter()
    with open(
        ROOT / "structures/databases/grnade_rnasolo/RNAsolo_processed_df.csv"
    ) as fh:
        for row in csv.DictReader(fh):
            try:
                for f in eval(row["rfam_list"]):
                    if f and f != "unknown":
                        fams[f] += 1
            except Exception:
                pass
    top = fams.most_common(15)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color="#2b6cb0")
    ax.set_xscale("log")
    ax.set_xlabel("clusters (log)")
    ax.set_title(f"gRNAde: top Rfam families with 3D ({len(fams)} families total)")
    fig.tight_layout()
    fig.savefig(FIG / "06_structures_rfam_coverage.png")
    plt.close(fig)

    return {
        "n_chains": len(lens),
        "n_with_res": len(res),
        "median_res": statistics.median(res) if res else None,
        "le4": sum(1 for r in res if r <= 4),
        "le2_5": sum(1 for r in res if r <= 2.5),
        "le2": sum(1 for r in res if r <= 2),
        "methods": dict(methods),
        "median_len": statistics.median(lens),
        "n_families": len(fams),
        "top_families": top[:10],
    }


# ─────────────────────────── 3. 2D BENCHMARKS ───────────────────────────
def explore_benchmarks():
    import pyarrow.parquet as pq

    b = ROOT / "benchmarks" / "secondary_structure"
    ds = {}
    for name, path in [
        ("ArchiveII", b / "archiveii/test.parquet"),
        ("bpRNA-spot train", b / "bprna_spot/train.parquet"),
        ("bpRNA-spot test", b / "bprna_spot/test.parquet"),
        ("bpRNA-new", b / "bprna_new/test.parquet"),
    ]:
        if path.exists():
            t = pq.read_table(path)
            lens = [len(s.as_py()) for s in t["sequence"]]
            ds[name] = {
                "n": t.num_rows,
                "median": statistics.median(lens),
                "mean": round(statistics.mean(lens), 1),
                "max": max(lens),
            }
    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = list(ds)
    meds = [ds[n]["median"] for n in names]
    maxs = [ds[n]["max"] for n in names]
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], meds, width=0.4, label="median nt", color="#2b6cb0")
    ax.bar([i + 0.2 for i in x], maxs, width=0.4, label="max nt", color="#ed8936")
    ax.set_yscale("log")
    ax.set_xticks(list(x), names, rotation=15)
    ax.legend()
    ax.set_title("2D structure benchmarks: sequence lengths")
    fig.tight_layout()
    fig.savefig(FIG / "07_benchmarks_2d_lengths.png")
    plt.close(fig)

    # RNAStrAlign + bpRNA-1m from json
    for name, path in [
        ("RNAStrAlign", b / "rnastralign/data.json"),
        ("bpRNA-1m", b / "bprna_full/data.json"),
    ]:
        if path.exists():
            d = json.load(open(path))
            ds[name] = {"n": len(d)}
    return ds


# ─────────────────────────── MAIN ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-per-chunk", type=int, default=50000)
    args = ap.parse_args()

    log("=== 1/3 elDORS sampling ===")
    lens, comp, per_chunk = sample_eldors(args.sample_per_chunk)
    log(f"sampled {len(lens):,} sequences total")
    fig_sequence_stats(lens, comp, per_chunk)

    log("=== 2/3 structures ===")
    s3d = explore_structures()

    log("=== 3/3 benchmarks ===")
    b2d = explore_benchmarks()

    # report
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eldors_sample": {
            "sequences_sampled": len(lens),
            "median_len": statistics.median(lens),
            "mean_len": round(statistics.mean(lens), 1),
            "min_len": min(lens),
            "max_len": max(lens),
            "composition_pct": {
                k: round(v / sum(comp.values()) * 100, 3) for k, v in comp.most_common()
            },
            "per_chunk": per_chunk,
        },
        "structures": s3d,
        "benchmarks_2d": b2d,
    }
    (REP / "exploration_stats.json").write_text(json.dumps(report, indent=2))
    log("wrote exploration_stats.json")
    log(f"figures: {len(list(FIG.glob('*.png')))}")


if __name__ == "__main__":
    main()
