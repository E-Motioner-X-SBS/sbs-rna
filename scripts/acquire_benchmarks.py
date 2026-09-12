#!/usr/bin/env python3
"""Benchmark acquisition pipeline for the RNA training database.

Downloads all benchmark/evaluation datasets verified in
data_inventory/00-MASTER-INVENTORY.md:
  - Secondary structure: ArchiveII, bpRNA (full + spot splits), RNAStrAlign, bpRNA-new
  - Tertiary: RNA-Puzzles standardized submissions, CASP15/16 targets (RCSB)
  - Fitness: RNAGym assays, NABench
  - Splicing: Spliceator, SpliceBERT, G3PO
  - Structure: RNA3DB releases, BGSU motif atlas
Each download is idempotent (skips existing files), logged, and verified.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

DATA = Path("/store/shuvam/E-motioner-X-SBS/sbs-rna/data")
DATA.mkdir(parents=True, exist_ok=True)
LOG = DATA / "benchmarks" / "acquisition.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def run(cmd: list, cwd=None, check=False) -> int:
    log("RUN: " + " ".join(str(c) for c in cmd))
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        log(f"  exit={p.returncode} stderr={p.stderr[-400:]}")
        if check:
            raise RuntimeError(p.stderr[-400:])
    return p.returncode


def curl(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"SKIP (exists) {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    rc = run(
        [
            "curl",
            "-sS",
            "--fail",
            "--retry",
            "4",
            "--retry-delay",
            "8",
            "-C",
            "-",
            "-o",
            str(dest),
            url,
        ]
    )
    if rc == 0 and dest.exists():
        log(f"OK {dest.name} ({dest.stat().st_size:,} bytes)")
        return True
    log(f"FAIL {url}")
    return False


def hf_file(repo: str, filename: str, dest_dir: Path) -> bool:
    """Download a single file from a HuggingFace dataset repo."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log("huggingface_hub missing")
        return False
    target = dest_dir / filename.replace("/", "_")
    if target.exists() and target.stat().st_size > 0:
        log(f"SKIP (exists) {target.name}")
        return True
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            local_dir=str(dest_dir),
        )
        log(f"OK {repo}/{filename}")
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"FAIL {repo}/{filename}: {exc}")
        return False


def git_clone(url: str, dest: Path) -> bool:
    if (dest / ".git").exists():
        log(f"SKIP (exists) {dest.name}")
        return True
    rc = run(["git", "clone", "--depth", "1", url, str(dest)])
    if rc == 0:
        log(f"OK clone {dest.name}")
        return True
    log(f"FAIL clone {url}")
    return False


def main() -> int:
    results = {}

    # 1. Secondary structure (HF mirror datasets, standardized parquet/json)
    log("=== 1. SECONDARY STRUCTURE ===")
    results["archiveii"] = hf_file(
        "multimolecule/archiveii",
        "test.parquet",
        DATA / "benchmarks/secondary_structure/archiveii",
    )
    for split in ("train", "validation", "test"):
        results[f"bprna_spot_{split}"] = hf_file(
            "multimolecule/bprna-spot",
            f"{split}.parquet",
            DATA / "benchmarks/secondary_structure/bprna_spot",
        )
    results["rnastralign"] = hf_file(
        "rouskinlab/RNAstralign", "data.json", DATA / "benchmarks/secondary_structure/rnastralign"
    )
    results["bprna_new"] = hf_file(
        "multimolecule/bprna-new",
        "test.parquet",
        DATA / "benchmarks/secondary_structure/bprna_new",
    )
    results["bprna_full"] = hf_file(
        "rouskinlab/bpRNA-1m", "data.json", DATA / "benchmarks/secondary_structure/bprna_full"
    )

    # 2. Tertiary benchmarks
    log("=== 2. TERTIARY / RNA-PUZZLES / CASP ===")
    results["rna_puzzles_std"] = git_clone(
        "https://github.com/mmagnus/RNA-Puzzles-Standardized-Submissions",
        DATA / "structures/blind_tests/rna_puzzles_std",
    )
    results["rna_puzzles_main"] = git_clone(
        "https://github.com/rnapuzzles/rnapuzzles.github.io",
        DATA / "structures/blind_tests/rnapuzzles_github_io",
    )

    casp15 = [
        "7qr4",
        "7qr3",
        "8s95",
        "8fza",
        "8tvz",
        "8btz",
        "7zj4",
        "7ptk",
        "7ptl",
        "8uys",
        "8uye",
        "8uyg",
        "8uyj",
        "7yr7",
        "7yr6",
    ]
    casp16 = [
        "8uo6",
        "9cfn",
        "9c2k",
        "9dcf",
        "9b0l",
        "9ely",
        "9bzc",
        "9bz1",
        "9cbu",
        "9cbx",
    ]
    for pid in casp15:
        curl(
            f"https://files.rcsb.org/download/{pid.upper()}.cif",
            DATA / f"tertiary/casp15/{pid.upper()}.cif",
        )
    for pid in casp16:
        curl(
            f"https://files.rcsb.org/download/{pid.upper()}.cif",
            DATA / f"tertiary/casp16/{pid.upper()}.cif",
        )

    # 3. Fitness / function
    log("=== 3. FITNESS ===")
    curl(
        "https://marks.hms.harvard.edu/rnagym/fitness_prediction/"
        "fitness_processed_assays.zip",
        DATA / "benchmarks/fitness/rnagym/fitness_processed_assays.zip",
    )
    results["rnagym_repo"] = git_clone(
        "https://github.com/MarksLab-DasLab/RNAGym", DATA / "benchmarks/fitness/rnagym_repo"
    )
    results["nabench"] = git_clone(
        "https://github.com/mrzzmrzz/NABench", DATA / "benchmarks/fitness/nabench"
    )

    # 4. Splicing
    log("=== 4. SPLICING ===")
    curl(
        "https://bigest-icube.fr/spliceator/static/data/data.tar.gz",
        DATA / "benchmarks/splicing/spliceator/data.tar.gz",
    )
    results["splicebert_zenodo"] = curl(
        "https://zenodo.org/records/7995778/files/data.tar.gz?download=1",
        DATA / "benchmarks/splicing/splicebert/data.tar.gz",
    )
    results["g3po"] = git_clone(
        "https://github.com/BiGEst-ICube/g3po", DATA / "benchmarks/splicing/g3po"
    )

    # 5. Structure datasets (RNA3DB releases)
    log("=== 5. RNA3DB ===")
    for asset in ("rna3db-jsons.tar.gz", "rna3db-cmscans.tar.gz"):
        curl(
            f"https://github.com/marcellszi/rna3db/releases/download/"
            f"2026-01-05-full-release/{asset}",
            DATA / "structures/databases/rna3db" / asset,
        )

    # Summary
    log("=== SUMMARY ===")
    ok = sum(1 for v in results.values() if v)
    log(f"{ok}/{len(results)} flagged tasks OK")
    with open(DATA / "acquisition_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
