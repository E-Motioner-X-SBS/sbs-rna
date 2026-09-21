#!/usr/bin/env python3
"""Content-level audit of the downloaded corpora.

Archive integrity was checked by verify_downloads.py; this checks what
the files *contain*: record counts, family counts, construct counts,
annotation coverage.  Each check reports measured vs documented.

    uv run python scripts/audit_data.py                 # everything
    uv run python scripts/audit_data.py --quick         # skip eldors/mars
    uv run python scripts/audit_data.py --only rmdb
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HUNTER = ROOT.parent / "pdb_hunter"
OUT = DATA / "acquisition" / "data_audit.json"

EXPECTED = {
    "eldors_sequences": 1_323_715_880,
    "mars_sequences": 1_730_000_000,
    "rnacentral_active": 46_210_324,
    "seed_families": 4_227,
    "constructs": 1_024,
    "reactivity_values": 520_709_190,
    "bprna_1m_dbn_zip": 102_318,
    "rna3db_chains": 15_441,
    "grnade_pdbs": 14_369,
    "casp15": 15,
    "casp16": 10,
    "raw_pdb_entries": 10_423,
    "pdb_hunter_entries": 10_452,
    "archiveii_parquet": 3_975,
    "rnastralign_parquet": 37_149,
    "bprna_new_parquet": 5_401,
    "hunter_dbn_entries": 9_347,
    "hunter_bb_xyz": 10_280,
}


def count_fasta(path: Path) -> int:
    if path.suffix == ".gz":
        op = lambda: gzip.open(path, "rt", errors="ignore")       # noqa: E731
    else:
        op = lambda: open(path, errors="ignore")                  # noqa: E731
    n = 0
    with op() as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
    return n


def count_fasta_zcat(path: Path) -> int:
    r = subprocess.run(f"zcat {path} | grep -c '^>'", shell=True,
                       capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def count_mars_part(path: Path) -> int:
    """Count FASTA headers in one MARS .tgz part.

    `tar -xzO` strips tar headers and concatenates member contents, so
    `grep -c '^>'` counts records at C speed (Python line iteration over
    1.5 TB of FASTA is the bottleneck otherwise).
    """
    r = subprocess.run(f"tar -xzOf {path} | grep -c '^>'", shell=True,
                       capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def count_tar_members(path: Path, suffix: str, mode: str = "r|gz") -> int:
    n = 0
    with tarfile.open(path, mode) as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(suffix):
                n += 1
    return n


def count_zip_members(path: Path, suffix: str = "") -> int:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(suffix)]
        return len(names)


def parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(str(path)).metadata.num_rows


# ── audits ──────────────────────────────────────────────────────────────

def audit_eldors() -> dict:
    files = sorted((DATA / "sequences/elDORS_v1").glob("elDORS_v1_0*.fasta.gz"))
    per = {}
    with ThreadPoolExecutor(max_workers=20) as pool:
        futs = {pool.submit(count_fasta_zcat, f): f.name for f in files}
        for fut in as_completed(futs):
            per[futs[fut]] = fut.result()
    return {"eldors_files": len(files), "eldors_sequences": sum(per.values()),
            "per_chunk": per}


def audit_mars() -> dict:
    files = sorted((DATA / "sequences/mars").glob("OMIX003037-*.tgz"))
    per = {}
    with ThreadPoolExecutor(max_workers=15) as pool:
        futs = {pool.submit(count_mars_part, f): f.name for f in files}
        for fut in as_completed(futs):
            per[futs[fut]] = fut.result()
    return {"mars_parts": len(files), "mars_sequences": sum(per.values()),
            "per_part": per}


def audit_rmdb() -> dict:
    rdat = sorted((DATA / "benchmarks/probing/rmdb/rdat").glob("*.rdat"))
    rows = 0
    values = 0
    titr = []
    legacy = 0
    for p in rdat:
        with open(p, errors="ignore") as fh:
            for line in fh:
                if line.startswith("REACTIVITY:"):
                    rows += 1
                    values += line.count("\t") - 1
                elif line.startswith("DATA:") or \
                        line.startswith("DATA_ANNOTATION:"):
                    legacy += 1
                    values += line.count("\t") - 1
        if "_MGTI_" in p.name or "_MGPH_" in p.name or "_MG50_" in p.name:
            titr.append(p.name)
    return {"constructs": len(rdat), "reactivity_rows": rows,
            "legacy_data_rows": legacy, "reactivity_values": values,
            "mg_titrations": len(titr),
            "mg_titration_examples": sorted(titr)[:5]}


def audit_rnacentral() -> dict:
    out = {}
    for f in ["rnacentral_active", "rnacentral_inactive",
              "rnacentral_species_specific_ids"]:
        p = DATA / "sequences/rnacentral" / f"{f}.fasta.gz"
        out[f] = count_fasta_zcat(p)
    return out


def audit_rfam() -> dict:
    seed = DATA / "families/rfam/Rfam.seed.gz"
    fams = 0
    with gzip.open(seed, "rt", errors="ignore") as fh:
        for line in fh:
            if line.startswith("//"):
                fams += 1
    full = list((DATA / "families/rfam/full_alignments").glob("*.sto"))
    return {"seed_families": fams, "full_alignments": len(full)}


def audit_raw_pdb() -> dict:
    files = list((DATA / "structures/raw_pdb_entries").glob("*.cif.gz"))
    return {"raw_pdb_entries": len(files),
            "raw_pdb_bytes": sum(p.stat().st_size for p in files)}


def audit_pdb_hunter() -> dict:
    root = HUNTER / "RNA_Database"
    entries = [d for d in root.iterdir() if d.is_dir()]
    stats = {"entries": len(entries), "complete_meta": 0,
             "hunter_dbn_entries": 0, "hunter_bpseq_entries": 0,
             "hunter_bb_xyz": 0, "hunter_raw_pdb": 0, "hunter_raw_cif": 0,
             "hunter_clean_cif_entries": 0}
    for d in entries:
        meta = d / f"{d.name}_meta.json"
        if meta.exists():
            try:
                if json.loads(meta.read_text()).get("complete"):
                    stats["complete_meta"] += 1
            except Exception:                                    # noqa: BLE001
                pass
        clean = d / "clean"
        if clean.is_dir():
            exts = {f.suffix for f in clean.iterdir()}
            if ".dbn" in exts:
                stats["hunter_dbn_entries"] += 1
            if ".bpseq" in exts:
                stats["hunter_bpseq_entries"] += 1
            if ".cif" in exts:
                stats["hunter_clean_cif_entries"] += 1
        if (d / f"{d.name}_bb.xyz").exists():
            stats["hunter_bb_xyz"] += 1
        if (d / f"{d.name}.pdb").exists():
            stats["hunter_raw_pdb"] += 1
        if (d / f"{d.name}.cif").exists():
            stats["hunter_raw_cif"] += 1
    return stats


def audit_benchmarks() -> dict:
    sec = DATA / "benchmarks/secondary_structure"
    out: dict = {}
    out["bprna_1m_dbn_zip"] = count_zip_members(
        sec / "bprna_full/bpRNA_1m_dbn.zip", ".dbn")
    for split in ("train", "validation", "test"):
        p = sec / f"bprna_spot/{split}.parquet"
        if p.exists():
            out[f"bprna_spot_{split}"] = parquet_rows(p)
    p = sec / "archiveii/test.parquet"
    if p.exists():
        out["archiveii_parquet"] = parquet_rows(p)
    p = sec / "rnastralign/train.parquet"
    if p.exists():
        out["rnastralign_parquet"] = parquet_rows(p)
    p = sec / "bprna_new/test.parquet"
    if p.exists():
        out["bprna_new_parquet"] = parquet_rows(p)
    out["archiveii_tar_pickles"] = count_tar_members(
        sec / "archiveii/ArchiveII.tar.gz", ".pickle")
    for split, f in [("train", "train"), ("val", "validation"),
                     ("test", "test")]:
        p = DATA / f"benchmarks/probing/eternabench-cm/{f}.parquet"
        if p.exists():
            out[f"eternabench_cm_{split}"] = parquet_rows(p)
    fit = DATA / "benchmarks/fitness"
    out["rnagym_csvs"] = count_zip_members(
        fit / "rnagym/fitness_processed_assays.zip", ".csv")
    nb = fit / "nabench"
    out["nabench_files"] = sum(1 for p in nb.rglob("*") if p.is_file()) \
        if nb.exists() else 0
    g3 = DATA / "benchmarks/splicing/g3po/G3PO.csv"
    if g3.exists():
        with open(g3, errors="ignore") as fh:
            out["g3po_species_rows"] = max(sum(1 for _ in fh) - 1, 0)
    for name in ("casp15", "casp16"):
        d = DATA / f"structures/blind_tests/{name}"
        out[name] = len(list(d.glob("*.cif"))) if d.exists() else 0
    out["ribonanza_quickstart"] = sum(
        1 for _ in open(DATA / "benchmarks/probing/ribonanza/"
                        "train_data_QUICK_START.csv")) - 1
    return out


def audit_structures() -> dict:
    out = {}
    out["rna3db_chains"] = count_tar_members(
        DATA / "structures/databases/rna3db/rna3db-mmcifs.tar.xz",
        ".cif", mode="r|xz")
    out["grnade_pdbs"] = count_tar_members(
        DATA / "structures/databases/grnade_rnasolo/"
        "RNASolo_31102023_raw.tar.gz", ".pdb")
    for res in ("1.5A", "3.0A", "20.0A"):
        p = DATA / f"structures/indices/bgsu_nrlist/nrlist_4.57_{res}.csv"
        if p.exists():
            with open(p, errors="ignore") as fh:
                out[f"bgsu_nrlist_{res}_rows"] = sum(1 for _ in fh)
    for t in ("il", "hl"):
        p = DATA / f"structures/indices/bgsu_motifs/motif_atlas_4.14_{t}.csv"
        if p.exists():
            with open(p, errors="ignore") as fh:
                out[f"bgsu_motifs_{t}_rows"] = max(sum(1 for _ in fh) - 1, 0)
    return out


CHECKS = {
    "eldors": audit_eldors,
    "mars": audit_mars,
    "rmdb": audit_rmdb,
    "rnacentral": audit_rnacentral,
    "rfam": audit_rfam,
    "raw_pdb": audit_raw_pdb,
    "pdb_hunter": audit_pdb_hunter,
    "benchmarks": audit_benchmarks,
    "structures": audit_structures,
}
SLOW = {"eldors", "mars", "structures"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow full-corpus counts")
    args = ap.parse_args()
    names = [args.only] if args.only else list(CHECKS)
    if args.quick:
        names = [n for n in names if n not in SLOW]
    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
    if OUT.exists():
        try:
            report["checks"] = json.loads(OUT.read_text()).get("checks", {})
        except Exception:                                        # noqa: BLE001
            pass
    for n in names:
        t0 = time.time()
        print(f"[audit] {n} ...", flush=True)
        try:
            res = CHECKS[n]()
        except Exception as exc:                                 # noqa: BLE001
            res = {"error": f"{type(exc).__name__}: {exc}"}
        res["seconds"] = round(time.time() - t0, 1)
        report["checks"][n] = res
        print(f"[audit] {n}: {json.dumps(res)[:400]}", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=1))

    print("\n=== measured vs documented ===")
    flat = {}
    for n, res in report["checks"].items():
        flat.update({k: v for k, v in res.items()
                     if isinstance(v, (int, float))})
    for key, exp in EXPECTED.items():
        got = flat.get(key)
        if got is None:
            continue
        delta = "" if got == exp else f"  (delta {got-exp:+,})"
        print(f"  {key:26s} measured {got:>15,}  expected {exp:>15,}{delta}")
    print(f"\nreport: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
