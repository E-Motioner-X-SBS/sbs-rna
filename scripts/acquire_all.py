#!/usr/bin/env python3
"""Acquire the full sbs-rna / PHAROS training-data corpus.

One resumable, idempotent, logged run over every source in
``data/catalog/README.md`` plus the pdb_hunter training corpora.

Design
------
* Every download lands in ``data/`` under the catalog's layout
  (``sequences/``, ``families/``, ``structures/``, ``benchmarks/``).
* A download writes ``<dest>.part`` and renames only when complete, so
  the existence of ``<dest>`` means "done".  Re-running costs one stat.
* HTTP transfers byte-range resume; transient failures retry with
  backoff up to a per-job budget.
* Jobs run in a thread pool; each has its own log under
  ``data/acquisition/logs/``.
* ``--group`` selects a slice (``sequence`` | ``catalog`` | ``long``),
  ``--only`` selects by name substring, ``--list`` prints the plan.

Ribonanza probing (needs Kaggle credentials) and RMDB Mg2+ titrations
(no bulk endpoint) are deliberately absent; see plans/14.

Reading `--list` on a machine that did not do the downloading
-------------------------------------------------------------
**`--list` under-reports on this machine and the data is not missing.** The
corpus was acquired on a different host and copied here; what came across was
`data/`, not `data/acquisition/`. Two consequences, and they cover every job
that reads as pending:

* a ``cmd`` job is DONE when ``data/acquisition/state/<name>.done`` exists, and
  that directory does not exist here at all -- so ``rmdb``, ``bgsu-*``,
  ``rfam-full-alignments``, ``rna-harvest`` and ``raw-pdb-entries`` all report
  pending while their output sits on disk. ``raw-pdb-entries`` is 10,527
  entries and 15 GB of it.
* an ``http`` job is DONE when its destination path exists, and the tree was
  reorganised after acquisition. ``ribonanza-quickstart`` wants
  ``benchmarks/probing/ribonanza/train_data_QUICK_START.csv``; the file is at
  ``benchmarks/chemical_probing/ribonanza_train_quickstart.csv``. EternaBench,
  gRNAde and the Rfam extras moved similarly.

**`scripts/sampling/audit_inventory_gap.py` is the authority on what is
present**, because it measures the tree as it is rather than as this file
expected to leave it: 18 sources, 17 present, 0 partial, **0 missing**, 1
absent by decision. Do not re-run a group here to "fill gaps" without checking
it first -- on this host that would re-download hundreds of gigabytes into a
second, parallel layout.

The one real absence is MARS: 30 jobs, 427.29 GB, skipped under D18 because
1.73B sequences yielded 0.17% diverse structured ncRNA. Documented total 833.3
GB, on disk 567.0 GB, and the 17 acquired sources exceed their documented size
because several are stored decompressed.

Usage
-----
    uv run python scripts/acquire_all.py --list
    uv run python scripts/acquire_all.py --group catalog --workers 12
    uv run python scripts/acquire_all.py --group sequence --workers 10
    uv run python scripts/acquire_all.py --only mars --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOGS = DATA / "acquisition" / "logs"
STATE = DATA / "acquisition" / "state"
HUNTER = ROOT.parent / "pdb_hunter"
UV = shutil.which("uv") or str(Path.home() / ".local/bin/uv")

S3 = "https://eldors-v1-database.s3.amazonaws.com/elDORS_v1/elDORS_v1_chunks"
MARS = "https://download.cncb.ac.cn/OMIX/OMIX003037"

# ── expected sizes (verified live) ──────────────────────────────────────

ELDORS_SIZES = {
    "001": 9384485602, "002": 9389192760, "003": 9458323752,
    "004": 9384587155, "005": 9389820793, "006": 9458208286,
    "007": 9384656600, "008": 9391707715, "009": 9456718189,
    "010": 9384873615, "011": 9404199920, "012": 9444055087,
    "013": 9385381770, "014": 9425023908, "015": 9423028547,
    "016": 9385081045, "017": 9448673962, "018": 9399651199,
    "019": 9384954447, "020": 3579817684,
}

MARS_SIZES = {
    "01": 13383880676, "02": 12627433835, "03": 14168160647,
    "04": 11941565088, "05": 12341239519, "06": 12612547707,
    "07": 12670490003, "08": 13018049436, "09": 13066938596,
    "10": 13454921156, "11": 14406242594, "12": 14399929729,
    "13": 13373410248, "14": 15639148800, "15": 15513464470,
    "16": 15560872492, "17": 15494583440, "18": 15447922446,
    "19": 15573193235, "20": 15401736219, "21": 15403184702,
    "22": 15417022713, "23": 15438754579, "24": 14868413110,
    "25": 14812939719, "26": 15111188153, "27": 14289941502,
    "28": 14419774014, "29": 14268729723, "30": 13168888863,
}

RFAM = "https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1"
RNA3DB = ("https://github.com/marcellszi/rna3db/releases/download/"
          "2026-01-05-full-release")
HF = "https://huggingface.co/datasets"

CASP15 = ["7qr4", "7qr3", "8s95", "8fza", "8tvz", "8btz", "7zj4", "7ptk",
          "7ptl", "8uys", "8uye", "8uyg", "8uyj", "7yr7", "7yr6"]
CASP16 = ["8uo6", "9cfn", "9c2k", "9dcf", "9b0l", "9ely", "9bzc", "9bz1",
          "9cbu", "9cbx"]


@dataclass
class Job:
    name: str
    group: str
    kind: str = "http"            # http | cmd | git
    url: Optional[str] = None
    dest: Optional[Path] = None
    expected: Optional[int] = None
    cmd: List[str] = field(default_factory=list)
    server: str = ""


def _dest(rel: str) -> Path:
    return DATA / rel


def build_jobs() -> List[Job]:
    J: List[Job] = []
    add = J.append

    # ── sequences (the long pole) ──────────────────────────────────────
    # Interleave elDORS and MARS so both make progress from the start.
    seq_jobs: List[Job] = []
    for i in range(1, 21):
        c = f"{i:03d}"
        seq_jobs.append(Job(f"eldors-{c}", "sequence", "http",
                f"{S3}/elDORS_v1_{c}.fasta.gz",
                _dest(f"sequences/elDORS_v1/elDORS_v1_{c}.fasta.gz"),
                ELDORS_SIZES[c], server="aws-s3"))
    seq_jobs.append(Job("eldors-manifest", "sequence", "http",
            f"{S3}/elDORS_v1_manifest.sha256",
            _dest("sequences/elDORS_v1/elDORS_v1_manifest.sha256"), 1780,
            server="aws-s3"))
    for i in range(1, 31):
        p = f"{i:02d}"
        seq_jobs.append(Job(f"mars-{p}", "sequence", "http",
                f"{MARS}/OMIX003037-{p}.tgz",
                _dest(f"sequences/mars/OMIX003037-{p}.tgz"),
                MARS_SIZES[p], server="ngdc"))
    seq_jobs.append(Job("rnacentral-active", "sequence", "http",
            "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/"
            "sequences/rnacentral_active.fasta.gz",
            _dest("sequences/rnacentral/rnacentral_active.fasta.gz"),
            10919860429, server="ebi"))
    seq_jobs.append(Job("rnacentral-inactive", "sequence", "http",
            "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/"
            "sequences/rnacentral_inactive.fasta.gz",
            _dest("sequences/rnacentral/rnacentral_inactive.fasta.gz"),
            1331040224, server="ebi"))
    seq_jobs.append(Job("rnacentral-species-specific", "sequence", "http",
            "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/"
            "sequences/rnacentral_species_specific_ids.fasta.gz",
            _dest("sequences/rnacentral/rnacentral_species_specific_ids.fasta.gz"),
            11692285341, server="ebi"))
    eldors = [j for j in seq_jobs if j.name.startswith("eldors-")]
    mars = [j for j in seq_jobs if j.name.startswith("mars-")]
    other = [j for j in seq_jobs if not j.name.startswith(("eldors-", "mars-"))]
    for pair in zip(eldors, mars):
        add(pair[0])
        add(pair[1])
    for j in eldors[len(mars):] + mars[len(eldors):] + other:
        add(j)

    # ── families ───────────────────────────────────────────────────────
    for fn, size in [("Rfam.seed.gz", 5928432), ("Rfam.cm.gz", 45753387),
                     ("Rfam.full_region.gz", 125363449),
                     ("Rfam.3d.seed.gz", 377401), ("Rfam.pdb.gz", 87290),
                     ("Rfam.seed_tree.tar.gz", 2633714),
                     ("Rfam.tar.gz", 46147784)]:
        add(Job(f"rfam-{fn.split('.')[1]}", "catalog", "http",
                f"{RFAM}/{fn}", _dest(f"families/rfam/{fn}"), size,
                server="ebi"))
    add(Job("rfam-full-alignments", "catalog", "cmd",
            cmd=[UV, "run", "python", "scripts/acquire_rfam_full.py",
                 "--workers", "12"],
            dest=_dest("families/rfam/full_alignments"), server="ebi"))

    # ── chemical probing / mapping ─────────────────────────────────────
    add(Job("rmdb", "catalog", "cmd",
            cmd=[UV, "run", "python", "scripts/acquire_rmdb.py",
                 "--workers", "8"],
            dest=_dest("benchmarks/probing/rmdb"), server="github"))
    add(Job("ribonanza-quickstart", "catalog", "http",
            f"{HF}/TerminatorJ/RNA_chemical_ribonanza/resolve/main/"
            "train_data_QUICK_START.csv",
            _dest("benchmarks/probing/ribonanza/train_data_QUICK_START.csv"),
            550743596, server="huggingface"))
    for split in ("train", "val", "test"):
        add(Job(f"pdb-ribonanzanet-{split}", "catalog", "http",
                f"{HF}/TerminatorJ/PDB_Ribonanzanet/resolve/main/{split}.csv",
                _dest(f"benchmarks/probing/ribonanzanet/{split}.csv"), None,
                server="huggingface"))
    for ds, files in [
        ("eternabench-cm", ["train.parquet", "test.parquet"]),
        ("eternabench-switch", ["train.parquet", "test.parquet"]),
        ("eternabench-external.300", ["test.parquet"]),
        ("eternabench-external.600", ["test.parquet"]),
        ("eternabench-external.900", ["test.parquet"]),
        ("eternabench-external.1200", ["test.parquet"]),
    ]:
        for fn in files:
            add(Job(f"{ds}-{fn.split('.')[0]}", "catalog", "http",
                    f"{HF}/multimolecule/{ds}/resolve/main/{fn}",
                    _dest(f"benchmarks/probing/{ds}/{fn}"), None,
                    server="huggingface"))

    # ── structure indices ──────────────────────────────────────────────
    add(Job("pdb-seqres", "catalog", "http",
            "https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz",
            _dest("structures/indices/pdb_seqres.txt.gz"), 66979554,
            server="rcsb"))
    add(Job("bgsu-rfam-map", "catalog", "http",
            "https://rna.bgsu.edu/data/pdb_chain_to_best_rfam.txt",
            _dest("structures/indices/pdb_chain_to_best_rfam.txt"), None,
            server="bgsu"))
    for res in ["1.5A", "2.0A", "2.5A", "3.0A", "3.5A", "4.0A", "20.0A"]:
        add(Job(f"bgsu-nrlist-{res}", "catalog", "cmd",
                cmd=[UV, "run", "pdb-hunter", "rna", "bgsu-nrlist",
                     "--resolution", res, "--format", "csv",
                     "--out", str(_dest(f"structures/indices/bgsu_nrlist/"
                                        f"nrlist_4.57_{res}.csv"))],
                dest=_dest(f"structures/indices/bgsu_nrlist/"
                           f"nrlist_4.57_{res}.csv"), server="bgsu"))
    for t in ["il", "hl"]:
        add(Job(f"bgsu-motifs-{t}", "catalog", "cmd",
                cmd=[UV, "run", "pdb-hunter", "rna", "motifs-download",
                     "--type", t, "--release", "4.14",
                     "--out", str(_dest(f"structures/indices/bgsu_motifs/"
                                        f"motif_atlas_4.14_{t}.csv"))],
                dest=_dest(f"structures/indices/bgsu_motifs/"
                           f"motif_atlas_4.14_{t}.csv"), server="bgsu"))

    # ── structure databases ────────────────────────────────────────────
    for asset, size in [("rna3db-jsons.tar.gz", 6851100),
                        ("rna3db-cmscans.tar.gz", 88160023),
                        ("rna3db-mmcifs.tar.xz", 2146440832)]:
        add(Job(f"rna3db-{asset.split('-')[1].split('.')[0]}", "catalog",
                "http", f"{RNA3DB}/{asset}",
                _dest(f"structures/databases/rna3db/{asset}"), size,
                server="github"))
    for fn, size in [("RNASolo_31102023_raw.tar.gz", 6863858264),
                     ("RNASolo_31102023_processed.pt", 2789581093),
                     ("RNAsolo_31102023_processed_df.csv", 3698514)]:
        add(Job(f"grnade-{fn.split('.')[0].lower()}", "catalog", "http",
                f"{HF}/chaitjo/gRNAde_datasets/resolve/main/{fn}",
                _dest(f"structures/databases/grnade_rnasolo/{fn}"), size,
                server="huggingface"))

    # ── blind tests ────────────────────────────────────────────────────
    for pid in CASP15:
        add(Job(f"casp15-{pid.upper()}", "catalog", "http",
                f"https://files.rcsb.org/download/{pid.upper()}.cif",
                _dest(f"structures/blind_tests/casp15/{pid.upper()}.cif"),
                None, server="rcsb"))
    for pid in CASP16:
        add(Job(f"casp16-{pid.upper()}", "catalog", "http",
                f"https://files.rcsb.org/download/{pid.upper()}.cif",
                _dest(f"structures/blind_tests/casp16/{pid.upper()}.cif"),
                None, server="rcsb"))
    add(Job("rna-puzzles-std", "catalog", "git",
            url="https://github.com/mmagnus/RNA-Puzzles-Standardized-Submissions",
            dest=_dest("structures/blind_tests/rna_puzzles_std"), server="github"))
    add(Job("rna-puzzles-io", "catalog", "git",
            url="https://github.com/rnapuzzles/rnapuzzles.github.io",
            dest=_dest("structures/blind_tests/rnapuzzles_github_io"),
            server="github"))

    # ── secondary structure benchmarks ─────────────────────────────────
    sec = "benchmarks/secondary_structure"
    for repo, fn, out in [
        ("multimolecule/archiveii", "test.parquet", "archiveii/test.parquet"),
        ("multimolecule/bprna-spot", "train.parquet", "bprna_spot/train.parquet"),
        ("multimolecule/bprna-spot", "validation.parquet", "bprna_spot/validation.parquet"),
        ("multimolecule/bprna-spot", "test.parquet", "bprna_spot/test.parquet"),
        ("multimolecule/bprna-new", "test.parquet", "bprna_new/test.parquet"),
        ("multimolecule/rnastralign", "train.parquet", "rnastralign/train.parquet"),
        ("rouskinlab/RNAstralign", "data.json", "rnastralign/data.json"),
        ("rouskinlab/bpRNA-1m", "data.json", "bprna_full/data.json"),
    ]:
        add(Job(f"hf-{out.replace('/', '-')}", "catalog", "http",
                f"{HF}/{repo}/resolve/main/{fn}",
                _dest(f"{sec}/{out}"), None, server="huggingface"))
    add(Job("rnastralign-bpseq", "catalog", "http",
            "https://ndownloader.figshare.com/files/45555225",
            _dest(f"{sec}/rnastralign/RNAStrAlign_bpseq.zip"), None,
            server="figshare"))
    add(Job("archiveii-ct", "catalog", "http",
            "https://zenodo.org/api/records/16730061/files/ArchiveII.tar.gz/content",
            _dest(f"{sec}/archiveii/ArchiveII.tar.gz"), None, server="zenodo"))
    add(Job("bprna-st", "catalog", "http",
            "https://zenodo.org/api/records/16730061/files/bpRNA_1m.tar.gz/content",
            _dest(f"{sec}/bprna_full/bpRNA_1m.tar.gz"), None, server="zenodo"))
    for fmt in ["dbn", "bpseq", "fasta", "ct", "st"]:
        add(Job(f"bprna-{fmt}", "catalog", "http",
                f"https://bprna.cgrb.oregonstate.edu/bpRNA_1m/{fmt}Files.zip",
                _dest(f"{sec}/bprna_full/bpRNA_1m_{fmt}.zip"), None,
                server="osu"))
    add(Job("spot-rna-bprna", "catalog", "http",
            "https://www.dropbox.com/s/w3kc4iro8ztbf3m/bpRNA_dataset.zip?dl=1",
            _dest(f"{sec}/bprna_spot/splits/bpRNA_dataset.zip"), None,
            server="dropbox"))
    add(Job("spot-rna-pdb", "catalog", "http",
            "https://www.dropbox.com/s/vnq0k9dg7vynu3q/PDB_dataset.zip?dl=1",
            _dest(f"{sec}/bprna_spot/splits/PDB_dataset.zip"), None,
            server="dropbox"))

    # ── fitness ────────────────────────────────────────────────────────
    fit = "benchmarks/fitness"
    add(Job("rnagym-processed", "catalog", "http",
            "https://marks.hms.harvard.edu/rnagym/fitness_prediction/"
            "fitness_processed_assays.zip",
            _dest(f"{fit}/rnagym/fitness_processed_assays.zip"), None,
            server="harvard"))
    add(Job("rnagym-raw", "catalog", "http",
            "https://marks.hms.harvard.edu/rnagym/fitness_prediction/"
            "fitness_raw_data.zip",
            _dest(f"{fit}/rnagym/fitness_raw_data.zip"), None,
            server="harvard"))
    add(Job("cpeb3", "catalog", "http",
            f"{HF}/Marks-lab/RNAgym/resolve/main/fitness_prediction/assays/"
            "Zhang_2020_cpeb3_ribozyme.parquet",
            _dest(f"{fit}/rnagym/Zhang_2020_cpeb3_ribozyme.parquet"), None,
            server="huggingface"))
    add(Job("rnagym-repo", "catalog", "git",
            url="https://github.com/MarksLab-DasLab/RNAGym",
            dest=_dest(f"{fit}/rnagym_repo"), server="github"))
    add(Job("nabench-repo", "catalog", "git",
            url="https://github.com/mrzzmrzz/NABench",
            dest=_dest(f"{fit}/nabench"), server="github"))

    # ── splicing ───────────────────────────────────────────────────────
    spl = "benchmarks/splicing"
    add(Job("spliceator-bigest", "catalog", "http",
            "https://bigest-icube.fr/spliceator/static/data/data.tar.gz",
            _dest(f"{spl}/spliceator/data.tar.gz"), None, server="bigest"))
    add(Job("splicebert-zenodo", "catalog", "http",
            "https://zenodo.org/records/7995778/files/data.tar.gz?download=1",
            _dest(f"{spl}/splicebert/data.tar.gz"), None, server="zenodo"))
    add(Job("g3po-repo", "catalog", "git",
            url="https://github.com/BiGEst-ICube/g3po",
            dest=_dest(f"{spl}/g3po"), server="github"))

    # ── long-running corpus builders ───────────────────────────────────
    add(Job("rna-harvest", "long", "cmd",
            cmd=[UV, "run", "pdb-hunter", "rna", "harvest",
                 "--out", str(HUNTER / "RNA_Database"),
                 "--workers", "12", "--nr-release", "4.57"],
            dest=HUNTER / "RNA_Database", server="mixed"))
    add(Job("raw-pdb-entries", "long", "cmd",
            cmd=[UV, "run", "python", "scripts/acquire_raw_pdb_entries.py",
                 "--workers", "6"],
            dest=DATA / "structures/raw_pdb_entries", server="rcsb"))

    return J


# ── engine ──────────────────────────────────────────────────────────────

_print_lock = threading.Lock()


def log(msg: str, job: Optional[str] = None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _print_lock:
        print(line, flush=True)
        if job:
            LOGS.mkdir(parents=True, exist_ok=True)
            with open(LOGS / f"{job}.log", "a") as fh:
                fh.write(line + "\n")


def complete(job: Job) -> bool:
    if job.kind == "cmd":
        # Directory-producing builders (harvest, raw PDB) are done only when
        # the orchestrator has recorded a successful exit; file-producing
        # commands (e.g. bgsu-nrlist) are done when the file exists.
        if job.dest is None:
            return False
        if job.dest.is_file():
            return job.dest.stat().st_size > 0
        return (STATE / f"{job.name}.done").exists()
    if job.kind == "git":
        return job.dest is not None and (job.dest / ".git").exists()
    if job.dest is None or not job.dest.exists():
        return False
    if job.expected:
        return job.dest.stat().st_size == job.expected
    return job.dest.stat().st_size > 0


def http_download(job: Job, max_attempts: int = 10000) -> None:
    assert job.url and job.dest
    dest = job.dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    attempt = 0                 # consecutive attempts without progress
    t0 = time.time()
    while attempt < max_attempts:
        attempt += 1
        have = part.stat().st_size if part.exists() else 0
        if job.expected and have >= job.expected:
            part.rename(dest)
            log(f"OK {dest.name} ({have:,} B, {(time.time()-t0)/60:.1f} min)")
            return
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(job.url, headers=headers, stream=True,
                              timeout=(30, 120), allow_redirects=True) as r:
                if have and r.status_code == 200:
                    have = 0
                    part.unlink(missing_ok=True)
                elif have and r.status_code == 416:
                    part.rename(dest)
                    log(f"OK (416, assumed complete) {dest.name}")
                    return
                r.raise_for_status()
                got = have
                prev = have
                last = time.time()
                with open(part, "ab" if have else "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            fh.write(chunk)
                            got += len(chunk)
                        now = time.time()
                        if now - last > 60:
                            rate = (got - prev) / max(now - last, 1e-9) / 1e6
                            log(f"  {dest.name}: {got/1e6:,.0f} MB "
                                f"(+{rate:.1f} MB/s)", job.name)
                            prev = got
                            last = now
        except Exception as exc:                       # noqa: BLE001
            size = part.stat().st_size if part.exists() else 0
            if size > have:
                attempt = 0     # made progress: fresh retry budget
            log(f"  {job.name}: attempt reset={attempt} "
                f"failed: {type(exc).__name__}: {str(exc)[:100]}", job.name)
            time.sleep(min(30, 3 * attempt))
            continue
        size = part.stat().st_size if part.exists() else 0
        if job.expected:
            if size == job.expected:
                part.rename(dest)
                dt = (time.time() - t0) / 60
                log(f"OK {dest.name} ({size:,} B in {dt:.1f} min)", job.name)
                return
            if size > job.expected:
                log(f"  {job.name}: oversize {size} > {job.expected}; "
                    f"restarting", job.name)
                part.unlink(missing_ok=True)
                continue
            log(f"  {job.name}: incomplete {size:,}/{job.expected:,} B; "
                f"resuming", job.name)
        elif size > 0:
            part.rename(dest)
            log(f"OK {dest.name} ({size:,} B)", job.name)
            return
        time.sleep(2)
    raise RuntimeError(f"{job.name}: gave up after {max_attempts} attempts")


def run_git(job: Job) -> None:
    assert job.dest and job.url
    job.dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["git", "clone", "--depth", "1", job.url,
                        str(job.dest)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git clone failed: {r.stderr[-300:]}")
    log(f"OK clone {job.dest.name}")


def run_cmd(job: Job) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(LOGS / f"{job.name}.log", "a") as fh:
        fh.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                 f"RUN {' '.join(job.cmd)}\n")
        fh.flush()
        r = subprocess.run(job.cmd, cwd=str(ROOT), stdout=fh,
                           stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{job.name}: exit {r.returncode} "
                           f"(see {LOGS / (job.name + '.log')})")


def write_raw_pdb_entrylist() -> None:
    """Derive the raw-PDB entry list from the pdb_hunter catalog index."""
    out = DATA / "structures/raw_pdb_entrylist.txt"
    if out.exists() and out.stat().st_size > 0:
        return
    index = DATA / "catalog/pdb_hunter_index.json"
    doc = json.loads(index.read_text())
    ids = sorted({e["pdb_id"].upper() for e in doc.get("entries", [])})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ids) + "\n")
    log(f"wrote {len(ids):,} ids to {out}")


def run_job(job: Job) -> str:
    if complete(job):
        return f"skip {job.name} (complete)"
    log(f"START {job.name} [{job.server}]", job.name)
    if job.dest is not None and job.kind == "cmd":
        job.dest.parent.mkdir(parents=True, exist_ok=True)
    if job.name == "raw-pdb-entries":
        write_raw_pdb_entrylist()
    if job.kind == "http":
        http_download(job)
    elif job.kind == "git":
        run_git(job)
    elif job.kind == "cmd":
        run_cmd(job)
        if job.dest is not None and job.dest.is_dir():
            STATE.mkdir(parents=True, exist_ok=True)
            (STATE / f"{job.name}.done").write_text(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    else:
        raise ValueError(job.kind)
    return f"done {job.name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the plan")
    ap.add_argument("--group", default=None,
                    help="sequence | catalog | long")
    ap.add_argument("--only", default=None,
                    help="substring match on job name")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--exclude", default=None,
                    help="substring to exclude")
    args = ap.parse_args()

    jobs = build_jobs()
    if args.group:
        jobs = [j for j in jobs if j.group == args.group]
    if args.only:
        jobs = [j for j in jobs if args.only in j.name]
    if args.exclude:
        jobs = [j for j in jobs if args.exclude not in j.name]

    if args.list:
        by_group: Dict[str, List[Job]] = {}
        for j in jobs:
            by_group.setdefault(j.group, []).append(j)
        for g, js in by_group.items():
            print(f"\n== {g} ({len(js)} jobs) ==")
            for j in js:
                state = "DONE" if complete(j) else "    "
                size = f"~{j.expected/1e9:.2f} GB" if j.expected else "size ?"
                print(f"  {state} {j.name:<28} {j.server:<12} {size}")
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs if not complete(j)]
    log(f"{len(jobs)} jobs selected, {len(jobs)-len(pending)} already complete, "
        f"{len(pending)} to run with {args.workers} workers")
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_job, j): j for j in pending}
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                log(fut.result())
            except Exception as exc:                   # noqa: BLE001
                failures.append(job.name)
                log(f"FAIL {job.name}: {exc}")
    if failures:
        log(f"{len(failures)} failures: {failures}")
        return 1
    log("all selected jobs complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
