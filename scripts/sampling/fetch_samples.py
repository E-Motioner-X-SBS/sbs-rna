#!/usr/bin/env python3
"""Fetch a small, representative working sample of the RNA data universe.

The bulk corpus lives on the server; this pulls ~100-200 real files so
architecture work on a laptop is grounded in actual data, not just the catalog.

Targets:
  1. BGSU non-redundant representative set (3.0A) -> chain list
  2. ~150 representative RNA structures as mmCIF (carry ions: MG/K/NA/ZN)
  3. Rfam seed alignments for families with 3D reps (coevolution signal)
  4. RNAcentral sequence sample
"""
from __future__ import annotations
import csv, json, re, subprocess, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "samples"
UA = {"User-Agent": "sbs-rna-architecture-research/0.1 (academic; contact via repo)"}
N_STRUCTURES = int(sys.argv[1]) if len(sys.argv) > 1 else 150


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------- 1. BGSU NR list ----------
def fetch_nrlist() -> list[tuple[str, str]]:
    d = OUT / "indices"; d.mkdir(parents=True, exist_ok=True)
    dest = d / "nrlist_3.0A.csv"
    if not dest.exists():
        log("fetching BGSU NR list 3.0A ...")
        dest.write_bytes(get("https://rna.bgsu.edu/rna3dhub/nrlist/download/current/3.0A/csv"))
    text = dest.read_text()
    if text.lstrip().startswith("<"):
        dest.unlink(missing_ok=True)
        raise SystemExit("BGSU returned HTML, not CSV - endpoint changed; aborting")
    rows = list(csv.reader(text.splitlines()))
    log(f"  NR list: {len(rows)} equivalence classes")
    PDB_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
    reps = []
    for r in rows:
        if len(r) < 2 or not r[1]:
            continue
        rep = r[1].split("|")          # e.g. 4V9F|1|0
        if len(rep) >= 3 and PDB_RE.match(rep[0]):
            reps.append((rep[0].upper(), rep[2]))   # (pdb_id, chain)
    if not reps:
        raise SystemExit("no valid PDB identifiers parsed from NR list")
    log(f"  parsed {len(reps)} valid representative chains")
    return reps


# ---------- 2. mmCIF structures ----------
def fetch_structures(reps: list[tuple[str, str]], n: int) -> dict:
    d = OUT / "structures"; d.mkdir(parents=True, exist_ok=True)
    seen, picks = set(), []
    for pdb, ch in reps:
        if pdb in seen:
            continue
        seen.add(pdb); picks.append((pdb, ch))
        if len(picks) >= n:
            break
    log(f"downloading {len(picks)} mmCIF structures ...")

    def one(item):
        pdb, ch = item
        dest = d / f"{pdb}.cif.gz"
        if dest.exists() and dest.stat().st_size > 0:
            return pdb, ch, dest.stat().st_size, "cached"
        try:
            dest.write_bytes(get(f"https://files.rcsb.org/download/{pdb}.cif.gz"))
            time.sleep(0.15)
            return pdb, ch, dest.stat().st_size, "ok"
        except Exception as e:
            return pdb, ch, 0, f"FAIL {e}"

    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, res in enumerate(ex.map(one, picks), 1):
            results.append(res)
            if i % 25 == 0:
                log(f"  {i}/{len(picks)}")
    ok = [r for r in results if r[3] in ("ok", "cached")]
    log(f"  structures: {len(ok)}/{len(picks)} ok, "
        f"{sum(r[2] for r in ok)/1e6:.1f} MB")
    (d / "manifest.json").write_text(json.dumps(
        [{"pdb": p, "rep_chain": c, "bytes": b, "status": s} for p, c, b, s in results], indent=2))
    return {"requested": len(picks), "ok": len(ok)}


# ---------- 3. Rfam seeds ----------
def fetch_rfam() -> None:
    d = OUT / "benchmarks"; d.mkdir(parents=True, exist_ok=True)
    # families with well-known 3D representatives
    fams = {
        "RF00001": "5S_rRNA", "RF00005": "tRNA", "RF00010": "RNaseP_bact_a",
        "RF00023": "tmRNA", "RF00050": "FMN_riboswitch", "RF00059": "TPP_riboswitch",
        "RF00162": "SAM_riboswitch", "RF00167": "Purine_riboswitch",
        "RF00174": "Cobalamin_riboswitch", "RF01739": "Glutamine_riboswitch",
        "RF00379": "ydaO_yuaA", "RF01831": "THF_riboswitch",
    }
    for acc, name in fams.items():
        dest = d / f"rfam_{acc}_{name}.stk"
        if dest.exists():
            continue
        try:
            dest.write_bytes(get(f"https://rfam.org/family/{acc}/alignment/stockholm"))
            log(f"  rfam {acc} {name}: {dest.stat().st_size/1024:.0f} KB")
            time.sleep(0.4)
        except Exception as e:
            log(f"  rfam {acc} FAIL {e}")


# ---------- 4. RNAcentral sequences ----------
def fetch_rnacentral(n: int = 200) -> None:
    """RNAcentral's API stalls on large page sizes; paginate at 100."""
    d = OUT / "sequences"; d.mkdir(parents=True, exist_ok=True)
    dest = d / "rnacentral_sample.json"
    if dest.exists():
        return
    PAGE = 100
    results, page = [], 1
    while len(results) < n:
        url = (f"https://rnacentral.org/api/v1/rna/"
               f"?page_size={PAGE}&page={page}&format=json")
        try:
            raw = subprocess.run(
                ["curl", "-sS", "--fail", "--compressed", "--max-time", "90", url],
                capture_output=True, check=True).stdout
            batch = json.loads(raw).get("results", [])
        except Exception as e:
            log(f"  rnacentral page {page} FAIL {e}")
            break
        if not batch:
            break
        results.extend(batch)
        log(f"  rnacentral page {page}: +{len(batch)} (total {len(results)})")
        page += 1
        time.sleep(0.5)
    if not results:
        log("  rnacentral: nothing fetched")
        return
    dest.write_text(json.dumps({"count": len(results), "results": results[:n]}, indent=2))
    log(f"  rnacentral: {len(results[:n])} sequences -> {dest.name}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    reps = fetch_nrlist()
    fetch_structures(reps, N_STRUCTURES)
    fetch_rfam()
    fetch_rnacentral()
    log("DONE")
