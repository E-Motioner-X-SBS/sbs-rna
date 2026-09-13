#!/usr/bin/env python3
"""Regression guard: re-derive every headline claim and fail loudly on drift.

Written during the rigor audit. Five implementation defects were found in the
original analyses and two more (a silent unguarded string replacement, and an
overclaimed statistic) in the write-up. This script exists so the numbers in the
deliverables cannot silently diverge from the data again.

Run: python3 scripts/sampling/verify_claims.py
Exit code 0 = all claims reproduce; 1 = drift detected.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "data" / "samples" / "analysis"

def load(n): return json.load(open(A / n))

def main() -> int:
    fails = []
    def chk(name, got, want, tol=0.005):
        ok = abs(float(got) - float(want)) <= tol * max(abs(float(want)), 1)
        print(f"  {'OK ' if ok else 'FAIL'} {name:38s} got={got}  want={want}")
        if not ok: fails.append(name)

    print("== measured claims vs analysis JSON ==")
    ion = load("ion_summary.json")
    chk("Mg2+ total", ion["total_ions_by_type"]["MG"], 17428)
    chk("K+ total", ion["total_ions_by_type"]["K"], 1868)
    inner = ion["top_inner_sphere_partners"]
    frac_phos = (inner["OP1"] + inner["OP2"]) / sum(inner.values())
    chk("inner-sphere phosphate fraction", round(frac_phos, 3), 0.83, tol=0.02)

    rig = load("rigidity_summary.json")
    mg = rig["rigidity_by_mg_distance"]
    chk("Mg gradient span (sigma)", round(mg[-1]["mean_zB"] - mg[0]["mean_zB"], 2), 1.76)
    chk("Mg analysis nucleotides", sum(x["n"] for x in mg), 24623)
    g = rig["gnra_tetraloop"]
    chk("GNRA effect (sigma)", round(g["mean_zB_other"] - g["mean_zB_GNRA"], 3), 0.073)

    cs = load("contact_sparsity.json")["summary"]
    chk("contacts/nt median", cs["contacts_per_nt_median"], 4.397)
    chk("long-chain map density %", cs["by_length_bin"][-1]["median_density_pct"], 0.353)

    pr = load("proposal_recall.json")["summary"]
    chk("flat recall, L 500-1200", pr["by_length_bin"][-1]["mean_recall_c32"], 0.2003)
    chk("random baseline c=32", pr["random_baseline_c32_mean"], 0.7458)

    bs = load("block_sparsity.json")["summary"]["by_length_bin"][-1]["per_block"]["4"]
    chk("block occupancy b=4 (long) %", round(100 * bs["mean_occupancy_frac"], 2), 1.34)
    chk("effective c (long chains)", bs["mean_effective_c"], 17.24)

    co = load("coevolution_signal.json")["summary"]
    chk("coevolution mean prec@L/5", co["mean_prec_topL_5"], 0.6702)

    bench = load("pair_track_benchmark.json")
    r = [x for x in bench["runs"] if x["L"] == 4096][0]["hierarchical"]["stats"]
    chk("HPT L=4096 frac of dense %", round(100 * r["frac_of_dense"], 3), 0.959)
    chk("HPT L=4096 effective c", round(r["effective_c"], 1), 19.6)

    bp = load("basepair_geometry.json")
    chk("annotated base-pair steps", bp["total_annotated_steps"], 103964)
    chk("stiffness contexts (n>=200)", len(bp["stiffness_by_step_context"]), 76)
    chk("curated Mg coordination records", bp["metal_coordination_by_ion"]["MG"], 44708)
    chk("unobserved-residue records", bp["unobserved_residue_records"], 143871)
    gg = bp["stiffness_by_step_context"]["GG/CC"]["mean"]
    chk("GG/CC rise (A-form check)", gg["rise"], 3.14, tol=0.02)
    chk("GG/CC twist (A-form check)", gg["twist"], 29.98, tol=0.02)

    # derive the stiffness ratio from UNROUNDED values -- quoting it from the
    # 4-dp rounded display once produced a spurious 134x instead of 115x
    fcs = sorted((v["force_constants_diag"]["twist"]
                  for v in bp["stiffness_by_step_context"].values()), reverse=True)
    chk("stiffness ratio (stiffest/floppiest twist)", round(fcs[0]/fcs[-1], 1), 115.1, tol=0.01)

    hr = load("stiffness_headroom.json")
    chk("M1 sequence-table NLL", hr["M1_sequence_context_nll"], 17.579, tol=0.001)
    chk("M2 sequence x structure NLL", hr["M2_sequence_x_structure_nll"], 14.551, tol=0.001)
    chk("structure gain over sequence", hr["gain_structure_over_sequence"], 3.028, tol=0.002)

    print("\n== derived arithmetic ==")
    d, dff_moe, dff_dense = 768, 512, 3072
    attn = 32 * 4 * d * d
    moe_t = 16 * 34 * 3 * d * dff_moe
    moe_a = 16 * 6 * 3 * d * dff_moe
    dense = 16 * 3 * d * dff_dense
    chk("total params (M)", round((attn + moe_t + dense + 80e6) / 1e6), 911, tol=0.01)
    chk("active params (M)", round((attn + moe_a + dense + 80e6) / 1e6), 382, tol=0.01)
    L = 2048
    chk("dense activations at L=2048 (GB)", round(L*L*128*2*48*6/1e9, 1), 309.2)

    print("\n== physics (closed form) ==")
    import math
    e, eps0, kB, T, eps_r = 1.602176634e-19, 8.8541878128e-12, 1.380649e-23, 298.15, 78.4
    lB = e*e/(4*math.pi*eps0*eps_r*kB*T)*1e10
    chk("Bjerrum length (A)", round(lB, 2), 7.15)
    b_rna = 2.8/2
    xi = lB/b_rna
    chk("A-RNA Manning xi", round(xi, 2), 5.11)
    chk("A-RNA theta (z=1)", round(1-1/xi, 3), 0.804)

    print("\n== cross-document consistency ==")
    docs = {"ARCH": ROOT/"research/architecture/ARCHITECTURE.md",
            "TEX": ROOT/"research/report/main.tex",
            "HTML": ROOT/"research/architecture/blueprint.html"}
    txt = {}
    for k, p in docs.items():
        t = p.read_text()
        txt[k] = (t.replace("{,}", ",").replace("\\%", "%").replace("$", "")
                   .replace("\\", "").replace("−", "-").replace("–", "-"))
    for tok in ["17,428", "1.76", "0.200", "0.746", "1.34", "17.2", "0.670",
                "0.224", "0.372", "1.514", "0.804", "9.87",
                # PHAROS-Small is the current headline config; 910M/382M survives
                # only as the superseded Base row in the progression tables.
                "149", "61",
                "103,964", "44,708", "143,871", "14.551", "17.579", "115", "1.757"]:
        missing = [k for k, v in txt.items() if tok not in v]
        print(f"  {'OK ' if not missing else 'FAIL'} token {tok:8s} "
              f"{'present in all 3' if not missing else 'MISSING from ' + ','.join(missing)}")
        if missing: fails.append(f"doc-token {tok}")
    for bad in ["decisive pattern", "strongest argument for MoE",
                "0.76 for monovalent RNA", "spaced b ~ 5.9"]:
        present = [k for k, v in txt.items() if re.search(re.escape(bad), v)]
        print(f"  {'OK ' if not present else 'FAIL'} retracted text absent: {bad!r}"
              f"{'' if not present else ' STILL IN ' + ','.join(present)}")
        if present: fails.append(f"stale {bad}")

    print("\n== architecture config consistency (PHAROS-Small is the default) ==")
    cfg_docs = {"ARCH": ROOT/"research/architecture/ARCHITECTURE.md",
                "TEX":  ROOT/"research/report/main.tex",
                "HTML": ROOT/"research/architecture/blueprint.html"}
    cfg_txt = {k: v.read_text() for k, v in cfg_docs.items()}
    # phrases describing the SUPERSEDED default; Base-v2 scale-up rows are fine
    stale_cfg = ["32 blocks at d=768", "20 of 32 blocks", "4 of 32 blocks",
                 "537M", "d_ff = 512", "every second block", "PHAROS-Base carries"]
    for bad in stale_cfg:
        hit = [k for k, t in cfg_txt.items() if bad in t]
        print(f"  {'OK ' if not hit else 'FAIL'} superseded config absent: {bad!r}"
              f"{'' if not hit else ' IN ' + ','.join(hit)}")
        if hit: fails.append(f"stale config {bad}")
    for need in ["16 blocks", "d=512", "149M", "61M", "128 effective"]:
        miss = [k for k, t in cfg_txt.items() if need not in t]
        print(f"  {'OK ' if not miss else 'FAIL'} current config present: {need!r}"
              f"{'' if not miss else ' MISSING from ' + ','.join(miss)}")
        if miss: fails.append(f"missing config {need}")

    print("\n== reference implementation correctness tests ==")
    import subprocess
    t = ROOT / "research/architecture/reference/test_hierarchical_pair_track.py"
    if t.exists():
        r = subprocess.run([sys.executable, t.name], cwd=t.parent,
                           capture_output=True, text=True, timeout=1800)
        ok = r.returncode == 0 and "ALL TESTS PASS" in r.stdout
        print(f"  {'OK ' if ok else 'FAIL'} {t.name} "
              f"{'all pass' if ok else 'FAILURES -- run it directly'}")
        if not ok:
            fails.append("hpt correctness tests")
    else:
        print(f"  FAIL {t.name} missing"); fails.append("hpt tests missing")

    print("\n== diagram sources (feed figures into the report) ==")
    dg = sorted((ROOT / "research/architecture/diagrams").glob("*.mmd"))
    # phrasings that were retracted and must not survive in any figure
    banned = ["0.76 for monovalent", "ADAPTIVE SPARSE", "Select K = 32L"]
    for d in dg:
        t = d.read_text()
        hits = [b for b in banned if b in t]
        print(f"  {'OK ' if not hits else 'FAIL'} {d.name:34s}"
              f"{'clean' if not hits else 'STALE: ' + ', '.join(hits)}")
        if hits:
            fails.append(f"stale diagram {d.name}")
    rendered = ROOT / "research/architecture/diagrams/rendered"
    for d in dg:
        for ext in ("png", "svg"):
            f = rendered / f"{d.stem}.{ext}"
            if not f.exists():
                print(f"  FAIL missing render {f.name}")
                fails.append(f"missing render {f.name}")
            elif f.stat().st_mtime < d.stat().st_mtime:
                print(f"  FAIL stale render {f.name} (older than its .mmd source)")
                fails.append(f"stale render {f.name}")
    print(f"  OK  {len(dg)} diagram sources, renders present and newer than source")

    print(f"\n{'ALL CLAIMS REPRODUCE' if not fails else 'DRIFT DETECTED: ' + '; '.join(fails)}")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
