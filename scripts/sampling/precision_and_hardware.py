#!/usr/bin/env python3
"""Cycle-3 P1/P3: hardware-explicit cost, and the precision budget RNA actually needs.

Fixes defect #15: the published cost is quoted in A100-hours while applying a
1.60x FP8 factor. A100 (Ampere, SM80) has NO FP8 tensor cores -- FP8 arrived
with Hopper. "79 A100-hours" is therefore a hybrid unit that exists on no single
machine. Every figure here names its hardware AND its numeric format.

Also tests H1/H2: is precision a bottleneck for RNA at all?
"""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "samples" / "analysis"

# Dense tensor-core peak, TFLOP/s. Formats a generation does not support are None.
HW = {
    #                       bf16    fp8     fp4
    "A100-80G  (Ampere)":  (312e12, None,   None),
    "H100-80G  (Hopper)":  (989e12, 1979e12, None),
    "B200      (Blackwell)": (2250e12, 4500e12, 9000e12),
}
MFU = 0.35

# PHAROS-Small
N_ACTIVE, N_TOTAL = 60.65e6, 148.73e6
LOOPS = 8          # defect #17: refinement loops DO cost compute
TOKENS = 25e9      # cycle-3 decision D2 (was 323e9)


def hours(flops, peak):
    return flops / (peak * MFU) / 3600


def main():
    F = LOOPS * 6 * N_ACTIVE * TOKENS
    print(f"PHAROS-Small: {N_ACTIVE/1e6:.1f}M active x {LOOPS} refinement loops, {TOKENS/1e9:.0f}B tokens")
    print(f"  training FLOPs (loops*6*N*T) = {F/1e21:.2f} ZFLOPs\n")

    print(f"{'hardware':24s}{'bf16':>12s}{'fp8':>12s}{'fp4':>12s}   (GPU-hours)")
    rows = {}
    for name, (bf16, fp8, fp4) in HW.items():
        cells, rows[name] = [], {}
        for tag, peak in (("bf16", bf16), ("fp8", fp8), ("fp4", fp4)):
            if peak is None:
                cells.append("n/a"); rows[name][tag] = None
            else:
                h = hours(F, peak); cells.append(f"{h:,.0f}"); rows[name][tag] = round(h, 1)
        print(f"{name:24s}" + "".join(f"{c:>12s}" for c in cells))

    print("\nDEFECT #15 — the published figure is not a real unit:")
    a100 = rows["A100-80G  (Ampere)"]["bf16"]
    h100_fp8 = rows["H100-80G  (Hopper)"]["fp8"]
    print(f"  A100 bf16 (no FP8 possible)          {a100:8,.0f} h")
    print(f"  H100 fp8                             {h100_fp8:8,.0f} h")
    print(f"  ratio                                {a100/h100_fp8:8.1f}x")
    print("  The doc applied a 1.60x FP8 lever and kept reporting A100-hours.")
    print("  '3.3 days on a single A100' is unreachable: A100 cannot run FP8 at all.")

    # ---- H1: how much information is in a nucleotide token? ----
    print("\n=== H1: is precision the bottleneck in the token path? ===")
    comp = {"A": 0.2797, "U": 0.2681, "G": 0.2280, "C": 0.2210, "N": 0.0031}  # elDORS 2M census
    H = -sum(p * math.log2(p) for p in comp.values() if p > 0)
    print(f"  measured elDORS composition entropy  {H:.4f} bits/nucleotide")
    for d, bits in ((512, 16), (512, 8), (512, 4)):
        print(f"  one token as {d}-dim @ {bits:2d}-bit      {d*bits:6,d} bits "
              f"= {d*bits/H:8,.0f}x the identity content")
    print("  => the token path is over-provisioned by 3-4 ORDERS OF MAGNITUDE.")
    print("     Precision is not the binding constraint on sequence identity.")

    # ---- H2: what coordinate precision does the data support? ----
    print("\n=== H2: what coordinate precision do the STRUCTURES support? ===")
    med_res = 3.10   # catalogue: median resolution, A
    print(f"  catalogue median resolution          {med_res:.2f} A")
    for name, step in (("fp16 (~1e-3 A at 100 A)", 1e-3), ("int8 over 0-256 A", 1.0),
                       ("int8 over 0-64 A", 0.25), ("int12 over 0-256 A", 256/4096)):
        ratio = med_res / step
        print(f"  {name:26s} step {step:6.3f} A  -> {ratio:8.0f}x finer than the noise floor")
    print("  => coordinate storage finer than ~0.1 A encodes experimental noise.")

    # ---- corrected lever model: hardware-independent vs hardware-dependent ----
    print("\n=== corrected cost model: levers separated by what they depend on ===")
    HW_INDEP = [("Muon optimiser", 2.00, "optimiser-level; reported ~2x, unbenchmarked here"),
                ("down-weight 151-nt read chunks", 1.19, "data-level; 16.2% of nucleotide mass")]
    print("  hardware-INDEPENDENT levers (apply on any generation):")
    f_indep = 1.0
    for nm, f, why in HW_INDEP:
        f_indep *= f
        print(f"    {nm:32s} {f:.2f}x   {why}")
    print(f"    {'combined':32s} {f_indep:.2f}x")
    print()
    print(f"  {'hardware + format':28s}{'raw h':>9s}{'with levers':>13s}   note")
    ladder = []
    for name, (bf16, fp8, fp4) in HW.items():
        for tag, peak in (("bf16", bf16), ("fp8", fp8), ("fp4", fp4)):
            if peak is None:
                continue
            raw = hours(F, peak); net = raw / f_indep
            note = {"bf16": "", "fp8": "needs Hopper+", "fp4": "needs Blackwell (NVFP4)"}[tag]
            print(f"  {name.split()[0]+' '+tag:28s}{raw:9,.0f}{net:13,.0f}   {note}")
            ladder.append({"hw": name.split()[0], "format": tag,
                           "raw_hours": round(raw, 1), "with_levers_hours": round(net, 1),
                           "note": note})
    print()
    g = {f"{r['hw']}_{r['format']}": r['with_levers_hours'] for r in ladder}
    print(f"  Honest headline at {TOKENS/1e9:.0f}B tokens x {LOOPS} loops:")
    print(f"    A100 bf16 {g['A100-80G_bf16']:.0f} h  |  H100 fp8 {g['H100-80G_fp8']:.0f} h"
          f"  |  B200 NVFP4 {g['B200_fp4']:.0f} h (not adopted -- see D3)")
    print("  Two corrections are baked in: defect #15 (A100 cannot run FP8) and")
    print("  defect #17 (refinement loops were omitted from the FLOP count entirely).")

    res = {"corrected_ladder": ladder,
           "hw_independent_factor": round(f_indep, 3),
           "defect_15": {"published_unit": "A100-hours with an FP8 lever applied",
                         "problem": "A100 (SM80) has no FP8 tensor cores; FP8 is Hopper+",
                         "a100_bf16_hours": a100, "h100_fp8_hours": h100_fp8},
           "hardware_explicit_hours": rows,
           "H1_token_entropy_bits": round(H, 4),
           "H1_overprovision_512d_bf16": round(512 * 16 / H),
           "H2_median_resolution_A": med_res}
    (OUT / "precision_hardware.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT/'precision_hardware.json'}")


if __name__ == "__main__":
    main()
