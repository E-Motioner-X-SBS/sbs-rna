#!/usr/bin/env python3
"""Cycle-3: DECIDE architecture size + numeric format jointly.

The two decisions cannot be made separately. "Low-Bit Quantization Favors
Undertrained LLMs" (ACL 2025, arXiv 2411.17691, 1500+ checkpoints) finds
quantization-induced degradation (QiD) is WORST for *small* models trained on
*many* tokens -- and mildest for large or undertrained ones.

PHAROS-Small is 60.65M active at 323B tokens. That is the regime the scaling law
flags as worst-case. This scores the whole size ladder against that law and the
measured information content of RNA, then states the decision.
"""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "samples" / "analysis"

CHINCHILLA = 20.0          # tokens per parameter, compute-optimal
LADDER = [  # name, d, blocks, loops, total_M, active_M
    ("PHAROS-Micro", 256,  8, 16,  23,  12),
    ("PHAROS-Mini",  384, 12, 12,  67,  30),
    ("PHAROS-Small", 512, 16,  8, 149,  61),
    ("PHAROS-Base-v2", 768, 32, 3, 1401, 269),
]
TOKENS = [5e9, 25e9, 100e9, 323e9]     # the staged budget already in the design


def main():
    print("=== token / active-parameter ratio across the ladder ===")
    print(f"  Chinchilla-optimal ~= {CHINCHILLA:.0f} tokens per parameter\n")
    print(f"{'model':16s}{'active':>9s}" + "".join(f"{t/1e9:>10.0f}B" for t in TOKENS))
    rows = []
    for name, d, blk, loops, tot, act in LADDER:
        cells = []
        for T in TOKENS:
            r = T / (act * 1e6)
            cells.append(f"{r:>10,.0f}")
        print(f"{name:16s}{act:>7d}M" + "".join(cells))
        rows.append({"model": name, "active_M": act,
                     "ratios": {f"{int(t/1e9)}B": round(t/(act*1e6), 1) for t in TOKENS}})
    print("\n  (values are tokens per active parameter)")

    print("\n=== where each sits relative to the QiD scaling law ===")
    print("  Law: QiD is WORST for small models with many tokens;")
    print("       mildest for large models or fewer tokens.")
    print("       'Smaller models require higher precision at high token counts.'\n")
    print(f"{'model':16s}{'tok/param @323B':>18s}{'x Chinchilla':>14s}   verdict for 4-bit")
    dec = []
    for name, d, blk, loops, tot, act in LADDER:
        r = 323e9 / (act * 1e6)
        x = r / CHINCHILLA
        if x > 200:   v = "WORST CASE - avoid 4-bit"
        elif x > 50:  v = "risky - validate QiD first"
        else:         v = "acceptable"
        print(f"{name:16s}{r:>18,.0f}{x:>13,.0f}x   {v}")
        dec.append({"model": name, "tok_per_param": round(r), "x_chinchilla": round(x), "verdict": v})

    print("\n=== the conflict, stated plainly ===")
    act_small = 61e6
    r = 323e9 / act_small
    print(f"  PHAROS-Small: {act_small/1e6:.0f}M active x 323B tokens = {r:,.0f} tok/param")
    print(f"                = {r/CHINCHILLA:,.0f}x Chinchilla-optimal")
    print("  The design simultaneously wants:")
    print("    (a) the SMALLEST viable model  (right-sizing, cycle 2)")
    print("    (b) the FULL 323B-token corpus (one epoch of elDORS)")
    print("    (c) 4-bit NVFP4 training/deployment (cycle 3 precision work)")
    print("  The scaling law says (a)+(b) together is precisely the regime where")
    print("  (c) degrades most. These three cannot all hold.")

    print("\n=== what RNA's information content adds ===")
    ent = 2.0167            # measured bits/nt, elDORS
    attr = 58.9             # measured bits/nt fully attributed
    print(f"  measured sequence entropy        {ent:.4f} bits/nt")
    print(f"  measured full attribute content  {attr:.1f} bits/nt")
    print("  RNA is far lower-entropy than natural language (~10-12 bits/token).")
    print("  So the 323B 'tokens' carry much less information than 323B LM tokens:")
    for T in (323e9,):
        print(f"    323B nt x {ent:.2f} bits = {T*ent/8/1e12:.1f} TB of raw sequence information")
        print(f"    against ~{T*11/8/1e12:.1f} TB for the same count of English tokens")
    print("  => the effective over-training is even worse than the raw ratio suggests,")
    print("     because each RNA token is ~5x less informative than an LM token.")

    res = {"chinchilla_tokens_per_param": CHINCHILLA,
           "ladder": rows, "qid_verdicts": dec,
           "sequence_entropy_bits": ent, "full_attribute_bits": attr,
           "conflict": "smallest model + full corpus + 4-bit are mutually incompatible"}
    (OUT / "precision_size_decision.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
