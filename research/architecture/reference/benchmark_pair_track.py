#!/usr/bin/env python3
"""Measure real peak memory and latency: hierarchical vs dense pair track.

Turns ARCHITECTURE.md section 5's cost arithmetic into measured numbers. The
dense baseline is expected to fail (OOM) at lengths the hierarchical track
handles comfortably -- that failure IS the result.
"""
from __future__ import annotations
import gc, json, resource, time, tracemalloc
from pathlib import Path

import torch

from hierarchical_pair_track import HPTConfig, HierarchicalPairTrack, DensePairTrack

OUT = Path(__file__).resolve().parents[3] / "data" / "samples" / "analysis"
LENGTHS = [128, 256, 512, 1024, 2048, 4096]
MEM_BUDGET_GB = 4.0        # refuse to try a config predicted to exceed this


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB -> GB


def run(model, tok, label):
    gc.collect()
    base = peak_rss_gb()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            out, idx, stats = model(tok)
        dt = time.perf_counter() - t0
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {"ok": True, "seconds": round(dt, 3),
                "torch_peak_rss_gb": round(max(peak_rss_gb() - base, 0), 3),
                "py_peak_mb": round(peak / 1e6, 1), "stats": stats}
    except (RuntimeError, MemoryError) as e:
        tracemalloc.stop()
        return {"ok": False, "error": type(e).__name__ + ": " + str(e)[:120]}


def main():
    cfg = HPTConfig()
    torch.manual_seed(0)
    hpt = HierarchicalPairTrack(cfg).eval()
    dense = DensePairTrack(cfg).eval()
    n_hpt = sum(p.numel() for p in hpt.parameters())
    n_dense = sum(p.numel() for p in dense.parameters())
    print(f"HPT params {n_hpt/1e6:.2f}M | Dense params {n_dense/1e6:.2f}M\n")

    rows = []
    for L in LENGTHS:
        tok = torch.randn(1, L, cfg.d_model)
        r = {"L": L}
        rh = run(hpt, tok, "hpt")
        r["hierarchical"] = rh
        if rh["ok"]:
            s = rh["stats"]
            print(f"L={L:5d} HPT   ok  {rh['seconds']:6.2f}s  "
                  f"peakRSS={rh['torch_peak_rss_gb']:5.2f}GB  "
                  f"l1={s['l1_entries_needed']:8,}  l3_pairs={s['l3_pairs']:9,}  "
                  f"={100*s['frac_of_dense']:6.3f}% dense  c={s['effective_c']:6.1f}")
        else:
            print(f"L={L:5d} HPT   FAILED  {rh['error']}")

        # predicted dense footprint; skip the run if it would thrash the machine
        pred_gb = L * L * cfg.d_pair * 4 * 6 / 1e9
        if pred_gb > MEM_BUDGET_GB:
            r["dense"] = {"ok": False, "skipped": True,
                          "predicted_min_gb": round(pred_gb, 2),
                          "reason": f"predicted >{MEM_BUDGET_GB} GB for activations alone"}
            print(f"L={L:5d} DENSE skipped -- predicted {pred_gb:.1f} GB activations "
                  f"(budget {MEM_BUDGET_GB} GB)")
        else:
            rd = run(dense, tok, "dense")
            r["dense"] = rd
            if rd["ok"]:
                print(f"L={L:5d} DENSE ok  {rd['seconds']:6.2f}s  "
                      f"peakRSS={rd['torch_peak_rss_gb']:5.2f}GB")
            else:
                print(f"L={L:5d} DENSE FAILED  {rd['error']}")
        rows.append(r)
        del tok; gc.collect()
        print()

    summary = {
        "note": "CPU-only machine (no CUDA); peak RSS is process-level and approximate",
        "hpt_params_M": round(n_hpt/1e6, 3),
        "config": {"b1": cfg.b1, "b2": cfg.b2, "d_pair": cfg.d_pair,
                   "keep_frac_l1": cfg.keep_frac_l1, "target_c": cfg.target_c},
        "runs": rows,
    }
    (OUT / "pair_track_benchmark.json").write_text(json.dumps(summary, indent=2))
    print("wrote", OUT / "pair_track_benchmark.json")


if __name__ == "__main__":
    main()
