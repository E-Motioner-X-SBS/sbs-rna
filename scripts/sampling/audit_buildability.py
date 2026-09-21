#!/usr/bin/env python3
"""Cycle-5 B1/B2: could this be implemented from the spec, without inventing numbers?

Defects #18 (attention head count) and #19 (decoder size) were both components
never specified at all, then silently assumed small. #19 showed they are not.
This enumerates every component the architecture needs end to end and greps the
spec for the concrete quantity each one requires.
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = {
    # the final specification, plus the v0.1 trail: some component detail was
    # written once, in v0.1, and never restated
    "ARCH": ROOT / "research/architecture/ARCHITECTURE.md",
    "ARCH_V01": ROOT / "history_of_failed_attempts/ARCHITECTURE_v0.1_with_corrections.md",
    "CFG_M": ROOT / "configs/model/pharos_small.yaml",
    "CFG_T": ROOT / "configs/train/pharos_small_stage1.yaml",
    "REF": ROOT / "research/architecture/reference/hierarchical_pair_track.py",
    "PHYS": ROOT / "src/pharos/physics/manning.py",
}
TEXT = {k: (p.read_text() if p.exists() else "") for k, p in SPEC.items()}
ALL = "\n".join(TEXT.values())

# component, the quantity an implementer must know, regex that would evidence it
COMPONENTS = [
    ("trunk: d_model",              "512",              r"d_model:\s*512|d\s*=\s*512"),
    ("trunk: block count",          "16",               r"n_blocks:\s*16|16 blocks"),
    ("trunk: attention heads",      "8 x 64",           r"n_heads:\s*8|8 heads"),
    ("trunk: mixer schedule",       "GDN/SWA/FULL order", r"mixer_pattern|\[GDN"),
    ("trunk: SWA window",           "128",              r"swa_window:\s*128|window 128"),
    ("GDN: state size",             "linear-attn state dim", r"gdn.*state|state.*size|d_state"),
    ("MoE: routed/shared/top-k",    "32/2/4",           r"routed_experts:\s*32"),
    ("MoE: expert width d_ff",      "128",              r"d_ff:\s*128"),
    ("MoE: balancing bias step",    "bias update rate", r"bias.*(update|step).*rate|balanc.*rate|update_rate"),
    ("router: input features",      "list",             r"router_inputs:"),
    ("router: recycle-0 behaviour", "first-pass rule",  r"recycle 0 the structural router|route on .*Neff.*first pass"),
    ("pair track: b1/b2",           "16/4",             r"b1:\s*16"),
    ("pair track: budget c",        "20",               r"target_c:\s*20|c = 20|c=20"),
    ("pair track: d_pair",          "128",              r"d_pair\s*[:=]\s*128"),
    ("pair track: triangle layers", "2",                r"n_tri_layers:\s*2"),
    ("motif bank: n classes",       "667",              r"667"),
    ("motif bank: key construction","how keys are built at INFERENCE", r"motif.*key.*infer|key.*computed.*recycle"),
    ("stiffness: parameterisation", "Cholesky 6x6",     r"Cholesky|cholesky"),
    ("stiffness: acceptance gate",  "15.3424",          r"15\.3424"),
    ("heads: count + sizes",        "9 heads itemised", r"Prediction heads \(9\)"),
    ("decoder: type + blocks",      "frame diffusion, 4", r"n_blocks:\s*4|4 IPA|4-block"),
    ("decoder: K states",           "3",                r"K\s*=\s*3|K=3|n_states:\s*3"),
    ("decoder: diffusion steps",    "n denoise steps",  r"diffusion.*steps|denois.*steps|n_steps"),
    ("loss: weights lambda_1..5",   "numeric values",   r"lambda_?1\s*[:=]\s*[0-9]|λ₁\s*=\s*[0-9]"),
    ("loss: block-occupancy weight","pos_weight rule",  r"pos_weight"),
    ("train: optimizer",            "Muon + AdamW",     r"name:\s*muon"),
    ("train: learning rate",        "peak LR",          r"learning_rate|lr:\s*[0-9]|peak_lr"),
    ("train: batch size",           "tokens/step",      r"batch_size|tokens_per_step|global_batch"),
    ("train: warmup / schedule",    "warmup steps",     r"warmup"),
    ("train: token budget",         "25B",              r"tokens_total:\s*25e9"),
    ("train: precision",            "bf16",             r"default:\s*bf16"),
    ("data: split rule",            "md5 % 100",        r"md5\(header\)|md5.*%\s*100"),
    ("data: chunk weighting",       "per-chunk weights", r"down_weight_151nt|chunk_weights"),
    ("data: max context",           "4096",             r"max_len:\s*4096"),
    ("data: tokenizer vocab",       "5 + extended",     r"pretraining:\s*5"),
    ("eval: blind sets",            "CASP15/16, RNA-Puzzles", r"CASP15|CASP16"),
    ("eval: stratified reporting",  "isolated vs complexed", r"isolated vs complexed|split by isolated"),
]

PROVISIONAL = {"loss: weights lambda_1..5", "motif bank: key construction"}
rows, missing, partial = [], [], []
for name, needs, rx in COMPONENTS:
    hit = bool(re.search(rx, ALL, re.I))
    # a deliberate deferral with a documented procedure is not a gap
    prov = bool(re.search(r"PROVISIONAL|fix_by", ALL)) and name in PROVISIONAL
    status = "SPECIFIED" if hit else ("PROVISIONAL" if prov else "MISSING")
    rows.append({"component": name, "needs": needs, "status": status})
    (partial if status == "SPECIFIED" else missing).append(name)

print(f"{'component':34s}{'must know':30s}status")
for r in rows:
    mark = {"SPECIFIED": "OK  ", "PROVISIONAL": "PROV", "MISSING": "MISS"}[r["status"]]
    print(f"  {mark} {r['component']:32s}{r['needs']:30s}{r['status']}")

print(f"\n  SPECIFIED {len(partial)} / {len(COMPONENTS)}")
print(f"  MISSING   {len(missing)}:")
for m in missing:
    print(f"    - {m}")

(ROOT / "data/samples/analysis/buildability_audit.json").write_text(
    json.dumps({"n_components": len(COMPONENTS), "n_specified": len(partial),
                "n_missing": len(missing), "missing": missing, "rows": rows}, indent=2))
