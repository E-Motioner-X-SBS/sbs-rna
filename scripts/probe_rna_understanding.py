#!/usr/bin/env python3
"""Does the model know anything about RNA, or is the accuracy an artefact?

Masked-token accuracy is easy to inflate and hard to interpret. Four things
can produce a respectable number with no understanding behind it:

1. **BERT's 80/10/10 corruption scores positions the model can see.** Of the
   selected positions, 10% are left UNCHANGED in the input and 10% are replaced
   with a random base. Both are scored. A model that simply copies its input
   gets every one of the unchanged positions right -- roughly a tenth of the
   headline accuracy, free.
2. **Base composition.** Always predicting the commonest base scores ~27% on a
   near-uniform alphabet, and more on a skewed corpus.
3. **Copying a neighbour.** RNA is locally correlated; predicting `x[i-1]`
   scores well above chance without any structural knowledge.
4. **Collapsing onto one or two classes**, which a pooled accuracy hides
   entirely and a per-class breakdown does not.

So this reports accuracy split by corruption type, against those baselines,
with a confusion matrix -- and then asks whether the model has learned
something that is specifically about RNA:

**Complementarity.** Watson-Crick pairing is the central fact of RNA structure.
If a base is masked and its partner is visible, a model that understands
pairing should predict the complement. This measures that directly on real
base pairs taken from deposited structures, against the same base's prediction
when its partner is ALSO masked -- so the comparison isolates the use of the
partner rather than the model's prior over bases.

    python scripts/probe_rna_understanding.py --ckpt <checkpoint>
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

BASES = "ACGU"
COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}


def _trainer():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_pm", ROOT / "scripts/pretrain_mlm.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["_pm"] = m
    spec.loader.exec_module(m)
    return m


@torch.no_grad()
def breakdown(model, pm, seqs, device, budget, n_loops, seed):
    """Accuracy split by what the model could actually see at each position."""
    from pharos.model.moe import RouterFeatures
    rng = np.random.default_rng(seed)
    bc = pm.BatchChemistry(pm.SYMBOLS, device)
    # kind: 0 = [MASK], 1 = kept unchanged (visible answer), 2 = random substitute
    stat = {k: [0, 0] for k in (0, 1, 2)}
    conf = np.zeros((4, 4), dtype=np.int64)
    copy_prev = [0, 0]
    groups, cur, cl = [], [], 0
    for s in sorted(seqs, key=len):
        L = pm._quantised(len(s))
        if cur and (len(cur) + 1) * max(cl, L) > budget:
            groups.append(cur); cur, cl = [], 0
        cur.append(s); cl = max(cl, L)
    if cur:
        groups.append(cur)

    for grp in groups:
        tok, mask, lengths = pm.encode_batch(grp)
        inp, tgt, sel = pm.apply_span_mask(tok.copy(), lengths, rng)
        if not sel.any():
            continue
        kind = np.full(inp.shape, -1, dtype=np.int8)
        kind[sel & (inp == pm.MASK_ID)] = 0
        kind[sel & (inp == tok)] = 1
        kind[sel & (inp != tok) & (inp != pm.MASK_ID)] = 2

        it = torch.as_tensor(inp, device=device)
        bm = torch.as_tensor(mask, device=device)
        ch = bc(it, bm)
        nr = bm.sum(1)
        ft = RouterFeatures(length=nr.float(),
                            chem_summary=(ch.sum(1) / nr.unsqueeze(1).clamp(min=1))[:, :5])
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            out = model(it, torch.zeros_like(it), ch, bm, feats=ft,
                        n_loops=n_loops, mlm=True)
        pred = out["mlm_logits"].float().argmax(-1).cpu().numpy()

        for k in (0, 1, 2):
            m = (kind == k)
            if m.any():
                stat[k][0] += int((pred[m] == tgt[m]).sum())
                stat[k][1] += int(m.sum())
        m0 = (kind == 0)
        for a, b in zip(tgt[m0], pred[m0]):
            if 0 <= a < 4 and 0 <= b < 4:
                conf[a, b] += 1
        # "copy the previous residue" baseline, on genuinely masked positions
        bi, pi = np.nonzero(m0)
        keep = pi > 0
        if keep.any():
            prev = tok[bi[keep], pi[keep] - 1]
            copy_prev[0] += int((prev == tgt[bi[keep], pi[keep]]).sum())
            copy_prev[1] += int(keep.sum())
    return stat, conf, copy_prev


@torch.no_grad()
def complementarity(model, pm, pairs_data, device, n_loops):
    """Does a visible partner change the prediction toward its complement?

    Two conditions on the same base pairs: partner VISIBLE and partner ALSO
    MASKED. The difference is the only thing that can be attributed to using
    the partner, because everything else -- the base's own context, the model's
    prior, the position -- is held fixed.
    """
    from pharos.model.moe import RouterFeatures
    bc = pm.BatchChemistry(pm.SYMBOLS, device)
    res = {}
    for cond in ("partner_visible", "partner_masked"):
        hit = tot = 0
        for seq, prs in pairs_data:
            tok, mask, lengths = pm.encode_batch([seq])
            inp = tok.copy()
            sel = np.zeros_like(inp, dtype=bool)
            for i, j in prs:
                if i >= inp.shape[1] or j >= inp.shape[1]:
                    continue
                inp[0, i] = pm.MASK_ID
                sel[0, i] = True
                if cond == "partner_masked":
                    inp[0, j] = pm.MASK_ID
            if not sel.any():
                continue
            it = torch.as_tensor(inp, device=device)
            bm = torch.as_tensor(mask, device=device)
            ch = bc(it, bm)
            nr = bm.sum(1)
            ft = RouterFeatures(length=nr.float(),
                                chem_summary=(ch.sum(1) / nr.unsqueeze(1).clamp(min=1))[:, :5])
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                o = model(it, torch.zeros_like(it), ch, bm, feats=ft,
                          n_loops=n_loops, mlm=True)
            pred = o["mlm_logits"].float().argmax(-1).cpu().numpy()
            for i, j in prs:
                if i >= inp.shape[1]:
                    continue
                hit += int(pred[0, i] == tok[0, i])
                tot += 1
        res[cond] = (hit, tot)
    return res


def real_base_pairs(limit_chains: int = 40, max_len: int = 400):
    """Real Watson-Crick pairs from deposited structures, with their sequences.

    Taken from the RNA-Puzzles reference structures via the full-atom detector,
    so these are pairs a crystallographer observed, not pairs a model or a
    folding program proposed.
    """
    from pharos.eval.base_pairs import full_atom_pairs
    from pharos.eval.blind_tests import rna_puzzles
    from pharos.eval.structure import read_structure
    out = []
    for t in rna_puzzles():
        try:
            s = read_structure(t.reference)
            prs = full_atom_pairs(t.reference)
        except Exception:                                    # noqa: BLE001
            continue
        if not prs or len(s) > max_len:
            continue
        seq = s.seq.replace("N", "A")
        # only canonical pairs: a non-canonical pair has no complement to predict
        keep = [(i, j) for i, j in prs
                if i < len(seq) and j < len(seq)
                and COMPLEMENT.get(seq[i]) == seq[j]]
        if keep:
            out.append((seq, keep))
        if len(out) >= limit_chains:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[ROOT / "data/derived/parquet_starter"])
    ap.add_argument("--n-seq", type=int, default=512)
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--n-loops", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from pharos.model.pharos import Pharos, PharosConfig
    pm = _trainer()
    device = torch.device(args.device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved = sd.get("cfg") or {}
    cfg = PharosConfig(**{k: v for k, v in saved.items()
                          if k in PharosConfig.__dataclass_fields__})
    for k, v in saved.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    model = Pharos(cfg).to(device).eval()
    model.load_state_dict({k.replace("_orig_mod.", ""): v
                           for k, v in sd["model"].items()}, strict=False)
    print(f"[probe] {args.ckpt.name}, step {sd.get('step','?')}, "
          f"{sd.get('tokens',0)/1e6:.0f}M tokens\n")

    import pyarrow.parquet as pq
    rng = np.random.default_rng(args.seed)
    seqs = []
    for d in args.corpus:
        for f in sorted(Path(d).glob("*.parquet")):
            col = pq.read_table(f, columns=["sequence"])["sequence"].to_pylist()
            col = [s.upper().replace("T", "U") for s in col if s]
            col = [s for s in col if 20 <= len(s) <= 1024]
            if not col:
                continue
            take = min(len(col), args.n_seq - len(seqs))
            seqs += [col[int(i)] for i in rng.choice(len(col), take, replace=False)]
            if len(seqs) >= args.n_seq:
                break
        if len(seqs) >= args.n_seq:
            break

    stat, conf, copy_prev = breakdown(model, pm, seqs, device,
                                      args.token_budget, args.n_loops, args.seed)
    names = {0: "[MASK]  (must infer)", 1: "kept    (answer VISIBLE)",
             2: "random  (wrong base shown)"}
    tot_h = sum(v[0] for v in stat.values())
    tot_n = sum(v[1] for v in stat.values())
    print("== accuracy split by what the model could see ==")
    for k in (0, 1, 2):
        h, n = stat[k]
        if n:
            print(f"  {names[k]:28s} {h/n:.4f}   {n:7,d} positions "
                  f"({100*n/tot_n:.1f}% of scored)")
    print(f"  {'POOLED (the headline)':28s} {tot_h/max(tot_n,1):.4f}   {tot_n:,d}")
    h0, n0 = stat[0]
    inflation = tot_h/max(tot_n,1) - h0/max(n0,1)
    print(f"\n  the headline is {inflation:+.4f} above the honest number, because "
          f"{100*stat[1][1]/max(tot_n,1):.0f}% of scored positions show the answer")

    print("\n== baselines, on genuinely masked positions only ==")
    counts = Counter(c for s in seqs for c in s if c in BASES)
    tot_b = sum(counts.values())
    major = max(counts.values()) / tot_b
    print(f"  always the commonest base      {major:.4f}")
    print(f"  copy the previous residue      {copy_prev[0]/max(copy_prev[1],1):.4f}")
    print(f"  MODEL                          {h0/max(n0,1):.4f}")

    print("\n== does it use all four bases, or collapse? ==")
    print(f"  {'true\\pred':>10s} " + " ".join(f"{b:>7s}" for b in BASES))
    for i, b in enumerate(BASES):
        row = conf[i]
        print(f"  {b:>10s} " + " ".join(f"{v:7,d}" for v in row)
              + f"   recall {row[i]/max(row.sum(),1):.3f}")
    pred_share = conf.sum(0) / max(conf.sum(), 1)
    print("  predicted share: " + ", ".join(
        f"{b} {100*p:.1f}%" for b, p in zip(BASES, pred_share)))

    print("\n== RNA-SPECIFIC: does a visible partner pull the prediction to its complement? ==")
    pd = real_base_pairs()
    if not pd:
        print("  (no reference structures available)")
    else:
        npair = sum(len(p) for _s, p in pd)
        print(f"  {len(pd)} chains, {npair} canonical Watson-Crick pairs from "
              f"deposited structures")
        r = complementarity(model, pm, pd, device, args.n_loops)
        vh, vt = r["partner_visible"]
        mh, mt = r["partner_masked"]
        print(f"  partner VISIBLE  {vh/max(vt,1):.4f}  ({vh}/{vt})")
        print(f"  partner MASKED   {mh/max(mt,1):.4f}  ({mh}/{mt})")
        d = vh/max(vt,1) - mh/max(mt,1)
        print(f"  difference       {d:+.4f}   <- attributable to USING the partner")
        print("\n  A model that has learned complementarity scores higher with the")
        print("  partner visible. Zero difference means it is not using pairing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
