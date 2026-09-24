#!/usr/bin/env python3
"""Tests for the frozen motif bank.

The properties worth pinning are the ones §8 argues for, because each of them
is a thing a later edit could quietly undo:

  1. IT IS THE ATLAS      -- 667 classes, 413 internal + 254 hairpin loops, the
                             numbers §8 states. If the compiler silently drops
                             classes the bank is a different object.
  2. IT IS FROZEN         -- the descriptors are facts about RNA and must be
                             buffers, not parameters. If they become trainable
                             the model is memorising motifs, which is the thing
                             §8 chose retrieval over.
  3. NO SEQUENCE PATH     -- §8's measured negative result is that GNRA k-mer
                             context predicts rigidity at 0.073 sigma. The bank
                             must be queryable ONLY from pair-derived features,
                             so there is no way to rebuild the thing that was
                             already shown not to work.
  4. RARE CLASSES ARE     -- 231 of 667 have >= 5 instances. A class seen twice
     DISTRUSTED             must not win against one seen 300 times on equal
                             evidence.
  5. IT STARTS CLOSED     -- an untrained bank must inject nothing into the pair
                             track.

Run: python3 src/pharos/model/test_motif_bank.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pharos.model.motif_bank import MotifBank, MotifBankConfig, load_bank  # noqa: E402

fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} {detail}")
    if not ok:
        fails.append(name)


def main() -> int:
    bank = load_bank()
    if bank is None:
        print("  FAIL motif bank not compiled -- run scripts/build_motif_bank.py")
        return 1
    meta = bank.meta

    print("== property 1: the bank is the atlas §8 describes ==")
    chk("667 motif classes", bank.n_motifs == 667, str(bank.n_motifs))
    chk("413 internal-loop + 254 hairpin-loop",
        (meta["n_internal_loop"], meta["n_hairpin_loop"]) == (413, 254),
        f"{meta['n_internal_loop']} + {meta['n_hairpin_loop']}")
    chk("release recorded", "4.12" in meta["release"], meta["release"])
    chk("the interaction vocabulary came from the data, not a literal",
        len(meta["bp_families"]) >= 15, f"{len(meta['bp_families'])} families")

    print("\n== property 2: descriptors are frozen ==")
    names = {n for n, _ in bank.named_parameters()}
    bufs = {n for n, _ in bank.named_buffers()}
    chk("descriptor is a buffer, not a parameter",
        "descriptor" in bufs and "descriptor" not in names, "")
    chk("no descriptor tensor requires grad",
        not bank.descriptor.requires_grad, "facts about RNA do not train")
    chk("the learned part is only the projections",
        names == {"log_temp", "key.weight", "value.weight", "query.weight",
                  "gate.weight", "gate.bias", "out_norm.weight", "out_norm.bias"},
        f"{len(names)} tensors")

    print("\n== property 3: there is no sequence query path ==")
    sig = inspect.signature(MotifBank.forward)
    chk("forward takes a pair-derived query only",
        list(sig.parameters)[1] == "pair_query",
        f"({', '.join(list(sig.parameters)[1:])})")
    src = Path(__file__).resolve().parents[0].joinpath("motif_bank.py").read_text()
    chk("the module builds no k-mer or sequence query",
        not any(t in src.lower() for t in ("kmer", "k_mer", "n_gram", "ngram")),
        "§8: GNRA k-mer context predicts rigidity at 0.073 sigma")

    print("\n== property 4: rare classes are distrusted ==")
    r = bank.reliability
    chk("reliability is bounded to (0, 1]",
        bool((r > 0).all() and (r <= 1).all()),
        f"min {float(r.min()):.2f} max {float(r.max()):.2f}")
    chk("231 classes reach full trust (>= 5 instances)",
        int((r >= 1.0).sum()) == 231, str(int((r >= 1.0).sum())))
    chk("reliability rises with instance count",
        bool((r[bank.n_instances.argmax()] >= r[bank.n_instances.argmin()])), "")

    print("\n== property 5: it starts closed ==")
    torch.manual_seed(0)
    q = torch.randn(32, bank.cfg.d_query)
    with torch.no_grad():
        out, info = bank(q)
    chk("gate is near zero before training", float(info["gate_mean"]) < 0.1,
        f"gate {float(info['gate_mean']):.3f}")
    chk("so the contribution is small", float(out.abs().mean()) < 0.2,
        f"mean |out| {float(out.abs().mean()):.3f}")

    print("\n== property 6: retrieval behaves ==")
    # `.eval()`, which this never called. `nn.Module` starts in training mode,
    # so a check named "deterministic in eval" was running the training path
    # and only passed because nothing in the bank had training-mode state yet.
    # The query-centring running mean does, and it caught this immediately.
    bank.eval()
    with torch.no_grad():
        o1, i1 = bank(q, return_index=True)
        o2, i2 = bank(q, return_index=True)
    chk("deterministic in eval", bool(torch.equal(o1, o2)), "")
    # and the running mean is a training-mode quantity, frozen in eval
    before = bank.query_mean.clone()
    with torch.no_grad():
        bank(torch.randn(32, bank.cfg.d_query))
    chk("eval does not move the query mean",
        bool(torch.equal(before, bank.query_mean)), "")
    bank.train()
    with torch.no_grad():
        bank(torch.randn(32, bank.cfg.d_query) + 5.0)
    chk("training does move it", not bool(torch.equal(before, bank.query_mean)),
        f"|delta| {float((bank.query_mean - before).norm()):.4f}")
    bank.eval()
    chk("top-k is k", i1["top_index"].shape[1] == bank.cfg.top_k,
        str(tuple(i1["top_index"].shape)))
    chk("indices are in range",
        bool((i1["top_index"] >= 0).all() and (i1["top_index"] < bank.n_motifs).all()), "")
    chk("retrieved ids resolve to real motifs",
        all(bank.describe(int(i))["motif_id"] for i in i1["top_index"][:, 0]), "")
    q2 = torch.randn(32, bank.cfg.d_query, requires_grad=True)
    out, _ = bank(q2)
    out.sum().backward()
    chk("gradient flows to the query, not into the descriptors",
        q2.grad is not None and bank.descriptor.grad is None, "")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
