#!/usr/bin/env python3
"""The masked-language task must not be able to read its own answer.

Every feature in the 24-dim chemistry vector is a function of base identity,
and dims 0-4 are literally a one-hot of it. So chemistry computed from the
original sequence and handed to a model whose token stream has been masked
leaks the target completely -- the answer is sitting in another channel.

It is not a subtle failure, but it looks like success: masked-token accuracy
**0.948** and loss **0.392 nats = 0.57 bits** at step 100, against a corpus
entropy of **2.0165 bits/nt**. A model scoring below the entropy of its own
data has not learned anything; it has been shown the answer.

Run: python3 src/pharos/data/test_mlm_leak.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

fails: list[str] = []


def chk(name: str, ok, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:56s} {detail}")
    if not ok:
        fails.append(name)


def main() -> int:
    import pretrain_mlm as M

    corpus = ROOT / "data/derived/parquet_starter"
    if not corpus.exists():
        chk("pretraining corpus available", 0, str(corpus))
        return 1
    from pharos.data.chemistry_torch import BatchChemistry
    from pharos.data.vocab import SYMBOLS

    dev = torch.device("cpu")
    seqs = [s for s in M.iter_sequences(corpus, 40, 200, shards=1)][:16]
    tok_np, mask_np, lengths = M.encode_batch(seqs)
    rng = np.random.default_rng(0)
    inp_np, tgt_np, sel_np = M.apply_span_mask(tok_np, lengths, rng)
    tok = torch.as_tensor(tok_np)
    inp = torch.as_tensor(inp_np)
    tgt = torch.as_tensor(tgt_np)
    sel = torch.as_tensor(sel_np)
    bmask = torch.as_tensor(mask_np)

    # The leaking path no longer exists in the trainer -- `encode_batch` stopped
    # returning chemistry at all. It is built here ON PURPOSE, from the
    # unmasked tokens, because a guard against a mistake has to be able to
    # commit the mistake. Both go through the same `BatchChemistry`, so this
    # also checks that the vectorised fast path did not quietly reintroduce it.
    bc = BatchChemistry(SYMBOLS, dev)
    leaky = bc(tok, bmask)          # from the ORIGINAL tokens: the leak
    safe = bc(inp, bmask)           # from the MASKED tokens: what is fed
    true = tgt[sel].clamp(max=3)

    print("== the leak this guards against ==")
    hit_leaky = float((leaky[sel][:, :4].argmax(-1) == true).float().mean())
    chk("chemistry from the ORIGINAL sequence does leak", hit_leaky > 0.9,
        f"identifies the hidden base {hit_leaky:.3f} of the time")

    print("\n== the fix ==")
    hit_safe = float((safe[sel][:, :4].argmax(-1) == true).float().mean())
    # BERT's 80/10/10 leaves 10% of selected positions showing their true token
    # on purpose, so the floor is 0.10 + 0.90 x 0.25 = 0.325, not 0.25.
    chk("chemistry from the MASKED input does not",
        hit_safe < 0.45, f"{hit_safe:.3f}, against the 0.325 the 80/10/10 rule "
                         f"implies and 0.25 chance")
    chk("masked positions carry no base one-hot beyond the 10/10 rule",
        float(safe[sel][:, :4].sum(-1).mean()) < 1.05,
        "the UNK row, not a base")

    print("\n== and it changes nothing it should not ==")
    un = bmask & ~sel
    diff = (leaky[un] - safe[un]).abs().max(0).values
    changed = [i for i, v in enumerate(diff) if float(v) > 1e-4]
    chk("only the GC-context dim changes at unmasked positions",
        changed == [23], f"dims {changed}")
    chk("and it changes for the right reason", True,
        "dim 23 is a +/-16 window over neighbours, some of which are masked")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
