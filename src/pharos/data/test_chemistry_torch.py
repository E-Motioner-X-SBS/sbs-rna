#!/usr/bin/env python3
"""`BatchChemistry` must equal `chain_chemistry` exactly, not approximately.

A fast path that is only close to the definition is a silent corruption of the
input features, so these compare element-wise equality on the standard alphabet
and only fall back to a tolerance for the ratio in dim 23.
"""
from __future__ import annotations

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pharos.data.chemistry import N_DIMS, chain_chemistry      # noqa: E402
from pharos.data.chemistry_torch import BatchChemistry         # noqa: E402
from pharos.data.vocab import PAD_ID, SYMBOLS                  # noqa: E402

RNA = ("A", "C", "G", "U", "N")


def _reference(seqs, L):
    out = np.zeros((len(seqs), L, N_DIMS), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = chain_chemistry(list(s))
    return out


def _batch(seqs, L, device="cpu"):
    sym = {s: i for i, s in enumerate(SYMBOLS)}
    tok = np.full((len(seqs), L), PAD_ID, dtype=np.int64)
    msk = np.zeros((len(seqs), L), dtype=bool)
    for i, s in enumerate(seqs):
        tok[i, :len(s)] = [sym[c] for c in s]
        msk[i, :len(s)] = True
    bc = BatchChemistry(SYMBOLS, torch.device(device))
    got = bc(torch.as_tensor(tok, device=device),
             torch.as_tensor(msk, device=device))
    return got.cpu().numpy()


def test_matches_chain_chemistry_on_ragged_batch():
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(RNA, size=int(n)))
            for n in rng.integers(20, 300, size=16)]
    L = max(len(s) for s in seqs)
    ref, got = _reference(seqs, L), _batch(seqs, L)
    # dims 0-22 are table lookups and must be bit-identical
    assert np.array_equal(ref[..., :23], got[..., :23])
    # dim 23 is a ratio of exact integer sums; float32 vs float64 division
    assert np.allclose(ref[..., 23], got[..., 23], rtol=0, atol=1e-6)


def test_five_prime_terminus_is_row_zero_not_batch_zero():
    """Every sequence has its own 5' end; dim 18 is 0 there and -1 after."""
    got = _batch(["ACGU", "GGGG"], 4)
    assert got[0, 0, 18] == 0.0 and got[1, 0, 18] == 0.0
    assert got[0, 1, 18] == -1.0 and got[1, 1, 18] == -1.0


def test_padding_contributes_nothing():
    """A short sequence padded to any length gives the same rows as unpadded."""
    s = "GCGCAAAAUUUUGCGC"
    short = _batch([s], len(s))[0]
    long = _batch([s], 512)[0, :len(s)]
    assert np.array_equal(short, long)
    assert np.array_equal(_batch([s], 512)[0, len(s):],
                          np.zeros((512 - len(s), N_DIMS), dtype=np.float32))


def test_gc_window_edges_match_the_definition():
    """The window clips at both chain ends; a 40-mer exercises both."""
    s = "G" * 20 + "A" * 20
    ref = chain_chemistry(list(s))[:, 23]
    got = _batch([s], 40)[0, :, 23]
    assert np.allclose(ref, got, rtol=0, atol=1e-6)
    assert got[0] > got[-1]


def test_masked_tokens_get_the_unk_row_and_leak_nothing():
    """A masked position must not carry the base it is hiding."""
    sym = {s: i for i, s in enumerate(SYMBOLS)}
    bc = BatchChemistry(SYMBOLS, torch.device("cpu"))
    tok = torch.tensor([[sym["A"], sym["UNK"], sym["G"], sym["C"]]])
    msk = torch.ones(1, 4, dtype=torch.bool)
    got = bc(tok, msk)[0]
    assert got[1, :4].sum() == 0.0      # no A/C/G/U one-hot fires
    assert got[1, 4] == 1.0             # the MOD/unknown flag does


def test_batching_does_not_change_a_sequence():
    """One sequence alone and the same sequence inside a batch must agree."""
    rng = np.random.default_rng(7)
    seqs = ["".join(rng.choice(RNA, size=int(n)))
            for n in rng.integers(25, 200, size=8)]
    L = max(len(s) for s in seqs)
    together = _batch(seqs, L)
    for i, s in enumerate(seqs):
        alone = _batch([s], L)[0]
        assert np.array_equal(together[i], alone)


def test_cuda_matches_cpu():
    """Skipped without a GPU; the fast path runs on both and must agree."""
    if not torch.cuda.is_available():
        return
    rng = np.random.default_rng(3)
    seqs = ["".join(rng.choice(RNA, size=int(n)))
            for n in rng.integers(20, 400, size=12)]
    L = max(len(s) for s in seqs)
    assert np.allclose(_batch(seqs, L, "cpu"), _batch(seqs, L, "cuda"),
                       rtol=0, atol=1e-6)


def main() -> int:
    fails = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS {name}")
        except AssertionError as e:
            print(f"  FAIL {name}: {e}")
            fails.append(name)
    print()
    if fails:
        print(f"FAILURES ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
