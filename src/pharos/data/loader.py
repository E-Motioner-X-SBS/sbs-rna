#!/usr/bin/env python3
"""Length-bucketed batching over the sharded 3D set.

RNA chain length spans 32 to 4,450 nt with a median near 120, so padding a
batch to its longest member wastes most of it: a uniform-random batch of 16 that
happens to catch one rRNA is 97% padding. Batches are therefore formed by
**token budget within a length bucket** -- every batch holds roughly the same
number of real residues, which keeps GPU memory flat across a run instead of
spiking whenever a ribosome is drawn.

Quality weighting (D16) is applied here rather than by filtering: `train_weight`
comes from the clashscore bands, and a weighted sampler makes a 40-clashscore
structure appear less often instead of never. A resolution cutoff would discard
the cryo-EM majority, which is 62% of the corpus.
"""
from __future__ import annotations

import json
import math
import random
from bisect import bisect_right
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

from .dataset import ShardReader, pad_batch


def split_digest(manifest: Dict) -> str:
    """Eight hex characters naming WHICH split assignment this corpus carries.

    The corpus has been rebuilt three times with the same 16,604 chains and a
    re-drawn split: v1 was train 11,723 / val 1,189 / test 1,450, and v2 and
    v3 are 10,641 / 853 / 4,081. **63.8% of the current test set -- 2,603 of
    4,081 chains -- was in v1's TRAIN set.**

    That is not a detail. `block_scorer.pt` was trained under v1, and
    `measure_cascade_recall.py` loads it and evaluates it on
    `Pharos3DDataset(DATA, split="test")`, which now resolves to v3's test
    set. The scheduled re-measurement on 2026-09-26 23:34 did exactly that
    and wrote the result over the valid one: a number computed on chains the
    model had been trained on, in the same file, under the same keys, with
    nothing to say it was different in kind rather than in value. The only
    guard that caught anything was a pin on `n_chains == 1450`, which reads
    as staleness and invites updating the pin.

    Nothing about the split is wrong -- family-disjoint (D17) is the right
    policy and redrawing it was a deliberate improvement. What was missing is
    that a model and an evaluation have to agree on which draw they mean, and
    neither recorded it. This is that record: a hash over the sorted
    (pdb, chain, split) triples, so any change in assignment changes it.
    """
    import hashlib
    rows = sorted(f"{c['pdb']}_{c['chain']}:{c.get('split')}"
                  for sh in manifest["shards"] for c in sh["chains"])
    return hashlib.sha1("\n".join(rows).encode()).hexdigest()[:8]


class Pharos3DDataset:
    """The whole sharded set, addressed as one flat sequence of chains."""

    def __init__(self, root: Path, split: Optional[str] = None,
                 min_length: int = 32, max_length: int = 4608):
        self.root = Path(root)
        man = json.loads((self.root / "manifest.json").read_text())
        self.manifest = man
        self._readers: Dict[int, ShardReader] = {}
        self._index: List[tuple] = []          # (shard, local_i)
        self.meta: List[Dict] = []
        for si, sh in enumerate(man["shards"]):
            for li, cm in enumerate(sh["chains"]):
                if split is not None and cm.get("split") != split:
                    continue
                if not (min_length <= cm["length"] <= max_length):
                    continue
                self._index.append((si, li))
                self.meta.append(cm)
        self._files = [self.root / sh["file"] for sh in man["shards"]]
        #: which draw of the split this is; see `split_digest`
        self.split_digest = split_digest(man)

    def __len__(self) -> int:
        return len(self._index)

    def prewarm(self, verbose: bool = False) -> float:
        """Decompress every shard up front; return the GiB now resident.

        Shards are decompressed on first touch, and an epoch's batches are drawn
        across all of them, so without this the first pass pays the
        decompression inside the training loop. Profiled: `collate` was **53% of
        step time** with the GPU at 13% utilisation, almost all of it zlib on
        first touch. The whole set is 1.34 GiB expanded, so paying it once at
        startup is strictly better than paying it scattered through epoch 0.
        """
        import resource
        for si in range(len(self._files)):
            self._reader(si)
        # ru_maxrss is in KILOBYTES on Linux, so GiB is /1024/1024, not /1e6.
        # /1e6 gives GB and the label said GiB -- a 7.4% overstatement in a
        # number that is printed next to a VRAM budget.
        gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
        if verbose:
            print(f"[data] {len(self._files)} shards resident, {gib:.2f} GiB RSS",
                  flush=True)
        return gib

    def _reader(self, si: int) -> ShardReader:
        r = self._readers.get(si)
        if r is None:
            r = ShardReader(self._files[si])
            self._readers[si] = r
        return r

    def __getitem__(self, i: int) -> Dict:
        si, li = self._index[i]
        item = self._reader(si)[li]
        item["meta"] = self.meta[i]
        return item

    # -- batching ---------------------------------------------------------
    def length_batches(self, token_budget: int = 16384, max_batch: int = 512,
                       shuffle: bool = True, seed: int = 0,
                       bucket_ratio: float = 1.5) -> List[List[int]]:
        """Indices grouped so each batch holds about `token_budget` residues.

        Buckets are geometric in length (`bucket_ratio`), so the padding waste
        inside a batch is bounded by that ratio regardless of where in the range
        the chains sit.

        `max_batch` defaults to 512, not 32. At 32 the cap binds long before the
        token budget does -- raising the budget from 16k to 131k left the worst
        batch at 32 chains / 101k tokens, so the extra budget bought nothing and
        an 80 GiB card ran at 13% utilisation. The token budget should be the
        thing that binds; `max_batch` is a guard against a pathological batch of
        thousands of 32-nt chains, not a tuning knob. Within a bucket, order is shuffled; across buckets, the
        batch order is shuffled too, so a step is not systematically short or
        long at any point in the epoch.
        """
        order = sorted(range(len(self)), key=lambda i: self.meta[i]["length"])
        if not order:
            return []
        bounds: List[int] = []
        lo = max(1, self.meta[order[0]]["length"])
        while lo <= self.meta[order[-1]]["length"]:
            bounds.append(lo)
            lo = int(math.ceil(lo * bucket_ratio))
        buckets: Dict[int, List[int]] = {}
        for i in order:
            buckets.setdefault(bisect_right(bounds, self.meta[i]["length"]), []).append(i)

        rng = random.Random(seed)
        out: List[List[int]] = []
        for _, idxs in sorted(buckets.items()):
            if shuffle:
                rng.shuffle(idxs)
            cur: List[int] = []
            cur_max = 0
            for i in idxs:
                L = self.meta[i]["length"]
                # A batch costs (n+1) * LONGEST, since everything pads to it --
                # and the longest has to be the running maximum, not
                # `cur[0]`. It was `cur[0]`, and the bucket is shuffled, so
                # the first chain is not the longest: the estimate ignored
                # every chain between the first and the longest. Measured on
                # the training split, **71% of batches exceeded the budget**,
                # the worst by 1.23x. The budget exists so an 80 GiB card is
                # not asked for more than it has; one that is exceeded by two
                # batches in three is not a budget.
                #
                # Bounded by the bucket ratio, so this was never unbounded --
                # but it is the difference between a declared limit and an
                # advisory one, and stage 5's OOM handler skips the batch,
                # which makes the loss silently length-correlated.
                nxt_max = L if L > cur_max else cur_max
                if cur and ((len(cur) + 1) * nxt_max > token_budget
                            or len(cur) >= max_batch):
                    out.append(cur)
                    cur, cur_max, nxt_max = [], 0, L
                cur.append(i)
                cur_max = nxt_max
            if cur:
                out.append(cur)
        if shuffle:
            rng.shuffle(out)
        return out

    def sample_weights(self) -> np.ndarray:
        """Per-chain sampling weight (D16): structure quality, not a cutoff."""
        return np.array([float(m.get("train_weight") or 1.0) for m in self.meta],
                        dtype=np.float64)

    def collate(self, idxs: Sequence[int]) -> Dict:
        items = [self[i] for i in idxs]
        batch = pad_batch(items)
        batch["meta"] = [it["meta"] for it in items]
        batch["weights"] = np.array(
            [float(it["meta"].get("train_weight") or 1.0) for it in items],
            dtype=np.float32)
        return batch

    def iter_batches(self, **kw) -> Iterator[Dict]:
        for b in self.length_batches(**kw):
            yield self.collate(b)


def describe(root: Path) -> Dict:
    """Split sizes and length distribution, for a build's own sanity check."""
    man = json.loads((Path(root) / "manifest.json").read_text())
    out = {"n_chains": man["n_chains"], "n_residues": man["n_residues"],
           "n_contacts": man["n_contacts"], "split": man["split"]}
    return out
