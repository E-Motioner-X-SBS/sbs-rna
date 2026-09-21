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

    def __len__(self) -> int:
        return len(self._index)

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
    def length_batches(self, token_budget: int = 16384, max_batch: int = 32,
                       shuffle: bool = True, seed: int = 0,
                       bucket_ratio: float = 1.5) -> List[List[int]]:
        """Indices grouped so each batch holds about `token_budget` residues.

        Buckets are geometric in length (`bucket_ratio`), so the padding waste
        inside a batch is bounded by that ratio regardless of where in the range
        the chains sit. Within a bucket, order is shuffled; across buckets, the
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
            for i in idxs:
                L = self.meta[i]["length"]
                # a batch costs (n+1) * longest, since everything pads to it
                if cur and ((len(cur) + 1) * max(L, self.meta[cur[0]]["length"])
                            > token_budget or len(cur) >= max_batch):
                    out.append(cur)
                    cur = []
                cur.append(i)
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
