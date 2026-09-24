#!/usr/bin/env python3
"""Frozen motif retrieval over the BGSU atlas (ARCHITECTURE v0.2 §8).

RNA reuses a small vocabulary of local structure, and §8's position is that the
model should **retrieve** it rather than memorise it: a frozen KV bank of 667
motif classes, queried by the pair track *after* pairing is estimated.

The constraint that shapes the whole module
-------------------------------------------
§8's measured negative result: GNRA k-mer context alone predicts rigidity at
**0.073 sigma** -- negligible. Motif identity requires the interaction graph,
not sequence n-grams. So this bank is deliberately **not queryable from
sequence**. `forward` takes a query built from the pair track's estimated
pairing, and there is no code path that builds one from raw k-mers. That is not
a stylistic preference: the sequence-only version of this idea has already been
measured and it does not work.

What is frozen and what is learned
----------------------------------
Frozen (from `data/derived/motif_bank/`, compiled by `build_motif_bank.py`):
the interaction-family histogram, the position-specific base profile, and the
size/type/instance scalars of each of the 667 classes. These are facts about
RNA, so they do not move during training.

Learned: the projections that turn those facts into keys and values, and the
query projection from the pair track. The model learns *how to look motifs up*
and what to do with them; it does not learn what the motifs are.

Retrieval is soft and sparse: top-`k` classes by scaled dot product, with a
learned temperature and a gate, so a region that matches nothing known
contributes nothing rather than being forced onto its nearest neighbour. 231 of
the 667 classes have five or more instances; the rest are rare and the gate is
how the model is allowed to distrust them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BANK_DIR = Path(__file__).resolve().parents[3] / "data/derived/motif_bank"


@dataclass
class MotifBankConfig:
    d_query: int = 128          # pair-track feature width
    d_key: int = 64
    d_value: int = 128
    top_k: int = 4
    #: rare classes are kept but down-weighted by their instance count; see the
    #: `reliability` buffer
    min_instances_full_trust: int = 5
    dropout: float = 0.0
    #: Subtract a running mean from the query before retrieval. Measured on
    #: real pair features at step 8,000: without it one motif takes 51.9% of
    #: all queries and the bank has 3.0 effective motifs out of 667; with it,
    #: 22.2. See `forward`.
    centre_query: bool = True
    centre_momentum: float = 0.01


class MotifBank(nn.Module):
    """667 frozen motif descriptors, retrieved by a pair-derived query."""

    def __init__(self, cfg: MotifBankConfig, bank_dir: Optional[Path] = None):
        super().__init__()
        self.cfg = cfg
        d = Path(bank_dir or BANK_DIR)
        npz = np.load(d / "bank.npz")
        meta = json.loads((d / "meta.json").read_text())
        self.meta = meta

        sig = torch.as_tensor(npz["signature"], dtype=torch.float32)      # (M, F)
        prof = torch.as_tensor(npz["profile"], dtype=torch.float32)       # (M, P, 4)
        scal = torch.as_tensor(npz["scalar"], dtype=torch.float32)        # (M, 4)
        M = sig.shape[0]
        desc = torch.cat([sig, prof.reshape(M, -1), scal], dim=-1)        # (M, D)
        # frozen: these are properties of RNA, not parameters
        self.register_buffer("descriptor", desc)
        self.register_buffer("n_instances", torch.as_tensor(
            [m["num_instances"] for m in meta["motifs"]], dtype=torch.float32))
        # how much a class deserves to be believed: saturating in instance count
        rel = (self.n_instances / cfg.min_instances_full_trust).clamp(max=1.0)
        self.register_buffer("reliability", rel)

        self.key = nn.Linear(desc.shape[-1], cfg.d_key, bias=False)
        self.value = nn.Linear(desc.shape[-1], cfg.d_value, bias=False)
        self.query = nn.Linear(cfg.d_query, cfg.d_key, bias=False)
        self.gate = nn.Linear(cfg.d_query, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))
        # Running mean of the query input, subtracted before projection.
        #
        # `self.query` has no bias, so it CANNOT remove a shared offset: for
        # `x = xbar + delta`, `W x = W xbar + W delta` and `W xbar` is the same
        # vector for every query in the batch. Measured on real pair features
        # from the step-8,000 trunk, that shared component has norm 5.67
        # against a per-pair deviation of 4.53 -- it is larger than the signal
        # -- and the dot-product retrieval is dominated by it: 39 distinct
        # motifs of 667 reach top-1, one of them takes 51.9% of the queries,
        # and the effective count is 3.0. Centring the query on the same
        # features gives 120 distinct, a 13.1% top share and 22.2 effective.
        #
        # A running mean rather than the batch mean, because the batch mean is
        # undefined at batch 1 and would make a single-chain prediction differ
        # from the same chain inside a batch. A LayerNorm does NOT work here
        # and was tried: it removes each row's own mean, not the direction
        # shared across rows, and leaves the collapse at 37 distinct / 51.4%.
        self.register_buffer("query_mean", torch.zeros(cfg.d_query))
        self.register_buffer("query_mean_n", torch.zeros(()))
        self.drop = nn.Dropout(cfg.dropout)
        self.out_norm = nn.LayerNorm(cfg.d_value)
        # start closed: retrieval must earn its way in, so an untrained bank
        # cannot inject noise into the pair track
        nn.init.constant_(self.gate.bias, -3.0)

    @property
    def n_motifs(self) -> int:
        return int(self.descriptor.shape[0])

    def forward(self, pair_query: torch.Tensor,
                return_index: bool = False) -> Tuple[torch.Tensor, Dict]:
        """Retrieve for each query row.

        `pair_query` is `(N, d_query)` built from the pair track's estimated
        pairing -- never from sequence, see the module docstring.
        """
        x = pair_query
        if self.cfg.centre_query:
            if self.training and x.shape[0] > 1:
                with torch.no_grad():
                    bm = x.detach().mean(0)
                    if float(self.query_mean_n) == 0.0:
                        self.query_mean.copy_(bm)
                    else:
                        self.query_mean.mul_(1 - self.cfg.centre_momentum).add_(
                            bm, alpha=self.cfg.centre_momentum)
                    self.query_mean_n.add_(1)
            if float(self.query_mean_n) > 0.0:
                x = x - self.query_mean
        q = self.query(x)                                       # N, dk
        k = self.key(self.descriptor)                           # M, dk
        v = self.value(self.descriptor)                         # M, dv
        logits = (q @ k.T) / (k.shape[-1] ** 0.5) * torch.exp(self.log_temp)
        # a class seen twice should not win on equal evidence with one seen 300
        # times; reliability enters as a log-prior, not as a hard filter
        logits = logits + torch.log(self.reliability.clamp_min(1e-3))[None]

        kk = min(self.cfg.top_k, logits.shape[-1])
        topv, topi = logits.topk(kk, dim=-1)
        w = torch.softmax(topv, dim=-1)
        retrieved = (w.unsqueeze(-1) * v[topi]).sum(1)          # N, dv
        gate = torch.sigmoid(self.gate(pair_query))
        out = self.out_norm(self.drop(retrieved)) * gate
        # Retrieval DIVERSITY, which nothing reported.
        #
        # `gate_mean` and `top_weight` were already computed here and the one
        # caller that matters -- `step_losses` -- discarded them with
        # `r, _ = model.motifs(pair)`. Neither would have caught the failure
        # that matters anyway: a bank that always returns the same motif adds a
        # constant vector to every pair, which is a bias term wearing 667
        # descriptors. `effective_motifs` is the participation ratio of the
        # top-1 distribution over the batch, so 1.0 is total collapse and M is
        # perfectly spread, and it is the number to watch.
        with torch.no_grad():
            t1 = topi[:, 0]
            cnt = torch.bincount(t1, minlength=logits.shape[-1]).float()
            pr = cnt / cnt.sum().clamp(min=1)
        info: Dict = {"gate_mean": gate.detach().mean(),
                      "top_weight": w[:, 0].detach().mean(),
                      "distinct_motifs": (cnt > 0).sum(),
                      "top_motif_share": pr.max(),
                      "effective_motifs": 1.0 / (pr ** 2).sum().clamp(min=1e-9)}
        if return_index:
            info["top_index"] = topi.detach()
            info["motif_ids"] = [self.meta["motifs"][int(i)]["motif_id"]
                                 for i in topi[:, 0].tolist()]
        return out, info

    def describe(self, index: int) -> Dict:
        return self.meta["motifs"][int(index)]


def load_bank(cfg: Optional[MotifBankConfig] = None,
              bank_dir: Optional[Path] = None) -> Optional[MotifBank]:
    """The bank if it has been compiled, else `None` — never a fabricated one."""
    d = Path(bank_dir or BANK_DIR)
    if not (d / "bank.npz").exists():
        return None
    return MotifBank(cfg or MotifBankConfig(), d)
