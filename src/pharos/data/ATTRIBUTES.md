# Per-nucleotide attribute vector

Measured (`scripts/sampling/audit_token_attributes.py`): a fully attributed
nucleotide carries **58.9 bits**; a 512-dim bf16 token holds **8,192 bits** —
**139x over-provisioned**. So `d_model` is *compute* space, not storage.

**Design rule that follows**: attributes enter as **input features through one
projection**. They do NOT motivate a wider `d_model`, because training FLOPs go
as `6 * N_active` and widening `d` is quadratic in the FFN.

## Available at INFERENCE for any sequence — 26.0 bits

| Attribute | bits | Source |
|---|---|---|
| nucleotide identity | 2.02 | sequence (measured entropy) |
| relative position | 6.00 | computed |
| local GC, +/-16 window | 4.00 | computed |
| predicted pairing state | 1.00 | 2D head, recycled |
| ionic condition (global, broadcast) | 12.00 | user-supplied `c_ion` |
| in-complex flag | 1.00 | user-supplied context |

## MSA-gated — only when an alignment exists

`Neff/L` (4), per-column conservation (4), coevolution partner score (4).
Router gates the coevolution expert on `Neff/L` (cycle-1 finding).

## Structure-only — SUPERVISION TARGETS, never inference inputs

Saenger class (4.86), Leontis-Westhof class (3.00), modified-residue flag,
disorder label, normalised B-factor, Mg2+ distance class.

> **This distinction is load-bearing.** B-factor and Mg distance are *outputs*.
> Feeding them as inputs would leak the answer. `attributes.py` must keep the
> two sets physically separate.

## Not acquired

SHAPE/DMS reactivity (4 bits). Prerequisite for the ion-conditioning claim.
