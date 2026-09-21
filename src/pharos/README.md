# `src/pharos/` — implementation

**Built, not trained.** Every module below exists and is under
`scripts/sampling/verify_claims.py`; what has not happened is a training run,
because the shared A100 has not been free. The trainers refuse to start on a
busy card rather than OOM mid-run.

The layout is flatter than v0.1's sketch, and deliberately: that sketch listed
files by *concept* (`attention_bias.py`, `debye.py`, `splits.py`) and several of
those concepts turned out to belong inside one object rather than beside it. A
map that names files which do not exist is worse than no map.

```
src/pharos/
├── model/
│   ├── attention.py      GDN (chunked delta rule, UT transform) / SWA / FULL
│   │                     with the physics bias.  15 properties tested
│   ├── moe.py            fine-grained MoE + shared expert; router conditioned
│   │                     on length, Neff/L, in-complex, chemistry (§5.3)
│   ├── trunk.py          16-block stack, §5.1 pattern, loops with the
│   │                     one-step gradient and deep supervision
│   ├── heads.py          all ten of §9
│   ├── motif_bank.py     667 frozen BGSU classes, queried from the PAIR track
│   ├── dynamics.py       per-step 6x6 stiffness, O(L) harmonic ensemble,
│   │                     disorder head
│   ├── block_scorer.py   the trainable L1/L2 selectors (R1's experiment)
│   └── pharos.py         assembly: embedding, electrostatic bias, tied MLM
│                         head, §5.4 parameter accounting
├── physics/
│   └── manning.py        xi, theta, kappa, q_eff, b_elec.  A-RNA theta=0.804
├── data/
│   ├── mmcif_entities.py THE canonical resolver: entities, geometry, entry
│   │                     composition, and the free per-residue labels
│   ├── vocab.py          parent base + modification id (VOCAB.md, D6)
│   ├── chemistry.py      the 24 dims of CHEMISTRY.md
│   ├── dataset.py        contacts, block labels, shards, batching
│   ├── loader.py         length-bucketed batching, quality weighting (D16)
│   └── rdat.py           RMDB parsing, both format versions
└── (tests live beside their module as test_*.py)
```

**Where the v0.1 sketch's files went.** `attention_bias.py` is
`ElectrostaticBias` in `pharos.py`, because the bias is applied inside FULL
attention and splitting it out only moved the coupling. `pair_track.py` is two
things that should not share a file: the cost/structure reference in
`research/architecture/reference/` and the trainable selector in
`block_scorer.py`. `stiffness.py` is `StiffnessField` in `dynamics.py`, since
the stiffness field exists to be diagonalised and the two are one computation.
`debye.py` and `energy.py` are `manning.py` plus `ElectrostaticBias`.
`tokenizer.py` is `vocab.py`. `splits.py` and `chunk_weights.py` are in
`scripts/build_dataset.py` and `loader.py`, because a split is a property of a
*built dataset* rather than of the model. `decoder.py`, `attributes.py`,
`train/precision.py`, `train/schedule.py` and the `eval/` package are **not
built**; precision and schedule are settled as decisions (PRECISION.md, D2/D3)
and the evaluation splits are enforced in `build_dataset.py` and reported by the
trainers.

## Training

| stage (§12.1) | script | data |
|---|---|---|
| 1 pretrain, MLM + spans | `scripts/pretrain_mlm.py` | elDORS, 12M sequences |
| 2 secondary structure | `scripts/train_sequence_stages.py` | bpRNA-SPOT 10,934 |
| 3 probing | same, co-trained | Ribonanza 335,616 |
| 4 physics | closed form, no stage | — |
| 5 3D | `scripts/train_pharos.py` | 16,604 chains |
| — block selector (R1) | `scripts/train_block_scorer.py` | same |

`scripts/await_gpu_and_train.sh` starts the block-scorer run when the shared
GPU frees.

## Where the measured findings bind


| Finding | Lands in |
|---|---|
| O(L) contact scaling; 1.67% block occupancy; max contacts/nt **7.66** | `model/block_scorer.py`, and the reference in `research/architecture/reference/` |
| Mg/rigidity coupling **1.523 sigma** (X-ray, 1535 structures) | `model/heads.py` rigidity head, `model/dynamics.py` |
| A-RNA theta = 0.804 (not the B-DNA 0.76) | `physics/manning.py` |
| Coevolution gated on Neff/L | `model/trunk.py` router input |
| Stiffness: sequence and structure near-equal | `model/dynamics.py` StiffnessField |
| **G2: 97.15% of RNA is in complex** (10,520 raw entries; isolated RNA is 2.83% of residues, 2.7x the v0.1 figure) | `scripts/train_pharos.py` — reports the split, never averages it |
| **G3: 85.94% of residues ribosomal** (down from 92.65% on n=180) | `scripts/build_dataset.py` — family-disjoint (D21, D25) |
| **Context 4,608**: longest RNA chain in the whole PDB is 4,450 nt (6HRM), closed | `model/trunk.py` (v0.2 D20) |
| **G7: 1.005% outside ACGU on raw PDB, 0.025% on derivatives** | `data/vocab.py` — vocab depends on SOURCE (v0.2 D6) |
| Token over-provisioned 139x | `data/chemistry.py`; precision settled in PRECISION.md (D3) |
| **Mg-rigidity 1.523 sigma, X-ray only** | `model/heads.py` rigidity head (v0.2 D12) |
| **Mg:K = 54:1, inner-sphere 77.9% to OP1/OP2** | `physics/manning.py` b_elec, `model/pharos.py` ElectrostaticBias |
| **target_c = 24, with overflow handling** | `model/block_scorer.py` budgets (v0.2 D9/D23) |
| **3D corpus saturated at 10,399 entries** | `scripts/train_pharos.py` — 3D is the smallest stage |
| Chemistry: H-bond edges, pKa, pucker, stacking | `data/chemistry.py` (v0.2 D13) |
