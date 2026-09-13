# `src/pharos/` — implementation

Nothing here is trained. This is the skeleton the design lands in, laid out so
that each measured finding has exactly one place to live.

```
src/pharos/
├── model/          the network
│   ├── trunk.py            hybrid mixer stack (GDN / SWA / FULL), MoE FFN
│   ├── attention_bias.py   B_wc + B_elec(c_ion) + B_motif + A^(l-1)
│   ├── pair_track.py       hierarchical L1->L2->L3  (reference impl. exists)
│   ├── motif_bank.py       frozen 667-class KV bank
│   ├── stiffness.py        per-step 6x6 F encoder (Cholesky-parameterised)
│   ├── heads.py            2D / contact / Mg / rigidity / disorder / reactivity
│   └── decoder.py          frame diffusion, K=3 states
├── physics/        closed-form, no learned parameters
│   ├── manning.py          xi, theta, kappa, q_eff from c_ion  (A-RNA theta=0.804)
│   ├── debye.py            screened Coulomb pair bias
│   └── energy.py           E_stack + E_pair + E_elec + E_Mg + E_excl + E_rigid
├── data/           the dataset side
│   ├── catalog.py          thin wrapper over data/catalog/catalog.sqlite
│   ├── tokenizer.py        vocab: see model/VOCAB.md -- NOT plain 5 symbols
│   ├── attributes.py       per-nucleotide feature vector (see ATTRIBUTES.md)
│   ├── chunk_weights.py    per-chunk corpus weighting (151-nt read down-weight)
│   └── splits.py           md5(header)%100 split; identity-dedup for 3D
├── train/
│   ├── loop.py             deep supervision per segment, one-step gradient
│   ├── losses.py           incl. block_occupancy_loss (implemented+tested)
│   ├── precision.py        bf16 / fp8 / NVFP4 selection -- see PRECISION.md
│   └── schedule.py         staged token budget 5B/25B/100B/323B
└── eval/
    ├── blind.py            CASP15/16, RNA-Puzzles -- never in training
    ├── stratified.py       MUST report isolated vs in-complex (finding G2)
    └── recall.py           block-detection recall per length bin and per level
```

## Where the measured findings bind

| Finding | Lands in |
|---|---|
| O(L) contact scaling; 1.34% block occupancy | `model/pair_track.py` |
| Mg/rigidity coupling 1.76 sigma | `physics/energy.py` (E_rigid), `model/heads.py` |
| A-RNA theta = 0.804 (not the B-DNA 0.76) | `physics/manning.py` |
| Coevolution gated on Neff/L | `model/trunk.py` router input |
| Stiffness: sequence and structure near-equal | `model/stiffness.py` |
| **G2: 98.96% of RNA is in complex** | `eval/stratified.py` — mandatory split |
| **G7: 8.9% of residues outside ACGU** | `data/tokenizer.py` — vocab is NOT 5 |
| Token over-provisioned 139x | `data/attributes.py`, `train/precision.py` |
