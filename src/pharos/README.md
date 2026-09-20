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
│   ├── heads.py            10 heads: contact / distance / 3D / 2D / Mg /
│   │                       rigidity (X-RAY ONLY, v0.2 D12) / reactivity /
│   │                       fitness / splicing / base-identity
│   └── decoder.py          frame diffusion, K=3 states
├── physics/        closed-form, no learned parameters
│   ├── manning.py          xi, theta, kappa, q_eff from c_ion  (A-RNA theta=0.804)
│   ├── debye.py            screened Coulomb pair bias
│   └── energy.py           E_stack + E_pair + E_elec + E_Mg + E_excl + E_rigid
├── data/           the dataset side
│   ├── catalog.py          thin wrapper over data/catalog/catalog.sqlite
│   ├── tokenizer.py        vocab: see VOCAB.md; N_seq and N_struct are
│   │                       DIFFERENT tokens (v0.2 D11)
│   ├── chemistry.py        24-dim per-residue chemistry (see CHEMISTRY.md)
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
| O(L) contact scaling; 1.67% block occupancy; max contacts/nt **7.66** | `model/pair_track.py` |
| Mg/rigidity coupling **1.523 sigma** (X-ray, 1535 structures) | `physics/energy.py` (E_rigid), `model/heads.py` |
| A-RNA theta = 0.804 (not the B-DNA 0.76) | `physics/manning.py` |
| Coevolution gated on Neff/L | `model/trunk.py` router input |
| Stiffness: sequence and structure near-equal | `model/stiffness.py` |
| **G2: 97.15% of RNA is in complex** (10,520 raw entries; isolated RNA is 2.83% of residues, 2.7x the v0.1 figure) | `eval/stratified.py` — mandatory split |
| **G3: 85.94% of residues ribosomal** (down from 92.65% on n=180) | `data/splits.py` — family-disjoint (v0.2 D21) |
| **Context 4,608**: longest RNA chain in the whole PDB is 4,450 nt (6HRM), closed | `model/trunk.py` (v0.2 D20) |
| **G7: 1.005% outside ACGU on raw PDB, 0.025% on derivatives** | `data/tokenizer.py` — vocab depends on SOURCE (v0.2 D6) |
| Token over-provisioned 139x | `data/attributes.py`, `train/precision.py` |
| **Mg-rigidity 1.523 sigma, X-ray only** | `model/heads.py` rigidity head (v0.2 D12) |
| **Mg:K = 54:1, inner-sphere 77.9% to OP1/OP2** | `physics/energy.py` E_Mg |
| **target_c = 24, with overflow handling** | `model/pair_track.py` (v0.2 D9) |
| **3D corpus saturated at 10,399 entries** | `train/schedule.py` — 3D is the smallest stage |
| Chemistry: H-bond edges, pKa, pucker, stacking | `data/chemistry.py` (v0.2 D13) |
