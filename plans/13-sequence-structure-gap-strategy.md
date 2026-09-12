# The Sequence-Structure Gap: Analysis and Closure Strategy

> **User observation**: we have 1.3B RNA sequences but only ~6.6k with 3D structure.

## The gap, quantified

| Quantity | Count | Coverage |
|---|---|---|
| RNA sequences held (elDORS + RNAcentral) | 1,369,926,204 | 100% |
| Sequences with **experimental 3D** (unique) | 6,661 | **0.00049%** (1 in 205,663) |
| Chain-level 3D structures | 27,452 | covering 6,846 PDB entries |
| Rfam families with ANY 3D representative | 100 of 4,227 | **2.4%** |
| Sequences mapped to Rfam families | 10,070,931 | family-level annotation |

### This is normal for structural biology, not a failure

| | Sequences available | 3D structures | Coverage |
|---|---|---|---|
| **Proteins** (AlphaFold era) | ~2,000,000,000 (BFD) | ~220,000 | 0.009% (1 in 11,000) |
| **RNA** (us) | 1,369,926,204 | 27,452 chains | 0.002% (1 in 49,900 chains) |
| **NucleicBERT** | ~30,000,000 | ~875 | 0.003% (1 in 34,000) |

RNA's gap is ~5x worse than proteins because RNA is harder to crystallize
(flexible, heterogeneous, charged), and the field started later.

## The strategy this implies (the AlphaFold playbook, adapted)

### Stage 1 — Representation learning on ALL sequences (self-supervised)
1.37B sequences / ~1.25T nucleotides feed masked-language-model pretraining.
NucleicBERT proved this learns secondary structure + contact attention
*without any labels* (their model trained on 30M; we have 46x more).

### Stage 2 — Supervised fine-tuning on gold 3D (the 6.6k)
The 27,452 chains → contact/distance maps (like NucleicBERT's 408+467 set,
but 31x bigger). With cropping/augmentation this yields millions of training
pairs. This is the *evaluation-grade* supervision.

### Stage 3 — Bridge the middle (concrete expansion levers)

| Lever | Source we already hold | Scale |
|---|---|---|
| **Fold inheritance** | Rfam full alignments: 100 families with 3D reps contain ~5.3M Rfam-mapped sequences (5S/tRNA/rRNA/U2/...); their fold is known by homology | ~5.3M weak labels |
| **Family-level structure** | 10.07M Rfam full-region hits, 4,096 families: 2D structure (dot-bracket consensus) + covariance models exist for all | ~10M |
| **MSA coevolution contacts** | Our K-map/DCA pipeline (the whole existing campaign) runs on Rfam alignments; no 3D needed for contact pseudo-labels | thousands of family maps |
| **Structure-prediction pseudo-labels** | Run teacher models (RhoFold+, NuFold exists: nufold.kiharalab.org, AlphaFold3, Boltz-2) on selected sequences | 10^5-10^7 |
| **Chemical probing** | RMDB/SHAPE datasets (next acquisition target) | per-nucleotide constraints |
| **Cryo-EM maps** | EMDB holds maps; many contain unmodeled RNA chains | more chains |

### What NOT to do
- Do not train 3D tasks *only* on the 6.6k and expect generalization.
- Do not treat the 1.37B as directly 3D-supervised: it is representation fuel.

## Bottom line for model design

```
Pre-training (MLM):      1.37B sequences       [have]
Fine-tune 2D structure:  ~160k bpRNA/ArchiveII [have]
Fine-tune 3D contacts:   27k chains / 6.6k uniq[h ave, augmented]
Weak supervision:        10M Rfam-mapped seqs  [have via alignments]
Pseudo-labels:           to be generated       [teacher models]
```

The ratio is not a wall: it is exactly the ratio proteins had when
AlphaFold2 was built (0.009%), and the field solved it with
self-supervision + coevolution + distillation. We have all three paths open,
and our K-map/coevolution work is directly one of them.
