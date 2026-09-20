# The Sequence-Structure Gap: Analysis and Closure Strategy

> **User observation**: we have 1.3B RNA sequences but only ~6.6k with 3D structure.

## The gap, quantified

| Quantity | Count | Coverage |
|---|---|---|
| RNA sequences held (elDORS + RNAcentral) | 1,369,926,204 | 100% |
| Sequences with **experimental 3D** (unique) | 6,661 | **0.00049%** (1 in 205,663) |
| Chain-level 3D structures, curated derivatives | 27,452 | covering 6,846 PDB entries |
| **RNA chains in the raw PDB archive** | **29,038** | **in 10,520 entries** |
| Rfam families with ANY 3D representative | 100 of 4,227 | **2.4%** |
| Sequences mapped to Rfam families | 10,070,931 | family-level annotation |

### Correction (raw-corpus pass): the curated derivatives are not the supply

The first two structural rows above describe **RNA3DB and gRNAde/RNASolo**, and
they were being read as if they described the PDB. They do not. Matching the
entry ids of both derivative sets against every entry in the raw archive:

| | entries |
|---|---|
| RNA3DB | 5,389 |
| gRNAde / RNASolo (2023-11 snapshot) | 6,156 |
| union of the two | 7,943 |
| **RNA-bearing entries in the PDB** | **10,520** |
| **in neither derivative** | **2,581 — 24.5%** |

**A quarter of the world's RNA structures were outside the training corpus**,
and all 2,581 are already on disk in `data/structures/raw_pdb_entries/`. What
they are:

| | count |
|---|---|
| RNA residues they hold | 1,372,786 (10.3% of all structural RNA) |
| median longest chain | 22 nt — mostly short duplexes and oligos |
| entries with a chain >= 64 nt | 693 |
| **entries with a chain inside the 64-3,000 training window** | **608** |
| **protein-free entries** | **925**, of which 159 have a chain >= 64 nt |
| ribosome-sized (>2,000 RNA residues) | 237 |

**This set contains the extremes, not just the leftovers.** Splitting every raw
chain by whether its entry is covered: the 12,573 chains from covered entries
reach a maximum `effective_c` of **21.14** — exactly the figure published from
the derivatives — while the 1,533 chains from uncovered entries reach **23.30**.
The top 30 chains in the entire archive are all uncovered: one deposition
campaign (9T-series E. coli ribosome PTC refinements, deposited 2025-10-21,
after RNASolo's 2023-11 snapshot). The first chain from any other structure is
rank 31. A parameter fitted to a curated derivative inherits that derivative's
snapshot date.

The median is 22 nt because the uncovered population is dominated by small
crystallographic oligos — which is presumably why the derivative pipelines
dropped them. But the tail is not small, and it is pointed at the two things
this corpus is shortest of:

* **608 entries carry a chain inside the training window** that neither
  derivative exposes (693 have one at least 64 nt long) — a 9% increase in usable structural entries, for zero acquisition
  cost, since they are already downloaded and parsed.
* **925 are protein-free.** D21 measured the isolated-RNA stratum at 2.83% of
  residues and made it the stratum the mandatory G2 evaluation split runs on.
  These entries grow exactly that stratum, and they are the clean,
  high-resolution, small-molecule-free structures that ribosomal cryo-EM is not.

Four entries appear in a derivative but not in the raw archive; they are
obsoleted or superseded depositions and are not a gap.

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
