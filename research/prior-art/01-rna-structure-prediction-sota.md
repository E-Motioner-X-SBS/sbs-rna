# Prior Art 1 — RNA Structure Prediction: State of the Art (as of 2026-09)

## 1.1 The headline problem

RNA 3D structure prediction has **not** had its AlphaFold2 moment. Protein
prediction crossed into experimental accuracy; RNA has not. The numbers below
are the bar any new architecture must clear.

### Benchmark reality (Nov 2025, non-redundant RNA test set)

| Method | Mean TM-score | Notes |
|---|---|---|
| trRosettaRNA | **0.548** | MSA + transformer restraints + folding |
| NuFold | 0.525 | AF2-style, nucleotide frames |
| AlphaFold3 (server) | 0.510 | diffusion, all-atom |
| AlphaFold3 (local) | 0.504 | |

A TM-score of ~0.5 is the *threshold for "roughly the right fold"*. Protein
prediction routinely exceeds 0.9. **RNA 3D prediction is still at the
"sometimes gets the topology" stage.** CASP16 reinforced this: the best RNA
groups (e.g. Vfold) won by *bolting AlphaFold3 onto physics-based modelling*,
not by pure deep learning.

> Architectural implication: a pure sequence->coordinates neural net is not
> obviously the winning bet. The CASP16 signal is that **physics + ML hybrids
> are currently ahead**, which is the premise this project is built on.

## 1.2 Why RNA lags — the causal chain

1. **Data scarcity, severe.** Our own catalog quantifies it: 6,661 unique
   sequences with experimental 3D, out of 1.37B sequences (0.0005%). Proteins
   had ~220k structures for ~2B sequences (0.009%) at the AF2 moment — RNA's
   gap is ~18x worse in ratio terms and ~33x worse in absolute structure count.
2. **Redundancy inside the little data there is.** 15,441 RNA3DB chains collapse
   to 3,157 unique sequences; 5S rRNA alone accounts for 5,976 BGSU clusters.
   The effective dataset is far smaller than the nominal one.
3. **Family coverage is nearly empty.** 100 of 4,227 Rfam families (2.4%) have
   any 3D representative.
4. **RNA is conformationally plastic.** A single sequence often has multiple
   functional folds (riboswitches are *defined* by this). Single-structure
   prediction is the wrong target for much of RNA.
5. **Structure is ion-dependent.** RNA is a polyanion; its fold is not defined
   without specifying the ionic environment. No current sequence->structure
   model takes ionic conditions as an input. **This is an open lane.**
6. **Resolution is poor.** Our catalog: median 3.10 A, only 13% of chains <= 2.5 A,
   62% cryo-EM. Training targets are themselves noisy.

## 1.3 Method families

### (a) MSA/coevolution + geometry (trRosettaRNA, DeepFoldRNA, RoseTTAFoldNA)
Predict inter-nucleotide geometric restraints (distance/orientation) from MSA
features, then fold by optimisation. **Currently the accuracy leader
(trRosettaRNA 0.548).** Weakness: needs deep MSAs; RNA MSAs are shallow for most
families, and the model degrades hard when the MSA is thin.

### (b) Language-model-driven (RhoFold+)
- RNA-FM language model pretrained on ~23.7M RNAcentral ncRNA sequences.
- **Rhoformer**: an Evoformer-analogue, 10 recycling iterations.
- Structure module with **invariant point attention (IPA)** producing backbone
  frames + torsions.
- RhoFold+ adds MSA clustering/sampling to emit *multi-conformational*
  predictions — an early acknowledgement that one-structure output is wrong.

### (c) Diffusion / all-atom generative (AlphaFold3, Boltz)
- **Pairformer** replaces Evoformer (48 blocks; pair rep (n,n,128), single rep
  (n,384)); triangular attention enforces triangle-inequality consistency.
- MSA module shrunk to **4 blocks** and moved out of the trunk.
- Diffusion module: 30 attention blocks (3 local -> 24 global -> 3 local)
  predicting raw atom coordinates, replacing AF2's frame/torsion structure module.
- Gets nucleic acids "for free" from a unified all-atom tokenisation, which is
  also why it is not especially good at them.

### (d) Chemical-probing-supervised (RibonanzaNet / RibonanzaNet2)
- Trained on **2.1M sequences with DMS + SHAPE (2A3) chemical mapping** from the
  Eterna/Kaggle dual crowdsourcing effort.
- RibonanzaNet2 trained on 256 A100s; SOTA on RNA folding tasks.
- **Key lesson: per-nucleotide experimental observables are a supervision
  channel that is ~300x more plentiful than 3D structures** (2.1M vs 6.6k).
  This is the single most underused label source in the field.

## 1.4 Where the openings are

| Opening | Why it is open |
|---|---|
| **Ionic conditions as a model input** | No mainstream predictor conditions on [Mg²⁺]/[K⁺]. RNA folding is *defined* by it. |
| **Chemical mapping as dense supervision** | 2.1M probed sequences vs 6.6k structures; mostly used only by RibonanzaNet. |
| **Conformational ensembles, not single structures** | Riboswitches have >=2 functional folds by definition; only RhoFold+ gestures at this. |
| **Motif-level priors as hard structural constraints** | Motif Atlas exists and is machine-readable but is not wired into any large model as an inductive bias. |
| **Efficiency** | Everything above is dense and quadratic. Nobody has applied sparse-MoE + hybrid attention to RNA. |

## Sources
- Deep learning for RNA structure prediction (Curr Opin Struct Biol 2025) https://www.sciencedirect.com/science/article/pii/S0959440X25000090
- CASP16 physics+ML integration https://pmc.ncbi.nlm.nih.gov/articles/PMC12354339/
- RhoFold+ (Nat Methods 2024) https://pmc.ncbi.nlm.nih.gov/articles/PMC11621015/
- AlphaFold3 (Nature 2024) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11168924/
- AF3 architectural highlights https://www.blopig.com/blog/2024/08/architectural-highlights-of-alphafold3/
- Ribonanza dual crowdsourcing https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10925082/
- AF3 comprehensive benchmarking (Brief Bioinform 2025) https://academic.oup.com/bib/article/26/6/bbaf616/8351050
