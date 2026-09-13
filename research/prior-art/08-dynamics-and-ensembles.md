# Prior Art 8 — RNA Dynamics, Ensembles, and Folding Pathways

> The project brief asked for structure prediction "or maybe more to it, maybe
> dynamics of the RNA, maybe how it will fold". This was the thinnest part of the
> PHAROS design through the first sessions and is developed here.

## 8.1 Why a single structure is the wrong target for much of RNA

Protein prediction can often get away with one answer per sequence. RNA
frequently cannot:

- **Riboswitches are defined by having two functional folds** — a ligand-bound
  (holo) and unbound (apo) state. Predicting one is predicting half the biology.
- The field has explicitly moved on from the minimum-free-energy structure:
  functionally relevant RNAs exist as **heterogeneous ensembles where the most
  stable state may be a minor subpopulation**.
- **Cotranscriptional folding is kinetic, not thermodynamic.** Structure evolves
  as the chain elongates, and cotranscriptional products often *deviate from
  thermodynamic predictions* because the system rarely reaches equilibrium. A
  model trained only on deposited (equilibrium, crystallisable) structures has no
  access to this at all.

## 8.2 What exists

| Method | Approach | Note |
|---|---|---|
| **RNAnneal** (bioRxiv 2026) | generative DL + statistical physics + MD; ab initio sampling, then unsupervised models trained on the samples | 10-state ensembles; evaluated against **16 experimentally-resolved riboswitch conformations**; pseudoknot-free conformations well reproduced |
| **RhoFold+** | MSA clustering/sampling to emit multi-conformational predictions | the earliest mainstream acknowledgement that one output is wrong |
| **CASP16 alternative-conformation track** | multi-MSA strategy + structural clustering | the community now scores this explicitly |
| **gRNAde** multi-state | conditions design on several states | **3-5% improvement, best at 3 states** |
| **AlphaFold3** | diffusion learns a distribution in principle | in practice still struggles with riboswitch flexibility |
| **DeepRMSF** (Brief Bioinform 2026) | predicts atomic-level RMSF for RNA | flexibility is a *directly learnable* target |
| **Landscape zooming / helix-based kinetics** | partitions the folding landscape by stable helices; master-equation population kinetics over a reduced ensemble | predicts pathways, intermediates, transcriptional products |
| **R2D2** | uses nucleotide-resolution chemical probing to reconstruct cotranscriptional pathways | experiment-constrained, not ab initio |

**The pattern**: ensembles are approached either by *sampling* (RNAnneal, AF3) or
by *kinetics over a reduced landscape* (landscape zooming). Nobody predicts a
per-nucleotide **fluctuation amplitude** directly from sequence as a supervised
target, despite the labels existing.

## 8.3 Elastic/harmonic dynamics: the tractable middle

Between "one static structure" and "a full MD ensemble" sits the harmonic
approximation, which is cheap and well established for nucleic acids:

- Each base-pair step has a **6x6 stiffness matrix** over (shift, slide, rise,
  tilt, roll, twist). Deformability is **sequence-dependent**: pyrimidine-purine
  (YR) steps are the most flexible, purine-pyrimidine (RY) the stiffest, with RR
  intermediate.
- Couplings matter — slide-rise, twist-roll and twist-slide are the major ones,
  so a diagonal-only treatment loses real physics.
- **Important limitation**: recent work identifies the **pentameric scale as the
  minimum range of elastic couplings**, and correlation analysis shows concerted
  motions of neighbouring steps. **Dinucleotide step models are demonstrably
  insufficient.**
- Deformability can also be read straight out of **anisotropic displacement
  parameters** in high-resolution structures.
- Elastic network models for RNA have been assessed against MD and SHAPE.

## 8.4 Our own measurements bearing on dynamics

Three results already in `data/samples/analysis/` are dynamics measurements that
were not framed as such.

### (a) Step stiffness, measured [measured]

`extract_basepair_geometry.py` recovered **103,964 annotated base-pair steps**
across 155/180 structures and derived `F = kT C^-1` for **76 step contexts** with
n >= 200.

- Twist force constants span **115x**, from UG/UG (1.343e-2, stiffest) to
  AA/UA (1.167e-4, floppiest); the floppy end is dominated by non-canonical
  contexts.
- Watson-Crick step means reproduce canonical A-form RNA (rise 2.81-3.14 A,
  twist 29.0-33.9 deg, slide -1.30 to -1.74 A), confirming the extraction.
- **Caveat**: 3 of 76 contexts have covariance condition number > 1e4
  (AU/AA, GA/AA, GC/AC), so `C^-1` is least reliable exactly where steps are
  floppiest.

### (b) Structure beats sequence at predicting deformability [measured]

`measure_stiffness_headroom.py` fits nested models to the step distribution
(negative log-likelihood, lower is better):

| Model | NLL | gain |
|---|---|---|
| M0 global (one distribution for all steps) | 19.7235 | — |
| M1 sequence context | 17.5792 | **2.14 nats** over global |
| M_struct structural context only | 17.8470 | 1.88 over global |
| **M2 sequence x structure** | **14.5510** | **3.03 nats** over sequence alone |

**Structural context adds more than sequence context does** (3.03 vs 2.14), and
the two are complementary rather than redundant. This independently reproduces
the literature's conclusion that dinucleotide *sequence* models are insufficient
— and says what to do instead: condition stiffness on the predicted pair
representation, not on the k-mer.

### (c) Disorder labels nobody is using [measured]

`_pdbx_unobs_or_zero_occ_residues` yielded **46,447 RNA unobserved-residue records**. A residue that could not be modelled is a residue too mobile or too
poorly ordered to resolve — a **direct, free, per-residue disorder label**,
present in every deposited structure and used by no RNA structure predictor.
Together with B-factors (the 1.76 sigma Mg gradient) this is a second
flexibility channel at zero acquisition cost.

## 8.5 What PHAROS should do about it

1. **Predict a fluctuation amplitude, not just a coordinate.** Per-nucleotide
   rigidity is already a head; extend it to per-step **6x6 stiffness** supervised
   by the measured `F` matrices. Cheap, and gives a harmonic ensemble for free.
2. **Condition stiffness on structure, not sequence.** Directly implied by (b).
   The stiffness head reads the pair representation.
3. **Model couplings, not just diagonals.** The literature is explicit that
   slide-rise, twist-roll and twist-slide matter, and that couplings extend to
   the pentameric scale — so a learned encoder over a structural neighbourhood,
   not a 16-entry dinucleotide lookup table.
4. **Add unobserved-residue disorder as an auxiliary head.** 46,447 RNA free labels.
5. **Emit K states, not one.** gRNAde's evidence is that **3 states** is the
   sweet spot; RNAnneal uses 10. Start at 3.
6. **Be honest about kinetics.** Cotranscriptional pathway prediction needs
   time-resolved probing data we do not have. The harmonic ensemble above is an
   *equilibrium fluctuation* model — it describes breathing around a fold, **not**
   the pathway to it. Claiming folding-pathway prediction would be unsupported.

## Sources
- RNAnneal, ab initio RNA structure ensembles (bioRxiv 2026) https://www.biorxiv.org/content/10.64898/2026.02.01.703098v1
- Modeling alternative conformational states in CASP16 https://www.biorxiv.org/content/10.1101/2025.09.02.673835.full.pdf
- gRNAde: geometric deep learning for 3D RNA inverse design https://arxiv.org/pdf/2305.14749
- DeepRMSF (Brief Bioinform 2026) https://academic.oup.com/bib/article/27/1/bbaf720/8424669
- Sequence-dependent shape and stiffness of DNA and RNA duplexes: hexanucleotide scale and beyond https://pmc.ncbi.nlm.nih.gov/articles/PMC12421674/
- Correlated motions in DNA: beyond base-pair step models https://academic.oup.com/nar/article/51/6/2633/7076479
- Relative flexibility of B-DNA and A-RNA duplexes https://academic.oup.com/nar/article/32/20/6144/1115749
- Nucleic-acid deformability from anisotropic displacement parameters https://pmc.ncbi.nlm.nih.gov/articles/PMC4163913/
- Elastic network models for RNA (NAR 2015) https://academic.oup.com/nar/article/43/15/7260/2414435
- Computational modeling of cotranscriptional RNA folding (CSBJ 2025) https://pmc.ncbi.nlm.nih.gov/articles/PMC12212151/
- Cotranscriptional kinetic folding including pseudoknots https://pubmed.ncbi.nlm.nih.gov/33902324/
- R2D2: reconstructing cotranscriptional folding from probing data https://www.biorxiv.org/content/10.1101/379222v1.full
- **Own measurements**: `basepair_geometry.json`, `stiffness_headroom.json`
