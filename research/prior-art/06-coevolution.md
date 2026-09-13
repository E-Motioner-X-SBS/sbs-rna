# Prior Art 6 — Coevolution: how much structural signal is really there

The user's brief asks specifically about "coevolutionary information of RNA...
how RNA sequences in the 3D space tend to be close together". This is the
oldest and best-established structural signal in the field, and also the one
with the sharpest failure mode.

## 6.1 The principle

If two nucleotides form a base pair, a mutation at one is only tolerated when
compensated at the other (G-C -> A-U). Across a family alignment this produces
**correlated substitution** at spatially paired columns. Direct Coupling
Analysis (DCA) disentangles *direct* couplings from indirect chains of
correlation, and its output is enriched in **tertiary** contacts, not just
secondary structure — which is exactly the 3D proximity signal we want.

## 6.2 Prior methods

| Method | Idea |
|---|---|
| **mfDCA / plmDCA** | mean-field / pseudo-likelihood DCA on RNA family alignments; established that nucleotide coevolution predicts 2D *and* 3D contacts |
| **DIRECT** | adds a Restricted Boltzmann Machine to reweight DCA by structural templates, addressing low accuracy when homologs are few |
| **CoCoNet** | a small CNN on top of DCA couplings; boosts contact precision |
| **RNAcmap / RNAcmap3** | fully automatic pipeline: homology search -> alignment -> coupling analysis -> contact map |
| **rMSA** | dedicated RNA homology search/alignment to *deepen* the MSA before coupling analysis |
| **Infernal / Rfam** | covariance models use sequence **and** secondary structure; far more sensitive than BLAST or profile HMMs. This is what built Rfam |
| **RNA-MSM** | MSA-transformer adapted to RNA, run on deep RNAcmap3 alignments |
| **CS-Fold** | phylogenetic modelling of compensatory mutations inside a neural network |

The recurring theme across all of them is that **RNA MSAs are shallow**, and
every method is really a strategy for coping with that.

## 6.3 Our own measurement

`scripts/sampling/measure_coevolution.py` computes sequence-reweighted,
pseudocounted, **APC-corrected mutual information** on the 12 downloaded Rfam
seed alignments and scores it against each family's curated consensus secondary
structure (`#=GC SS_cons`), restricted to `|i-j| >= 4`.

> **Implementation note.** A first version of this measurement reported a mean
> precision of 0.039. That was a bug: the outer-product term in the MI
> normalisation broadcast to shape `(C,q,1)` instead of `(C,q,q)`, so only the
> diagonal was used. After fixing it and adding standard DCA sequence
> reweighting at 80% identity plus pseudocounts, precision rose **17-fold** to
> 0.670. The lesson is worth recording: a plausible-looking low number was
> a silent implementation error, not a property of RNA.

### Results

| Family | N seqs | Neff | L | precision@L/5 | recall@L |
|---|---|---|---|---|---|
| tRNA (RF00005) | 954 | **245.9** | 71 | **1.000** | **1.000** |
| THF riboswitch (RF01831) | 101 | 27.9 | 100 | 1.000 | 0.906 |
| Purine riboswitch (RF00167) | 133 | **102.7** | 102 | 0.950 | 0.955 |
| 5S rRNA (RF00001) | 712 | 10.3 | 119 | 0.826 | 0.765 |
| RNase P bact (RF00010) | 458 | 11.6 | 367 | 0.753 | 0.784 |
| SAM riboswitch (RF00162) | 457 | 19.9 | 108 | 0.714 | 0.697 |
| Cobalamin (RF00174) | 434 | 63.1 | 189 | 0.676 | 0.667 |
| Glutamine (RF01739) | 940 | 15.6 | 53 | 0.500 | 0.692 |
| ydaO/yuaA (RF00379) | 106 | 18.4 | 134 | 0.462 | 0.615 |
| TPP riboswitch (RF00059) | 115 | 41.3 | 105 | 0.429 | 0.926 |
| tmRNA (RF00023) | 477 | 74.6 | 378 | 0.400 | 0.670 |
| FMN riboswitch (RF00050) | 146 | 21.2 | 139 | 0.333 | 0.536 |
| **mean** | | | | **0.670** | **0.768** |

### Depth and signal: a suggestive but statistically weak association

> **Revised after audit.** An earlier version of this document called the depth
> split "the decisive pattern". That was an overclaim; the corrected assessment
> follows.

| Alignment depth | n families | mean precision@L/5 |
|---|---|---|
| Neff/L >= 1 (deep) | **2** | 0.975 |
| Neff/L < 1 (shallow) | 10 | 0.609 |

The mean difference is +0.366. Tested properly, it is weak evidence:

| Test | Result |
|---|---|
| Exact permutation test (all 66 splits) | **p = 0.0455** |
| Pearson(Neff/L, precision), n=12 | +0.498 |
| **Spearman(Neff/L, precision), n=12** | **+0.224** |

Three reasons not to lean on this:
1. The "deep" arm has **n = 2**.
2. The Neff/L = 1 threshold was chosen **post hoc**, so p = 0.0455 is optimistic.
3. A *shallow* family — THF riboswitch, Neff/L = 0.279 — attains precision
   **1.000**, so depth is plainly not necessary for a strong signal.

The rank correlation of **+0.224** is the most honest single summary: a weak
positive tendency, not a gate. tRNA (Neff/L = 3.5) does reach perfect precision,
but n=12 cannot separate that from family-specific effects.

**Note how few families are deep.** Only 2 of 12 clear Neff/L >= 1, and these are
hand-curated Rfam *seed* alignments — among the best-conditioned RNA alignments
that exist. Raw homology search on an arbitrary sequence will usually do worse.
That observation stands on its own and does not depend on the split above.

## 6.4 Architectural consequences

1. **Coevolution is a strong signal.** Mean precision@L/5 = 0.670 across 12
   families, and 1.000 for tRNA. This is not a weak feature to be blended in at
   low weight — for some families it is very nearly sufficient on its own.
2. **Its strength varies enormously between families** (0.333 to 1.000). Whatever
   drives that variation, a model that applies coevolution at a *fixed* weight
   is leaving accuracy on the table for both the strong and the weak cases.
3. **`Neff/L` is a reasonable router feature, on theoretical grounds.** It is
   cheaply computable before the model runs, and the published literature is
   unambiguous that shallow RNA MSAs degrade coupling analysis (this is the
   stated motivation for rMSA, RNAcmap3 and RNA-MSM). **Our own n=12 measurement
   supports this only weakly** (Spearman +0.224; see the revised section above),
   so the architectural case rests on the literature and on the observed
   between-family variance, *not* on our depth split.
4. It also explains why RhoFold+ pairs a language model *with* MSA features
   rather than choosing one: the two have complementary failure modes. PHAROS
   makes that complementarity explicit and routed rather than concatenated.
5. **What would settle it**: measuring precision against alignment depth across
   hundreds of families, not twelve, with the threshold fixed in advance.

## 6.5 Caveats

- The ground truth here is Rfam's **consensus secondary structure**, not 3D
  contacts. It measures how well coevolution recovers curated pairing, which is
  the signal the expert consumes; it is not a direct 3D contact benchmark.
- 12 families is a small sample, and they are seed alignments (curated,
  favourable). Numbers on automatically-generated alignments would be lower.
- APC-corrected MI is a *simplification* of full DCA; plmDCA would do better.
  These values are therefore a reasonable lower bound for the signal available.
- **The ground truth is incomplete.** `ss_pairs()` parses only the nested WUSS
  brackets `()<>[]{}` and ignores the alphabetic pseudoknot brackets (Aa, Bb, Cc,
  Dd), which account for **53 of 537 pairs (9.87%)** across these alignments. A
  correctly-ranked pseudoknot pair is therefore scored as a false positive, so
  the reported precision is **deflated**. Direction of the bias is conservative.
- Only 12 families, all curated seeds, and the depth split has n=2 in one arm.

## Sources
- Direct-Coupling Analysis of nucleotide coevolution facilitates RNA secondary and tertiary structure prediction https://pmc.ncbi.nlm.nih.gov/articles/PMC4666395/
- DIRECT: RNA contact predictions by integrating structural patterns https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3099-4
- CoCoNet — boosting RNA contact prediction by CNNs https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8682773/
- RNAcmap https://academic.oup.com/bioinformatics/article/37/20/3494/6281070
- rMSA https://www.sciencedirect.com/science/article/pii/S0022283622005241
- CS-Fold https://www.biorxiv.org/content/10.1101/2025.04.27.650904.full.pdf
- ML for RNA secondary structure prediction: review https://arxiv.org/html/2511.02622
- Predicting RNA Structure Utilizing Attention from Pretrained LMs (single-sequence, no MSA) https://pmc.ncbi.nlm.nih.gov/articles/PMC12264945/
- **Own measurement**: `data/samples/analysis/coevolution_signal.json`
