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

### The decisive pattern: depth gates the signal

| Alignment depth | n families | mean precision@L/5 |
|---|---|---|
| Neff/L >= 1 (deep) | 2 | **0.975** |
| Neff/L < 1 (shallow) | 10 | 0.609 |

tRNA, with Neff = 245.9 over 71 columns (Neff/L = 3.5), reaches **perfect**
precision and recall. Families with Neff/L well below 1 degrade substantially.

**Note how few families are deep.** Only 2 of 12 clear Neff/L >= 1, and these
are hand-curated Rfam *seed* alignments — among the best-conditioned RNA
alignments that exist. Raw homology search on an arbitrary sequence will
usually do worse.

## 6.4 Architectural consequences

1. **Coevolution is a strong signal — when it is available.** At mean
   precision@L/5 = 0.670 across families and 1.000 for tRNA, this is not a weak
   feature to be blended in; where the alignment is deep it is nearly
   sufficient on its own.
2. **It is conditionally available**, and the condition (`Neff/L`) is
   **cheaply computable before running the model**.
3. **This is the cleanest possible justification for MoE routing.** The router
   should gate the coevolution expert on measured alignment depth, and fall
   back to the language-model pathway when the MSA is shallow. That is not a
   heuristic bolted on — it is a case where the correct expert genuinely
   depends on an observable property of the input, which is exactly what a
   mixture of experts is for.
4. **`Neff/L` should be an explicit input feature to the router**, alongside the
   structural-regime features from §4.3 of the architecture.
5. It also explains why RhoFold+ pairs a language model *with* MSA features
   rather than choosing one: the two have complementary failure modes. PHAROS
   makes that complementarity explicit and routed rather than concatenated.

## 6.5 Caveats

- The ground truth here is Rfam's **consensus secondary structure**, not 3D
  contacts. It measures how well coevolution recovers curated pairing, which is
  the signal the expert consumes; it is not a direct 3D contact benchmark.
- 12 families is a small sample, and they are seed alignments (curated,
  favourable). Numbers on automatically-generated alignments would be lower.
- APC-corrected MI is a *simplification* of full DCA; plmDCA would do better.
  These values are therefore a reasonable lower bound for the signal available.

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
