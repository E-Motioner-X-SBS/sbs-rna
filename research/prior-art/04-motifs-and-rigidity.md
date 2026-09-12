# Prior Art 4 — RNA Motifs, Modularity, and Local Rigidity

## 4.1 The motif concept

Atomic-resolution RNA structures show that many internal and hairpin loops are
**modular, recurrent, and structured by conserved non-Watson-Crick base pairs**.
An RNA 3D motif is a set of nucleotides forming a *dense, connected graph of
interactions* — a self-contained structural unit.

A motif is **recurrent** when instances appear in non-homologous locations of the
same or different RNAs. Two nucleotide sets are the *same* recurrent motif when
they share both the interaction pattern and the overall geometry.

This is the RNA analogue of protein secondary structure, but stronger: a GNRA
tetraloop or a kink-turn has a **near-invariant 3D geometry** wherever it occurs.
That invariance is exactly the kind of hard prior a neural network should not
have to rediscover from 6,661 training sequences.

### The RNA 3D Motif Atlas

- BGSU, release 4.12 (already in our catalog, `BGSU_motifs`, 6,326 records).
- Internal loops: 2,972 loop instances -> 413 motif classes.
- Hairpin loops: 2,020 instances -> 254 motif classes.
- Machine-readable (CSV + JSON), instance-level, linked to PDB chains.
- Stated applications include **"bioinformatic analysis to improve RNA 3D
  structure prediction from sequence"** — i.e. the authors intend this use, but
  no large model has actually wired it in.

**Compression ratio: ~4,992 instances -> 667 classes (7.5:1).** RNA tertiary
structure is far more repetitive than its sequence diversity suggests. A model
with a motif vocabulary gets most of that repetition for free.

## 4.2 Rigidity: theory and prior methods

The premise "certain RNA motifs are stiff" is well supported:

- **Elastic network models (ENM)**: macromolecule as nodes + harmonic springs;
  a few low-frequency normal modes reproduce observed conformational change.
  Assessed for RNA against MD and SHAPE (NAR 2015); a 2026 essential-dynamics
  refined ENM now covers DNA/RNA/protein-NA complexes.
- **Constraint counting / rigidity theory** (pebble game, FIRST-style): decomposes
  a structure into rigid clusters and flexible hinges directly from the bond
  network. Applied to RNA in Biophys J 2008.
- **RPflex**: coarse-grained network model for RNA pocket flexibility — finds
  most **ligand-binding pockets are relatively rigid**, while protein-binding
  pockets are relatively flexible.
- **DeepRMSF** (Brief Bioinform 2026): deep learning predicting atomic-level
  RMSF (flexibility) for RNA — flexibility is now a directly learnable target.
- **Weighted persistent homology** ML for RNA flexibility.

So rigidity is (a) real, (b) computable from structure, (c) already a supervised
learning target. What does not exist is **rigidity as an inductive bias inside a
sequence-to-structure model.**

## 4.3 Our own measurement of RNA rigidity

`scripts/sampling/analyze_rigidity.py` on the 180 sampled BGSU NR structures.
Rigidity proxy = crystallographic B-factor, **z-normalised within each structure**
(so refinement protocols are comparable). Lower z_B = more rigid.
**X-ray only (31 structures)** — cryo-EM ADPs are not comparable, so the other
106 parsed structures are excluded from this analysis.

### Result 1 — packing density predicts rigidity

Pearson r(local density, z_B) = **-0.294** over 25,760 nucleotides.
Denser local packing => lower B => more rigid. Directionally correct and
moderately strong for a single crude descriptor.

| Neighbours within 10 A of C1' | n | mean z_B |
|---|---|---|
| 0-10 | 23,932 | +0.030 |
| 10-20 | 1,828 | **-0.394** |

### Result 2 — Mg²⁺ proximity is a strong, monotonic rigidity signal

Over **24,623 nucleotides** in X-ray structures containing Mg²⁺:

| Distance to nearest Mg²⁺ | n | mean z_B |
|---|---|---|
| 0-4 A | 319 | **-0.883** |
| 4-6 A | 2,441 | -0.673 |
| 6-8 A | 7,590 | -0.415 |
| 8-12 A | 8,115 | +0.004 |
| 12-20 A | 2,984 | +0.756 |
| > 20 A | 3,174 | **+0.877** |

**A monotonic gradient spanning 1.76 standard deviations.** Nucleotides within
4 A of a Mg²⁺ are ~0.9 sigma more ordered than average; nucleotides more than
20 A away are ~0.9 sigma less ordered.

This is the empirical heart of the architecture. It says the two "physics"
ingredients the user asked for — **ion effects and motif rigidity — are not
independent features. Ion coordination is a principal *cause* of local
rigidity.** A model that predicts ion sites gets a rigidity map nearly for free,
and vice versa. That mutual information is what justifies routing them into a
*shared* physics expert rather than two disconnected heads.

> Honest caveats: (i) B-factor also absorbs resolution, occupancy and refinement
> choices; (ii) Mg²⁺ and dense packing are correlated (both are markers of a
> folded core), so this is **association, not isolated causation** — the gradient
> partly restates that structured cores bind ions and are ordered; (iii) n=319
> in the closest bin is small. It is strong evidence for *using* the coupling as
> a prior, not a proof of mechanism.

### Result 3 — GNRA sequence context alone is a weak rigidity signal (negative result)

Matching the 4-mers {GAAA, GCAA, GAGA, GUGA, GGAA, GCGA, GUAA, GAAG} anywhere in
a chain:

| Context | n nucleotides | mean z_B |
|---|---|---|
| GNRA-like 4-mer | 4,944 | -0.082 |
| all other | 95,768 | -0.009 |

Effect size **0.073 sigma** — negligible. **Reported as a negative result.**

The likely reason is methodological: a GNRA *sequence* is only a GNRA *tetraloop*
when it actually caps a hairpin. Matching 4-mers linearly counts mostly helical
and junction occurrences, which dilutes the signal to nothing.

**The architectural lesson is the important part**: sequence context alone does
not identify a motif. Motif identity requires the *interaction graph*
(base-pairing + stacking), which is why the motif expert must be conditioned on
predicted secondary structure / pair representation, **not on raw k-mers**. This
negative result directly rules out the naive "learn motifs from sequence n-grams"
design.

## 4.4 Architectural consequences

1. **Motif vocabulary as a discrete latent.** 667 Motif Atlas classes -> a
   learned motif codebook. Motif assignment is a routing signal (which expert)
   *and* a structural constraint (geometry prior).
2. **Rigidity as a predicted per-nucleotide scalar**, supervised by B-factor /
   RMSF where available. It gates how much the structure module is allowed to
   move a nucleotide — a *learned* local temperature.
3. **Couple ions and rigidity in one expert.** Result 2 says they share
   information; separate heads would waste it.
4. **Do not route motifs from sequence k-mers.** Result 3 says that fails. Route
   on the pair representation after secondary structure is available.
5. **Rigid motifs are compressible.** A near-invariant motif geometry does not
   need full quadratic attention every layer — it needs *recall* of a stored
   geometry. This is the direct justification for the motif KV-cache / retrieval
   mechanism the user asked about.

## Sources
- RNA 3D Motif Atlas (Methods 2016) https://pmc.ncbi.nlm.nih.gov/articles/PMC4921307/
- Automated classification of RNA 3D motifs (RNA 2013) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3854523/
- Motif Atlas site https://rna.bgsu.edu/rna3dhub/motifs
- Elastic network models for RNA (NAR 2015) https://academic.oup.com/nar/article/43/15/7260/2414435
- Essential-dynamics ENM for DNA/RNA (bioRxiv 2026) https://www.biorxiv.org/content/10.64898/2026.03.11.710985v1.full
- Flexibility of RNA by constraint counting (Biophys J 2008) https://www.cell.com/fulltext/S0006-3495(08)70077-8
- RPflex https://pmc.ncbi.nlm.nih.gov/articles/PMC10058308/
- DeepRMSF (Brief Bioinform 2026) https://academic.oup.com/bib/article/27/1/bbaf720/8424669
- **Own measurement**: `data/samples/analysis/rigidity_summary.json`
