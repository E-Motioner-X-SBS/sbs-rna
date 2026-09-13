# Prior Art 3 — Ion-Mediated Effects, Manning Condensation, and the RNA Hamiltonian

> This is the **primary differentiator** of the proposed architecture. No
> mainstream RNA structure predictor takes ionic conditions as an input, yet RNA
> tertiary structure does not exist without counterions.

## 3.1 The physics

RNA is a **polyanion**: one formal -1 charge per phosphate. Manning's `b` is the
**axial** charge spacing — charges projected onto the helix axis — *not* the
through-backbone P-P contour distance (~5.9-7.0 A), which is a much larger number
and a common source of error. For A-form RNA the rise is ~2.8 A per base pair
carrying 2 phosphate charges, so **b ~ 1.40 A**. Folding a polyanion means bringing like charges
together, which costs enormous electrostatic energy. That cost is paid by
counterions. **No ions, no tertiary structure** — this is not a correction term,
it is a leading-order term.

### Manning counterion condensation

Manning's parameter for a line charge:

```
xi = l_B / b
```

with Bjerrum length `l_B = e^2 / (4 pi eps_0 eps_r k_B T)` ~ 7.1 A in water at
298 K, and `b` the axial charge spacing. For A-form RNA, `xi > 1`, so the chain
is **above the condensation threshold**: a fraction of counterions condenses onto
the chain and is not free, renormalising the effective charge to

```
theta = 1 - 1/(z * xi)        (fraction of charge neutralised)
q_eff = -(1 - theta) per phosphate
```

Worked values [DERIVED, and cross-checked against the literature]:

| Form | rise | b (axial) | xi = l_B/b | theta (z=1) | theta (z=2) |
|---|---|---|---|---|---|
| B-DNA | 3.4 A/bp | 1.70 A | 4.21 | **0.762** | 0.881 |
| **A-form RNA** | 2.8 A/bp | **1.40 A** | **5.11** | **0.804** | 0.902 |

So for **RNA** roughly **80%** of the backbone charge is screened by condensed
counterions before any structure-specific electrostatics is considered, giving
`q_eff = -0.196` per phosphate. The frequently quoted `theta ~ 0.76` is the
**B-DNA** value; A-form RNA has a shorter axial charge spacing, hence a higher
Manning parameter and stronger condensation. For divalent Mg²⁺ (z=2) condensation
is stronger still (theta ~ 0.90), which is why Mg²⁺ is the folding switch.

Literature check: B-DNA `xi = 4.2`, `theta = 1 - 1/xi = 0.76`; A-form RNA ion
atmosphere reported to neutralise **~0.8** of the phosphate charge (0.7-0.8
monovalent cations bound per phosphate). Both match the derivation above.

The classical result: the melting temperature of a nucleic-acid helix is linear
in `log[salt]`, a direct consequence of condensation theory.

**Limitation to respect**: Manning theory assumes an *infinite line charge in
infinitely dilute solution*. Real folded RNA is a compact 3D object with grooves,
pockets and chelation sites. Mean-field condensation captures the *diffuse*
atmosphere well but fails at *site-specific* binding, which is where the
structural action is.

## 3.2 STEM (2026) — the current best synthesis, and our template

**Structural Topology-based Electrostatic Model** (PMC13345221) does exactly the
hybrid split we should imitate:

| Ion | Treatment | Why |
|---|---|---|
| **Mg²⁺** | **explicit** | ion-ion correlations and site-specific chelation are outside mean-field theory |
| **K⁺ / monovalent** | **implicit**, via generalised Manning condensation | mean-field captures KCl well; explicit K⁺ is wasted compute |

Its central physical claim: the driver of RNA folding dynamics is the **exchange
between inner-sphere (direct, chelated) and outer-sphere (water-mediated) Mg²⁺-
phosphate coordination** — i.e. structure is controlled not by *whether* an ion
is bound but by *which coordination mode* it is in.

Validated against: crystallographic ion-binding sites, experimental preferential
ion-interaction coefficients, SAXS radii of gyration. Case study: a 58-nt rRNA
fragment where a chelated Mg²⁺-mediated tertiary contact drives the
intermediate -> native transition.

Related result: in the SAM-II riboswitch, Mg²⁺ **remodels the free-energy
landscape**, shifting equilibrium from extended/partially-unfolded toward a
compact pre-organised state, anchoring an A-minor twist. Mg²⁺ is not a spectator;
it selects the fold.

## 3.3 Empirical grounding from our own sampled structures

Measured directly on **180 BGSU non-redundant representative structures (<=3.0 A)**
downloaded for this project (`scripts/sampling/analyze_ions_motifs.py`,
179 parsed, 307,965 RNA residues):

### Ion inventory

| Ion | Count |
|---|---|
| **Mg²⁺** | **17,428** |
| K⁺ | 1,868 |
| Zn²⁺ | 359 |
| Na⁺ | 161 |
| Ca²⁺ | 29 |
| Mn²⁺ | 12 |
| Ba²⁺ | 3 |

127/179 structures contain at least one ion; **96/179 contain Mg²⁺**.
Mg²⁺ outnumbers every other cation ~9:1. Any ion-aware model that handles only
one ion species should handle Mg²⁺.

### Coordination shells (Mg²⁺ to RNA electronegative atoms)

- Inner sphere (<= 2.6 A): mean **0.77** RNA contacts per Mg²⁺
- Outer sphere (3.5-5.0 A): mean **7.00** RNA contacts per Mg²⁺

**Inner-sphere partner atoms** (which atoms chelate Mg²⁺ directly):

| Atom | Count | Interpretation |
|---|---|---|
| OP2 | 5,890 | phosphate non-bridging O |
| OP1 | 5,144 | phosphate non-bridging O |
| O6 | 757 | guanine major groove |
| O4 | 511 | uracil |
| N7 | 447 | purine major groove |
| O2' | 196 | ribose 2'-OH |

**83% of direct Mg²⁺ coordination is to the two non-bridging phosphate oxygens
(OP1/OP2).** The next largest class — G(O6)/G(N7) — is the textbook major-groove
guanine site. Outer-sphere contacts are more diffuse and add O5'/O3' backbone and
N1/N3/N4 base atoms.

### Stratification (the honest version)

The aggregate numbers are **ribosome-dominated** and must not be quoted flat:

- Top 5 structures (7RQE, 7RQB, 5J8B, 8JDJ, 7PZY) hold **39.4%** of all Mg²⁺.
- Mg²⁺ per structure: median **72.5**, mean 181.5, max 2,493 (7RQE).
- Mg²⁺ per nucleotide: median **0.046** (~1 Mg²⁺ per 22 nt); range 0.001-0.28.
- Fraction of Mg²⁺ with any inner-sphere RNA contact:
  **0.511 overall**, but only **0.296** for structures < 500 nt (n=36).

So: large, densely-packed RNAs chelate a much higher fraction of their ions;
small RNAs hold most of their Mg²⁺ in the water-mediated outer shell. **The
inner/outer ratio is itself a function of tertiary compactness** — which makes it
a *predictable target*, not just an input.

> **Caveat that must appear in the report**: crystallographic Mg²⁺ counts are
> *resolved, site-bound* ions only. The diffuse Manning atmosphere is invisible
> to crystallography. So 0.046 Mg/nt is a lower bound on site binding and says
> nothing about the ~76% of charge neutralised diffusely. The two channels
> (explicit sites, implicit atmosphere) must be modelled separately — which is
> precisely the STEM split.

## 3.4 Learned ion-site prediction (prior art we can absorb)

| Method | Approach |
|---|---|
| **MgNet** | graph CNN over RNA structure; geometric + electrostatic features; predicts Mg²⁺ *density distribution* |
| **RMSIF** | geometric DL on molecular-surface fingerprints; beats MetalionRNA, MgNet, Metal3DRNA |
| **IonNet** | DL Mg²⁺ site prediction |
| **Metal3DRNA / MetalionRNA** | earlier baselines |

All of these predict ions **given a structure**. Our setting is the inverse and
harder: predict structure *conditioned on ionic composition*, with ion sites as a
latent/auxiliary variable. That inversion is the novel contribution.

## 3.5 Architectural consequences

1. **Ionic condition is a model input.** `([Mg²⁺], [K⁺], [Na⁺], T)` -> a
   conditioning vector, like a diffusion timestep embedding. Same sequence at
   0 mM and 10 mM Mg²⁺ should yield different predictions. Nothing does this today.
2. **Two-channel electrostatics**, mirroring STEM:
   - *implicit*: a Manning-condensation screening term modulating the pair
     representation as a smooth function of ionic strength (Debye-Hückel-like
     `exp(-kappa r)` with `kappa` from salt).
   - *explicit*: a predicted **Mg²⁺ site head** with an inner/outer coordination
     classifier, since inner vs outer is the mechanistically decisive variable.
3. **Auxiliary supervision is free.** We have 17,428 Mg²⁺ positions with
   coordination labels from 180 structures alone; scaled to the server's 27,452
   chains this is a large auxiliary label set that costs nothing to extract.
4. **Physics-derived attention bias.** The screened Coulomb term between
   phosphates is a *computable* pair prior — the ion-aware generalisation of
   ERNIE-RNA's hand-set base-pairing bias.

## Sources
- STEM: Structural Topology-based Electrostatic Model (2026) https://pmc.ncbi.nlm.nih.gov/articles/PMC13345221/
- Generalized Manning Condensation Model Captures the RNA Ion Atmosphere https://pubmed.ncbi.nlm.nih.gov/26197147/
- Critical Role of Mg²⁺ Ions in RNA Folding Transitions (SAM-II) https://pubs.acs.org/doi/10.1021/acs.jpcb.5c02586
- Mg-induced triplex pre-organizes the SAM-II riboswitch https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5352136/
- Monovalent salt corrections in ViennaRNA https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10386259/
- MgNet / graph DL locates Mg in RNA https://pubmed.ncbi.nlm.nih.gov/37292390/
- RMSIF geometric DL for Mg sites https://pubmed.ncbi.nlm.nih.gov/38365157/
- **Own measurement**: `data/samples/analysis/ion_summary.json` (180 BGSU NR structures)
