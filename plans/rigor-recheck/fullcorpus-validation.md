# Validating the architectural decisions against the full server corpus

**Date:** Sep 20, 2026
**Basis:** 29,807 chains (RNA3DB 15,441 per-chain mmCIFs + RNASolo 14,366 PDBs),
238M nt sampled from four elDORS chunks, and the catalog.
**Contrast:** every geometric number in `ARCHITECTURE.md` was measured on **180**
BGSU structures fetched onto the laptop. The corpus is on the server. This is
the first time the two have met.

---

## The pattern, stated first

**Every central estimate reproduced. Both tail estimates failed.**

That is not a coincidence and it is the useful finding. An n=180 sample
estimates a mean well and cannot estimate a p99.9 at all. Two load-bearing
design parameters were set from maxima observed on 180 structures, and both are
wrong on 29,807 — in each case the mean they sat next to was fine.

| | published (n=180) | full corpus | verdict |
|---|---|---|---|
| **Central** | | | |
| mean effective c, long chains | 17.2 | **17.14** | reproduces |
| b=4 occupancy, long chains | 1.34% | 1.67% | same order |
| sequence entropy | 2.0167 bits/nt | **2.0165** | reproduces to 4 dp |
| **Tail** | | | |
| max effective c | 19.04 | **21.14** | **breaches `target_c = 20`** |
| chains over 4096 nt | **0** | **8** (max 4,450) | **falsifies "covers every chain"** |

---

## 1. `target_c = 20` is breached — OQ-7 closed negatively

20,266 chains pass the published filters (8.0 A any-atom contact, |i-j| >= 4,
64 <= L <= 3000, >= 20 contacts). Worst chain **effective c = 21.14**
(`7PAS_1_3`), **5.7% over budget**. One chain in 20,266; p99 19.13, p99.9 19.52.

The mean reproduced (17.14 vs 17.2), so nothing in the bin-level tables was
wrong — only the maximum, which C14 had already flagged as the quantity being
misread ("a mean read as if it bounded the maximum").

**The true tail is likely worse.** RNA3DB and RNASolo are pre-normalised and
carry 0.025% non-ACGU residues against 1.05% in raw-PDB BGSU — a 42x
under-representation of *exactly* the driver OQ-7 named. Both residue rules give
identical maxima (21.14) here because there are almost no modified residues left
to disagree about.

**Action:** raise `target_c` to 24 (clears the observed maximum by 12%, costing
more only in the L3 refinement) or give the track an overflow path. Decide it on
raw PDB entries, not on normalised derivatives.

Script: `scripts/sampling/recheck_block_sparsity_fullcorpus.py`.
Data: `data/samples/analysis/block_sparsity_fullcorpus.json`.

## 2. The 4096 context does not cover every chain — G1 corrected

Longest-chain distribution over all 29,807: p50 **95**, p90 2,831, p99 3,562,
p99.9 3,773, **max 4,450**.

Eight chain files, **five unique structures**, exceed 4096 — 6HRM (4,450),
7UPH (4,438), 4V6X (4,298), 8TOC (4,269), 7LHD (4,217) — each present in both
corpora. All are large ribosomal subunit rRNAs. The published table read `0`.

The breach is 0.027% of chains, so the design is not endangered; the *sentence*
is false. State coverage as **99.97%**, or raise the context to 4,608.

## 3. The vocabulary premise is source-dependent — D6

G7's **8.90%** non-ACGU is a **raw-PDB** figure. The structural data on disk is
pre-normalised: **0.025%** over 1,800 sampled structures / 1.26M residues, top
species `N` (245 of 310 instances).

So the second vocabulary is needed only if training consumes raw PDB entries.
Against RNA3DB/RNASolo a 5-symbol vocabulary is very nearly exact. **Vocabulary
and data source must be decided together.**

### `N` is two different things sharing one token

| | sequence corpus (elDORS) | structural corpus |
|---|---|---|
| meaning | ambiguous base call | base identity unassigned by depositor |
| rate | **0.203%** of nt (README says ~0.1%) | 0.025% of residues |
| range | 0.51% chunk 001 -> 0.0077% chunk 006 (67x) | — |
| geometry | none | **complete** — `8vvt_ZA` residue 1248 carries a full ribose incl. O2' plus a partial base ring |
| sequences affected | 1.45% carry >= 1 N | — |

The structural case is **free supervision**: those residues can train the
geometry heads normally, and recovering their base identity is a legitimate
auxiliary task. Conflating them with sequence-`N` discards that.

## 4. What the corpus cannot check

Both structural corpora have **zero HETATM**. RNA3DB additionally zeroes every
B-factor; RNASolo retains real ones.

| claim | status on this corpus |
|---|---|
| Mg/rigidity gradient (1.76 sigma) | **uncheckable** — no ions |
| ion inventory (Mg:K 9:1, 83% to OP1/OP2) | **uncheckable** — no ions |
| Mg-site supervision head | **no training data on disk** |
| ionic-condition input (the headline novelty) | **no training data on disk** (RMDB absent too) |
| chemical probing (SHAPE/DMS) | absent — Ribonanza not acquired |

These are not refutations. They mean two of PHAROS's three distinctive claims
currently have nothing to learn from, and closing that needs ~7k raw PDB entries
re-fetched with HETATM plus RMDB, or an explicit demotion to future work.

## 5. Decisions that validate unchanged

| Decision | Check | Result |
|---|---|---|
| D1 PHAROS-Small sizing | 13 arithmetic identities | all exact |
| D2 25B token budget | entropy 2.0167 bits/nt | **2.0165** on 238M independent nt |
| D5 attribute budget | 512-dim bf16 = 8,192 bits; 139x over 58.9 | exact |
| D7 cost | 25B/323B ratio 13.0x vs token ratio 12.9x; 8x loop factor | consistent |
| — | `verify_claims.py` | passes, and correctly exits 1 on drift |
| — | `test_manning.py` | 4/4, A-RNA theta = 0.804 |

`test_mmcif_entities.py` **cannot run here**: `data/samples/structures/` holds
only `analysis/`, so the canonical loader — the resolver five defects were fixed
into — has no test coverage on the machine that holds the data.

---

## Recommended actions

1. **Raise `target_c` to 24**, or implement overflow. (§1)
2. **Restate G1 coverage as 99.97%**, or context 4,608. (§2)
3. **Split the `N` token**; add base-identity imputation as an auxiliary head. (§3)
4. **Decide vocabulary jointly with data source.** (§3)
5. **Re-fetch raw PDB entries for the ~7k unique 3D sequences** if the ion heads
   are to survive; otherwise demote them explicitly. (§4)
6. **Re-derive every remaining max/min in ARCHITECTURE.md on the full corpus.**
   Two of two tested tail claims failed; the rest should be assumed suspect
   until measured.
7. **Restore the 180 sample structures to the server** so the canonical-loader
   tests run where the data lives. (§5)
