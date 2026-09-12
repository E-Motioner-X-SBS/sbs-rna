# Prior Art 2 — RNA Language Models and the NucleicBERT Baseline

## 2.1 The field in one table

| Model | Params | Pretraining data | Key idea |
|---|---|---|---|
| RNABERT | ~1M | small | first RNA BERT |
| RNA-FM | 100M | 23.7M RNAcentral ncRNA | the workhorse; backs RhoFold+ |
| RNA-MSM | — | MSA-based | MSA-aware LM |
| **NucleicBERT** | **404M** | ~30M MARS ncRNA | 32 layers, 32 heads, d=1024, **vocab 25**, maxlen 1024 |
| UNI-RNA | 400M | large | scaling study |
| RiNALMo | 650M | large | largest dense RNA LM |
| **ERNIE-RNA** | **86M** | 20.4M RNAcentral | **structure-biased attention**; SOTA despite being smallest |
| AIDO.RNA | — | MARS | scaling |
| RibonanzaNet2 | — | 2.1M chemically probed | probing-supervised |

## 2.2 ERNIE-RNA is the most important result here

**86M parameters beats 650M RiNALMo and 400M UNI-RNA.** That is a 7.5x
parameter disadvantage overturned by *one architectural choice*.

### Architecture
- 12 transformer blocks, 12 heads, d=768, maxlen 1024.
- Pretraining: MLM, 15% masking, 20.4M ncRNA from RNAcentral.

### The structural-prior attention bias (the whole trick)

Layer 1:
```
Attention(Q,K) = QK^T / sqrt(d) + B_pair(i,j)
```
Layers l >= 2 accumulate the previous layer's attention:
```
Attention_l(Q,K) = QK^T / sqrt(d) + Attention_{l-1}(Q,K)
```

`B_pair` is a **hand-set base-pairing prior**:

| Pair | Bias |
|---|---|
| C-G | 3 |
| A-U | 2 |
| G-U | alpha (tunable, init 0.8) |
| other | lower |
| diagonal | 0 |

### Results

| Task | ERNIE-RNA | Best competitor |
|---|---|---|
| 2D structure F1 (bpRNA-1m) | **0.873** (attn-map) / 0.862 (emb) | RiNALMo 0.850 (650M) |
| Cross-family F1 (bpRNA-new) | **0.646** | RiNALMo 0.355 |
| **Contact map Top-L/1 precision** | **0.68** (zero-shot attention map) | RNA-FM 0.42 |

The cross-family number is the striking one: **0.646 vs 0.355**. A fixed
physical prior nearly doubles generalisation to unseen families, because the
prior is *true regardless of family* while learned statistics are not.

> **This is the single strongest evidence for the thesis of this project.**
> A crude, hand-set, three-valued physical prior — not even a good one — buys
> more than 7x the parameters. The obvious next move is to replace that crude
> prior with a *rich, learned, physics-grounded* one: motif rigidity, ion
> coordination, coevolution. That is exactly what the proposed MoE does.

### Its limitations = our opening
1. The prior is **static** — it cannot know that a GU pair inside a kink-turn
   behaves differently from a GU pair in a free helix.
2. It is **nearest-neighbour-blind** — real pairing energy is stacking-dependent
   (the Turner nearest-neighbour model), not per-pair.
3. It encodes **only canonical pairing**. Non-canonical pairs define RNA 3D
   motifs; they carry most of the tertiary information and are entirely absent.
4. It ignores **ions**, which is what actually stabilises tertiary contacts.

## 2.3 NucleicBERT — the model to beat

- 404M params, 32 layers, 32 heads, d=1024, **vocab 25**, maxlen 1024.
- Pretrained on ~30M ncRNA from MARS, 80/20 split, 300 epochs, **192 A100s**,
  MLM accuracy 83.1%.
- Downstream: 2D structure (RNAStrAlign + bpRNA-1m TR0 -> ArchiveII/TS0),
  contact/distance maps from **only ~875 structures** (467 BGSU + 408
  NucleoSeeker), splice sites (Spliceator), fitness (CPEB3 ribozyme).

### Concrete weaknesses we can exploit

| NucleicBERT choice | Problem | Our counter |
|---|---|---|
| vocab 25 (IUPAC degenerate) | elDORS is already normalised to 5 symbols; 20 tokens are dead weight | vocab 5 + specials |
| maxlen 1024 | truncates >50% of sequences in our long chunks (6, 9 median ~2.2-2.4k nt) | 2048 core / 4096 lossless |
| dense 404M, all tokens all layers | every nucleotide pays full cost regardless of whether it is in a rigid helix or a complex junction | sparse MoE: cost follows structural complexity |
| 3D trained on ~875 structures | tiny | 27,452 chains / 6,661 unique (31x / 7.6x) |
| no physics | — | ion Hamiltonian + motif rigidity |
| 192 A100s | not reproducible for us | MoE gives more capacity per FLOP |

## 2.4 What the benchmarks say about LM-only approaches

Comprehensive benchmarking (Brief Bioinform 2025) of RNABERT, RNA-FM, RNA-MSM,
ERNIE-RNA, RNAErnie, RiNALMo found ERNIE-RNA, RiNALMo, RNA-MSM best per motif.
Zero-shot benchmarking (Brief Bioinform 2026) found RiNALMo and ERNIE-RNA
separate RNA families cleanly in embedding space.

**But**: every one of these is a *secondary-structure*-grade result. None of them
closes the 3D gap. Scaling dense RNA LMs is producing diminishing returns —
ERNIE-RNA proves the gains are now coming from *inductive bias*, not scale.

## Sources
- ERNIE-RNA (Nat Commun 2025) https://pmc.ncbi.nlm.nih.gov/articles/PMC12627772/
- Comprehensive benchmarking of LLMs for RNA 2D (Brief Bioinform 2025) https://academic.oup.com/bib/article/26/2/bbaf137/8109668
- Zero-shot benchmarking of RNA LMs (Brief Bioinform 2026) https://academic.oup.com/bib/article/27/2/bbag098/8509095
- NucleicBERT https://github.com/KIT-MBS/NucleicBERT (facts via plans/12-nucleicbert-data-scanner.md)
- RhoFold+/RNA-FM https://pmc.ncbi.nlm.nih.gov/articles/PMC11621015/
