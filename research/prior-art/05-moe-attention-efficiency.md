# Prior Art 5 — Mixture-of-Experts, Hybrid Attention, and Retrieval

## 5.1 MoE in biological sequence models

| Model | Domain | Design |
|---|---|---|
| **AIDO.Protein** | protein | **first MoE protein LM**; 16B params, 8 experts/block, **top-2** routing |
| BALM-MoE | antibody | top-2 MoE; beats its dense counterpart at equal *active* params |
| EssLM-MoE | protein essentiality | MoE fusing multiple biological information levels |
| MERA | protein active sites | multimodal MoE + retrieval; 3 orthogonal retrieval experts |
| Mol-MoE | molecules | multi-view experts over SMILES / SELFIES / graph |

**There is no MoE RNA structure model.** The nearest neighbours are AIDO.Protein
(sparse MoE for sequence) and MERA (multimodal MoE + retrieval). The
combination the user is asking for — *sparse MoE routed over heterogeneous
biological modalities, with physics experts, for RNA structure* — is unoccupied.

Key transferable finding: **BALM-MoE beats the dense model at equal active
parameters.** MoE is not only a compute trick; the specialisation itself helps.
That matters more in RNA than in protein because RNA data is genuinely
heterogeneous (reads vs transcripts vs riboswitches vs rRNA).

## 5.2 MoE design lessons worth copying (DeepSeekMoE / DeepSeek-V3)

1. **Fine-grained expert segmentation.** Split the FFN intermediate dimension to
   make more, smaller experts at constant parameter count, and activate more of
   them. Lets knowledge decompose more precisely; each expert stays more
   specialised.
2. **Shared expert isolation.** A few always-on experts (bypassing the gate)
   absorb generic high-frequency patterns, so routed experts do not each have to
   relearn the basics. For RNA the generic pattern is Watson-Crick helix
   formation — enormously frequent and shared by everything. **Helix handling
   belongs in a shared expert; the routed experts should specialise in what is
   rare and hard: junctions, pseudoknots, ion pockets.**
3. **Auxiliary-loss-free load balancing.** DeepSeek-V3 adjusts a per-expert bias
   term up/down each step to equalise utilisation, instead of adding an
   auxiliary loss that fights the main objective. Directly applicable, and
   important here: our data is *intrinsically* imbalanced (5S rRNA alone is
   5,976 of the 3D clusters), so a naive auxiliary balance loss would actively
   fight the biology. Bias-based balancing is the right tool.

> Design consequence: **route on structural context, not on family identity.**
> If routing keys on family, the router collapses onto rRNA. If it keys on local
> structural regime (helix / junction / ion pocket / single-strand), the load is
> far more even and the experts learn transferable physics.

## 5.3 Hybrid and linear attention

The user asked specifically about interleaving self-attention, linear attention,
cross-attention, and KV caching. The 2025-2026 state of that art:

- **Gated DeltaNet** (ICLR 2025) outperforms Mamba2 and DeltaNet on language
  modelling, in-context retrieval, length extrapolation and long context. Hybrids
  that combine Gated DeltaNet layers with sliding-window attention or Mamba2
  layers improve training efficiency further.
- **Jamba**: Mamba + attention + MoE interleaved at a **7:1 ratio**; 3x throughput
  of Mixtral, 256K context with only **4 GB KV cache**.
- **Layer interleaving** generally: alternate sparse (sliding-window softmax) and
  linear (Mamba / DeltaNet) layers, with occasional **dense "reset" layers** to
  restore global mixing.
- **HyLo**: hybrid of MLA + linear blocks (Mamba2 or Gated DeltaNet) with staged
  long-context training and teacher-guided distillation; extends usable context
  up to 90%, 2M-token prefill.
- **iFlame**: interleaving full and linear attention specifically for *mesh
  generation* — i.e. the interleaving idea already transfers to 3D geometry.
- **Oryx**: switches mixer type *within* a sequence (quadratic attention where
  rich context is needed, linear recurrence where efficiency is needed), tying
  >=90% of parameters across mixers.

**Oryx is the most relevant.** RNA has exactly this structure: long stretches of
regular A-form helix (low information, predictable, linear-attention-friendly)
punctuated by junctions, loops and pseudoknots (high information, long-range,
needing full attention). The mixer should follow the biology.

## 5.4 Why the RNA-specific case for hybrid attention is unusually strong

A quantitative argument from our own corpus analysis:

- elDORS sequences run to **4,096 nt**; chunks 6 and 9 have median length
  ~2,200-2,400 nt. NucleicBERT's 1,024 limit truncates most of that.
- Full attention at 4,096 nt is 16.8M pair entries per sequence per layer.
- But RNA pairing is **overwhelmingly local plus a sparse set of long-range
  contacts**. Secondary structure is dominated by nested stems; tertiary contacts
  (pseudoknots, kissing loops, A-minor) are few but decisive.
- So the pair matrix is *sparse and structured*, not dense. Paying dense
  quadratic cost everywhere is exactly wrong.

## 5.5 KV caching and retrieval — the RNA reading

The user's framing ("at the end of the day RNA is just a bunch of data",
"token-based KV caching for faster retrieval") maps onto a real mechanism:

- **MERA** already shows retrieval experts working in a protein MoE.
- RNA has something proteins do not: a **closed, finite, curated motif
  vocabulary** (667 Motif Atlas classes) with near-invariant geometry.
- Therefore: precompute a **frozen motif KV bank** — keys = motif signatures,
  values = motif geometry embeddings. Cross-attend to it. This is retrieval
  against a *structural* memory, not a sequence memory, and it is cheap because
  the bank is small (~10^3 entries), fixed, and shareable across all sequences.

This also answers the data-scarcity problem in a different way from pretraining:
instead of hoping 6,661 structures teach the model motif geometry implicitly, we
**hand it the geometry explicitly** and let it learn only *where* motifs go.

## 5.6 Summary of what to borrow

| Source | Borrowed element |
|---|---|
| DeepSeekMoE | fine-grained experts + shared-expert isolation |
| DeepSeek-V3 | auxiliary-loss-free (bias-based) load balancing |
| AIDO.Protein / BALM-MoE | top-2 sparse routing works in biological sequence models |
| Jamba | interleave linear/attention/MoE; small KV cache at long context |
| Gated DeltaNet | the linear-attention layer of choice |
| Oryx | mixer choice varies *within* a sequence |
| MERA | retrieval experts inside an MoE |
| ERNIE-RNA | structural prior as an additive attention bias |
| AlphaFold3 | Pairformer triangle updates; diffusion decoder |
| STEM | implicit-Manning + explicit-Mg²⁺ split |

## Sources
- AIDO.Protein / MoE for protein (bioRxiv 2024) https://www.biorxiv.org/content/10.1101/2024.11.29.625425v1
- Expert specialization in MoE antibody LMs https://pmc.ncbi.nlm.nih.gov/articles/PMC13131559/
- MERA multimodal MoE + retrieval https://arxiv.org/html/2603.01511v2
- DeepSeekMoE https://arxiv.org/html/2401.06066v1
- DeepSeek-V3 technical report https://arxiv.org/pdf/2412.19437
- MoE load-balancing review https://huggingface.co/blog/NormalUhr/moe-balance
- Gated Delta Networks (ICLR 2025) https://arxiv.org/pdf/2412.06464
- Speed Always Wins: survey of efficient LLM architectures https://arxiv.org/pdf/2508.09834
- iFlame interleaved full/linear attention https://arxiv.org/pdf/2503.16653
