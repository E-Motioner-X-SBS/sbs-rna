# Continuation State — sbs-rna

## Session Summary
| Field | Value |
|-------|-------|
| Phase | **v0.2a SPECIFIED, MEASURED, BUILT AND TRAINING.** The §12.1 curriculum runs chained and unattended; the block scorer is trained and measured end to end. One item is still blocked externally: the rest of Ribonanza (Kaggle credentials). |
| What I did | Collected and verified 256.6 GB of RNA data; closed D9/D10/G1/G2/G3 on the raw archive; **built the model** (hybrid trunk, MoE, 667-class motif bank, harmonic ensemble, ten heads) and the **16,604-chain training set** with its free supervision; acquired the **RMDB ionic ladders**; re-derived S7.1 on 2,994 chains |
| What worked | elDORS S3 anonymous download (SHA256 verified); HF/GitHub direct fetches; catalog + loader API; parquet ETL; exploration pipeline |
| Errors | none outstanding (all downloads completed byte-exact; catalog rebuilt from new location) |
| Corpus | **v0.2a: the pretraining corpus was a length band.** elDORS is sorted by length and ships as 20 chunks; the starter took c001-c008, so stage 1 saw sequences averaging 850 nt at GC 0.474 and never saw the other half of the corpus, averaging 233 nt at GC 0.572. Rebuilt across all 20 chunks: 25M sequences, 20/20 coverage, median 522 -> 261. Shard order is now shuffled and `LEN_QUANTUM` is 64 rather than 128, both re-measured. |
| Next priorities | 1) **Restart stage 1 on the corrected corpus** to 2e9 tokens (31.5 tok/param) at ~36.5k tok/s, resuming every 250 steps; stages 2-3 and 5 follow chained. The 101.9M tokens it reached before the corpus was fixed were trained on the long-sequence slice and are discarded. 2) **Retrain the block scorer with the L1 diagonal admitted** -- the cascade measurement showed 18.9% of true L2 blocks are unreachable at any budget because L1 clamps its minimum separation to a 16-residue block where L2 clamps to a 4-residue one, and the current weights cannot exploit the fixed mask because they never scored a diagonal block. 3) **Re-probe router specialisation** as stage 1 proceeds: at 55.4M tokens the per-token routing entropy was 99.0% of uniform, i.e. the MoE was a dense feed-forward. `probe_router_specialisation.py` appends to a history keyed on token count and can be run against a live checkpoint. 4) **Compile stage 5 too.** Prepared, not yet done: the 3D set has median 121 but p90 2,854 (the ribosome tail), and at a 32,768-token budget quantising the width to 128 gives **17 distinct widths at 7.6% padding** -- the same shape of bargain stage 1 took, and few enough graphs to compile. Needs measuring on stage 5's own shapes before enabling, since SWA there is O(L^2) with a materialised band mask at a 4x longer context and may dominate differently. 5) Open action 3 (Ribonanza beyond 335,616) needs the user's Kaggle credentials; the public mirrors are the same file we already hold. |
| Blockers | none for training. The A100 is shared and gets taken back without warning, which is why every stage resumes and the runner requires 60 GiB free -- 40 was too low, stage 1 peaks at 51.8 and the old floor let runs start on cards they would then OOM on. |
| Audit status | v0.1 audited (28 defects, 11 cycles); v0.2 validated on 29,807 chains + 10,520 raw entries. **C15** (entry counter keyed `label_asym_id` against auth-keyed entity declarations -- 3,254 entries silently read zero) and **C16** (NMR ensembles stacked 20 models into one residue -- 424 atoms/residue on 1ARJ) found and fixed in the raw pass. v0.2a adds four more, all from measuring things that had been assumed: the stage-1 packer was **42.9% padding**, the chemistry was built **twice per step and used once**, `train_flops` charged every refinement loop a full backward the one-step gradient does not do (**1.8x overcharge**), and `MFU = 0.35` had never been checked against a run (**measured 3.6%, now 6.6%**). `verify_claims.py` pins **301** checks across 12 suites. |

## State of the data (all verified)
- Sequences: 1,369,926,204 (elDORS 1,323,715,880 SHA256-verified + RNAcentral 46,210,324)
- 3D, curated derivatives: 27,452 chains (RNA3DB 15,441 + gRNAde 12,011) = 6,661 unique sequences
- **3D, raw archive: 29,038 RNA chains in 10,520 entries, 13,348,166 residues.** The
  derivatives cover 7,943 of those entries; **2,581 (24.5%) are in neither**, all
  already on disk -- 608 with a chain in the 64-3,000 window, 925 protein-free,
  and they hold the entire top-30 of the effective_c distribution.
- Benchmarks: ~160k 2D annotations, >3.6M fitness measurements, 147-species splicing
- Rfam 15.1: 4,227 families; 100 with 3D representatives; 10.07M Rfam-mapped sequences
- Catalog: data/catalog/catalog.sqlite (47,471 files indexed)
- **RMDB ionic titrations** (195 MB, `data/benchmarks/rmdb/`): 24 Mg2+ ladders,
  527 concentration points, **all 24 crossing sub-millimolar** -- the regime the
  PDB cannot contain because unfolded RNA is not deposited. RMDB's bulk endpoint
  is now GitHub release assets, not the old Django site.
- **PHAROS 3D training set** (`data/derived/pharos3d/`, 33 shards): 16,604 chains,
  13,172,991 residues, 63,298,800 contacts, with Mg sites, X-ray B-factors,
  N_struct and disorder labels extracted from the entries themselves.
- **PDB Chemical Component Dictionary** (114 MB, `data/structures/ccd/components.cif.gz`,
  wwPDB 2026-09-12): the only source of a modified residue's parent base -- entry
  files do not carry the field. Resolves all 370 modification species the archive
  contains; 88.6% of modified residues map to a standard A/C/G/U parent.

## Key documents
| File | Content |
|------|---------|
| data/catalog/README.md | source-by-source inventory |
| data/catalog/VERIFICATION.md | all verification checks |
| data/catalog/3d_data_statistics.json | 3D quality tiers |
| data/exploration/reports/00-EXPLORATION-SUMMARY.md | findings + recommendations |
| plans/12-nucleicbert-data-scanner.md | acquisition log + recipes |
| plans/13-sequence-structure-gap-strategy.md | strategy for the 3D gap |

## How to resume training

Training was stopped by request at **188,601,237 tokens, step 7,500** (9.4% of
the 2e9 stage-1 budget), with optimiser state, on a clean signal. Nothing was
lost: the checkpoint's step matches the last logged step exactly.

```bash
crontab -e            # uncomment the #STOPPED-BY-REQUEST line
```

That is all. The next fire resumes stage 1 from its checkpoint, sizes the batch
from whatever VRAM is free, and carries on to stages 2-3 and 5. To run it now
instead of waiting for the half-hour tick:

```bash
./scripts/gpu_cron_runner.sh >> data/samples/analysis/cron/cron.log 2>&1 &
```

Every stage resumes: model, optimiser, and for stage 5 and the scorer the
OneCycleLR state too. `--restart` renames rather than overwrites.

## Where the training record lives

`data/samples/analysis/runs/` — a CSV per stage, appended and flushed per row,
so a killed run keeps everything up to its last logged step:

```
runs/stage1_mlm.csv            every row every stage-1 run ever logged
runs/stage1_mlm/<run_id>.csv   one run alone
runs/stage1_mlm/<run_id>.json  the manifest: argv, git commit, config,
                               params, device, resume point
```

23 columns for stage 1: tokens, padded tokens, CE in nats and bits, the MoE
balance term, total loss, learning rate, masked accuracy, real and padded
throughput, MFU, padding fraction, token budget, OOM count, peak GiB. `kind`
distinguishes `step` / `epoch` / `eval` / `event` rows, so an OOM or a resume
appears in the same file as the metrics around it.

This exists because it did not before: history was serialised once, at the end
of a run, and on a shared card runs do not end. Stage 1 reached 188.6M tokens
across several fires and produced no history file at all.

## The `docs/acquisition-inventory` branch is superseded -- DO NOT MERGE IT

Its five useful files (`scripts/acquire_all.py`, `build_raw_pdb_entrylist.py`,
`watch_downloads.py`, `pyproject.toml`, `uv.lock`) were cherry-picked onto main
on 2026-09-21. The branch itself is a snapshot from before the model existed:
it is **47 commits behind main**, and merging it would **delete 65 files**,
including all of `src/pharos/`. Opening the PR that its URL suggests would
propose exactly that deletion.

Nothing on it is now missing from main, and it is archived at
`archive/acquisition-inventory/` -- the branch's own copies of the three
scripts, its commit list, its full file list, and a README recording what was
taken and how to use `acquire_all.py` to fetch datasets (including MARS, the
one source skipped by decision). The branch can be deleted whenever you like;
until then it must not be merged.

## Repo
- Local: /store/shuvam/E-motioner-X-SBS/sbs-rna
- Remote: https://github.com/E-Motioner-X-SBS/sbs-rna (public, main)
- Versioned: catalog + exploration + READMEs + scripts (33 files)
- Ignored: bulk data (sequences/structures/benchmarks/families/derived)

## Continuation prompt hints
- Start from model design now that data is complete: tokenizer (5 symbols),
  context window (2048 recommended), pretraining objective (MLM + span masking
  like NucleicBERT), and the 3D fine-tuning head (contact/distance maps from
  RNA3DB with the documented split).
- If curation decided: write the filter to `scripts/` and regenerate the
  pretraining parquet starter pack with the chosen chunk weighting.

## Unattended training (installed 2026-09-21)

A **system cron** entry, not an in-session watcher, because the shared A100 has
stayed occupied for many hours at a stretch and a watcher inside a Claude
session dies with the session:

```
9,39 * * * * .../scripts/gpu_cron_runner.sh >> .../cron/cron.log 2>&1
```

Every 30 minutes it checks the card, and when it is genuinely free it runs the
**whole §12.1 curriculum in order, passing weights forward**:

```
verify_claims.py
  -> pretrain_mlm.py        stage 1, MLM        resumes from its own checkpoint
  -> train_sequence_stages  stages 2-3          --init-from pretrain_small.pt
  -> train_block_scorer.py  R1                  standalone BY DESIGN: a conv
                                                residue encoder, not the trunk
  -> train_pharos.py        stage 5             --init-from seqstages_small.pt
  -> verify_claims.py
```

Chaining is the point. Stages that do not pass weights are four unrelated runs,
and stage 5 from random init is what produced r = 0.049 on unseen folds.
`--init-from` refuses rather than proceeds if more than half the tensors fail to
load, and `pretrain_mlm.py` strips `torch.compile`'s `_orig_mod.` prefix when it
saves, so an uncompiled stage 5 finds the tensors it expects. State lands in
`data/samples/analysis/cron/status.json`; one log per run beside it.

Four guards, each for a failure this job can actually hit:

* **flock** -- cron fires every 30 min and a run takes hours, so without a lock
  the second fire starts a second run on the same card and both OOM.
* **sustained free-check** -- a dip between two phases of somebody else's job
  looks exactly like a free card, so the memory floor must hold across a
  re-check 90 s later before anything starts.
* **done markers** -- cron fires forever; a finished stage is not repeated.
* **validation first** -- if `verify_claims.py` fails, the architecture is
  inconsistent with its own measurements and training it would produce a number
  about the wrong model, so the run stops there.

To check on it: `cat data/samples/analysis/cron/status.json`.
To stop it: `crontab -e` and delete the line.
