#!/usr/bin/env bash
# Stage 1 (masked language modelling) for shared400 -- the corrected pipeline.
#
# Every flag below that is not a default exists because something was measured
# and found wrong; `research/architecture/AUDIT_2026-09-25.md` has the evidence.
#
#   --tokens 8e9        4e9 was 0.37 of ONE epoch. The corpus is 40.3M
#                       sequences and 10.94B usable tokens after the 20-1024
#                       filter, and 325.7M active parameters want ~6.5B by
#                       Chinchilla. 8B is 0.73 epochs and 1.2x Chinchilla.
#                       The LR schedule is a cosine over TOKENS, so this
#                       number is a commitment: stopping early leaves the
#                       schedule undecayed.
#   --n-loops 3
#   --sample-loops      Free. Uniform 1..3 has mean 2.0 and costs
#                       6+2(2-1) = 8 FLOPs per active parameter per token,
#                       identical to the fixed 2 the last run used -- and the
#                       model ends up usable at 1, 2 and 3 loops instead of
#                       collapsing at everything but 2 (R7).
#   --interleave 8      A pool was 0.54 of ONE length-sorted shard: total
#                       variation 0.342 from the corpus, pool mean 201 nt
#                       against 320. Round-robin over 8 shards gives 0.112 (R9).
#   --max-batch 2048    512 bound on the sequence COUNT, not the token budget:
#                       92.5% of batches capped on short shards and the step
#                       carried 83,899 real tokens against 194,607.
#   --optimizer muon    Measured better than adamw on held-out bits.
#   --no-compile        dynamic=True never finished; dynamic=False recompiled.
#
# Held-out is scored by scripts/heldout_watch.sh with --split stratified: the
# legacy sample was 52.7% 20-79 nt against the corpus's 6.5% and overstated
# the model by ~0.19 bits (R11).
set -eu
REPO=/store/shuvam/E-motioner-X-SBS/sbs-rna
cd "$REPO"

# Pin the allocator. Do NOT inherit it.
#
# The throughput of this run depended on WHO STARTED IT. cron does not source
# the user profile; an interactive shell does, and the profile exports
# `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Measured on 2026-09-30,
# same checkpoint, same step, same code:
#
#     launched by cron (run 2)           ~2.0 s/step     15.9k tok/s
#     launched from a shell, inherited   37.8 s/step      4.4k tok/s
#
# 3.5x, from an environment variable nobody passed on purpose. Expandable
# segments are a good default when there is room; at a 60 GiB peak on an
# 80 GiB card they leave the allocator mapping and unmapping at the ceiling,
# which is what the `expandable_segments: memory mapping failed with OOM
# while trying to map 20971520 bytes` warnings were -- and memory sat at
# 67.8 GiB against run 2's 60.0 GiB peak for the same budget.
#
# Set explicitly so the two launch paths cannot diverge again.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# Overridable by the cron runner, defaulted here so the file is the source of
# truth when it is run directly.
: "${MLM_TOKENS:=8e9}"
: "${MLM_SIZE:=shared400}"
: "${NEED_GIB:=60}"
: "${MLM_OPT:=muon}"
: "${MUON_LR:=0.02}"
# Fixed budget, pinned at the level the MEASUREMENTS support -- not the one
# the previous comment claimed they did.
#
# That comment said the auto-tune "grew to 228,352 (peak 58.8 GiB)". It did
# not. 58.8 is the peak at **207,872**, from six samples. Run 2's own
# telemetry, grouped by the budget each step actually ran at:
#
#     budget    n   peak mean   peak MAX
#     143,360    6      41.3       42.9
#     157,696    6      45.3       48.8
#     173,056    6      49.4       52.0
#     180,224   17      59.6       60.0
#     189,440    6      53.9       57.8
#     207,872    6      57.7       58.8
#     228,352   87      65.4       69.6
#
# 69.6 GiB at 228,352, from the best-sampled row in the table. Against an
# 80 GiB card with a 1.7 GiB neighbour and allocator fragmentation, that does
# not fit -- and on 2026-09-30 it did not: five OOMs at step 13,773 took the
# budget 228,352 -> 99,328, worse than run 2's floor, and throughput fell to
# 13.8k tok/s against run 2's 15.9k. The card being free does not buy a
# bigger batch, because the neighbour was never the binding constraint; the
# trainer's own peak on long-sequence pools was.
#
# Note how weakly peak tracks budget -- 189,440 peaks LOWER than 180,224 --
# because memory is driven by B*L^2 and which length band the pool drew, not
# by the budget alone. That is why the MAX column over many samples is the
# only column worth sizing against, and why the six-sample rows are not
# evidence of headroom.
#
# 180,224 ran 1,650 steps with zero OOMs at 15.9k tok/s, and its 60.0 GiB
# max now sits under 80 rather than under ~71, so it has MORE margin than it
# had in run 2. Max speed here is the exclusive card, not a bigger batch.
: "${MLM_TOKEN_BUDGET:=180224}"
: "${MLM_VRAM_TARGET:=0.65}"
exec /store/shuvam/.venv/bin/python -u scripts/pretrain_mlm.py \
    --device cuda --min-free-gib "${NEED_GIB:-60}" --size "${MLM_SIZE:-shared400}" \
    --tokens "${MLM_TOKENS:-8e9}" \
    --token-budget "${MLM_TOKEN_BUDGET:-180224}" \
    --vram-target "${MLM_VRAM_TARGET:-0.65}" \
    --corpus data/derived/parquet_starter data/derived/parquet_mars \
    --n-loops 3 --sample-loops \
    --interleave 8 --max-batch 2048 \
    --optimizer "${MLM_OPT:-muon}" --muon-lr "${MUON_LR:-0.02}" \
    --no-compile "$@"
