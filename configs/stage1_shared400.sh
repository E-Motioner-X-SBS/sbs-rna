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
# Overridable by the cron runner, defaulted here so the file is the source of
# truth when it is run directly.
: "${MLM_TOKENS:=8e9}"
: "${MLM_SIZE:=shared400}"
: "${NEED_GIB:=60}"
: "${MLM_OPT:=muon}"
: "${MUON_LR:=0.02}"
# Fixed budget and a vram target that keeps the auto-tune from growing into
# the OOM that cost run 2 half its batch. 0 restores auto-tuning.
#
# The auto-tune grew to 228,352 (peak 58.8 GiB), then took one more 10% step
# and peak went past 79 -- memory is not linear in the budget, because
# attention is B*L^2 and the pool composition moves. Four OOMs at one step
# followed, each cutting 15%, ending at 117,760. 180,224 is 79% of the level
# that survived; at an estimated 46.4 GiB peak it sits above
# 0.85 * 0.65 * 79.3 = 43.8, so the growth check never fires.
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
