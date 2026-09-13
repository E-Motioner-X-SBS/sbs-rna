# Rigor Recheck — PHAROS empirical claims

**Note on location**: the repo's `plans/` already holds RNA data-acquisition
documents (12-*, 13-*, CONTINUATION_STATE). Rigor-mode state lives in this
subfolder to avoid collision.

| Field | Value |
|---|---|
| Task | Recheck every empirical claim in the PHAROS architecture work |
| Cycle | 1 |
| Phase | 0 DECOMPOSE -> 4 IMPLEMENT |
| Started | 2026-09-13 |
| Open TODO | see todo.md |

## Scope

Artifacts under audit:
- `scripts/sampling/*.py` (7 analysis scripts)
- `research/architecture/reference/*.py` (2)
- `data/samples/analysis/*.json` (11 outputs)
- `research/architecture/ARCHITECTURE.md`
- `research/report/main.tex` -> main.pdf (21pp)
- `research/architecture/blueprint.html` (published artifact v5)
- `research/prior-art/01-06`

## Prior known defects (found during session 1, already fixed)
1. BGSU nrlist endpoint returned HTML, parsed as CSV -> 0 structures
2. `_exptl.method_details` overwrote `_exptl.method` -> 0 X-ray detected
3. MI outer-product broadcast (C,q,1) vs (C,q,q) -> precision 17x too low
4. HPT L3 expansion took block diagonal only -> quarter of pair budget
5. HPT L2 budget clamped by masking -> effective c pinned at 5.0

Five defects in one session is a high base rate. Assume more exist.
