# Rigor Recheck — PHAROS empirical claims

**Note on location**: the repo's `plans/` already holds RNA data-acquisition
documents (12-*, 13-*, CONTINUATION_STATE). Rigor-mode state lives in this
subfolder to avoid collision.

| Field | Value |
|---|---|
| Task | Recheck every empirical claim in the PHAROS architecture work |
| Cycle | **5** — buildability audit |
| Phase | 6 AUDIT (cycle 5) |
| Started | 2026-09-13 (cycle 1), cycle 2 same day |
| Open TODO | 0 open, 2 deferred to the 5B checkpoint |

## Why cycle 2 exists

Cycle 1 exited macro-audit PASS with scope fixed at: prior-art 01-06,
ARCHITECTURE.md, main.tex (**21pp**), blueprint (**v5**). Since that audit
closed, 16 commits added **6,259 insertions across 55 files**. main.tex is now
**33pp** and the blueprint is **v14**. The following entered *unaudited*:

| Added since cycle 1 | Status |
|---|---|
| mmCIF mining: 103,964 steps, 44,708 metalc, 143,871 unobs | UNAUDITED |
| Stiffness encoder (§7c) + headroom NLL model | UNAUDITED |
| **Dynamics module (§7d) + prior-art 08 + diagram 08 + blueprint §13** | UNAUDITED (newest) |
| Right-sizing to PHAROS-Small (149M/61M) | partially self-audited |
| Training cost model, Muon/FP8/HRM claims | flagged "reported, must benchmark" |
| 115x stiffness correction (mine, cycle-2 pre-work) | self-verified only |
| Gradient fix: score gating + aux occupancy loss | fix verified; aux loss untested |
| 5 new scripts, 2 new prior-art docs | UNAUDITED |

## Scope — cycle 2

- `scripts/sampling/{extract_basepair_geometry,measure_stiffness_headroom,
  optimize_architecture,parameter_budget_justification,training_cost_model,
  test_residue_guard_bias}.py`
- `research/architecture/reference/{hierarchical_pair_track,
  test_hierarchical_pair_track}.py`
- `research/prior-art/07`, `research/prior-art/08`
- ARCHITECTURE.md §7c, §7d, §10b-d; main.tex new sections; blueprint §09-13
- `data/samples/analysis/{basepair_geometry,stiffness_headroom,
  architecture_optimization,training_cost_base,residue_guard_bias}.json`

## Prior defect base rate

Cycle 0 (build): 5 silent defects. Cycle 1 (audit): 4 more. Pre-cycle-2: 1
(stiffness rounding). **Ten defects, every one producing a plausible number
rather than an error.** Assume more exist in the 6,259 unaudited lines.
