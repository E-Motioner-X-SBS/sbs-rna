# Plan — Cycle 1 (recorded after execution)

**Deviation noted honestly**: this cycle went from DECOMPOSE straight to
IMPLEMENT without writing plan.md first, because the task was *auditing existing
work* rather than building something new — the "plan" was the todo list itself.
Recorded here for the next session rather than backfilled as if written first.

## Method that worked, and should be reused

1. **Re-derive with an INDEPENDENT implementation.** Re-running the original
   script reproduces its own bugs. Every headline number was recomputed with a
   separately-written parser. This is what caught the metal/ligand and row-wrap
   defects.
2. **Chase discrepancies, however small.** The audit opened on a 31-vs-63
   mismatch that turned out to be benign — but the habit of chasing it is what
   surfaced the genuine sample mis-statement next to it.
3. **Test the confound before claiming the effect.** Partial correlation and a
   within-structure estimate turned the headline 1.76 sigma claim from
   "association that might be packing" into a much stronger result.
4. **Test the statistic, not just the number.** The coevolution depth split was
   arithmetically correct and rhetorically wrong. Exact permutation tests and
   Spearman exposed it.
5. **Cross-document scanning catches silent edit failures.** An unguarded
   `str.replace` had left the report a novelty row short. Every edit now asserts.
6. **A weak or surprising result is a suspected bug.** The MI precision of 0.039
   and the effective-c plateau at 5.0 were both defects, not findings.

## Next cycle, if one is needed
- Train the block-detection scorer and measure block recall (risk R1, still open).
- Benchmark Muon on the actual parameter shapes before assuming the ~2x.
- Acquire RMDB Mg2+ titration series (prerequisite for the ionic claim).
