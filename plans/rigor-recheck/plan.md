# Plan — Cycle 2

## Hypothesis being tested

That the 6,259 lines added after cycle 1 closed are as sound as the audited
core. Given a base rate of ten silent defects, the prior is that they are not.

## Method (carried forward from cycle 1, DEC-2)

Re-derive every headline number with an **independently written** check, never
by re-running the script that produced it. Chase every discrepancy however
small. Treat a weak or surprising result as a suspected bug.

## Step-locked order (highest suspicion first)

| Step | Target | Why this order |
|---|---|---|
| S1 | `_pdbx_unobs_or_zero_occ_residues` = 143,871 **RNA** residues? | The category covers **all polymers**. The 180 structures include ribosomes with many protein chains. If the extractor did not filter to RNA, this label count — now a *novelty claim* — is inflated. **Highest suspicion.** |
| S2 | A100-hour internal consistency: table says PHAROS-Small **299**, text says **~79** | Two different numbers for one configuration in the same document. Either a lever stack is being applied in one place and not the other, or one is wrong. |
| S3 | Right-sizing arithmetic: 149M/61M, 128 vs 96 effective layers, 9.4x | The whole model was resized on these. Must reproduce exactly. |
| S4 | Stiffness headroom NLL gains (2.1443 / 3.0282 / 1.8765) | Load-bearing for "learned, not tabulated" and for §7d. |
| S5 | Aux block-occupancy loss: specified, or actually implemented and tested? | The commit says "both implemented". R1 downgrade depends on it. |
| S6 | Literature claims in prior-art 07/08 not verified this session | gRNAde 3-5%/3 states; RNAnneal 16 ERCs/10 states; pentameric couplings; Muon ~2x; FP8 <0.25%; HRM +13pp. Anti-hallucination rule 1 and 10. |
| S7 | Cross-document consistency after 16 commits | Cycle 1 found a silent edit failure this way. |

## Success criterion per step

A step is done when an independent check reproduces the number, or a defect is
logged with a correction applied to all three deliverables and the regression
guard updated.
