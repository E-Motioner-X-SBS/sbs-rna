# Architecture A/Bs against the three-run baseline

Each arm: 40 epochs, 16,384-token budget, lr 2e-4, from the same stage-4 init, scored on 17 RNA-Puzzles targets.

The baselines span **0.0046** at identical settings and the init-RNG noise floor measured between two of them is **0.0023**, so a mean difference under about 0.0046 is inside the spread the settings already produce. On 17 targets, two-sided p < 0.05 needs **13 wins**.

| arm | baseline | n | arm TM | base TM | delta | wins | p |
|---|---|---|---|---|---|---|---|
| 100  GeometricBias | run 1 (blind_tests_v2) | 17 | 0.0907 | 0.0913 | -0.0006 | 6/17 | 0.332 |
| 100  GeometricBias | run 2 (blind_ab_A) | 17 | 0.0907 | 0.0890 | +0.0017 | 11/17 | 0.332 |
| 100  GeometricBias | run 3 (blind_tests_v3) | 17 | 0.0907 | 0.0936 | -0.0029 | 5/17 | 0.143 |
| 103  bidirectional gdn | run 1 (blind_tests_v2) | 17 | 0.0862 | 0.0913 | -0.0051 | 6/17 | 0.332 |
| 103  bidirectional gdn | run 2 (blind_ab_A) | 17 | 0.0862 | 0.0890 | -0.0028 | 7/17 | 0.629 |
| 103  bidirectional gdn | run 3 (blind_tests_v3) | 17 | 0.0862 | 0.0936 | -0.0074 | 6/17 | 0.332 |
| 112  union coevolution | run 1 (blind_tests_v2) | 17 | 0.0867 | 0.0913 | -0.0045 | 7/17 | 0.629 |
| 112  union coevolution | run 2 (blind_ab_A) | 17 | 0.0867 | 0.0890 | -0.0022 | 10/17 | 0.629 |
| 112  union coevolution | run 3 (blind_tests_v3) | 17 | 0.0867 | 0.0936 | -0.0069 | 4/17 | 0.049 |
| 92   bilinear contact bias | run 1 (blind_tests_v2) | 17 | 0.0861 | 0.0913 | -0.0052 | 5/17 | 0.143 |
| 92   bilinear contact bias | run 2 (blind_ab_A) | 17 | 0.0861 | 0.0890 | -0.0029 | 6/17 | 0.332 |
| 92   bilinear contact bias | run 3 (blind_tests_v3) | 17 | 0.0861 | 0.0936 | -0.0075 | 5/17 | 0.143 |

## Verdicts

| arm | mean delta | best p | verdict |
|---|---|---|---|
| 100  GeometricBias | -0.0006 | 0.143 | **NO EFFECT** — inside the spread three identical runs give |
| 103  bidirectional gdn | -0.0051 | 0.332 | **WORSE** — below the baselines by more than their spread |
| 112  union coevolution | -0.0046 | 0.049 | **WORSE, weakly** — mean -0.0046 is inside the 0.0046 spread, but a paired sign test reaches p = 0.049; directionally negative, not neutral |
| 92   bilinear contact bias | -0.0052 | 0.143 | **WORSE** — below the baselines by more than their spread |

Not yet run: 109  triangle update.
