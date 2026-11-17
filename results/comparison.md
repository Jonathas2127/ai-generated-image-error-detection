# Recomputed historical results

All results use the same 90-image exploratory dataset. No new model calls were made.
Undefined precision/recall/F1 = 0. Historical parsed predictions are unchanged.

| Run | Exact match | Macro-F1 | Micro-F1 | wrong F1 | Empty | Other rule violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| always_wrong | 7.78% | 0.0833 | 0.1765 | 0.3333 | 0 | 0 |
| always_weird | 2.22% | 0.0794 | 0.1667 | 0.0000 | 0 | 0 |
| always_unrealistic | 26.67% | 0.1667 | 0.4412 | 0.0000 | 0 | 0 |
| always_other | 37.78% | 0.1371 | 0.3333 | 0.0000 | 0 | 0 |
| test_07 | 25.56% | 0.5324 | 0.5714 | 0.3784 | 14 | 6 |
| test_09 | 20.00% | 0.4671 | 0.5116 | 0.2667 | 34 | 7 |
| test_10 | 18.89% | 0.3107 | 0.3800 | 0.2581 | 28 | 2 |
| test_11 | 15.56% | 0.3688 | 0.4444 | 0.2727 | 39 | 3 |
| test_12 | 16.67% | 0.2622 | 0.3711 | 0.1053 | 28 | 0 |
| test_13 | 16.67% | 0.2600 | 0.3696 | 0.1053 | 31 | 0 |
| test_17 | 30.00% | 0.3598 | 0.4786 | 0.0000 | 0 | 10 |
| test_19 | 27.78% | 0.5147 | 0.5704 | 0.2051 | 0 | 12 |

Constant-label baselines are descriptive comparisons on this dataset, not held-out results.
See summary.json for per-label precision, recall, F1 and TP/FP/FN/TN counts.
