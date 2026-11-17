# Research notes and improvement plan

## Historical observations

The study compared prompt and pipeline changes on a small exploratory dataset. Adding questions, retrying outputs, or passing vision descriptions to a language model did not uniformly improve the four classes. Test 17 has higher exact match than tests 7 and 19 but zero F1 for `wrong`; selecting it as the best error detector would be misleading.

## Engineering findings

1. **Parser/format interaction:** For `**Categories:** Wrong, Weird, Unrealistic`, splitting on `Categories:` leaves `** Wrong` as the first token. The historical exact-token filter drops that token. Parsing an entire analysis response by comma can also drop a prefixed first category. The final text classification response is not consistently retained, so not all historical parsing can be replayed.
2. **Mixed runs:** Appending to one `model_analysis.txt` allows repeated batches to accumulate. Four recovered files have duplicate filenames. A future exporter should assign an immutable run ID and record boundaries explicitly.
3. **Missing configuration:** Model IDs, prompts, temperatures and timestamps were not consistently stored with each run. The current source's temperature or model must not be assumed to describe every historical log.
4. **Metric conventions:** Historical `zero_division=1` may report perfect precision when a class is never predicted. The offline evaluator uses zero and reports support and predicted counts. All four classes occur in this dataset, so this convention does not change their F1 values for the archived runs.
5. **Confusion matrices:** Flattening image-label decisions produces an aggregated binary 2×2 matrix, not a mutually exclusive four-class matrix. The new summary reports per-label confusion counts.

## Next research steps (not completed)

- Use structured outputs with schema validation; distinguish refusal, invalid format, transient API failures and valid labels. Never silently map failures to `other`.
- Create a configurable, current inference adapter with run manifests, bounded retries, checkpoints, and no reference labels in model prompts. A model migration is a new experiment.
- Refine the label guide and collect independent annotations to evaluate agreement.
- Acquire a separate test set. The existing data have already influenced prompt development and should not be presented as an untouched test set after retrospective splitting.
- Compare fixed-model prompt strategies under controlled sampling settings and repeated runs, then estimate uncertainty. Include constant-label baselines and all per-class results.
- Investigate whether staged error-versus-style decisions help; this is a proposal, not an implemented or validated contribution.

## Audit inclusion

| Source log | Records | Unique images | Offline comparison |
| --- | ---: | ---: | --- |
| test 7 | 90 | 90 | Included |
| test 8 | 107 | 90 | Excluded: duplicate filenames |
| test 9 | 90 | 90 | Included |
| test 10 | 90 | 90 | Included |
| test 11 | 90 | 90 | Included |
| test 12 | 90 | 90 | Included |
| test 13 | 90 | 90 | Included |
| test 15 | 270 | 90 | Excluded: duplicate filenames |
| test 16 | 131 | 90 | Excluded: duplicate filenames |
| test 17 | 90 | 90 | Included |
| test 18 | 109 | 90 | Excluded: duplicate filenames |
| test 19 | 90 | 90 | Included |

Logs for tests 1–6 and 14 were not found in the available project. Integrity checks establish only that filenames and annotations align; they do not establish experimental comparability, independence, correctness of model reasoning, or statistical significance.
