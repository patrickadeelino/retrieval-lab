# Evaluation guide

The pilot compares four strategies over 53 frozen Portuguese GitHub documentation chunks and ten information needs. All 530 query/chunk pairs have grades assigned by the author. This is a local reference, not independent multi-annotator ground truth.

## Relevance rubric

| Grade | Meaning |
| --- | --- |
| 0 | Does not help answer the specific information need, even if topically related. |
| 1 | Useful but incomplete supporting information. |
| 2 | Directly answers the information need or a central part of it. |

Judge the available chunk text, not unseen linked content. Reasons and borderline decisions live in [pilot-proposed.jsonl](../data/qrels/pilot-proposed.jsonl). Its name is historical; it contains reviewed grades. Preserve previous reports when changing the reference.

## Metrics

| Measure | What it reveals | Limitation |
| --- | --- | --- |
| Recall@5 / @10 | Fraction of all grade-1/2 chunks in the first k results | Ignores ordering within k; treats grades 1/2 equally. |
| nDCG@5 / @10 | Graded ordering relative to the ideal ranking; gain `2^grade − 1`, logarithmic discount | Depends on judgments; averages can hide regressions. |
| Grade 2 at rank 1 | Queries with a directly relevant first result | Does not describe the remaining ranking. |
| Candidate coverage | Relevant chunks in each retrieval/fusion stage | Reranking cannot recover absent candidates. |
| Wins / ties / losses | Per-query ColBERT change versus hybrid | Descriptive, not statistical significance. |
| Typical latency | Median of per-query medians after warmup | Small local sample; excludes HTTP and cold start. |

## Last validated result and trade-off

The clean-clone run at commit `5e9dc69` showed mean nDCG@5 increasing from 0.797 (hybrid) to 0.868 (ColBERT), with typical latency increasing from 15.94 ms to 171.93 ms, about 10.8×. Recall@10 stays at 94.3% because both return the same ten IDs. Equal Recall@10 means equal coverage at that cutoff, not equal ordering or equal Recall@5. All 40 query/strategy rankings and quality metrics matched the previous validated report; absolute latency varied between runs. The canonical artifacts are [HTML](../reports/baseline/index.html) and [JSON](../reports/baseline/pilot.json).

This is an ordering-versus-latency trade-off observed in this pilot, not evidence of universal superiority, production throughput or financial cost. The ten hand-authored queries and author-assigned judgments limit generalization. Treat the results as descriptive and compare only runs with matching corpus, candidate limits and timing methodology. The root [README](../README.md) summarizes the latest clean-clone run; see the canonical [HTML report](../reports/baseline/index.html) and [JSON data](../reports/baseline/pilot.json).

## Judgment provenance

The project author wrote the ten queries after reviewing the frozen corpus and assigned all 530 query/chunk grades using the rubric above. The queries are purpose-built for these documents; they are not a sample of production user traffic or a held-out benchmark. The qrels are single-annotator judgments, not independent multi-annotator ground truth.

The repository preserves the current qrels snapshot and records changes made after the change log was introduced. It does not provide a row-by-row history of earlier judgment revisions. Future changes should record the query and chunk IDs, old and new grades, rationale, date, and whether retrieval results had been inspected. See the [qrels change log](../data/qrels/CHANGELOG.md). Interpret the reported metrics as results against these author-assigned judgments.

## Reproduction and provenance

Use the [reproduction and recovery procedure](reproduction-and-recovery.md) to rebuild and evaluate the release index in an isolated Compose project. `--reviewed-qrels` records author review; it does not perform one.

The evaluator verifies corpus, IDs, payloads, vectors, settings and snapshots, then runs the shared service with warmup and three timed repetitions per query/strategy. JSON preserves input hashes, settings, rankings, grades, latency samples and environment. Skipped chunks remain in the relevance denominator. See [index/token policy](index-validation-and-token-budget.md).

The HTML shows aggregate quality/time, candidate coverage, paired changes and individual rankings. Saved HTML remains Portuguese; the current renderer uses English interface text with unchanged Portuguese data. Rendering saved JSON does not require another model evaluation.
