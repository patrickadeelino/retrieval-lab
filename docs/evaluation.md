# Evaluation guide

The pilot compares four strategies over 53 frozen Portuguese GitHub documentation chunks and ten information needs. All 530 query/chunk pairs have grades. The author reviewed them, including a second-opinion audit; this is a local reference, not independent multi-annotator ground truth.

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

The last validated run showed mean nDCG@5 increasing from 0.797 (hybrid) to 0.868 (ColBERT), with typical latency increasing from 13.95 ms to 154.46 ms, about 11.1×. Recall@10 stays at 94.3% because both return the same ten IDs. Equal Recall@10 means equal coverage at that cutoff, not equal ordering or equal Recall@5. The old report bundles were removed; the final clean-clone report is pending approval and will be the only HTML/JSON pair under `reports/baseline/`.

This is an ordering-versus-latency trade-off for this pilot, not universal superiority, production throughput or financial cost. Ten queries and no independent held-out benchmark limit generalization. The original [conclusions](experiment-conclusions.md) remain in Portuguese. Earlier runs used different timing summaries or candidate counts; compare like-for-like settings.

## Reproduction and provenance

Use the [reproduction and recovery procedure](reproduction-and-recovery.md) to rebuild and evaluate the release index in an isolated Compose project. `--reviewed-qrels` records author review; it does not perform one.

The evaluator verifies corpus, IDs, payloads, vectors, settings and snapshots, then runs the shared service with warmup and three timed repetitions per query/strategy. JSON preserves input hashes, settings, rankings, grades, latency samples and environment. Skipped chunks remain in the relevance denominator. See [index/token policy](index-validation-and-token-budget.md).

The HTML shows aggregate quality/time, candidate coverage, paired changes and individual rankings. Saved HTML remains Portuguese; the current renderer uses English interface text with unchanged Portuguese data. Rendering saved JSON does not require another model evaluation.
