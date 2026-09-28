# Reviewed relevance judgments

`pilot-proposed.jsonl` retains its historical filename but contains the author's reviewed judgments for the ten current queries, `q01`–`q10`: 530 query/chunk pairs, with 53 chunks per query. Each pair has an explicit grade of `0`, `1`, or `2`; a missing row would be a missing judgment, not a zero. Positive rows include a `reason`. The corpus is in [`../corpus/chunks.jsonl`](../corpus/chunks.jsonl), and the information needs are in [`../queries/pilot-queries.jsonl`](../queries/pilot-queries.jsonl).

## Rubric

- `2`: The chunk directly answers the question or sufficiently explains a plausible cause in the context of an open-ended question. The chunk text must support the explanation on its own.
- `1`: The chunk provides a useful check, step, or part of a solution, but does not resolve the main question by itself.
- `0`: The chunk adds no useful explanation or action for the information need; topical or lexical overlap alone is not enough.

The user's question defines the focus. `information_need` clarifies that focus rather than adding a second required question. For `q04`, the main answer is whether GitHub retries automatically; redelivery instructions are supporting context. For `q06`, a complete explanation of one plausible reason for an incomplete listing can receive grade 2. For `q10`, distinguishing pagination from permissions requires information about both.

`Recall@k` treats grades 1 and 2 as relevant; `nDCG@k` uses the graded labels. The current query IDs map to [`pilot-queries.jsonl`](../queries/pilot-queries.jsonl). These are author-assigned local judgments, not consensus among independent annotators. See the [judgment change log](CHANGELOG.md) and the [evaluation provenance notes](../../docs/evaluation.md#judgment-provenance).
