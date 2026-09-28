# Index identity and E5 token budgets

The Qdrant collection metadata stores a compact manifest: corpus SHA-256, indexed and skipped chunk IDs, BM25 average length, encoder and chunking configuration, and model revisions. Before evaluation, the runner checks the corpus hash, configuration, complete accounting of source chunk IDs, exact point count and deterministic point IDs. It does not read every payload or vector back from Qdrant.

The indexer validates the corpus and prepares encoder outputs before deleting and recreating the fixed `github_docs_pilot` collection. Stop the API before reindexing because search resources are cached per process. A failure after deletion can leave the collection incomplete; rerun indexing before restarting the API. This deliberate trade-off keeps the local retrieval lab focused on Build, Search, Inspect and Evaluate rather than production recovery.

## Token policy

E5 uses `AutoTokenizer` loaded from the same snapshot as the embedding model. Counts include `query: ` or `passage: ` and special tokens, with truncation disabled. The budget is 512 tokens.

- A query above the budget raises `TokenLimitError`; the search service converts it to `SearchValidationError`, and HTTP returns 422 with an English explanation. The raw query is not logged.
- A passage above the budget is skipped during ingestion. The logger records its ID, count and limit in `index.chunk_skipped`. The CLI summarizes skipped IDs, and the manifest preserves them.
- If every passage is skipped, indexing raises an error before replacing the existing collection.
- Skipped passages remain in the evaluation's relevance judgments and denominator. Their absence can reduce recall. The HTML reports skipped IDs when present.
- BM25 does not acquire an artificial E5 limit when used alone. Dense, hybrid and hybrid-ColBERT requests enforce the E5 budget when encoding the dense query.

The tokenizer API is documented in [Hugging Face AutoTokenizer](https://huggingface.co/docs/transformers/model_doc/auto) and [tokenization options](https://huggingface.co/docs/transformers/main_classes/tokenizer).

## Verification

See the [development guide](development.md) for current commands to run unit and integration tests, lint, formatting and strict typing. The canonical [evaluation report](../reports/baseline/) records the frozen corpus, qrels, configuration and rankings used for comparison.
