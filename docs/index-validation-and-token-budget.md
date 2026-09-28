# Index identity and E5 token budgets

The index stores a versioned manifest in Qdrant collection metadata. It records the source corpus SHA-256, indexed IDs, skipped chunks and their token counts, BM25 average length, encoder settings and package versions, and the exact Hugging Face snapshot revisions resolved by FastEmbed. A separate SHA-256 covers the IDs, payloads and vectors read back from Qdrant after indexing.

Before evaluating, the runner checks the corpus hash, encoder configuration, complete accounting of indexed/skipped IDs, payloads, and stored-vector checksum. Equal point counts are insufficient. Missing manifests and mismatches fail before any timed search. Search resources also reject model revisions that differ from the index. Older collections must be reindexed once.

The checksum scans the full index. This is deliberately appropriate for the small local pilot; a larger deployment would need a different verification cost model. Indexing writes a uniquely named physical collection and verifies it before switching the `github_docs_pilot_active` alias in one Qdrant alias-update operation. A failed build leaves the active generation intact; the prior generation is removed only after a successful switch.

## Token policy

E5 uses `AutoTokenizer` loaded locally from the same snapshot as the embedding model. Counts include `query: ` or `passage: ` and special tokens, with truncation disabled. The budget is 512 tokens.

- A query above the budget raises `TokenLimitError`; the search service converts it to `SearchValidationError`, and HTTP returns 422 with an English explanation. The raw query is not logged.
- A passage above the budget is skipped during ingestion. The logger records its ID, count and limit in `index.chunk_skipped`. The CLI summarizes skipped IDs, and the manifest preserves them.
- If every passage is skipped, indexing raises an error before creating or switching the active collection. If candidate creation, upload, or verification fails, the candidate is removed and the active alias remains unchanged.
- Skipped passages remain in the evaluation's relevance judgments and denominator. Their absence can reduce recall. The HTML reports skipped IDs when present.
- BM25 does not acquire an artificial E5 limit when used alone. Dense, hybrid and hybrid-ColBERT requests enforce the E5 budget when encoding the dense query.

The tokenizer API is documented in [Hugging Face AutoTokenizer](https://huggingface.co/docs/transformers/model_doc/auto) and [tokenization options](https://huggingface.co/docs/transformers/main_classes/tokenizer).

## Verification

51 unit tests passed. The six integration tests passed against real Qdrant and encoders, including a mixed valid/oversized corpus, preservation of an existing collection when every chunk is oversized, manifest verification, and rejection of an oversized query. Ruff lint/format and strict mypy passed over 37 source/script files.

The clean-clone pilot rerun and comparison are recorded in the [evaluation guide](evaluation.md), with the canonical artifacts in [`reports/baseline/`](../reports/baseline/).
