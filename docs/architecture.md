# Architecture and API

HTTP and offline evaluation share `SearchService`. It validates query and limit values, resolves the requested strategy through `SearchStrategyFactory`, executes it and applies the response limit. Each strategy implements the `SearchStrategy` protocol; the factory maps the request name to its implementation. HTTP request validation, serialization and status mapping remain in `api/`.

```text
src/hybrid_retrieval_lab/
  api/                    HTTP adapter and request/response models
  encoders/               BM25, E5, ColBERT, snapshots and token validation
  ingestion/              Corpus loader, indexer and identity manifest
  services/search/
    service.py            Application flow
    resources.py          Lazy encoders and snapshot checks
    models.py             Typed hits and inspection data
    config.py             Candidate limits and RRF constant
    exceptions.py         Application errors
    strategy/
      protocol.py         SearchStrategy interface
      factory.py          Explicit name-to-strategy registry
      bm25.py             Sparse retrieval
      dense.py            Dense retrieval
      hybrid.py           RRF fusion
      colbert.py          Candidate-constrained reranking
  evaluation/             Judgments, metrics, runner and report
  logger.py               JSON logging and rotation
  cli.py                  Index and evaluate commands
tests/
  unit/
    api/                  Mirrors api/
    encoders/             Mirrors encoders/
    evaluation/           Mirrors evaluation/
    ingestion/            Mirrors ingestion/
    services/search/      Mirrors services/search/
      strategy/           Mirrors search/strategy/
    test_logger.py        Covers the package-level logger.py module
  integration/
    services/search/      End-to-end retrieval against real Qdrant/encoders
```

## Strategies

| API value | Pipeline | Score type |
| --- | --- | --- |
| `bm25` | Portuguese `Qdrant/bm25`, top 10 | `bm25` |
| `dense` | `intfloat/multilingual-e5-small`, top 10 | `cosine` |
| `hybrid` | 10 BM25 + 10 dense → RRF → 10 | `rrf` |
| `hybrid_colbert` | Same hybrid 10 → `jinaai/jina-colbert-v2` | `maxsim` |

RRF sums `1 / (60 + rank)` using one-based ranks; absent candidates contribute zero. Ties resolve by chunk ID. ColBERT uses Qdrant MaxSim over stored multivectors, filtered to the selected IDs; the returned candidate set must remain unchanged. Scores across strategies are not directly comparable.

E5 uses `query: ` and `passage: ` prefixes. Its 512-token budget includes prefixes and special tokens, counted without truncation by the snapshot tokenizer. Oversized queries fail; oversized chunks are skipped individually. BM25-only queries do not inherit the E5 limit.

## HTTP contract

`POST /search`:

| Field | Contract |
| --- | --- |
| `query` | Required string, 1–2,000 characters; blank text rejected. |
| `strategy` | One of the four values above; defaults to `bm25`. |
| `limit` | Integer 1–10; defaults to 5. |
| `inspect` | Boolean; defaults to false. |

Response fields: `query`, `strategy`, `results`, optional `inspection`. Hits include ID, rank, text, source ID/URL, title, section, score and score type. Inspection shows intermediate lists and fusion contributions. A smaller response limit does not reduce the internal reranking pool.

Validation errors map to 422, known dependency failures to 503, unexpected errors to 500 with a generic message. `X-Request-ID` correlates logs.

## Index lifecycle

Ingestion validates the corpus and prepares all encoder outputs before replacing the fixed `github_docs_pilot` collection. Its compact manifest records the corpus hash, indexed and skipped chunk IDs, chunking settings, encoder configuration and model revisions. Evaluation checks this identity, the exact point count and deterministic point IDs without reading back payloads or vectors. The API must be stopped during reindexing because search resources are cached per process. If collection creation or writing fails after replacement starts, the index remains unavailable until indexing succeeds again; automatic recovery is outside this local lab's scope.
