# Search observability

The search flow writes one JSON object per line to `logs/hybrid-retrieval-lab.log` when run locally, or to `/app/logs/hybrid-retrieval-lab.log` inside the API container. Compose bind-mounts `./logs` so the file is visible on the host. The log directory is ignored by Git.

| Setting | Default | Purpose |
| --- | --- | --- |
| `LOG_FILE` | `logs/hybrid-retrieval-lab.log` | File path outside Compose; Compose sets `/app/logs/hybrid-retrieval-lab.log`. |
| `LOG_LEVEL` | `INFO` | Accepts `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |
| `LOG_MAX_BYTES` | `10485760` | Rotates after 10 MiB by default. |
| `LOG_BACKUP_COUNT` | `5` | Number of rotated files to keep. |

Every HTTP request receives a generated `request_id`, also returned in `X-Request-ID`. A request's records share this ID. The API records method, path, status, duration, strategy, result count, score type, and returned chunk IDs. It does **not** record the raw query, chunk text, request body, or full HTTP response. `query_chars` helps distinguish empty/large inputs without storing content.

Unexpected exceptions return HTTP 500 with a generic English message and the same `X-Request-ID`. The file records the exception type and traceback, then a final completion event with status 500.

| Event | Level | Meaning and useful fields |
| --- | --- | --- |
| `http.request.received` | INFO | Method and path at ingress. |
| `search.started` | INFO | Strategy selected, limit, inspection flag, query length. |
| `encoder.loaded` | INFO | Lazy model initialization and duration; appears only on first use per process. |
| `search.bm25.completed` / `search.dense.completed` | INFO | First-stage candidate counts and durations, including query encoding and Qdrant call. |
| `search.rrf.completed` | INFO | BM25/dense input counts, fused count, RRF k, fusion duration. |
| `search.colbert.completed` | INFO | Candidate count, result count, moved positions, reranking duration. |
| `search.completed` | INFO | Application-level total duration and returned count. |
| `search.response.ready` | INFO | Ordered returned chunk IDs and score type. |
| `http.request.completed` | INFO or ERROR | Final HTTP status and duration; 4xx/5xx use ERROR and include a known error type when available. |
| `search.dependency_failed` / `http.request.failed` | ERROR | Exception type and traceback for dependency or unexpected failures. |
| `search.query_rejected` | ERROR | Model, token count and limit for a query that exceeds the E5 window; HTTP returns 422. |
| `index.chunk_skipped` | ERROR | Chunk ID, model, token count and limit; ingestion continues with other chunks. |
| `index.*` / `cli.command.*` | INFO or ERROR | Corpus validation, vector preparation, generation creation/verification, alias switch, completion, and CLI failure. |

`index.completed` also records `skipped_count`, `skipped_ids`, and the stored-index checksum. The collection manifest preserves these exclusions so evaluation does not silently remove them from the relevance denominator.

The search total includes lazy model initialization; a first-stage duration does not, because the encoder is loaded before that function starts. Compare warm requests when judging retrieval latency. The file logger is process-local and uses standard `RotatingFileHandler`; Compose currently runs one API worker.

Run one request and inspect the file on the host:

```bash
docker compose up -d --build api
curl -sS -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"How do I validate a GitHub webhook?","strategy":"hybrid_colbert","limit":5}'
tail -n 20 logs/hybrid-retrieval-lab.log
```

This event inventory is specific to Hybrid Retrieval Lab. No external dashboard or Sentry event contract is part of this phase.
