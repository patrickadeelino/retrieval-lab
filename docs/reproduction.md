# Reproduce the retrieval evaluation

The canonical token-only baseline is stored in `reports/baseline/`. Rebuild it from the repository root with Docker Compose:

```bash
docker compose build
docker compose up -d qdrant
docker compose run --rm -e LOG_FILE=/tmp/index-run.log api \
  python -m hybrid_retrieval_lab.cli index
docker compose up -d api
docker compose run --rm -e LOG_FILE=/tmp/evaluation-run.log api \
  python -m hybrid_retrieval_lab.cli evaluate \
  --qrels data/qrels/pilot-token-only.jsonl --output reports/baseline
```

The evaluator checks the corpus and index manifest before timing searches. Its JSON records input hashes, settings, rankings, grades, latency samples and environment. The HTML report can be opened without downloading models:

```bash
python -m http.server 8770 --bind 127.0.0.1 --directory reports/baseline
```

## Reindexing

The API caches the collection and encoder resources, so stop it before indexing:

```bash
docker compose stop api
docker compose run --rm -e LOG_FILE=/tmp/index-run.log api \
  python -m hybrid_retrieval_lab.cli index
docker compose up -d api
```

Indexing validates all input chunks and prepares their vectors before deleting and recreating the fixed `github_docs_pilot` collection. The API is unavailable during this operation. If Qdrant fails after collection deletion, rerun indexing; there is no automatic alias recovery or previous generation to restore. Existing model downloads remain in the Compose `model_cache` volume.

For unit/integration commands and type/lint checks, see the [development guide](development.md). For the manifest and oversized chunk behavior, see [index identity and token budgets](index-validation-and-token-budget.md).
