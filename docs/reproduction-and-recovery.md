# Reproduction and index recovery

This guide defines the repeatable local run and the safe response to an interrupted index replacement. The current token-only baseline is stored at `reports/baseline/`; its judgments include grades carried forward from the previous corpus snapshot, as documented in the [evaluation guide](evaluation.md#judgment-provenance).

## Reproduce from the release commit

Run this procedure from a clean clone of the commit that contains the locked install and recovery changes. It creates a separate Compose project, Qdrant volume, model cache and host ports, so an existing local stack is left untouched.

```bash
git clone https://github.com/patrickadeelino/retrieval-lab.git
cd retrieval-lab

uv sync --locked --extra dev
export GIT_COMMIT="$(git rev-parse HEAD)"
if [ -z "$(git status --porcelain)" ]; then
  export GIT_TREE_STATE=clean
else
  export GIT_TREE_STATE=dirty
fi
export UV_VERSION="$(uv --version)"
export QDRANT_HOST_PORT=16333
export API_HOST_PORT=18000

docker compose -p retrieval-lab-baseline build --no-cache
docker compose -p retrieval-lab-baseline up -d qdrant
docker compose -p retrieval-lab-baseline run --rm \
  -e LOG_FILE=/tmp/index-run.log api \
  python -m hybrid_retrieval_lab.cli index
docker compose -p retrieval-lab-baseline up -d api

for attempt in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:18000/openapi.json >/dev/null; then
    break
  fi
  sleep 1
done

docker compose -p retrieval-lab-baseline run --rm \
  -e LOG_FILE=/tmp/evaluation-run.log api \
  python -m hybrid_retrieval_lab.cli evaluate \
  --qrels data/qrels/pilot-token-only.jsonl --output reports/baseline
```

The evaluator verifies the corpus, judgments, stored vectors and index manifest before timing. It writes one HTML report and its source JSON. The runtime section records Python/platform, CPU count, Qdrant and uv versions, the `uv.lock` SHA-256, Git revision/tree state and encoder snapshots. The HTML can be viewed without loading the models:

```bash
python -m http.server 8770 --bind 127.0.0.1 --directory reports/baseline
```

When finished, remove only this isolated stack and its downloaded model cache:

```bash
docker compose -p retrieval-lab-baseline down -v
```

The report lives in the host checkout and remains available. A fresh model download needs several gigabytes of free disk space. Do not run another model-heavy benchmark at the same time; its timings would share the machine.

## Recover after an indexing error

Reindexing replaces a physical Qdrant collection and changes the active alias. `SearchResources` caches its manifest and encoders, so use a maintenance window: stop the API, run indexing, then start the API again.

```bash
docker compose stop api
docker compose run --rm -e LOG_FILE=/tmp/index-run.log api \
  python -m hybrid_retrieval_lab.cli index
docker compose up -d api
```

The indexer validates the new generation before switching the alias. It then reads the alias back if the update call fails:

- If the alias still points to the previous generation, the candidate is discarded when possible and the original error is returned. The previous index remains active.
- If the alias points to the candidate, the switch succeeded despite the client-side error. The indexer treats it as published and continues cleanup.
- If the alias cannot be read or points somewhere unexpected, the indexer retains both generations and raises an explicit recovery error. Do not delete either collection automatically.

For an ambiguous result, keep the API stopped and inspect the alias and collections:

```bash
curl -fsS http://127.0.0.1:${QDRANT_HOST_PORT:-6333}/aliases
curl -fsS http://127.0.0.1:${QDRANT_HOST_PORT:-6333}/collections
docker compose logs --tail=100 qdrant
```

Match `github_docs_pilot_active` to the physical collection in the Qdrant alias response. Use the `alias`, `previous_collection`, `candidate_collection`, `active_collection` and `observed_collection` fields in the JSON logs to identify the two generations. If the alias points to the candidate, restart the API and run an evaluation; the evaluator revalidates the active index. If it points to the previous generation, verify Qdrant health before retrying the index command. If the target is still ambiguous, preserve both collections and the Qdrant volume until the alias can be read reliably. Only remove an inactive generation after the active target has been verified.

When indexing is not running, the previous successful generation is not modified. An index error or client timeout is not sufficient evidence that Qdrant rejected an alias update; always reconcile from Qdrant state before cleanup.
