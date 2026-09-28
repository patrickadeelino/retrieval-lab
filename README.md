# Retrieval Lab

**Four retrieval strategies. One frozen Portuguese corpus. A reproducible evaluation.**

BM25, dense embeddings, RRF fusion and ColBERT reranking, compared through one Python API. The service returns ranked chunks from six GitHub Docs pages; it does not generate answers.

[![Unit test coverage](https://img.shields.io/badge/unit%20coverage-79.45%25-brightgreen)](docs/development.md#coverage)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

This is the implementation companion to the [Fundamentos de Information Retrieval blog series](https://www.patrickadelino.com.br/series/information-retrieval/) (Portuguese). The articles explain the concepts; this repository brings them together and measures their behavior on a judged corpus. See the [article-to-code map](#companion-to-the-blog-series).

[Results](#results) · [How it works](#how-it-works) · [Quickstart](#quickstart) · [Method and limitations](#method-and-limitations) · [Documentation](docs/README.md)

## Results

The pilot uses the same 53 chunks, 10 queries and 530 relevance judgments for every strategy. The API selects a strategy with one request parameter.

| Strategy | Pipeline | Mean nDCG@5 | Mean Recall@10 | Typical latency |
| --- | --- | ---: | ---: | ---: |
| `bm25` | BM25 | 0.777 | 91.0% | 1.96 ms |
| `dense` | E5 | 0.713 | 84.7% | 15.45 ms |
| `hybrid` | BM25 + E5 → RRF | 0.797 | 94.3% | 15.94 ms |
| `hybrid_colbert` | BM25 + E5 → RRF → ColBERT | **0.868** | 94.3% | 171.93 ms |

- **Fusion increased Recall@10 in this pilot.** Hybrid retrieved 94.3% of the judged relevant chunks, compared with 91.0% for BM25 and 84.7% for dense retrieval.
- **ColBERT changed ordering, not candidate coverage.** It reranks the same ten hybrid candidates, so Recall@10 is equal by construction. Mean nDCG@5 rose from 0.797 to 0.868, with about **10.8×** the typical latency.
- **Results varied by query.** Dense scored 0.101 on q10, where BM25 scored 0.899. ColBERT raised q09 (an exact error message) from 0.562 to 0.847, and lowered q01 from 0.695 to 0.647.

With ten queries and one annotator, these results are diagnostic rather than a benchmark; differences are suggestive, not conclusive. Typical latency is the median of per-query medians across three timed runs after warmup. It includes query encoding and Qdrant, but excludes HTTP.

See the [HTML report](reports/baseline/index.html), [source JSON](reports/baseline/pilot.json), and [evaluation method](docs/evaluation.md).

## How it works

The four API strategies are stopping points along one retrieval pipeline. BM25 and dense return their own rankings; hybrid fuses both candidate lists; hybrid_colbert adds a reranking step.

```mermaid
flowchart LR
    Q([Query]) --> BM25[BM25<br/>lexical · top 10]
    Q --> E5[E5<br/>semantic · top 10]
    BM25 --> RRF[RRF fusion<br/>k = 60 · top 10]
    E5 --> RRF
    RRF --> COL[ColBERT<br/>rerank the same 10]
```

`bm25` returns after lexical retrieval, `dense` after E5, `hybrid` after RRF, and `hybrid_colbert` after ColBERT.

- **One service, two callers.** `POST /search` and the offline evaluator share `SearchService`, so the report exercises the retrieval service used by the API.
- **Strategy protocol and factory.** Each strategy implements the same protocol and the factory selects it from the request value.
- **Three representations in each indexed Qdrant collection.** Chunks have a sparse BM25 vector, a 384-dimensional E5 vector and a ColBERT multivector with 128 dimensions per token.
- **ColBERT reuses the hybrid candidates.** Qdrant applies MaxSim to the fused candidate IDs; reranking is checked against the original candidate set.
- **Verified index publication.** Indexing builds and verifies a versioned collection against a manifest before atomically switching the active alias.

See the [architecture and HTTP contract](docs/architecture.md).

## Quickstart

Requires Docker with Compose and internet access for the initial model downloads. The ColBERT model is about 2.2 GB.

```bash
docker compose build
docker compose up -d qdrant
docker compose run --rm -e LOG_FILE=/tmp/index-run.log api python -m hybrid_retrieval_lab.cli index
docker compose up -d api
```

Query the API:

```bash
curl -sS http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Para que serve X-GitHub-Delivery?","strategy":"hybrid_colbert","limit":5,"inspect":true}'
```

The query is in Portuguese to match the corpus. Set `inspect` to `true` to see the intermediate BM25 and dense rankings and each chunk's RRF contribution. Open the interactive [API docs](http://127.0.0.1:8000/docs).

Before reindexing, stop the API with `docker compose stop api`; see [reproduction and recovery](docs/reproduction-and-recovery.md). To develop locally, run `uv sync --locked --extra dev`, then follow the [development guide](docs/development.md) for lint, strict typing and unit/integration tests.

### Reproduce the report

The saved baseline can be viewed without running the models:

```bash
python -m http.server 8770 --bind 127.0.0.1 --directory reports/baseline
```

For a clean-clone reproduction, follow the [reproduction procedure](docs/reproduction-and-recovery.md). It records the Git revision, worktree state, lockfile hash, model snapshots and input hashes. Avoid running other model workloads during timing.

## Method and limitations

### What supports this comparison

- **Frozen, hashed corpus.** Six source pages produce 53 chunks; the manifest records SHA-256 hashes.
- **Exhaustive judgments.** All 530 query/chunk pairs have an explicit grade from 0 to 2, so the evaluation has no unjudged chunks within this corpus.
- **Verified index identity.** Evaluation stops if the corpus, encoder snapshots or stored vectors differ from the manifest.
- **Clean-clone reproduction.** All 40 query/strategy rankings matched the previous validated report. The report records the commit, `uv.lock` hash and model snapshots; absolute latency varied between runs.

### What it cannot establish

- The queries were written by the author after reviewing the corpus, and all grades were assigned by one annotator. They are not production traffic or an independent benchmark. See [judgment provenance](docs/evaluation.md#judgment-provenance).
- The corpus has only 53 chunks from one domain.
- Model size and retrieval technique are confounded: `jina-colbert-v2` is much larger than `multilingual-e5-small`, so the observed gain cannot be attributed to late interaction alone.
- ColBERT reranks only ten candidates; this experiment does not separate reranking quality from candidate generation over the full corpus.
- Latency comes from one local machine and excludes HTTP.
- `jinaai/jina-colbert-v2` is licensed CC BY-NC 4.0, which restricts commercial use.

The [HTML report](reports/baseline/index.html) contains per-query rankings, candidate coverage, latency samples and configuration details. See [evaluation methodology and limitations](docs/evaluation.md).

## Companion to the blog series

The series builds from retrieval fundamentals to hybrid search. These articles connect those concepts to their implementation in this repository.

| Article (PT) | Where it appears in this repository |
| --- | --- |
| [O que é Information Retrieval?](https://www.patrickadelino.com.br/series/information-retrieval/o-que-sao-information-retrieval/introducao/) | Overall flow: query → candidates → ranking |
| [Busca Baseada em Termos: TF-IDF, índice invertido e BM25](https://www.patrickadelino.com.br/series/information-retrieval/term-based-retrieval/introducao/) | [`encoders/bm25.py`](src/hybrid_retrieval_lab/encoders/bm25.py), [`strategy/bm25.py`](src/hybrid_retrieval_lab/services/search/strategy/bm25.py) |
| [Busca Semântica: embeddings, vetores e similaridade de cosseno](https://www.patrickadelino.com.br/series/information-retrieval/embeddings-busca-semantica/introducao/) | [`encoders/e5.py`](src/hybrid_retrieval_lab/encoders/e5.py), [`strategy/dense.py`](src/hybrid_retrieval_lab/services/search/strategy/dense.py) |
| [Chunking: Tamanho fixo, estrutural e semântico](https://www.patrickadelino.com.br/series/information-retrieval/chunking/introducao/) | [`ingestion/chunker.py`](src/hybrid_retrieval_lab/ingestion/chunker.py) (section-aware, capped at 1,600 characters) |
| [Vector Databases: armazenamento, indexação e busca vetorial com Qdrant](https://www.patrickadelino.com.br/series/information-retrieval/vector-databases/por-que-usar-um-vector-database/) | [`ingestion/indexer.py`](src/hybrid_retrieval_lab/ingestion/indexer.py), [`ingestion/identity.py`](src/hybrid_retrieval_lab/ingestion/identity.py) |
| [Hybrid Search: BM25 e embeddings, RRF e reranking](https://www.patrickadelino.com.br/series/information-retrieval/hybrid-search-reranking/quando-a-pergunta-pede-mais-de-um-sinal/) | [`strategy/hybrid.py`](src/hybrid_retrieval_lab/services/search/strategy/hybrid.py), [`strategy/colbert.py`](src/hybrid_retrieval_lab/services/search/strategy/colbert.py) |

More writing at [patrickadelino.com.br](https://www.patrickadelino.com.br/).

## Repository map

```text
src/hybrid_retrieval_lab/
  api/          HTTP adapter (FastAPI)
  encoders/     BM25, E5 and ColBERT adapters, token budget
  ingestion/    chunking, indexing, index manifest and verification
  services/     SearchService, strategy protocol, factory and retrieval
  evaluation/   metrics, runner and HTML report
data/           corpus, queries and relevance judgments
reports/        canonical baseline (HTML and JSON)
tests/          unit (doubles) and integration (Qdrant and encoders)
docs/           architecture, evaluation, observability and reproduction
```

## License

The original code is licensed under the [MIT License](LICENSE). The license does not apply to third-party models or the GitHub Docs corpus (CC BY 4.0); see [third-party notices](docs/third-party-notices.md).
