from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hybrid_retrieval_lab.evaluation import runner
from hybrid_retrieval_lab.services.search.models import SearchHit, StrategyResult


@pytest.fixture
def evaluation_case(tmp_path, monkeypatch):
    rows = [
        {
            "id": key,
            "text": "Chunk " + key,
            "title": "Title",
            "section": "Section",
            "source_id": "source",
            "source_url": "https://example.com",
        }
        for key in ("a", "b", "c")
    ]
    corpus, queries, qrels = [tmp_path / f"{name}.jsonl" for name in ("corpus", "queries", "qrels")]
    for path, records in (
        (corpus, rows),
        (queries, [{"id": "q01", "query": "question", "information_need": "Find the answer"}]),
        (
            qrels,
            [{"query_id": "q01", "chunk_id": key, "grade": grade} for key, grade in (("a", 2), ("b", 1), ("c", 0))],
        ),
    ):
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
    calls = []
    rankings = {
        "bm25": ["a", "b", "c"],
        "dense": ["b", "a", "c"],
        "hybrid": ["c", "b", "a"],
        "hybrid_colbert": ["a", "b", "c"],
    }

    def search(query, strategy, limit, inspect):
        calls.append(strategy)
        hits = [
            SearchHit(key, rank, "Chunk " + key, "source", "https://example.com", "Title", "Section", 0.5)
            for rank, key in enumerate(rankings[strategy], 1)
        ]
        return StrategyResult(
            hits,
            "bm25",
            {"bm25": rankings["bm25"], "dense": rankings["dense"], "colbert_candidates_before": rankings["hybrid"]},
        )

    client = SimpleNamespace(info=lambda: SimpleNamespace(version="test"))
    monkeypatch.setattr(runner, "QdrantClient", lambda **kwargs: client)
    monkeypatch.setattr(
        runner,
        "verify_index",
        lambda *args: SimpleNamespace(indexed_ids=["a", "b", "c"], skipped_chunks=[], model_dump=lambda: {}),
    )
    monkeypatch.setattr(runner, "create_search_service", lambda **kwargs: SimpleNamespace(search=search))
    return SimpleNamespace(paths=(corpus, queries, qrels), qrels=qrels, calls=calls, rankings=rankings)
