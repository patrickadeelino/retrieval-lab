from __future__ import annotations

from types import SimpleNamespace

import pytest

from hybrid_retrieval_lab.services.search.models import SearchHit
from hybrid_retrieval_lab.services.search.strategy.colbert import rerank_colbert
from hybrid_retrieval_lab.services.search.strategy.hybrid import FusionEntry


def candidate(identifier: str, rank: int) -> FusionEntry:
    hit = SearchHit(identifier, rank, identifier, "source", "url", "title", "section", 0.1)
    return FusionEntry(hit, rank, None, 0.1, 0.0)


class StubEncoder:
    def query(self, query: str) -> list[list[float]]:
        return [[1.0]]


class StubClient:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.arguments = None

    def query_points(self, **kwargs: object) -> SimpleNamespace:
        self.arguments = kwargs
        return SimpleNamespace(
            points=[SimpleNamespace(id=identifier, score=score) for identifier, score in self.scores.items()]
        )


def test_colbert_reorders_only_fused_ids_with_stable_tie_break() -> None:
    from hybrid_retrieval_lab.ingestion.identity import point_id

    candidates = [candidate("b", 1), candidate("a", 2)]
    client = StubClient({point_id("a"): 0.9, point_id("b"): 0.9})
    ranked = rerank_colbert(client, StubEncoder(), "collection", "question", candidates)
    assert [hit.id for hit in ranked] == ["a", "b"]
    assert [hit.rank for hit in ranked] == [1, 2]
    assert client.arguments["limit"] == 2
    assert {str(identifier) for identifier in client.arguments["query_filter"].must[0].has_id} == set(client.scores)


def test_colbert_rejects_missing_candidate() -> None:
    from hybrid_retrieval_lab.ingestion.identity import point_id

    client = StubClient({point_id("a"): 0.9})
    with pytest.raises(RuntimeError, match="exactly the hybrid candidates"):
        rerank_colbert(client, StubEncoder(), "collection", "question", [candidate("a", 1), candidate("b", 2)])


def test_colbert_skips_encoder_and_client_for_empty_candidates() -> None:
    assert rerank_colbert(StubClient({}), StubEncoder(), "collection", "question", []) == []
