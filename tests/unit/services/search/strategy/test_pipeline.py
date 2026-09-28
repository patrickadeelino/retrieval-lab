from __future__ import annotations

from types import SimpleNamespace

from hybrid_retrieval_lab.ingestion.identity import point_id
from hybrid_retrieval_lab.services.search.service import SearchService
from hybrid_retrieval_lab.services.search.strategy.factory import SearchStrategyFactory


class RecordingQdrant:
    def __init__(self) -> None:
        self.queries: list[dict[str, object]] = []
        self.bm25_ids = [f"chunk-{index:02d}" for index in range(10)]
        self.dense_ids = [f"chunk-{index:02d}" for index in range(5, 15)]

    @staticmethod
    def _point(chunk_id: str, score: float) -> SimpleNamespace:
        return SimpleNamespace(
            id=point_id(chunk_id),
            score=score,
            payload={
                "chunk_id": chunk_id,
                "text": f"Text for {chunk_id}",
                "source_id": "github-docs",
                "source_url": "https://docs.github.com/",
                "title": "GitHub Docs",
                "section": "Search",
            },
        )

    def query_points(self, **kwargs: object) -> SimpleNamespace:
        self.queries.append(kwargs)
        using = kwargs["using"]
        if using == "bm25":
            ids = self.bm25_ids
            return SimpleNamespace(points=[self._point(chunk_id, 1.0 / rank) for rank, chunk_id in enumerate(ids, 1)])
        if using == "dense":
            ids = self.dense_ids
            return SimpleNamespace(points=[self._point(chunk_id, 1.0 / rank) for rank, chunk_id in enumerate(ids, 1)])

        condition = kwargs["query_filter"].must[0]
        candidate_ids = [str(identifier) for identifier in condition.has_id]
        return SimpleNamespace(
            points=[
                SimpleNamespace(id=identifier, score=float(rank))
                for rank, identifier in enumerate(reversed(candidate_ids), start=1)
            ]
        )


def test_hybrid_colbert_search_retrieves_fuses_reranks_same_ten_then_applies_limit() -> None:
    client = RecordingQdrant()
    resources = SimpleNamespace(
        client=client,
        collection="pilot",
        bm25_encoder=SimpleNamespace(query=lambda query: {"sparse": query}),
        dense_encoder=SimpleNamespace(query=lambda query: [0.1, 0.2]),
        colbert_encoder=SimpleNamespace(query=lambda query: [[0.1, 0.2]]),
    )
    service = SearchService(SearchStrategyFactory(resources))

    result = service.search("How do I find the relevant chunks?", "hybrid_colbert", limit=5, inspect=True)

    lexical_queries = [query for query in client.queries if query["using"] == "bm25"]
    dense_queries = [query for query in client.queries if query["using"] == "dense"]
    colbert_queries = [query for query in client.queries if query["using"] == "colbert"]
    assert len(lexical_queries) == len(dense_queries) == len(colbert_queries) == 1
    assert lexical_queries[0]["limit"] == dense_queries[0]["limit"] == 10
    assert colbert_queries[0]["limit"] == 10

    assert result.score_type == "maxsim"
    assert len(result.hits) == 5  # The API response limit applies after reranking.
    assert result.inspection is not None
    before = result.inspection["colbert_candidates_before"]
    after = [item["id"] for item in result.inspection["colbert_after"]]
    assert len(before) == len(set(before)) == 10
    assert set(before) == set(after)
    assert [item["id"] for item in result.inspection["fusion"]] == before
