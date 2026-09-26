from __future__ import annotations

import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from hybrid_retrieval_lab.services.search.service import create_search_service


def test_real_qdrant_search_keeps_colbert_within_rrf_top_10(
    indexed_collection: tuple[QdrantClient, str],
) -> None:
    client, collection = indexed_collection
    service = create_search_service(client, collection)
    query = json.loads(Path("data/queries/pilot-queries.jsonl").read_text(encoding="utf-8").splitlines()[0])["query"]
    results = {
        name: service.search(query, name, limit=10, inspect=True)
        for name in ("bm25", "dense", "hybrid", "hybrid_colbert")
    }
    assert all(len(result.hits) == 10 for result in results.values())
    assert [hit.rank for hit in results["bm25"].hits] == list(range(1, 11))
    assert [hit.rank for hit in results["dense"].hits] == list(range(1, 11))
    hybrid_ids = [hit.id for hit in results["hybrid"].hits]
    colbert_ids = [hit.id for hit in results["hybrid_colbert"].hits]
    assert len(set(hybrid_ids)) == len(set(colbert_ids)) == 10
    assert set(hybrid_ids) == set(colbert_ids)
    assert results["hybrid"].inspection["candidates_per_strategy"] == 10
    assert results["hybrid"].inspection["fused_candidates"] == 10
    assert results["hybrid_colbert"].inspection["colbert_candidates_before"] == hybrid_ids
    assert [item["id"] for item in results["hybrid_colbert"].inspection["colbert_after"]] == colbert_ids


def test_query_over_model_budget_is_rejected(indexed_collection: tuple[QdrantClient, str]) -> None:
    from hybrid_retrieval_lab.services.search.exceptions import SearchValidationError

    client, collection = indexed_collection
    with pytest.raises(SearchValidationError, match="512-token limit"):
        create_search_service(client, collection).search("webhook " * 200, "dense")
