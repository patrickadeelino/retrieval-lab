from __future__ import annotations

import logging
import time

from qdrant_client import QdrantClient

from hybrid_retrieval_lab.encoders.e5 import VECTOR_NAME, E5Encoder
from hybrid_retrieval_lab.services.search.config import CANDIDATES_PER_STRATEGY
from hybrid_retrieval_lab.services.search.models import SearchHit, StrategyResult, point_to_hit
from hybrid_retrieval_lab.services.search.resources import SearchResources

logger = logging.getLogger(__name__)


def search_dense(client: QdrantClient, encoder: E5Encoder, collection: str, query: str, limit: int) -> list[SearchHit]:
    started = time.perf_counter()
    result = client.query_points(
        collection_name=collection,
        query=encoder.query(query),
        using=VECTOR_NAME,
        limit=limit,
        with_payload=True,
    )
    hits = [point_to_hit(point, rank) for rank, point in enumerate(result.points, start=1)]
    logger.info(
        "search.dense.completed",
        extra={
            "fields": {
                "collection": collection,
                "requested_limit": limit,
                "candidate_count": len(hits),
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        },
    )
    return hits


class DenseStrategy:
    def __init__(self, resources: SearchResources) -> None:
        self.resources = resources

    def search(self, query: str, inspect: bool) -> StrategyResult:
        hits = search_dense(
            self.resources.client,
            self.resources.dense_encoder,
            self.resources.collection,
            query,
            CANDIDATES_PER_STRATEGY,
        )
        return StrategyResult(hits, "cosine", {"dense": [hit.id for hit in hits]} if inspect else None)
