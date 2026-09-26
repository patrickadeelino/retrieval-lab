from __future__ import annotations

import logging
import time

from qdrant_client import QdrantClient, models

from hybrid_retrieval_lab.encoders.colbert import VECTOR_NAME, ColBERTEncoder
from hybrid_retrieval_lab.ingestion.identity import point_id
from hybrid_retrieval_lab.services.search.models import SearchHit, StrategyResult
from hybrid_retrieval_lab.services.search.strategy.hybrid import FusionEntry, HybridStrategy

logger = logging.getLogger(__name__)


def rerank_colbert(
    client: QdrantClient,
    encoder: ColBERTEncoder,
    collection: str,
    query: str,
    candidates: list[FusionEntry],
) -> list[SearchHit]:
    if not candidates:
        return []
    started = time.perf_counter()
    by_point_id = {point_id(entry.hit.id): entry.hit for entry in candidates}
    response = client.query_points(
        collection_name=collection,
        query=encoder.query(query),
        using=VECTOR_NAME,
        query_filter=models.Filter(must=[models.HasIdCondition(has_id=list(by_point_id))]),
        limit=len(candidates),
        with_payload=False,
    )
    if {str(point.id) for point in response.points} != set(by_point_id):
        raise RuntimeError("ColBERT did not return exactly the hybrid candidates")
    scores = {str(point.id): point.score for point in response.points}
    ordered_ids = sorted(by_point_id, key=lambda identifier: (-scores[identifier], by_point_id[identifier].id))
    ranked = [
        SearchHit(**{**by_point_id[identifier].__dict__, "rank": rank, "score": scores[identifier]})
        for rank, identifier in enumerate(ordered_ids, start=1)
    ]
    logger.info(
        "search.colbert.completed",
        extra={
            "fields": {
                "collection": collection,
                "candidate_count": len(candidates),
                "result_count": len(ranked),
                "moved_count": sum(hit.id != entry.hit.id for hit, entry in zip(ranked, candidates, strict=True)),
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        },
    )
    return ranked


class HybridColBERTStrategy(HybridStrategy):
    def search(self, query: str, inspect: bool) -> StrategyResult:
        bm25_hits, dense_hits, fused = self._candidates(query)
        ranked = rerank_colbert(
            self.resources.client,
            self.resources.colbert_encoder,
            self.resources.collection,
            query,
            fused,
        )
        details = None
        if inspect:
            details = self._inspection(bm25_hits, dense_hits, fused)
            details["colbert_candidates_before"] = [entry.hit.id for entry in fused]
            details["colbert_after"] = [{"id": hit.id, "rank": hit.rank, "score": hit.score} for hit in ranked]
        return StrategyResult(ranked, "maxsim", details)
