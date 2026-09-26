from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from hybrid_retrieval_lab.services.search.config import CANDIDATES_PER_STRATEGY, FUSED_CANDIDATES, RRF_K
from hybrid_retrieval_lab.services.search.models import FusionInspection, SearchHit, SearchInspection, StrategyResult
from hybrid_retrieval_lab.services.search.resources import SearchResources
from hybrid_retrieval_lab.services.search.strategy.bm25 import search_bm25
from hybrid_retrieval_lab.services.search.strategy.dense import search_dense

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusionEntry:
    hit: SearchHit
    bm25_rank: int | None
    dense_rank: int | None
    bm25_contribution: float
    dense_contribution: float

    def inspection(self) -> FusionInspection:
        return {
            "id": self.hit.id,
            "rank": self.hit.rank,
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
            "bm25_contribution": self.bm25_contribution,
            "dense_contribution": self.dense_contribution,
            "rrf_score": self.hit.score,
        }


def fuse_rrf(bm25_hits: list[SearchHit], dense_hits: list[SearchHit], k: int = RRF_K) -> list[FusionEntry]:
    if k <= 0:
        raise ValueError("RRF k must be positive")
    bm25_by_id = {hit.id: hit for hit in bm25_hits}
    dense_by_id = {hit.id: hit for hit in dense_hits}
    entries: list[FusionEntry] = []
    for chunk_id in bm25_by_id.keys() | dense_by_id.keys():
        bm25 = bm25_by_id.get(chunk_id)
        dense = dense_by_id.get(chunk_id)
        bm25_contribution = 1 / (k + bm25.rank) if bm25 else 0.0
        dense_contribution = 1 / (k + dense.rank) if dense else 0.0
        source = bm25 or dense
        assert source is not None
        hit = SearchHit(
            id=chunk_id,
            rank=0,
            text=source.text,
            source_id=source.source_id,
            source_url=source.source_url,
            title=source.title,
            section=source.section,
            score=bm25_contribution + dense_contribution,
        )
        entries.append(
            FusionEntry(
                hit, bm25.rank if bm25 else None, dense.rank if dense else None, bm25_contribution, dense_contribution
            )
        )
    entries.sort(key=lambda entry: (-entry.hit.score, entry.hit.id))
    return [
        FusionEntry(
            SearchHit(**{**entry.hit.__dict__, "rank": rank}),
            entry.bm25_rank,
            entry.dense_rank,
            entry.bm25_contribution,
            entry.dense_contribution,
        )
        for rank, entry in enumerate(entries, start=1)
    ]


class HybridStrategy:
    def __init__(self, resources: SearchResources) -> None:
        self.resources = resources

    def _candidates(self, query: str) -> tuple[list[SearchHit], list[SearchHit], list[FusionEntry]]:
        bm25_hits = search_bm25(
            self.resources.client,
            self.resources.bm25_encoder,
            self.resources.collection,
            query,
            CANDIDATES_PER_STRATEGY,
        )
        dense_hits = search_dense(
            self.resources.client,
            self.resources.dense_encoder,
            self.resources.collection,
            query,
            CANDIDATES_PER_STRATEGY,
        )
        started = time.perf_counter()
        fused = fuse_rrf(bm25_hits, dense_hits)[:FUSED_CANDIDATES]
        logger.info(
            "search.rrf.completed",
            extra={
                "fields": {
                    "bm25_count": len(bm25_hits),
                    "dense_count": len(dense_hits),
                    "fused_count": len(fused),
                    "rrf_k": RRF_K,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            },
        )
        return bm25_hits, dense_hits, fused

    @staticmethod
    def _inspection(
        bm25_hits: list[SearchHit], dense_hits: list[SearchHit], fused: list[FusionEntry]
    ) -> SearchInspection:
        return {
            "rrf_k": RRF_K,
            "candidates_per_strategy": CANDIDATES_PER_STRATEGY,
            "fused_candidates": FUSED_CANDIDATES,
            "bm25": [hit.id for hit in bm25_hits],
            "dense": [hit.id for hit in dense_hits],
            "fusion": [entry.inspection() for entry in fused],
        }

    def search(self, query: str, inspect: bool) -> StrategyResult:
        bm25_hits, dense_hits, fused = self._candidates(query)
        details = self._inspection(bm25_hits, dense_hits, fused) if inspect else None
        return StrategyResult([entry.hit for entry in fused], "rrf", details)
