from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qdrant_client.models import ScoredPoint
from typing_extensions import TypedDict

ScoreType = Literal["bm25", "cosine", "rrf", "maxsim"]
StrategyName = Literal["bm25", "dense", "hybrid", "hybrid_colbert"]


class FusionInspection(TypedDict):
    id: str
    rank: int
    bm25_rank: int | None
    dense_rank: int | None
    bm25_contribution: float
    dense_contribution: float
    rrf_score: float


class ColbertInspectionHit(TypedDict):
    id: str
    rank: int
    score: float


class SearchInspection(TypedDict, total=False):
    bm25: list[str]
    dense: list[str]
    rrf_k: int
    candidates_per_strategy: int
    fused_candidates: int
    fusion: list[FusionInspection]
    colbert_candidates_before: list[str]
    colbert_after: list[ColbertInspectionHit]


@dataclass(frozen=True)
class SearchHit:
    id: str
    rank: int
    text: str
    source_id: str
    source_url: str
    title: str
    section: str
    score: float


@dataclass(frozen=True)
class StrategyResult:
    hits: list[SearchHit]
    score_type: ScoreType
    inspection: SearchInspection | None = None


def point_to_hit(point: ScoredPoint, rank: int) -> SearchHit:
    payload = point.payload or {}

    def required_text(field: str) -> str:
        value = payload.get(field)
        if not isinstance(value, str):
            raise ValueError(f"Search result is missing a string payload field: {field}")
        return value

    return SearchHit(
        id=required_text("chunk_id"),
        rank=rank,
        text=required_text("text"),
        source_id=required_text("source_id"),
        source_url=required_text("source_url"),
        title=required_text("title"),
        section=required_text("section"),
        score=point.score,
    )
