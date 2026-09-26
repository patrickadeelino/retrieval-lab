from pydantic import BaseModel, Field

from hybrid_retrieval_lab.services.search.models import ScoreType, SearchInspection, StrategyName


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    strategy: StrategyName = "bm25"
    limit: int = Field(default=5, ge=1, le=10)
    inspect: bool = False


class ChunkResult(BaseModel):
    id: str
    rank: int
    text: str
    source_id: str
    source_url: str
    title: str
    section: str
    score: float
    score_type: ScoreType


class SearchResponse(BaseModel):
    query: str
    strategy: StrategyName
    results: list[ChunkResult]
    inspection: SearchInspection | None = None
