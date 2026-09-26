from __future__ import annotations

from typing import NotRequired, TypedDict


class QueryInput(TypedDict):
    id: str
    query: str
    information_need: str


class JudgmentInput(TypedDict):
    query_id: str
    chunk_id: str
    grade: int


class RankedChunk(TypedDict):
    id: str
    rank: int
    score: float
    grade: int
    title: str
    text: str
    source_url: str


class RelevantChunk(TypedDict):
    id: str
    grade: int


class LatencyObservation(TypedDict):
    runs: list[float]
    median: float


class EvaluationInspection(TypedDict, total=False):
    bm25_ids: list[str]
    dense_ids: list[str]
    candidate_ids_before: list[str]


class EvaluatedStrategy(TypedDict):
    ranking: list[RankedChunk]
    metrics: dict[str, float]
    latency_ms: LatencyObservation
    inspection: EvaluationInspection


class QueryReport(TypedDict):
    id: str
    query: str
    information_need: str
    relevant: list[RelevantChunk]
    missing_before_colbert: list[str]
    strategies: dict[str, EvaluatedStrategy]
    coverage_before_colbert_at_10: NotRequired[float]
    coverage_before_colbert_at_20: NotRequired[float]


class InputHashes(TypedDict):
    corpus: str
    corpus_sha256: str
    queries: str
    queries_sha256: str
    qrels: str
    qrels_sha256: str


class EvaluationConfig(TypedDict):
    collection: str
    point_count: int
    source_chunk_count: NotRequired[int]
    skipped_chunk_ids: NotRequired[list[str]]
    index_manifest: NotRequired[dict[str, object]]
    runtime: NotRequired[dict[str, str]]
    strategies: tuple[str, ...]
    candidates_per_strategy: int
    rrf_k: int
    fused_candidates: int
    colbert_candidates: int
    e5_model: str
    colbert_model: str
    warmups_per_query_strategy: int
    timed_runs_per_query_strategy: int
    latency_scope: str


class PilotReport(TypedDict):
    created_at_utc: str
    provisional_qrels: bool
    inputs: InputHashes
    config: EvaluationConfig
    queries: list[QueryReport]
