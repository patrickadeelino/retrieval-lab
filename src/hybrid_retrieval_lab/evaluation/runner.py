from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from qdrant_client import QdrantClient

from hybrid_retrieval_lab.encoders.colbert import MODEL_NAME as COLBERT_MODEL
from hybrid_retrieval_lab.encoders.e5 import MODEL_NAME as E5_MODEL
from hybrid_retrieval_lab.evaluation.metrics import ndcg_at, recall_at
from hybrid_retrieval_lab.evaluation.models import (
    EvaluatedStrategy,
    EvaluationInspection,
    JudgmentInput,
    PilotReport,
    QueryInput,
    QueryReport,
)
from hybrid_retrieval_lab.ingestion.identity import verify_index
from hybrid_retrieval_lab.ingestion.loader import load_chunks
from hybrid_retrieval_lab.services.search.config import CANDIDATES_PER_STRATEGY, FUSED_CANDIDATES, RRF_K
from hybrid_retrieval_lab.services.search.models import SearchHit
from hybrid_retrieval_lab.services.search.service import create_search_service

STRATEGIES = ("bm25", "dense", "hybrid", "hybrid_colbert")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value: object = json.loads(line)
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise ValueError(f"Expected an object with string keys on line {line_number} of {path}")
        records.append(cast(dict[str, object], value))
    return records


def _load_queries(path: Path) -> list[QueryInput]:
    queries = []
    for record in _read_jsonl(path):
        if not all(isinstance(record.get(field), str) for field in ("id", "query", "information_need")):
            raise ValueError("Each query must contain id, query, and information_need strings")
        queries.append(
            QueryInput(
                id=cast(str, record["id"]),
                query=cast(str, record["query"]),
                information_need=cast(str, record["information_need"]),
            )
        )
    return queries


def _load_judgments(path: Path) -> list[JudgmentInput]:
    judgments = []
    for record in _read_jsonl(path):
        if (
            not isinstance(record.get("query_id"), str)
            or not isinstance(record.get("chunk_id"), str)
            or type(record.get("grade")) is not int
        ):
            raise ValueError("Each judgment must contain query_id, chunk_id, and an integer grade")
        judgments.append(
            JudgmentInput(
                query_id=cast(str, record["query_id"]),
                chunk_id=cast(str, record["chunk_id"]),
                grade=cast(int, record["grade"]),
            )
        )
    return judgments


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    repeats: int = 3,
    reviewed_qrels: bool = False,
) -> PilotReport:
    if repeats < 1:
        raise ValueError("Repeats must be positive")
    chunks = load_chunks(corpus_path)
    queries = _load_queries(queries_path)
    qrel_rows = _load_judgments(qrels_path)
    chunk_ids = {chunk.id for chunk in chunks}
    query_ids = {item["id"] for item in queries}
    if len(query_ids) != len(queries):
        raise ValueError("Duplicate query IDs")
    grades: dict[str, dict[str, int]] = {query_id: {} for query_id in query_ids}
    for row in qrel_rows:
        query_id, chunk_id, grade = row["query_id"], row["chunk_id"], row["grade"]
        if (
            query_id not in grades
            or chunk_id not in chunk_ids
            or grade not in (0, 1, 2)
            or chunk_id in grades[query_id]
        ):
            raise ValueError("Invalid, unknown, or duplicate relevance judgment")
        grades[query_id][chunk_id] = grade
    if any(set(items) != chunk_ids for items in grades.values()):
        raise ValueError("Every query must have a judgment for every chunk")
    if any(not any(grade > 0 for grade in items.values()) for items in grades.values()):
        raise ValueError("Every query must have at least one relevant chunk")

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    collection = os.getenv("QDRANT_COLLECTION", "github_docs_pilot_active")
    manifest = verify_index(client, collection, corpus_path, chunks)
    count = len(manifest.indexed_ids)
    service = create_search_service(client=client, collection=collection, corpus_path=corpus_path)

    def search(strategy: str, query: str) -> tuple[list[SearchHit], EvaluationInspection]:
        result = service.search(query, strategy, limit=10, inspect=True)
        details = result.inspection or {}
        if strategy == "hybrid":
            return result.hits, {"bm25_ids": details["bm25"], "dense_ids": details["dense"]}
        if strategy == "hybrid_colbert":
            return result.hits, {"candidate_ids_before": details["colbert_candidates_before"]}
        return result.hits, {}

    reports: list[QueryReport] = []
    for item in queries:
        query_id, query = item["id"], item["query"]
        strategy_results: dict[str, EvaluatedStrategy] = {}
        for strategy in STRATEGIES:
            search(strategy, query)  # Warm model and local caches; not timed.
            runs_ms = []
            rankings = []
            inspection: EvaluationInspection = {}
            for _ in range(repeats):
                start = time.perf_counter()
                hits, inspection = search(strategy, query)
                runs_ms.append((time.perf_counter() - start) * 1000)
                rankings.append([hit.id for hit in hits])
            if any(ranking != rankings[0] for ranking in rankings[1:]):
                raise RuntimeError(f"Unstable ranking for {query_id}/{strategy}")
            ids = rankings[0]
            strategy_results[strategy] = {
                "ranking": [
                    {
                        "id": hit.id,
                        "rank": hit.rank,
                        "score": hit.score,
                        "grade": grades[query_id][hit.id],
                        "title": hit.title,
                        "text": hit.text,
                        "source_url": hit.source_url,
                    }
                    for hit in hits
                ],
                "metrics": {f"recall_at_{k}": recall_at(ids, grades[query_id], k) for k in (5, 10)}
                | {f"ndcg_at_{k}": ndcg_at(ids, grades[query_id], k) for k in (5, 10)},
                "latency_ms": {"runs": runs_ms, "median": statistics.median(runs_ms)},
                "inspection": inspection,
            }
        before = strategy_results["hybrid_colbert"]["inspection"]["candidate_ids_before"]
        after = [row["id"] for row in strategy_results["hybrid_colbert"]["ranking"]]
        if set(before) != set(after):
            raise RuntimeError(f"ColBERT changed the candidate set for {query_id}")
        relevant = {chunk_id for chunk_id, grade in grades[query_id].items() if grade >= 1}
        reports.append(
            {
                "id": query_id,
                "query": query,
                "information_need": item["information_need"],
                "relevant": [{"id": chunk_id, "grade": grades[query_id][chunk_id]} for chunk_id in sorted(relevant)],
                "coverage_before_colbert_at_10": len(relevant.intersection(before)) / len(relevant),
                "missing_before_colbert": sorted(relevant.difference(before)),
                "strategies": strategy_results,
            }
        )
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provisional_qrels": not reviewed_qrels,
        "inputs": {
            "corpus": str(corpus_path),
            "corpus_sha256": _sha256(corpus_path),
            "queries": str(queries_path),
            "queries_sha256": _sha256(queries_path),
            "qrels": str(qrels_path),
            "qrels_sha256": _sha256(qrels_path),
        },
        "config": {
            "collection": collection,
            "point_count": count,
            "source_chunk_count": len(chunks),
            "skipped_chunk_ids": [chunk.id for chunk in manifest.skipped_chunks],
            "index_manifest": manifest.model_dump(),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "qdrant": client.info().version,
            },
            "strategies": STRATEGIES,
            "candidates_per_strategy": CANDIDATES_PER_STRATEGY,
            "rrf_k": RRF_K,
            "fused_candidates": FUSED_CANDIDATES,
            "colbert_candidates": FUSED_CANDIDATES,
            "e5_model": E5_MODEL,
            "colbert_model": COLBERT_MODEL,
            "warmups_per_query_strategy": 1,
            "timed_runs_per_query_strategy": repeats,
            "latency_scope": "offline retrieval including query encoding and Qdrant, excluding HTTP",
        },
        "queries": reports,
    }
