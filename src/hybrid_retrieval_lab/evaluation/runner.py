from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
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
    RankedChunk,
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
        if not isinstance(value, dict) or not _has_string_keys(value):
            raise ValueError(f"Expected an object with string keys on line {line_number} of {path}")
        records.append(cast(dict[str, object], value))
    return records


def _has_string_keys(value: dict[object, object]) -> bool:
    return all(isinstance(key, str) for key in value)


def _has_query_fields(record: dict[str, object]) -> bool:
    return all(isinstance(record.get(field), str) for field in ("id", "query", "information_need"))


def _load_queries(path: Path) -> list[QueryInput]:
    queries = []
    for record in _read_jsonl(path):
        if not _has_query_fields(record):
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


def _has_positive_grade(grades: dict[str, int]) -> bool:
    return any(grade > 0 for grade in grades.values())


def _has_query_without_relevance(grades_by_query: dict[str, dict[str, int]]) -> bool:
    return any(not _has_positive_grade(grades) for grades in grades_by_query.values())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command_output(command: list[str], project_root: Path) -> str | None:
    try:
        result = subprocess.run(command, cwd=project_root, capture_output=True, check=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _runtime_metadata(project_root: Path, qdrant_version: str) -> dict[str, str]:
    git_commit = os.getenv("GIT_COMMIT")
    git_tree_state = os.getenv("GIT_TREE_STATE")
    if git_commit is None:
        git_commit = _command_output(["git", "rev-parse", "HEAD"], project_root) or "unknown"
    if git_tree_state is None:
        status = _command_output(["git", "status", "--porcelain"], project_root)
        git_tree_state = "unknown" if status is None else "dirty" if status else "clean"
    uv_version = os.getenv("UV_VERSION") or _command_output(["uv", "--version"], project_root) or "unavailable"
    lockfile = project_root / "uv.lock"
    lock_hash = _sha256(lockfile) if lockfile.is_file() else "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": str(os.cpu_count() or "unknown"),
        "qdrant": qdrant_version,
        "uv": uv_version,
        "uv_lock_sha256": lock_hash,
        "git_commit": git_commit,
        "git_tree_state": git_tree_state,
    }


def _ranked_ids(hits: list[SearchHit]) -> list[str]:
    return [hit.id for hit in hits]


def _evaluate_strategy(
    search: Callable[[str, str], tuple[list[SearchHit], EvaluationInspection]],
    query_id: str,
    query: str,
    strategy: str,
    grades: dict[str, int],
    repeats: int,
) -> EvaluatedStrategy:
    search(strategy, query)  # Warm model and local caches; not timed.
    runs_ms: list[float] = []
    rankings: list[list[str]] = []
    hits_for_report: list[SearchHit] = []
    inspection: EvaluationInspection = {}
    for _ in range(repeats):
        start = time.perf_counter()
        hits_for_report, inspection = search(strategy, query)
        runs_ms.append((time.perf_counter() - start) * 1000)
        rankings.append(_ranked_ids(hits_for_report))
    if any(ranking != rankings[0] for ranking in rankings[1:]):
        raise RuntimeError(f"Unstable ranking for {query_id}/{strategy}")

    ids = rankings[0]
    ranking: list[RankedChunk] = [
        {
            "id": hit.id,
            "rank": hit.rank,
            "score": hit.score,
            "grade": grades[hit.id],
            "title": hit.title,
            "text": hit.text,
            "source_url": hit.source_url,
        }
        for hit in hits_for_report
    ]
    metrics = {f"recall_at_{k}": recall_at(ids, grades, k) for k in (5, 10)}
    metrics |= {f"ndcg_at_{k}": ndcg_at(ids, grades, k) for k in (5, 10)}
    return {
        "ranking": ranking,
        "metrics": metrics,
        "latency_ms": {"runs": runs_ms, "median": statistics.median(runs_ms)},
        "inspection": inspection,
    }


def _evaluate_query_strategies(
    search: Callable[[str, str], tuple[list[SearchHit], EvaluationInspection]],
    query_id: str,
    query: str,
    grades: dict[str, int],
    repeats: int,
) -> dict[str, EvaluatedStrategy]:
    results: dict[str, EvaluatedStrategy] = {}
    for strategy in STRATEGIES:
        results[strategy] = _evaluate_strategy(search, query_id, query, strategy, grades, repeats)
    return results


def _build_query_report(
    item: QueryInput, strategy_results: dict[str, EvaluatedStrategy], grades: dict[str, dict[str, int]]
) -> QueryReport:
    query_id = item["id"]
    before = strategy_results["hybrid_colbert"]["inspection"]["candidate_ids_before"]
    hybrid_ids = _ranked_row_ids(strategy_results["hybrid"]["ranking"])
    after = _ranked_row_ids(strategy_results["hybrid_colbert"]["ranking"])
    if before != hybrid_ids:
        raise RuntimeError(f"ColBERT candidate pool differs from hybrid ranking for {query_id}")
    if len(before) != len(set(before)) or set(before) != set(after):
        raise RuntimeError(f"ColBERT changed the candidate set for {query_id}")
    relevant = {chunk_id for chunk_id, grade in grades[query_id].items() if grade >= 1}
    return {
        "id": query_id,
        "query": item["query"],
        "information_need": item["information_need"],
        "relevant": [{"id": chunk_id, "grade": grades[query_id][chunk_id]} for chunk_id in sorted(relevant)],
        "coverage_before_colbert_at_10": len(relevant.intersection(before)) / len(relevant),
        "missing_before_colbert": sorted(relevant.difference(before)),
        "strategies": strategy_results,
    }


def _ranked_row_ids(rows: list[RankedChunk]) -> list[str]:
    return [row["id"] for row in rows]


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
    if _has_query_without_relevance(grades):
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
        strategy_results = _evaluate_query_strategies(search, query_id, query, grades[query_id], repeats)
        reports.append(_build_query_report(item, strategy_results, grades))
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
            "runtime": _runtime_metadata(Path(__file__).resolve().parents[3], client.info().version),
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
