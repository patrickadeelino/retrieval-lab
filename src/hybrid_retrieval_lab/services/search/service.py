from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException

from hybrid_retrieval_lab.encoders.exceptions import TokenLimitError
from hybrid_retrieval_lab.services.search.exceptions import SearchUnavailableError, SearchValidationError
from hybrid_retrieval_lab.services.search.models import StrategyResult
from hybrid_retrieval_lab.services.search.resources import SearchResources
from hybrid_retrieval_lab.services.search.strategy.factory import SearchStrategyFactory

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self, factory: SearchStrategyFactory) -> None:
        self.factory = factory

    def search(self, query: str, strategy: str, limit: int = 5, inspect: bool = False) -> StrategyResult:
        started = time.perf_counter()
        if not query.strip():
            raise SearchValidationError("Query must not be empty")
        if not 1 <= limit <= 10:
            raise SearchValidationError("Limit must be between 1 and 10")
        selected = self.factory.create(strategy)
        logger.info(
            "search.started",
            extra={"fields": {"strategy": strategy, "limit": limit, "inspect": inspect, "query_chars": len(query)}},
        )
        try:
            result = selected.search(query, inspect)
        except TokenLimitError as exc:
            logger.error(
                "search.query_rejected",
                extra={
                    "fields": {
                        "strategy": strategy,
                        "token_count": exc.token_count,
                        "max_tokens": exc.max_tokens,
                        "model": exc.model,
                    }
                },
            )
            raise SearchValidationError(str(exc)) from exc
        except (ApiException, ResponseHandlingException, ConnectionError, TimeoutError, OSError) as exc:
            logger.exception("search.dependency_failed", extra={"fields": {"strategy": strategy}})
            raise SearchUnavailableError("Search dependency is unavailable") from exc
        response = StrategyResult(result.hits[:limit], result.score_type, result.inspection)
        logger.info(
            "search.completed",
            extra={
                "fields": {
                    "strategy": strategy,
                    "candidate_count": len(result.hits),
                    "result_count": len(response.hits),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            },
        )
        return response


def create_search_service(
    client: QdrantClient | None = None,
    collection: str | None = None,
    corpus_path: Path | None = None,
) -> SearchService:
    resources = SearchResources(
        client or QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333")),
        collection if collection is not None else os.environ.get("QDRANT_COLLECTION", "github_docs_pilot_active"),
        corpus_path or Path(os.getenv("CORPUS_PATH", "data/corpus/chunks.jsonl")),
    )
    return SearchService(SearchStrategyFactory(resources))
