from __future__ import annotations

import logging
import time
import uuid
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from hybrid_retrieval_lab.api.models import ChunkResult, SearchRequest, SearchResponse
from hybrid_retrieval_lab.logger import configure_logging, request_id_context
from hybrid_retrieval_lab.services.search.exceptions import (
    SearchUnavailableError,
    SearchValidationError,
    UnknownStrategyError,
)
from hybrid_retrieval_lab.services.search.service import (
    SearchService,
    create_search_service,
)

configure_logging()
logger = logging.getLogger(__name__)
app = FastAPI(title="Retrieval Lab", version="0.1.0")


@app.middleware("http")
async def log_http_request(request: Request, call_next: RequestResponseEndpoint) -> Response:
    request_id = uuid.uuid4().hex
    token = request_id_context.set(request_id)
    started = time.perf_counter()
    request.state.request_id = request_id
    fields = {"method": request.method, "path": request.url.path}
    logger.info("http.request.received", extra={"fields": fields})
    try:
        try:
            response = await call_next(request)
        except Exception as exc:
            request.state.error_type = type(exc).__name__
            logger.exception(
                "http.request.failed",
                extra={"fields": fields | {"duration_ms": round((time.perf_counter() - started) * 1000, 3)}},
            )
            response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
        response.headers["X-Request-ID"] = request_id
        completed = fields | {
            "status_code": response.status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        if hasattr(request.state, "result_count"):
            completed["result_count"] = request.state.result_count
        if hasattr(request.state, "error_type"):
            completed["error_type"] = request.state.error_type
        log_method = logger.error if response.status_code >= 400 else logger.info
        log_method("http.request.completed", extra={"fields": completed})
        return response
    finally:
        request_id_context.reset(token)


@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    return create_search_service()


@app.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    http_request: Request,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> SearchResponse:
    try:
        result = service.search(request.query, request.strategy, request.limit, request.inspect)
    except (UnknownStrategyError, SearchValidationError) as exc:
        http_request.state.error_type = type(exc).__name__
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SearchUnavailableError as exc:
        http_request.state.error_type = type(exc).__name__
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    http_request.state.result_count = len(result.hits)
    logger.info(
        "search.response.ready",
        extra={
            "fields": {
                "strategy": request.strategy,
                "score_type": result.score_type,
                "result_count": len(result.hits),
                "result_ids": [hit.id for hit in result.hits],
            }
        },
    )
    return SearchResponse(
        query=request.query,
        strategy=request.strategy,
        results=[ChunkResult(**hit.__dict__, score_type=result.score_type) for hit in result.hits],
        inspection=result.inspection,
    )
