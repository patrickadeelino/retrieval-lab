from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from hybrid_retrieval_lab.api.app import app, get_search_service
from hybrid_retrieval_lab.logger import configure_logging
from hybrid_retrieval_lab.services.search.exceptions import SearchUnavailableError
from hybrid_retrieval_lab.services.search.models import SearchHit, StrategyResult


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_http_flow_is_correlated_without_logging_query_text(tmp_path: Path) -> None:
    path = tmp_path / "requests.log"
    configure_logging(path)

    def search(query: str, strategy: str, limit: int, inspect: bool) -> StrategyResult:
        hit = SearchHit("chunk-1", 1, "text", "source", "url", "title", "section", 0.5)
        return StrategyResult([hit], "bm25")

    app.dependency_overrides[get_search_service] = lambda: SimpleNamespace(search=search)
    try:
        with TestClient(app) as client:
            response = client.post("/search", json={"query": "PRIVATE_QUERY_TEXT", "strategy": "bm25"})
    finally:
        app.dependency_overrides.clear()
        configure_logging()
    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    recorded = events(path)
    assert [event["event"] for event in recorded] == [
        "http.request.received",
        "search.response.ready",
        "http.request.completed",
    ]
    assert {event["request_id"] for event in recorded} == {response.headers["X-Request-ID"]}
    assert recorded[1]["result_ids"] == ["chunk-1"]
    assert recorded[2]["result_count"] == 1
    assert "PRIVATE_QUERY_TEXT" not in path.read_text(encoding="utf-8")


def test_http_errors_use_error_level_and_type(tmp_path: Path) -> None:
    path = tmp_path / "errors.log"
    configure_logging(path)

    def fail(*args: object) -> None:
        raise SearchUnavailableError("Search dependency is unavailable")

    app.dependency_overrides[get_search_service] = lambda: SimpleNamespace(search=fail)
    try:
        with TestClient(app) as client:
            response = client.post("/search", json={"query": "question"})
    finally:
        app.dependency_overrides.clear()
        configure_logging()
    assert response.status_code == 503
    completed = events(path)[-1]
    assert completed["event"] == "http.request.completed"
    assert completed["level"] == "ERROR"
    assert completed["error_type"] == "SearchUnavailableError"


def test_unexpected_error_returns_correlated_500_and_exception_log(tmp_path: Path) -> None:
    path = tmp_path / "unexpected.log"
    configure_logging(path)

    def fail(*args: object) -> None:
        raise RuntimeError("unexpected bug")

    app.dependency_overrides[get_search_service] = lambda: SimpleNamespace(search=fail)
    try:
        with TestClient(app) as client:
            response = client.post("/search", json={"query": "question"})
    finally:
        app.dependency_overrides.clear()
        configure_logging()
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    recorded = events(path)
    assert [event["event"] for event in recorded] == [
        "http.request.received",
        "http.request.failed",
        "http.request.completed",
    ]
    assert {event["request_id"] for event in recorded} == {response.headers["X-Request-ID"]}
    assert recorded[-1]["error_type"] == "RuntimeError"
