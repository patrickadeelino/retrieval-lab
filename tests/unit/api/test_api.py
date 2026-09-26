from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hybrid_retrieval_lab.api.app import app, get_search_service
from hybrid_retrieval_lab.services.search.exceptions import (
    SearchUnavailableError,
    SearchValidationError,
    UnknownStrategyError,
)
from hybrid_retrieval_lab.services.search.models import SearchHit, StrategyResult


@pytest.fixture
def client() -> TestClient:
    def search(query: str, strategy: str, limit: int, inspect: bool) -> StrategyResult:
        hit = SearchHit("a", 1, "text", "source", "url", "title", "section", 0.5)
        return StrategyResult([hit], "bm25", {"bm25": ["a"]} if inspect else None)

    app.dependency_overrides[get_search_service] = lambda: SimpleNamespace(search=search)
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def test_search_serializes_contract_and_inspection(client: TestClient) -> None:
    response = client.post("/search", json={"query": "question", "strategy": "bm25", "inspect": True})
    assert response.status_code == 200
    assert response.json()["results"][0] == {
        "id": "a",
        "rank": 1,
        "text": "text",
        "source_id": "source",
        "source_url": "url",
        "title": "title",
        "section": "section",
        "score": 0.5,
        "score_type": "bm25",
    }
    assert response.json()["inspection"] == {"bm25": ["a"]}


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "question", "limit": 0},
        {"query": "question", "limit": 11},
        {"query": "question", "strategy": "invalid"},
    ],
)
def test_request_validation_returns_422(client: TestClient, payload: dict) -> None:
    assert client.post("/search", json=payload).status_code == 422


@pytest.mark.parametrize(
    "error,status",
    [
        (SearchValidationError("invalid query"), 422),
        (UnknownStrategyError("unknown strategy"), 422),
        (SearchUnavailableError("dependency unavailable"), 503),
    ],
)
def test_known_service_errors_are_mapped(client: TestClient, error: Exception, status: int) -> None:
    def fail(*args: object) -> None:
        raise error

    app.dependency_overrides[get_search_service] = lambda: SimpleNamespace(search=fail)
    response = client.post("/search", json={"query": "question"})
    assert response.status_code == status
    assert response.json()["detail"] == str(error)


def test_unexpected_service_error_is_500(client: TestClient) -> None:
    def fail(*args: object) -> None:
        raise RuntimeError("bug")

    app.dependency_overrides[get_search_service] = lambda: SimpleNamespace(search=fail)
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/search", json={"query": "question"}).status_code == 500
