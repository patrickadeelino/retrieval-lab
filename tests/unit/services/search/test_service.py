from __future__ import annotations

from types import SimpleNamespace

import pytest

from hybrid_retrieval_lab.encoders.exceptions import TokenLimitError
from hybrid_retrieval_lab.services.search.exceptions import SearchUnavailableError, SearchValidationError
from hybrid_retrieval_lab.services.search.models import SearchHit, StrategyResult
from hybrid_retrieval_lab.services.search.service import SearchService


def test_service_limits_hits_but_preserves_inspection() -> None:
    hits = [SearchHit(str(i), i, "text", "source", "url", "title", "section", 1.0) for i in range(1, 4)]
    calls = []

    class StubStrategy:
        def search(self, query: str, inspect: bool) -> StrategyResult:
            calls.append((query, inspect))
            return StrategyResult(hits, "rrf", {"fusion": ["1", "2", "3"]} if inspect else None)

    factory = SimpleNamespace(create=lambda name: StubStrategy())
    result = SearchService(factory).search(" question ", "hybrid", limit=2, inspect=True)
    assert [hit.id for hit in result.hits] == ["1", "2"]
    assert result.inspection == {"fusion": ["1", "2", "3"]}
    assert calls == [(" question ", True)]


@pytest.mark.parametrize("query,limit", [("  ", 5), ("question", 0), ("question", 11)])
def test_service_rejects_invalid_input_before_factory(query: str, limit: int) -> None:
    factory = SimpleNamespace(create=lambda name: pytest.fail("factory should not run"))
    with pytest.raises(SearchValidationError):
        SearchService(factory).search(query, "bm25", limit)


def test_service_maps_dependency_failures_but_does_not_hide_programming_errors() -> None:
    class FailingStrategy:
        def __init__(self, error: Exception) -> None:
            self.error = error

        def search(self, query: str, inspect: bool) -> StrategyResult:
            raise self.error

    factory = SimpleNamespace(create=lambda name: FailingStrategy(ConnectionError("down")))
    with pytest.raises(SearchUnavailableError, match="unavailable"):
        SearchService(factory).search("question", "bm25")
    factory = SimpleNamespace(create=lambda name: FailingStrategy(RuntimeError("bug")))
    with pytest.raises(RuntimeError, match="bug"):
        SearchService(factory).search("question", "bm25")


def test_service_maps_token_limit_to_validation_error() -> None:
    def search(query: str, inspect: bool) -> None:
        raise TokenLimitError("e5", "Query", 600, 512)

    factory = SimpleNamespace(create=lambda name: SimpleNamespace(search=search))
    with pytest.raises(SearchValidationError, match="600.*512-token"):
        SearchService(factory).search("question", "dense")
