from __future__ import annotations

from types import SimpleNamespace

import pytest

from hybrid_retrieval_lab.services.search.exceptions import UnknownStrategyError
from hybrid_retrieval_lab.services.search.strategy.bm25 import BM25Strategy
from hybrid_retrieval_lab.services.search.strategy.colbert import HybridColBERTStrategy
from hybrid_retrieval_lab.services.search.strategy.dense import DenseStrategy
from hybrid_retrieval_lab.services.search.strategy.factory import SearchStrategyFactory
from hybrid_retrieval_lab.services.search.strategy.hybrid import HybridStrategy


@pytest.mark.parametrize(
    "name,expected",
    [
        ("bm25", BM25Strategy),
        ("dense", DenseStrategy),
        ("hybrid", HybridStrategy),
        ("hybrid_colbert", HybridColBERTStrategy),
    ],
)
def test_factory_selects_each_strategy_without_loading_models(name: str, expected: type) -> None:
    resources = SimpleNamespace()
    strategy = SearchStrategyFactory(resources).create(name)
    assert isinstance(strategy, expected)
    assert strategy.resources is resources


def test_factory_rejects_unknown_name() -> None:
    with pytest.raises(UnknownStrategyError, match="unknown"):
        SearchStrategyFactory(SimpleNamespace()).create("unknown")
