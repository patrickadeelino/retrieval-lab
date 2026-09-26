from __future__ import annotations

from collections.abc import Callable

from hybrid_retrieval_lab.services.search.exceptions import UnknownStrategyError
from hybrid_retrieval_lab.services.search.resources import SearchResources
from hybrid_retrieval_lab.services.search.strategy.bm25 import BM25Strategy
from hybrid_retrieval_lab.services.search.strategy.colbert import HybridColBERTStrategy
from hybrid_retrieval_lab.services.search.strategy.dense import DenseStrategy
from hybrid_retrieval_lab.services.search.strategy.hybrid import HybridStrategy
from hybrid_retrieval_lab.services.search.strategy.protocol import SearchStrategy


class SearchStrategyFactory:
    def __init__(self, resources: SearchResources) -> None:
        self.resources = resources
        self._registry: dict[str, Callable[[SearchResources], SearchStrategy]] = {
            "bm25": BM25Strategy,
            "dense": DenseStrategy,
            "hybrid": HybridStrategy,
            "hybrid_colbert": HybridColBERTStrategy,
        }

    def create(self, name: str) -> SearchStrategy:
        try:
            return self._registry[name](self.resources)
        except KeyError as exc:
            raise UnknownStrategyError(f"Unknown search strategy: {name}") from exc
