from __future__ import annotations

from typing import Protocol

from hybrid_retrieval_lab.services.search.models import StrategyResult


class SearchStrategy(Protocol):
    def search(self, query: str, inspect: bool) -> StrategyResult: ...
