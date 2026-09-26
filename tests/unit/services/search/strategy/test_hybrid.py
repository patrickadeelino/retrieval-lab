from __future__ import annotations

import pytest

from hybrid_retrieval_lab.services.search.models import SearchHit
from hybrid_retrieval_lab.services.search.strategy.hybrid import fuse_rrf


def hit(identifier: str, rank: int) -> SearchHit:
    return SearchHit(identifier, rank, identifier, "source", "https://example.com", "Title", "Section", 1.0)


def test_rrf_adds_contributions_and_records_source_ranks() -> None:
    entries = fuse_rrf([hit("a", 1), hit("b", 2)], [hit("b", 1), hit("c", 2)], k=60)
    assert [entry.hit.id for entry in entries] == ["b", "a", "c"]
    assert [entry.hit.rank for entry in entries] == [1, 2, 3]
    assert entries[0].bm25_rank == 2
    assert entries[0].dense_rank == 1
    assert entries[0].bm25_contribution == pytest.approx(1 / 62)
    assert entries[0].dense_contribution == pytest.approx(1 / 61)
    assert entries[0].hit.score == pytest.approx(1 / 62 + 1 / 61)
    assert entries[1].dense_contribution == 0


def test_rrf_breaks_equal_scores_by_chunk_id() -> None:
    assert [entry.hit.id for entry in fuse_rrf([hit("z", 1)], [hit("a", 1)])] == ["a", "z"]


def test_rrf_accepts_empty_lists_and_rejects_nonpositive_k() -> None:
    assert fuse_rrf([], []) == []
    assert [entry.hit.id for entry in fuse_rrf([hit("a", 1)], [])] == ["a"]
    with pytest.raises(ValueError, match="positive"):
        fuse_rrf([], [], k=0)
    with pytest.raises(ValueError, match="positive"):
        fuse_rrf([], [], k=-1)
