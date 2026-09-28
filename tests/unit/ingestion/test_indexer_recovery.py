from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import models

from hybrid_retrieval_lab.encoders.colbert import DIMENSIONS as COLBERT_DIMENSIONS
from hybrid_retrieval_lab.encoders.e5 import DIMENSIONS as E5_DIMENSIONS
from hybrid_retrieval_lab.ingestion import indexer


class AliasRecoveryClient:
    def __init__(self, failure_mode: str) -> None:
        self.failure_mode = failure_mode
        self.aliases = {"active": "old_generation"}
        self.collections = {"old_generation"}
        self.readback_failed = False

    def get_aliases(self) -> SimpleNamespace:
        if self.readback_failed:
            raise OSError("Alias readback failed")
        return SimpleNamespace(
            aliases=[SimpleNamespace(alias_name=name, collection_name=target) for name, target in self.aliases.items()]
        )

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, *, collection_name: str, **_: object) -> None:
        self.collections.add(collection_name)

    def upsert(self, *, collection_name: str, points: list[models.PointStruct], wait: bool) -> None:
        assert wait
        self.points = points

    def count(self, *, collection_name: str, exact: bool) -> SimpleNamespace:
        assert exact
        return SimpleNamespace(count=len(self.points))

    def update_collection(self, collection_name: str, *, metadata: dict[str, object]) -> None:
        self.metadata = metadata

    def update_collection_aliases(
        self,
        *,
        change_aliases_operations: list[models.DeleteAliasOperation | models.CreateAliasOperation],
    ) -> None:
        if self.failure_mode in {"before_commit", "cleanup_fails"}:
            raise TimeoutError("Alias update timed out")
        create = change_aliases_operations[-1]
        assert isinstance(create, models.CreateAliasOperation)
        self.aliases[create.create_alias.alias_name] = create.create_alias.collection_name
        if self.failure_mode == "after_commit":
            raise TimeoutError("Alias update timed out")
        if self.failure_mode == "readback_failed":
            self.readback_failed = True
            raise TimeoutError("Alias update timed out")

    def delete_collection(self, collection_name: str) -> None:
        if self.failure_mode == "cleanup_fails" and collection_name != "old_generation":
            raise OSError("Generation cleanup failed")
        self.collections.remove(collection_name)


class DenseEncoder:
    revision = "e5-test"

    def validate_passage(self, _: str) -> None:
        return None

    def passages(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * E5_DIMENSIONS for _ in texts]


class SparseEncoder:
    revision = "bm25-test"

    def __init__(self, avg_len: float) -> None:
        self.avg_len = avg_len

    def documents(self, texts: list[str]) -> list[models.SparseVector]:
        return [models.SparseVector(indices=[1], values=[1.0]) for _ in texts]


class ColbertEncoder:
    revision = "colbert-test"

    def passages(self, texts: list[str]) -> list[list[list[float]]]:
        return [[[0.1] * COLBERT_DIMENSIONS] for _ in texts]


@pytest.fixture
def corpus_path(tmp_path: Path) -> Path:
    chunk = {
        "id": "chunk-1",
        "text": "A small source chunk.",
        "title": "Title",
        "section": "Section",
        "source_id": "source",
        "source_url": "https://example.com",
    }
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def stub_encoders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(indexer, "E5Encoder", DenseEncoder)
    monkeypatch.setattr(indexer, "BM25Encoder", SparseEncoder)
    monkeypatch.setattr(indexer, "ColBERTEncoder", ColbertEncoder)
    monkeypatch.setattr(indexer, "stored_index_hash", lambda *_: "index-hash")
    monkeypatch.setattr(indexer, "verify_index", lambda *_: SimpleNamespace(model_dump=lambda: {}))


def test_committed_alias_timeout_is_reconciled_without_deleting_active_generation(
    corpus_path: Path, stub_encoders: None
) -> None:
    client = AliasRecoveryClient("after_commit")

    indexed_count = indexer.index_corpus(client, "active", corpus_path)  # type: ignore[arg-type]

    active_generation = client.aliases["active"]
    assert indexed_count == 1
    assert active_generation in client.collections
    assert active_generation != "old_generation"


def test_uncommitted_alias_timeout_preserves_previous_generation(corpus_path: Path, stub_encoders: None) -> None:
    client = AliasRecoveryClient("before_commit")

    with pytest.raises(TimeoutError, match="Alias update timed out"):
        indexer.index_corpus(client, "active", corpus_path)  # type: ignore[arg-type]

    assert client.aliases["active"] == "old_generation"
    assert client.collections == {"old_generation"}


def test_cleanup_failure_does_not_hide_original_alias_error(corpus_path: Path, stub_encoders: None) -> None:
    client = AliasRecoveryClient("cleanup_fails")

    with pytest.raises(TimeoutError, match="Alias update timed out"):
        indexer.index_corpus(client, "active", corpus_path)  # type: ignore[arg-type]

    assert client.aliases["active"] == "old_generation"
    assert "old_generation" in client.collections
    assert len(client.collections) == 2


def test_unreadable_alias_state_retains_both_generations_for_manual_recovery(
    corpus_path: Path, stub_encoders: None
) -> None:
    client = AliasRecoveryClient("readback_failed")

    with pytest.raises(RuntimeError, match="publication outcome could not be reconciled"):
        indexer.index_corpus(client, "active", corpus_path)  # type: ignore[arg-type]

    assert client.aliases["active"] in client.collections
    assert "old_generation" in client.collections
    assert len(client.collections) == 2
