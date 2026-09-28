from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models

from hybrid_retrieval_lab.encoders.colbert import DIMENSIONS as COLBERT_DIMENSIONS
from hybrid_retrieval_lab.encoders.e5 import DIMENSIONS as E5_DIMENSIONS
from hybrid_retrieval_lab.encoders.e5 import MAX_TOKENS
from hybrid_retrieval_lab.encoders.e5 import MODEL_NAME as E5_MODEL
from hybrid_retrieval_lab.encoders.exceptions import TokenLimitError
from hybrid_retrieval_lab.ingestion.identity import METADATA_KEY, point_id
from hybrid_retrieval_lab.ingestion.indexer import index_corpus


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def delete_collection(self, collection_name: str) -> None:
        self.deleted.append(collection_name)
        del self.collections[collection_name]

    def create_collection(self, *, collection_name: str, **_: object) -> None:
        self.collections[collection_name] = {"points": [], "metadata": None}

    def upsert(self, *, collection_name: str, points: list[models.PointStruct], wait: bool) -> None:
        assert wait
        self.collections[collection_name]["points"] = points

    def count(self, *, collection_name: str, exact: bool) -> SimpleNamespace:
        assert exact
        return SimpleNamespace(count=len(self.collections[collection_name]["points"]))

    def update_collection(self, collection_name: str, *, metadata: dict[str, object]) -> None:
        self.collections[collection_name]["metadata"] = metadata

    def get_collection(self, collection_name: str) -> SimpleNamespace:
        metadata = self.collections[collection_name]["metadata"] or {}
        return SimpleNamespace(config=SimpleNamespace(metadata=metadata))

    def scroll(
        self,
        collection_name: str,
        *,
        limit: int,
        offset: object = None,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[list[SimpleNamespace], None]:
        assert not with_payload
        assert not with_vectors
        points = self.collections[collection_name]["points"]
        return [SimpleNamespace(id=point.id) for point in points[:limit]], None


class DenseEncoder:
    revision = "e5-test-revision"

    def validate_passage(self, text: str) -> None:
        if text == "oversized":
            raise TokenLimitError(E5_MODEL, "Passage", MAX_TOKENS + 1, MAX_TOKENS)

    def passages(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * E5_DIMENSIONS for _ in texts]


class SparseEncoder:
    revision = "bm25-test-revision"

    def __init__(self, avg_len: float) -> None:
        self.avg_len = avg_len

    def documents(self, texts: list[str]) -> list[models.SparseVector]:
        return [models.SparseVector(indices=[1], values=[1.0]) for _ in texts]


class ColbertEncoder:
    revision = "colbert-test-revision"

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
def encoders(monkeypatch: pytest.MonkeyPatch) -> None:
    from hybrid_retrieval_lab.ingestion import indexer

    monkeypatch.setattr(indexer, "E5Encoder", DenseEncoder)
    monkeypatch.setattr(indexer, "BM25Encoder", SparseEncoder)
    monkeypatch.setattr(indexer, "ColBERTEncoder", ColbertEncoder)


def test_index_rebuilds_fixed_collection_with_a_simple_manifest(corpus_path: Path, encoders: None) -> None:
    client = FakeQdrant()
    client.create_collection(collection_name="github_docs_pilot")
    client.collections["github_docs_pilot"]["points"] = [SimpleNamespace(id=point_id("stale"))]

    indexed_count = index_corpus(client, "github_docs_pilot", corpus_path)  # type: ignore[arg-type]

    collection = client.collections["github_docs_pilot"]
    manifest = collection["metadata"][METADATA_KEY]
    assert indexed_count == 1
    assert client.deleted == ["github_docs_pilot"]
    assert client.count(collection_name="github_docs_pilot", exact=True).count == 1
    assert manifest["corpus_sha256"]
    assert manifest["indexed_ids"] == ["chunk-1"]
    assert manifest["chunking_config"]["max_tokens"] == MAX_TOKENS
    assert manifest["model_revisions"] == {
        "bm25": SparseEncoder.revision,
        "dense": DenseEncoder.revision,
        "colbert": ColbertEncoder.revision,
    }
    assert "index_sha256" not in manifest


def test_reindex_replaces_previous_points_with_current_corpus(corpus_path: Path, encoders: None) -> None:
    client = FakeQdrant()
    index_corpus(client, "github_docs_pilot", corpus_path)  # type: ignore[arg-type]
    replacement = json.loads(corpus_path.read_text(encoding="utf-8")) | {
        "id": "replacement-chunk",
        "text": "The replacement source chunk.",
    }
    corpus_path.write_text(json.dumps(replacement) + "\n", encoding="utf-8")

    assert index_corpus(client, "github_docs_pilot", corpus_path) == 1  # type: ignore[arg-type]

    stored_ids = {point.id for point in client.collections["github_docs_pilot"]["points"]}
    assert stored_ids == {point_id("replacement-chunk")}
    assert client.deleted == ["github_docs_pilot"]


def test_oversized_corpus_does_not_delete_existing_index(corpus_path: Path, encoders: None) -> None:
    oversized_chunk = json.loads(corpus_path.read_text(encoding="utf-8"))
    oversized_chunk["text"] = "oversized"
    corpus_path.write_text(json.dumps(oversized_chunk) + "\n", encoding="utf-8")
    client = FakeQdrant()
    client.create_collection(collection_name="github_docs_pilot")
    previous_point = SimpleNamespace(id=point_id("previous-index"))
    client.collections["github_docs_pilot"]["points"] = [previous_point]

    with pytest.raises(ValueError, match="No chunks fit"):
        index_corpus(client, "github_docs_pilot", corpus_path)  # type: ignore[arg-type]

    assert client.deleted == []
    assert client.collections["github_docs_pilot"]["points"] == [previous_point]
