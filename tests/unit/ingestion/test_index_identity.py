from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import models

from hybrid_retrieval_lab.ingestion.identity import (
    METADATA_KEY,
    IndexManifest,
    SkippedChunk,
    chunk_payload,
    corpus_hash,
    encoding_config,
    point_id,
    stored_index_hash,
    verify_index,
)
from hybrid_retrieval_lab.ingestion.loader import Chunk


@pytest.fixture
def index(tmp_path: Path) -> tuple:
    chunk = Chunk("one", "text", "title", "section", "source", "https://example.com")
    path = tmp_path / "corpus.jsonl"
    path.write_text("frozen source", encoding="utf-8")
    records = [models.Record(id=point_id(chunk.id), payload=chunk_payload(chunk), vector={"dense": [0.1, 0.2]})]
    metadata = {}
    client = SimpleNamespace(
        scroll=lambda *args, **kwargs: (records, None),
        get_collection=lambda name: SimpleNamespace(config=SimpleNamespace(metadata=metadata)),
    )
    manifest = IndexManifest(
        corpus_sha256=corpus_hash(path),
        indexed_ids=[chunk.id],
        skipped_chunks=[],
        bm25_avg_len=1.0,
        encoding_config=encoding_config(),
        model_revisions={"dense": "revision"},
        index_sha256=stored_index_hash(client, "test", [chunk]),
    )
    metadata[METADATA_KEY] = manifest.model_dump()
    return client, path, chunk, records, metadata


def test_matching_index_is_accepted(index: tuple) -> None:
    client, path, chunk, _, _ = index
    assert verify_index(client, "test", path, [chunk]).indexed_ids == ["one"]


@pytest.mark.parametrize("change", ["payload", "vector", "corpus", "configuration", "missing_manifest"])
def test_same_point_count_cannot_hide_a_different_index(index: tuple, change: str) -> None:
    client, path, chunk, records, metadata = index
    if change == "payload":
        records[0].payload["text"] = "different text"
    elif change == "vector":
        records[0].vector = {"dense": [0.9, 0.8]}
    elif change == "corpus":
        path.write_text("different corpus", encoding="utf-8")
    elif change == "configuration":
        metadata[METADATA_KEY]["encoding_config"]["e5_model"] = "another model"
    else:
        metadata.clear()
    with pytest.raises(ValueError):
        verify_index(client, "test", path, [chunk])


def test_manifest_accounts_for_skipped_chunks_without_requiring_them_in_qdrant(index: tuple) -> None:
    client, path, chunk, _, metadata = index
    skipped = Chunk("too-long", "long text", "title", "section", "source", "https://example.com")
    metadata[METADATA_KEY]["skipped_chunks"] = [
        SkippedChunk(id=skipped.id, token_count=600, max_tokens=512).model_dump()
    ]
    assert verify_index(client, "test", path, [chunk, skipped]).skipped_chunks[0].id == skipped.id
