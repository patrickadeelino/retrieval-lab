from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid_retrieval_lab.ingestion.identity import (
    METADATA_KEY,
    IndexManifest,
    SkippedChunk,
    chunking_config,
    corpus_hash,
    encoding_config,
    point_id,
    verify_index,
)
from hybrid_retrieval_lab.ingestion.loader import Chunk


@pytest.fixture
def index(tmp_path: Path) -> tuple:
    chunk = Chunk("one", "text", "title", "section", "source", "https://example.com")
    path = tmp_path / "corpus.jsonl"
    path.write_text("frozen source", encoding="utf-8")
    records = [SimpleNamespace(id=point_id(chunk.id))]
    metadata: dict[str, object] = {}
    client = SimpleNamespace(
        scroll=lambda *args, **kwargs: (records, None),
        count=lambda *args, **kwargs: SimpleNamespace(count=len(records)),
        get_collection=lambda name: SimpleNamespace(config=SimpleNamespace(metadata=metadata)),
    )
    manifest = IndexManifest(
        corpus_sha256=corpus_hash(path),
        indexed_ids=[chunk.id],
        skipped_chunks=[],
        bm25_avg_len=1.0,
        chunking_config=chunking_config("revision"),
        encoding_config=encoding_config(),
        model_revisions={"dense": "revision"},
    )
    metadata[METADATA_KEY] = manifest.model_dump()
    return client, path, chunk, records, metadata


def test_matching_corpus_and_point_ids_are_accepted(index: tuple) -> None:
    client, path, chunk, _, _ = index
    assert verify_index(client, "test", path, [chunk]).indexed_ids == ["one"]


@pytest.mark.parametrize("change", ["corpus", "configuration", "chunking", "missing_manifest"])
def test_evaluation_rejects_corpus_or_configuration_mismatch(index: tuple, change: str) -> None:
    client, path, chunk, _, metadata = index
    if change == "corpus":
        path.write_text("different corpus", encoding="utf-8")
    elif change == "configuration":
        manifest_data = metadata[METADATA_KEY]
        manifest_data["encoding_config"]["e5_model"] = "another model"
    elif change == "chunking":
        manifest_data = metadata[METADATA_KEY]
        manifest_data["chunking_config"]["strategy"] = "character_based"
    else:
        metadata.clear()
    with pytest.raises(ValueError):
        verify_index(client, "test", path, [chunk])


def test_evaluation_rejects_missing_or_unexpected_point_ids(index: tuple) -> None:
    client, path, chunk, records, _ = index
    records[0].id = point_id("unexpected")
    with pytest.raises(ValueError, match="point IDs"):
        verify_index(client, "test", path, [chunk])


def test_evaluation_rejects_qdrant_point_count_mismatch(index: tuple) -> None:
    client, path, chunk, records, _ = index
    records.append(SimpleNamespace(id=point_id("unexpected")))
    with pytest.raises(ValueError, match="point count"):
        verify_index(client, "test", path, [chunk])


def test_manifest_accounts_for_skipped_chunks_without_requiring_them_in_qdrant(index: tuple) -> None:
    client, path, chunk, _, metadata = index
    skipped = Chunk("too-long", "long text", "title", "section", "source", "https://example.com")
    metadata[METADATA_KEY]["skipped_chunks"] = [
        SkippedChunk(id=skipped.id, token_count=600, max_tokens=512).model_dump()
    ]
    assert verify_index(client, "test", path, [chunk, skipped]).skipped_chunks[0].id == skipped.id
