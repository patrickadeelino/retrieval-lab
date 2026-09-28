from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import cast

import pytest
from qdrant_client import QdrantClient, models

from hybrid_retrieval_lab.encoders.bm25 import VECTOR_NAME as BM25_VECTOR
from hybrid_retrieval_lab.encoders.colbert import VECTOR_NAME as COLBERT_VECTOR
from hybrid_retrieval_lab.encoders.e5 import DIMENSIONS, MAX_TOKENS, E5Encoder
from hybrid_retrieval_lab.encoders.e5 import VECTOR_NAME as E5_VECTOR
from hybrid_retrieval_lab.ingestion.chunker import Page, chunk_pages
from hybrid_retrieval_lab.ingestion.identity import point_id, verify_index
from hybrid_retrieval_lab.ingestion.indexer import _publish_alias, index_corpus
from hybrid_retrieval_lab.ingestion.loader import load_chunks


def test_qdrant_alias_switches_generations_in_one_operation() -> None:
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=30)
    suffix = uuid.uuid4().hex
    alias = f"test_alias_{suffix}"
    old_collection = f"test_old_{suffix}"
    new_collection = f"test_new_{suffix}"
    try:
        client.create_collection(
            old_collection, vectors_config={"dense": models.VectorParams(size=2, distance=models.Distance.COSINE)}
        )
        client.create_collection(
            new_collection, vectors_config={"dense": models.VectorParams(size=2, distance=models.Distance.COSINE)}
        )
        _publish_alias(client, alias, old_collection, previous=None)
        assert (
            next(item for item in client.get_aliases().aliases if item.alias_name == alias).collection_name
            == old_collection
        )

        _publish_alias(client, alias, new_collection, previous=old_collection)

        assert (
            next(item for item in client.get_aliases().aliases if item.alias_name == alias).collection_name
            == new_collection
        )
    finally:
        aliases = [item for item in client.get_aliases().aliases if item.alias_name == alias]
        if aliases:
            client.update_collection_aliases(
                change_aliases_operations=[
                    models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
                ]
            )
        for collection_name in (old_collection, new_collection):
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
        client.close()


def test_all_corpus_points_have_deterministic_ids_payload_and_three_vectors(
    indexed_collection: tuple[QdrantClient, str],
) -> None:
    client, collection = indexed_collection
    expected_count = len(load_chunks(Path("data/corpus/chunks.jsonl")))
    assert client.count(collection_name=collection, exact=True).count == expected_count
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection, limit=20, offset=offset, with_payload=True, with_vectors=True
        )
        points.extend(batch)
        if offset is None:
            break
    assert len(points) == expected_count
    assert len({str(point.id) for point in points}) == expected_count
    for point in points:
        payload = point.payload
        assert payload is not None
        assert str(point.id) == point_id(payload["chunk_id"])
        assert all(payload[field] for field in ("text", "title", "section", "source_id", "source_url"))
        assert point.vector is not None
        assert set(point.vector) == {BM25_VECTOR, E5_VECTOR, COLBERT_VECTOR}
        assert len(point.vector[E5_VECTOR]) == DIMENSIONS
        assert point.vector[COLBERT_VECTOR]


def test_oversized_e5_passage_is_rejected_before_collection_replacement(
    indexed_collection: tuple[QdrantClient, str],
    tmp_path: Path,
) -> None:
    client, collection = indexed_collection
    chunk = {
        "id": "too-long",
        "text": "documentation " * (MAX_TOKENS + 100),
        "title": "Title",
        "section": "Section",
        "source_id": "source",
        "source_url": "https://example.com",
    }
    path = tmp_path / "oversized.jsonl"
    path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No chunks fit"):
        index_corpus(client, collection, path)
    expected_count = len(load_chunks(Path("data/corpus/chunks.jsonl")))
    assert client.count(collection_name=collection, exact=True).count == expected_count


def test_mixed_corpus_skips_only_oversized_chunk(indexed_collection: tuple[QdrantClient, str], tmp_path: Path) -> None:
    client, existing = indexed_collection
    collection = existing + "_mixed"
    valid = {
        "id": "valid",
        "text": "GitHub webhook validation",
        "title": "Title",
        "section": "Section",
        "source_id": "source",
        "source_url": "https://example.com",
    }
    oversized = valid | {"id": "too-long", "text": "documentation " * (MAX_TOKENS + 100)}
    path = tmp_path / "mixed.jsonl"
    path.write_text(json.dumps(valid) + "\n" + json.dumps(oversized) + "\n", encoding="utf-8")
    try:
        assert index_corpus(client, collection, path) == 1
        manifest = verify_index(client, collection, path, load_chunks(path))
        assert manifest.indexed_ids == ["valid"]
        assert [chunk.id for chunk in manifest.skipped_chunks] == ["too-long"]
    finally:
        aliases = [item for item in client.get_aliases().aliases if item.alias_name == collection]
        physical_collections = {item.collection_name for item in aliases}
        if aliases:
            client.update_collection_aliases(
                change_aliases_operations=[
                    models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=collection))
                ]
            )
        elif client.collection_exists(collection):
            physical_collections.add(collection)
        for physical_collection in physical_collections:
            if client.collection_exists(physical_collection):
                client.delete_collection(physical_collection)


def test_pilot_corpus_fits_e5_budget() -> None:
    chunks = load_chunks(Path("data/corpus/chunks.jsonl"))
    encoder = E5Encoder()
    encoder.validate_passages([(chunk.id, chunk.text) for chunk in chunks])
    pages = cast(
        list[Page],
        [json.loads(line) for line in Path("data/corpus/pages.jsonl").read_text(encoding="utf-8").splitlines()],
    )
    regenerated = chunk_pages(pages, token_count=encoder.passage_token_count, max_tokens=MAX_TOKENS)
    assert regenerated == [
        {
            "id": chunk.id,
            "source_id": chunk.source_id,
            "source_url": chunk.source_url,
            "title": chunk.title,
            "section": chunk.section,
            "text": chunk.text,
        }
        for chunk in chunks
    ]
