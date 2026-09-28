from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient, models

from hybrid_retrieval_lab.encoders.bm25 import LANGUAGE
from hybrid_retrieval_lab.encoders.bm25 import MODEL_NAME as BM25_MODEL
from hybrid_retrieval_lab.encoders.colbert import MODEL_NAME as COLBERT_MODEL
from hybrid_retrieval_lab.encoders.e5 import MAX_TOKENS
from hybrid_retrieval_lab.encoders.e5 import MODEL_NAME as E5_MODEL
from hybrid_retrieval_lab.ingestion.loader import Chunk

ID_NAMESPACE = uuid.UUID("321a2d3f-19c1-42c0-a2db-4659c932a486")
METADATA_KEY = "hybrid_retrieval_lab"


class SkippedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    token_count: int
    max_tokens: int


class IndexManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[2] = 2
    corpus_sha256: str
    indexed_ids: list[str]
    skipped_chunks: list[SkippedChunk]
    bm25_avg_len: float
    encoding_config: dict[str, str | int]
    chunking_config: dict[str, str | int | bool]
    model_revisions: dict[str, str]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, chunk_id))


def corpus_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoding_config() -> dict[str, str | int]:
    return {
        "bm25_model": BM25_MODEL,
        "bm25_language": LANGUAGE,
        "e5_model": E5_MODEL,
        "e5_max_tokens": MAX_TOKENS,
        "e5_query_prefix": "query: ",
        "e5_passage_prefix": "passage: ",
        "colbert_model": COLBERT_MODEL,
    }


def chunking_config(tokenizer_revision: str) -> dict[str, str | int | bool]:
    return {
        "strategy": "section_aware_token_budget",
        "max_tokens": MAX_TOKENS,
        "tokenizer_model": E5_MODEL,
        "tokenizer_revision": tokenizer_revision,
        "includes_passage_prefix_and_special_tokens": True,
    }


def read_manifest(client: QdrantClient, collection: str) -> IndexManifest:
    metadata = client.get_collection(collection).config.metadata or {}
    if METADATA_KEY not in metadata:
        raise ValueError("Collection has no index manifest; reindex the corpus before evaluation")
    return IndexManifest.model_validate(metadata[METADATA_KEY])


def stored_point_ids(client: QdrantClient, collection: str) -> set[str]:
    ids: set[str] = set()
    offset: models.ExtendedPointId | None = None
    while True:
        batch, next_offset = client.scroll(collection, limit=256, offset=offset, with_payload=False, with_vectors=False)
        for record in batch:
            key = str(record.id)
            if key in ids:
                raise ValueError("Index contains duplicate point IDs; reindex before evaluation")
            ids.add(key)
        if next_offset is None:
            break
        offset = next_offset
    return ids


def verify_index(client: QdrantClient, collection: str, corpus_path: Path, chunks: list[Chunk]) -> IndexManifest:
    manifest = read_manifest(client, collection)
    source_ids = {chunk.id for chunk in chunks}
    indexed_ids = set(manifest.indexed_ids)
    skipped_ids = {chunk.id for chunk in manifest.skipped_chunks}
    if (
        manifest.corpus_sha256 != corpus_hash(corpus_path)
        or manifest.encoding_config != encoding_config()
        or manifest.chunking_config != chunking_config(manifest.model_revisions.get("dense", ""))
    ):
        raise ValueError("Corpus or encoder configuration differs from the index manifest; reindex before evaluation")
    if (
        not indexed_ids
        or len(skipped_ids) != len(manifest.skipped_chunks)
        or indexed_ids & skipped_ids
        or indexed_ids | skipped_ids != source_ids
        or len(indexed_ids) != len(manifest.indexed_ids)
    ):
        raise ValueError("Index manifest does not account for every corpus chunk exactly once")
    point_count = client.count(collection_name=collection, exact=True).count
    if point_count != len(indexed_ids):
        raise ValueError("Qdrant point count differs from the index manifest; reindex before evaluation")
    expected_point_ids = {point_id(chunk_id) for chunk_id in indexed_ids}
    if stored_point_ids(client, collection) != expected_point_ids:
        raise ValueError("Qdrant point IDs differ from the index manifest; reindex before evaluation")
    return manifest
