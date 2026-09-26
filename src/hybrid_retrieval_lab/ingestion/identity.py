from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict
from importlib.metadata import version
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
    schema_version: Literal[1] = 1
    corpus_sha256: str
    indexed_ids: list[str]
    skipped_chunks: list[SkippedChunk]
    bm25_avg_len: float
    encoding_config: dict[str, str | int]
    model_revisions: dict[str, str]
    index_sha256: str


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
        **{package: version(package) for package in ("fastembed", "tokenizers", "transformers", "onnxruntime")},
    }


def chunk_payload(chunk: Chunk) -> dict[str, str]:
    payload = asdict(chunk)
    payload["chunk_id"] = payload.pop("id")
    return payload


def read_manifest(client: QdrantClient, collection: str) -> IndexManifest:
    metadata = client.get_collection(collection).config.metadata or {}
    if METADATA_KEY not in metadata:
        raise ValueError("Collection has no index manifest; reindex the corpus before evaluation")
    return IndexManifest.model_validate(metadata[METADATA_KEY])


def stored_index_hash(client: QdrantClient, collection: str, chunks: list[Chunk]) -> str:
    expected = {point_id(chunk.id): chunk_payload(chunk) for chunk in chunks}
    records: dict[str, models.Record] = {}
    offset: models.ExtendedPointId | None = None
    while True:
        batch, next_offset = client.scroll(collection, limit=16, offset=offset, with_payload=True, with_vectors=True)
        for record in batch:
            key = str(record.id)
            if key in records or key not in expected or record.payload != expected[key]:
                raise ValueError("Indexed IDs or payloads do not match the corpus; reindex before evaluation")
            records[key] = record
        if next_offset is None:
            break
        offset = next_offset
    if set(records) != set(expected):
        raise ValueError("Indexed chunk set does not match the manifest; reindex before evaluation")
    digest = hashlib.sha256()
    for key in sorted(records):
        record = records[key]
        digest.update(
            json.dumps(
                {"id": key, "payload": record.payload, "vector": record.model_dump(mode="json")["vector"]},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def verify_index(client: QdrantClient, collection: str, corpus_path: Path, chunks: list[Chunk]) -> IndexManifest:
    manifest = read_manifest(client, collection)
    source_ids = {chunk.id for chunk in chunks}
    indexed_ids = set(manifest.indexed_ids)
    skipped_ids = {chunk.id for chunk in manifest.skipped_chunks}
    if manifest.corpus_sha256 != corpus_hash(corpus_path) or manifest.encoding_config != encoding_config():
        raise ValueError("Corpus or encoder configuration differs from the index manifest; reindex before evaluation")
    if (
        not indexed_ids
        or len(skipped_ids) != len(manifest.skipped_chunks)
        or indexed_ids & skipped_ids
        or indexed_ids | skipped_ids != source_ids
        or len(indexed_ids) != len(manifest.indexed_ids)
    ):
        raise ValueError("Index manifest does not account for every corpus chunk exactly once")
    indexed_chunks = [chunk for chunk in chunks if chunk.id in indexed_ids]
    if stored_index_hash(client, collection, indexed_chunks) != manifest.index_sha256:
        raise ValueError("Stored vectors differ from the index manifest; reindex before evaluation")
    return manifest
