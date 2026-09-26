from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from hybrid_retrieval_lab.encoders.bm25 import VECTOR_NAME, BM25Encoder
from hybrid_retrieval_lab.encoders.colbert import DIMENSIONS as COLBERT_DIMENSIONS
from hybrid_retrieval_lab.encoders.colbert import VECTOR_NAME as COLBERT_VECTOR_NAME
from hybrid_retrieval_lab.encoders.colbert import ColBERTEncoder
from hybrid_retrieval_lab.encoders.e5 import DIMENSIONS, E5Encoder
from hybrid_retrieval_lab.encoders.e5 import VECTOR_NAME as DENSE_VECTOR_NAME
from hybrid_retrieval_lab.encoders.exceptions import TokenLimitError
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
from hybrid_retrieval_lab.ingestion.loader import load_chunks

logger = logging.getLogger(__name__)


def _alias_target(client: QdrantClient, alias: str) -> str | None:
    for item in client.get_aliases().aliases:
        if item.alias_name == alias:
            return item.collection_name
    return None


def _publish_alias(client: QdrantClient, alias: str, target: str, previous: str | None) -> None:
    operations: list[models.DeleteAliasOperation | models.CreateAliasOperation] = []
    if previous is not None:
        operations.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    operations.append(
        models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=target, alias_name=alias))
    )
    client.update_collection_aliases(change_aliases_operations=operations)


def _temporary_collection_name(alias: str) -> str:
    return f"{alias}_index_{uuid.uuid4().hex}"


def index_corpus(
    client: QdrantClient,
    collection: str,
    corpus_path: Path,
    legacy_collection: str | None = None,
) -> int:
    started = time.perf_counter()
    logger.info("index.started", extra={"fields": {"collection": collection, "corpus_path": str(corpus_path)}})
    source_hash = corpus_hash(corpus_path)
    previous_target = _alias_target(client, collection)
    if previous_target is None and legacy_collection and client.collection_exists(legacy_collection):
        previous_target = legacy_collection
        _publish_alias(client, collection, previous_target, previous=None)
    target = _temporary_collection_name(collection)
    source_chunks = load_chunks(corpus_path)
    dense_encoder = E5Encoder()
    chunks = []
    skipped = []
    for chunk in source_chunks:
        try:
            dense_encoder.validate_passage(chunk.text)
        except TokenLimitError as exc:
            skipped.append(SkippedChunk(id=chunk.id, token_count=exc.token_count, max_tokens=exc.max_tokens))
            logger.error(
                "index.chunk_skipped",
                extra={
                    "fields": {
                        "chunk_id": chunk.id,
                        "token_count": exc.token_count,
                        "max_tokens": exc.max_tokens,
                        "model": exc.model,
                    }
                },
            )
            continue
        chunks.append(chunk)
    if not chunks:
        raise ValueError("No chunks fit the encoder token budget; the existing collection was preserved")
    logger.info("index.corpus_validated", extra={"fields": {"collection": collection, "chunk_count": len(chunks)}})
    avg_len = sum(len(chunk.text.split()) for chunk in chunks) / len(chunks)
    encoder = BM25Encoder(avg_len=avg_len)
    vectors = encoder.documents([chunk.text for chunk in chunks])
    dense_vectors = dense_encoder.passages([chunk.text for chunk in chunks])
    colbert_encoder = ColBERTEncoder()
    colbert_vectors = colbert_encoder.passages([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError("BM25 produced an unexpected number of vectors")
    if len(dense_vectors) != len(chunks) or any(len(vector) != DIMENSIONS for vector in dense_vectors):
        raise RuntimeError("E5 produced an unexpected vector count or dimension")
    if len(colbert_vectors) != len(chunks) or any(
        not vector or any(len(token) != COLBERT_DIMENSIONS for token in vector) for vector in colbert_vectors
    ):
        raise RuntimeError("ColBERT produced an unexpected multivector count or dimension")
    logger.info("index.vectors_prepared", extra={"fields": {"collection": collection, "chunk_count": len(chunks)}})
    try:
        client.create_collection(
            collection_name=target,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(size=DIMENSIONS, distance=models.Distance.COSINE),
                COLBERT_VECTOR_NAME: models.VectorParams(
                    size=COLBERT_DIMENSIONS,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
            },
            sparse_vectors_config={VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        logger.info(
            "index.generation_created",
            extra={"fields": {"alias": collection, "physical_collection": target}},
        )
        points = [
            models.PointStruct(
                id=point_id(chunk.id),
                vector={VECTOR_NAME: vector, DENSE_VECTOR_NAME: dense_vector, COLBERT_VECTOR_NAME: colbert_vector},
                payload=chunk_payload(chunk),
            )
            for chunk, vector, dense_vector, colbert_vector in zip(
                chunks, vectors, dense_vectors, colbert_vectors, strict=True
            )
        ]
        client.upsert(collection_name=target, points=points, wait=True)
        count = client.count(collection_name=target, exact=True).count
        manifest = IndexManifest(
            corpus_sha256=source_hash,
            indexed_ids=sorted(chunk.id for chunk in chunks),
            skipped_chunks=skipped,
            bm25_avg_len=avg_len,
            encoding_config=encoding_config(),
            model_revisions={
                "bm25": encoder.revision,
                "dense": dense_encoder.revision,
                "colbert": colbert_encoder.revision,
            },
            index_sha256=stored_index_hash(client, target, chunks),
        )
        client.update_collection(target, metadata={METADATA_KEY: manifest.model_dump()})
        verify_index(client, target, corpus_path, source_chunks)
        logger.info(
            "index.generation_verified",
            extra={"fields": {"alias": collection, "physical_collection": target, "point_count": count}},
        )
    except Exception:
        if client.collection_exists(target):
            client.delete_collection(target)
        raise
    try:
        _publish_alias(client, collection, target, previous_target)
        logger.info(
            "index.alias_switched",
            extra={
                "fields": {
                    "alias": collection,
                    "previous_collection": previous_target,
                    "active_collection": target,
                }
            },
        )
    except Exception:
        client.delete_collection(target)
        raise
    if previous_target is not None and previous_target != target and client.collection_exists(previous_target):
        try:
            client.delete_collection(previous_target)
        except Exception:
            logger.exception(
                "index.previous_collection_cleanup_failed",
                extra={"fields": {"alias": collection, "collection": previous_target}},
            )
    logger.info(
        "index.completed",
        extra={
            "fields": {
                "collection": collection,
                "physical_collection": target,
                "previous_collection": previous_target,
                "point_count": count,
                "skipped_count": len(skipped),
                "skipped_ids": [chunk.id for chunk in skipped],
                "index_sha256": manifest.index_sha256,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        },
    )
    return count
