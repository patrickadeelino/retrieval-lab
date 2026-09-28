from __future__ import annotations

import logging
import time
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
    chunking_config,
    corpus_hash,
    encoding_config,
    point_id,
    verify_index,
)
from hybrid_retrieval_lab.ingestion.loader import load_chunks

logger = logging.getLogger(__name__)


def index_corpus(
    client: QdrantClient,
    collection: str,
    corpus_path: Path,
) -> int:
    started = time.perf_counter()
    logger.info("index.started", extra={"fields": {"collection": collection, "corpus_path": str(corpus_path)}})
    source_hash = corpus_hash(corpus_path)
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

    manifest = IndexManifest(
        corpus_sha256=source_hash,
        indexed_ids=sorted(chunk.id for chunk in chunks),
        skipped_chunks=skipped,
        bm25_avg_len=avg_len,
        chunking_config=chunking_config(dense_encoder.revision),
        encoding_config=encoding_config(),
        model_revisions={
            "bm25": encoder.revision,
            "dense": dense_encoder.revision,
            "colbert": colbert_encoder.revision,
        },
    )

    points = [
        models.PointStruct(
            id=point_id(chunk.id),
            vector={VECTOR_NAME: vector, DENSE_VECTOR_NAME: dense_vector, COLBERT_VECTOR_NAME: colbert_vector},
            payload={
                "chunk_id": chunk.id,
                "text": chunk.text,
                "title": chunk.title,
                "section": chunk.section,
                "source_id": chunk.source_id,
                "source_url": chunk.source_url,
            },
        )
        for chunk, vector, dense_vector, colbert_vector in zip(
            chunks, vectors, dense_vectors, colbert_vectors, strict=True
        )
    ]

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
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
    client.upsert(collection_name=collection, points=points, wait=True)
    count = client.count(collection_name=collection, exact=True).count
    if count != len(chunks):
        raise RuntimeError("Qdrant indexed an unexpected number of chunks")
    client.update_collection(collection, metadata={METADATA_KEY: manifest.model_dump()})
    verify_index(client, collection, corpus_path, source_chunks)
    logger.info(
        "index.completed",
        extra={
            "fields": {
                "collection": collection,
                "point_count": count,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        },
    )
    return count
