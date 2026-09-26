from __future__ import annotations

import logging
import time
from functools import cached_property
from pathlib import Path

from qdrant_client import QdrantClient

from hybrid_retrieval_lab.encoders.bm25 import BM25Encoder
from hybrid_retrieval_lab.encoders.colbert import ColBERTEncoder
from hybrid_retrieval_lab.encoders.e5 import E5Encoder
from hybrid_retrieval_lab.ingestion.identity import IndexManifest, read_manifest
from hybrid_retrieval_lab.services.search.exceptions import SearchUnavailableError

logger = logging.getLogger(__name__)


class SearchResources:
    def __init__(self, client: QdrantClient, collection: str, corpus_path: Path) -> None:
        self.client = client
        self.collection = collection
        self.corpus_path = corpus_path

    @cached_property
    def manifest(self) -> IndexManifest:
        return read_manifest(self.client, self.collection)

    def _verify_revision(self, name: str, revision: str) -> None:
        if self.manifest.model_revisions.get(name) != revision:
            raise SearchUnavailableError(f"The {name} model revision differs from the index; reindex before searching")

    @cached_property
    def bm25_encoder(self) -> BM25Encoder:
        started = time.perf_counter()
        manifest = self.manifest
        encoder = BM25Encoder(avg_len=manifest.bm25_avg_len)
        self._verify_revision("bm25", encoder.revision)
        logger.info(
            "encoder.loaded",
            extra={
                "fields": {
                    "encoder": "bm25",
                    "corpus_chunks": len(manifest.indexed_ids),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            },
        )
        return encoder

    @cached_property
    def dense_encoder(self) -> E5Encoder:
        started = time.perf_counter()
        encoder = E5Encoder()
        self._verify_revision("dense", encoder.revision)
        logger.info(
            "encoder.loaded",
            extra={"fields": {"encoder": "e5", "duration_ms": round((time.perf_counter() - started) * 1000, 3)}},
        )
        return encoder

    @cached_property
    def colbert_encoder(self) -> ColBERTEncoder:
        started = time.perf_counter()
        encoder = ColBERTEncoder()
        self._verify_revision("colbert", encoder.revision)
        logger.info(
            "encoder.loaded",
            extra={"fields": {"encoder": "colbert", "duration_ms": round((time.perf_counter() - started) * 1000, 3)}},
        )
        return encoder
