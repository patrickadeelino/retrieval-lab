from __future__ import annotations

import os
from pathlib import Path

from fastembed import SparseTextEmbedding
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from qdrant_client import models

from hybrid_retrieval_lab.encoders.artifacts import snapshot_directory

MODEL_NAME = "Qdrant/bm25"
LANGUAGE = "portuguese"
VECTOR_NAME = "bm25"


class BM25Encoder:
    def __init__(self, avg_len: float) -> None:
        cache = Path(os.getenv("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed")))
        model_cache = cache / "models--Qdrant--bm25"
        revision_file = model_cache / "refs" / "main"
        model_path = None
        if revision_file.is_file():
            snapshot = model_cache / "snapshots" / revision_file.read_text().strip()
            if (snapshot / "portuguese.txt").is_file():
                model_path = str(snapshot)
        self.model = SparseTextEmbedding(
            model_name=MODEL_NAME,
            language=LANGUAGE,
            avg_len=avg_len,
            specific_model_path=model_path,
        )
        self.revision = snapshot_directory(self.model.model).name

    def documents(self, texts: list[str]) -> list[models.SparseVector]:
        return [self._as_vector(item) for item in self.model.embed(texts)]

    def query(self, text: str) -> models.SparseVector:
        return self._as_vector(next(iter(self.model.query_embed(text))))

    @staticmethod
    def _as_vector(value: SparseEmbedding) -> models.SparseVector:
        return models.SparseVector(
            indices=[int(index) for index in value.indices],
            values=[float(weight) for weight in value.values],
        )
