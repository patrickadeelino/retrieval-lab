from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastembed import LateInteractionTextEmbedding

from hybrid_retrieval_lab.encoders.artifacts import snapshot_directory

MODEL_NAME = "jinaai/jina-colbert-v2"
VECTOR_NAME = "colbert"
DIMENSIONS = 128


class ColBERTEncoder:
    def __init__(self) -> None:
        try:
            self.model = LateInteractionTextEmbedding(model_name=MODEL_NAME)
        except Exception as exc:
            if "External data path validation failed" not in str(exc):
                raise
            self._materialize_onnx_files()
            self.model = LateInteractionTextEmbedding(model_name=MODEL_NAME)
        self.revision = snapshot_directory(self.model.model).name

    @staticmethod
    def _materialize_onnx_files() -> None:
        cache = Path(os.getenv("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed")))
        snapshots = cache.glob("models--jinaai--jina-colbert-v2/snapshots/*/onnx")
        for directory in snapshots:
            for filename in ("model.onnx", "model.onnx_data"):
                path = directory / filename
                if path.is_symlink():
                    temporary = path.with_name(f"{filename}.materializing")
                    shutil.copyfile(path, temporary)
                    temporary.replace(path)

    def passages(self, texts: list[str]) -> list[list[list[float]]]:
        return [
            [[float(value) for value in token] for token in vector] for vector in self.model.embed(texts, batch_size=2)
        ]

    def query(self, text: str) -> list[list[float]]:
        return [[float(value) for value in token] for token in next(iter(self.model.query_embed(text)))]
