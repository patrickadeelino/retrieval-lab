from __future__ import annotations

from typing import Protocol, cast

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from transformers import AutoTokenizer

from hybrid_retrieval_lab.encoders.artifacts import snapshot_directory
from hybrid_retrieval_lab.encoders.exceptions import TokenLimitError

MODEL_NAME = "intfloat/multilingual-e5-small"
VECTOR_NAME = "dense"
DIMENSIONS = 384
MAX_TOKENS = 512


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool, truncation: bool, verbose: bool) -> list[int]: ...


class TokenizerLoader(Protocol):
    def __call__(self, path: str, *, local_files_only: bool, use_fast: bool) -> Tokenizer: ...


class E5Encoder:
    def __init__(self) -> None:
        if not any(item["model"].lower() == MODEL_NAME.lower() for item in TextEmbedding.list_supported_models()):
            TextEmbedding.add_custom_model(
                model=MODEL_NAME,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=MODEL_NAME),
                dim=DIMENSIONS,
                model_file="onnx/model.onnx",
                description="Multilingual E5 small with mean pooling and L2 normalization",
                license="mit",
            )
        self.model = TextEmbedding(model_name=MODEL_NAME)
        directory = snapshot_directory(self.model.model)
        self.revision = directory.name
        load_tokenizer = cast(TokenizerLoader, AutoTokenizer.from_pretrained)
        self.tokenizer = load_tokenizer(str(directory), local_files_only=True, use_fast=True)

    def _token_count(self, text: str, prefix: str) -> int:
        return len(self.tokenizer.encode(f"{prefix}: {text}", add_special_tokens=True, truncation=False, verbose=False))

    def passage_token_count(self, text: str) -> int:
        """Count passage tokens exactly as the E5 model receives them."""
        return self._token_count(text, "passage")

    def _validate(self, text: str, prefix: str, input_kind: str) -> int:
        count = self._token_count(text, prefix)
        if count > MAX_TOKENS:
            raise TokenLimitError(MODEL_NAME, input_kind, count, MAX_TOKENS)
        return count

    def validate_passage(self, text: str) -> int:
        return self._validate(text, "passage", "Passage")

    def validate_passages(self, chunks: list[tuple[str, str]]) -> None:
        for _, text in chunks:
            self.validate_passage(text)

    def passages(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            self.validate_passage(text)
        return [
            [float(value) for value in vector] for vector in self.model.embed([f"passage: {text}" for text in texts])
        ]

    def query(self, text: str) -> list[float]:
        self._validate(text, "query", "Query")
        return [float(value) for value in next(iter(self.model.embed([f"query: {text}"])))]
