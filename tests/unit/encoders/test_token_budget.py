from __future__ import annotations

from types import SimpleNamespace

import pytest

from hybrid_retrieval_lab.encoders.e5 import MAX_TOKENS, E5Encoder
from hybrid_retrieval_lab.encoders.exceptions import TokenLimitError


@pytest.mark.parametrize("count,accepted", [(MAX_TOKENS, True), (MAX_TOKENS + 1, False)])
def test_query_budget_includes_prefix_and_special_tokens_before_embedding(count: int, accepted: bool) -> None:
    calls = []

    class FakeTokenizer:
        def encode(self, text: str, **kwargs: object) -> list[int]:
            assert text == "query: question"
            assert kwargs == {"add_special_tokens": True, "truncation": False, "verbose": False}
            return [1] * count

    def embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[0.5]]

    encoder = E5Encoder.__new__(E5Encoder)
    encoder.tokenizer = FakeTokenizer()
    encoder.model = SimpleNamespace(embed=embed)
    if accepted:
        assert encoder.query("question") == [0.5]
        assert calls == [["query: question"]]
    else:
        with pytest.raises(TokenLimitError, match="513.*512-token"):
            encoder.query("question")
        assert calls == []
