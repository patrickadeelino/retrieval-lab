from __future__ import annotations

import pytest

from hybrid_retrieval_lab.ingestion.chunker import Page, chunk_pages


def passage_token_count(text: str) -> int:
    return len(f"passage: {text}".split()) + 2


def test_chunk_pages_preserves_text_metadata_and_stable_ids() -> None:
    original_text = "one two three four five six seven eight nine ten eleven twelve"
    page: Page = {
        "id": "source",
        "title": "Title",
        "url": "https://example.com",
        "sections": [{"heading": "Section", "blocks": [original_text]}],
    }

    chunks = chunk_pages([page], token_count=passage_token_count, max_tokens=7)

    assert len(chunks) > 1
    assert " ".join(chunk["text"] for chunk in chunks).split() == original_text.split()
    assert all(passage_token_count(chunk["text"]) <= 7 for chunk in chunks)
    assert all(chunk["source_url"] == page["url"] for chunk in chunks)
    assert all(chunk["section"] == "Section" for chunk in chunks)
    assert [chunk["id"] for chunk in chunks] == [f"source-{index:03d}" for index in range(1, len(chunks) + 1)]
    assert chunk_pages([page], token_count=passage_token_count, max_tokens=7) == chunks


def test_chunk_pages_keeps_ordinals_sequential_across_sections_and_pages() -> None:
    pages: list[Page] = [
        {
            "id": "first",
            "title": "First page",
            "url": "https://example.com/first",
            "sections": [
                {"heading": "One", "blocks": ["alpha beta"]},
                {"heading": "Two", "blocks": ["gamma delta"]},
            ],
        },
        {
            "id": "second",
            "title": "Second page",
            "url": "https://example.com/second",
            "sections": [{"heading": "Three", "blocks": ["omega"]}],
        },
    ]

    chunks = chunk_pages(pages, token_count=passage_token_count, max_tokens=512)

    assert [(chunk["id"], chunk["section"], chunk["text"]) for chunk in chunks] == [
        ("first-001", "One", "alpha beta"),
        ("first-002", "Two", "gamma delta"),
        ("second-001", "Three", "omega"),
    ]


def test_chunk_pages_splits_blocks_at_the_model_token_budget() -> None:
    page: Page = {
        "id": "source",
        "title": "Title",
        "url": "https://example.com",
        "sections": [{"heading": "Section", "blocks": ["one two three four five six"]}],
    }

    chunks = chunk_pages([page], token_count=passage_token_count, max_tokens=7)

    assert [chunk["text"] for chunk in chunks] == ["one two three four", "five six"]
    assert all(passage_token_count(chunk["text"]) <= 7 for chunk in chunks)
    assert " ".join(chunk["text"] for chunk in chunks).split() == "one two three four five six".split()


def test_chunk_pages_splits_when_joined_paragraphs_exceed_the_model_budget() -> None:
    page: Page = {
        "id": "source",
        "title": "Title",
        "url": "https://example.com",
        "sections": [{"heading": "Section", "blocks": ["one two", "three four"]}],
    }

    chunks = chunk_pages([page], token_count=passage_token_count, max_tokens=5)

    assert [chunk["text"] for chunk in chunks] == ["one two", "three four"]
    assert all(passage_token_count(chunk["text"]) <= 5 for chunk in chunks)


def test_chunk_pages_keeps_an_indivisible_over_budget_word_in_its_own_chunk() -> None:
    page: Page = {
        "id": "source",
        "title": "Title",
        "url": "https://example.com",
        "sections": [{"heading": "Section", "blocks": ["short oversize tail"]}],
    }

    def token_count(text: str) -> int:
        return 10 if "oversize" in text else passage_token_count(text)

    chunks = chunk_pages([page], token_count=token_count, max_tokens=5)

    assert [chunk["text"] for chunk in chunks] == ["short", "oversize", "tail"]


def test_chunk_pages_rejects_nonpositive_token_budgets() -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        chunk_pages([], token_count=passage_token_count, max_tokens=0)
