from __future__ import annotations

from hybrid_retrieval_lab.ingestion.chunker import Page, chunk_pages


def test_chunk_pages_preserves_text_metadata_and_stable_ids() -> None:
    original_text = "one two three four five six seven eight nine ten eleven twelve"
    page: Page = {
        "id": "source",
        "title": "Title",
        "url": "https://example.com",
        "sections": [{"heading": "Section", "blocks": [original_text]}],
    }

    chunks = chunk_pages([page], max_chars=24)

    assert len(chunks) > 1
    assert " ".join(chunk["text"] for chunk in chunks).split() == original_text.split()
    assert all(len(chunk["text"]) <= 24 for chunk in chunks)
    assert all(chunk["source_url"] == page["url"] for chunk in chunks)
    assert all(chunk["section"] == "Section" for chunk in chunks)
    assert [chunk["id"] for chunk in chunks] == [f"source-{index:03d}" for index in range(1, len(chunks) + 1)]
    assert chunk_pages([page], max_chars=24) == chunks


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

    chunks = chunk_pages(pages, max_chars=12)

    assert [(chunk["id"], chunk["section"], chunk["text"]) for chunk in chunks] == [
        ("first-001", "One", "alpha beta"),
        ("first-002", "Two", "gamma delta"),
        ("second-001", "Three", "omega"),
    ]
