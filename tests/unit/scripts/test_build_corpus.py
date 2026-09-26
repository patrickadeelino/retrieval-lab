from __future__ import annotations

import pytest


def test_extraction_keeps_source_text_and_removes_controls(load_script) -> None:
    builder = load_script("build_corpus")
    html = b"""<h1>Original title</h1><div id="article-contents"><div class="markdown-body">
    <h2>Delivery</h2><p>Keep this text.<button>Copy</button><svg>icon</svg></p>
    <ul><li>First rule</li><li>Second rule</li></ul>
    <h2>Further reading</h2><p>Excluded navigation</p></div></div>"""
    page = builder.extract(html, "webhooks", "https://docs.github.com/pt/webhooks")
    assert page["title"] == "Original title"
    assert page["sections"] == [{"heading": "Delivery", "blocks": ["Keep this text.", "First rule", "Second rule"]}]
    with pytest.raises(ValueError, match="Article body missing"):
        builder.extract(b"<h1>Not an article</h1>", "webhooks", "https://example.com")


def test_chunking_preserves_words_source_and_stable_ids(load_script, monkeypatch) -> None:
    builder = load_script("build_corpus")
    monkeypatch.setattr(builder, "MAX_CHARS", 24)
    text = "one two three four five six seven eight nine ten eleven twelve"
    pages = [
        {
            "id": "source",
            "title": "Title",
            "url": "https://example.com",
            "sections": [{"heading": "Section", "blocks": [text]}],
        }
    ]
    chunks = builder.make_chunks(pages)
    assert len(chunks) > 1
    assert " ".join(chunk["text"] for chunk in chunks).split() == text.split()
    assert all(len(chunk["text"]) <= 24 and chunk["source_url"] == pages[0]["url"] for chunk in chunks)
    assert [chunk["id"] for chunk in chunks] == [f"source-{i:03d}" for i in range(1, len(chunks) + 1)]
    assert builder.make_chunks(pages) == chunks
