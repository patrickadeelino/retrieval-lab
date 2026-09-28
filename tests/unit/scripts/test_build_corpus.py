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
