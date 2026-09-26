from __future__ import annotations

from bs4 import BeautifulSoup

from hybrid_retrieval_lab.evaluation.report import render_html
from hybrid_retrieval_lab.evaluation.runner import evaluate


def test_html_preserves_metrics_and_escapes_source_text(evaluation_case) -> None:
    report = evaluate(*evaluation_case.paths, repeats=2, reviewed_qrels=True)
    unsafe = '<script>alert("source text")</script>'
    report["queries"][0]["query"] = unsafe
    report["queries"][0]["strategies"]["bm25"]["ranking"][0]["text"] = unsafe
    html = render_html(report)
    document = BeautifulSoup(html, "html.parser")
    assert document.find("script") is None
    assert unsafe in document.get_text()
    assert "Author-reviewed judgments" in document.get_text()
    assert "2 timed runs" in document.get_text()
    assert "Recall@10" in document.get_text() and "nDCG@5" in document.get_text()
    assert report["inputs"]["qrels_sha256"] in document.get_text()
    assert document.find_all("table")
