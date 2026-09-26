from __future__ import annotations

import io
import json

import pytest


@pytest.fixture
def review(load_script, tmp_path, monkeypatch):
    module = load_script("review_qrels")
    for name, rows in {
        "CORPUS": [{"id": "a"}, {"id": "b"}],
        "QUERIES": [{"id": "q01"}],
        "QRELS": [{"query_id": "q01", "chunk_id": chunk, "grade": 0} for chunk in ("a", "b")],
    }.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        monkeypatch.setattr(module, name, path)
    return module


def patch(review, payload):
    handler = review.Handler.__new__(review.Handler)
    body = json.dumps(payload).encode()
    handler.path = "/api/judgment"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    responses = []
    handler.send_json = lambda status, value: responses.append((status, value))
    handler.do_PATCH()
    return responses[0]


def test_patch_updates_only_selected_judgment(review) -> None:
    before = review.load_data()
    status, response = patch(review, {"query_id": "q01", "chunk_id": "a", "grade": 2, "reason": " Direct answer "})
    assert status == 200
    assert response["judgment"]["reason"] == "Direct answer"
    after = review.load_data()
    assert after["judgments"][0]["grade"] == 2
    assert after["judgments"][1] == before["judgments"][1]
    assert after["chunks"] == before["chunks"] and after["queries"] == before["queries"]


@pytest.mark.parametrize("grade", [True, -1, 3, "2"])
def test_invalid_grade_does_not_modify_judgments(review, grade) -> None:
    original = review.QRELS.read_bytes()
    status, _ = patch(review, {"query_id": "q01", "chunk_id": "a", "grade": grade, "reason": "reason"})
    assert status == 400
    assert review.QRELS.read_bytes() == original


def test_failed_atomic_save_preserves_original_file(review, monkeypatch) -> None:
    original = review.QRELS.read_bytes()

    def fail_replace(*args):
        raise OSError("Simulated disk failure")

    monkeypatch.setattr(review.os, "replace", fail_replace)
    status, _ = patch(review, {"query_id": "q01", "chunk_id": "a", "grade": 2, "reason": "reason"})
    assert status == 500
    assert review.QRELS.read_bytes() == original
    assert not list(review.QRELS.parent.glob(".pilot-proposed-*"))


def test_incomplete_judgment_grid_is_rejected(review) -> None:
    review.QRELS.write_text(json.dumps({"query_id": "q01", "chunk_id": "a", "grade": 0}) + "\n")
    with pytest.raises(ValueError, match="do not match"):
        review.load_data()
