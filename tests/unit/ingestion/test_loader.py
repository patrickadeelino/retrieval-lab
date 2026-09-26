from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_retrieval_lab.ingestion.loader import load_chunks

VALID = {
    "id": "chunk-1",
    "text": "A useful chunk",
    "title": "Title",
    "section": "Section",
    "source_id": "source",
    "source_url": "https://example.com/source",
}


def write_corpus(path: Path, *records: object) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return path


@pytest.mark.parametrize("field", ["id", "text", "title", "section", "source_id", "source_url"])
def test_loader_rejects_missing_or_blank_required_field(tmp_path: Path, field: str) -> None:
    record = VALID | {field: "  "}
    with pytest.raises(ValueError, match="non-empty string fields"):
        load_chunks(write_corpus(tmp_path / "corpus.jsonl", record))
    record.pop(field)
    with pytest.raises(ValueError, match="non-empty string fields"):
        load_chunks(write_corpus(tmp_path / "corpus.jsonl", record))


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate IDs"):
        load_chunks(write_corpus(tmp_path / "corpus.jsonl", VALID, VALID))


def test_loader_rejects_invalid_json_with_line_number(tmp_path: Path) -> None:
    path = write_corpus(tmp_path / "corpus.jsonl", VALID)
    path.write_text(path.read_text(encoding="utf-8") + "\n{", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_chunks(path)


def test_loader_accepts_valid_record_and_ignores_blank_lines(tmp_path: Path) -> None:
    path = write_corpus(tmp_path / "corpus.jsonl", VALID)
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert load_chunks(path)[0].id == "chunk-1"
