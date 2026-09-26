from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    title: str
    section: str
    source_id: str
    source_url: str


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    required = ("id", "text", "title", "section", "source_id", "source_url")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on corpus line {line_number}") from exc
        if not isinstance(record, dict) or any(
            not isinstance(record.get(field), str) or not record[field].strip() for field in required
        ):
            raise ValueError(f"Corpus line {line_number} must have non-empty string fields: {', '.join(required)}")
        chunks.append(Chunk(**{field: record[field] for field in required}))
    ids = [chunk.id for chunk in chunks]
    if not chunks or len(ids) != len(set(ids)):
        raise ValueError("Corpus is empty or contains duplicate IDs")
    return chunks
