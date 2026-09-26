#!/usr/bin/env python3
"""Local editor for the proposed pilot relevance judgments."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import NotRequired, TypedDict, cast
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/corpus/chunks.jsonl"
QUERIES = ROOT / "data/queries/pilot-queries.jsonl"
QRELS = ROOT / "data/qrels/pilot-proposed.jsonl"
HTML = ROOT / "scripts/review_qrels.html"
WRITE_LOCK = Lock()


class ChunkRow(TypedDict):
    id: str


class QueryRow(TypedDict):
    id: str


class JudgmentRow(TypedDict):
    query_id: str
    chunk_id: str
    grade: int
    reason: NotRequired[str]


class ReviewData(TypedDict):
    chunks: list[ChunkRow]
    queries: list[QueryRow]
    judgments: list[JudgmentRow]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value: object = json.loads(line)
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise ValueError(f"Invalid JSON object on line {line_number} of {path}")
        records.append(cast(dict[str, object], value))
    return records


def load_data() -> ReviewData:
    chunks: list[ChunkRow] = []
    for row in read_jsonl(CORPUS):
        if not isinstance(row.get("id"), str):
            raise ValueError("Corpus chunk is missing a string ID")
        chunks.append(cast(ChunkRow, row))
    queries: list[QueryRow] = []
    for row in read_jsonl(QUERIES):
        if not isinstance(row.get("id"), str):
            raise ValueError("Query is missing a string ID")
        queries.append(cast(QueryRow, row))
    judgments: list[JudgmentRow] = []
    for row in read_jsonl(QRELS):
        if not isinstance(row.get("query_id"), str) or not isinstance(row.get("chunk_id"), str):
            raise ValueError("Judgment is missing query_id or chunk_id")
        if type(row.get("grade")) is not int:
            raise ValueError("Judgment grade must be an integer")
        judgments.append(cast(JudgmentRow, row))
    expected = {(query["id"], chunk["id"]) for query in queries for chunk in chunks}
    actual = [(row["query_id"], row["chunk_id"]) for row in judgments]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Judgments do not match the pilot chunks and queries")
    return {"chunks": chunks, "queries": queries, "judgments": judgments}


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, value: Mapping[str, object]) -> None:
        self.send_bytes(
            status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"
        )

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self.send_bytes(200, HTML.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/data":
            try:
                self.send_json(200, load_data())
            except (OSError, ValueError) as exc:
                self.send_json(500, {"error": str(exc)})
        else:
            self.send_json(404, {"error": "Route not found"})

    def do_PATCH(self) -> None:
        if urlsplit(self.path).path != "/api/judgment":
            self.send_json(404, {"error": "Route not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 8192:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Invalid request body")
            query_id = payload.get("query_id")
            chunk_id = payload.get("chunk_id")
            if not isinstance(query_id, str) or not isinstance(chunk_id, str):
                raise ValueError("query_id and chunk_id must be strings")
            grade = payload.get("grade")
            reason = payload.get("reason")
            if type(grade) is not int or grade not in (0, 1, 2):
                raise ValueError("Grade must be 0, 1, or 2")
            if not isinstance(reason, str) or len(reason) > 2000:
                raise ValueError("Reason must be at most 2000 characters")
            with WRITE_LOCK:
                data = load_data()
                row = next(
                    (
                        item
                        for item in data["judgments"]
                        if item["query_id"] == query_id and item["chunk_id"] == chunk_id
                    ),
                    None,
                )
                if row is None:
                    self.send_json(404, {"error": "Query and chunk pair not found"})
                    return
                row["grade"] = grade
                if reason.strip():
                    row["reason"] = reason.strip()
                else:
                    row.pop("reason", None)
                content = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in data["judgments"])
                fd, tmp = tempfile.mkstemp(prefix=".pilot-proposed-", suffix=".jsonl", dir=QRELS.parent)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as file:
                        file.write(content)
                        file.flush()
                        os.fsync(file.fileno())
                    os.replace(tmp, QRELS)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
            self.send_json(200, {"judgment": row})
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except OSError as exc:
            self.send_json(500, {"error": str(exc)})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    load_data()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Local review: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()
