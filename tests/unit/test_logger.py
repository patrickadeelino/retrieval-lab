from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from hybrid_retrieval_lab.logger import configure_logging, request_id_context


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_logger_writes_json_with_level_and_request_id(tmp_path: Path) -> None:
    path = tmp_path / "events.log"
    configure_logging(path, level="ERROR", max_bytes=10000, backup_count=1)
    token = request_id_context.set("request-123")
    try:
        logger = logging.getLogger("hybrid_retrieval_lab.test")
        logger.info("ignored.event")
        logger.error("failure.recorded", extra={"fields": {"strategy": "bm25"}})
    finally:
        request_id_context.reset(token)
        configure_logging()
    assert [event["event"] for event in events(path)] == ["failure.recorded"]
    assert events(path)[0]["request_id"] == "request-123"
    assert events(path)[0]["strategy"] == "bm25"
    assert events(path)[0]["level"] == "ERROR"


def test_logger_rotates_file_and_rejects_unsupported_level(tmp_path: Path) -> None:
    path = tmp_path / "rotating.log"
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        configure_logging(path, level="DEBUG")
    configure_logging(path, level="INFO", max_bytes=300, backup_count=1)
    try:
        logger = logging.getLogger("hybrid_retrieval_lab.test")
        logger.info("first.record", extra={"fields": {"padding": "x" * 200}})
        logger.info("second.record", extra={"fields": {"padding": "x" * 200}})
    finally:
        configure_logging()
    assert path.with_name("rotating.log.1").exists()
    assert events(path)[0]["event"] == "second.record"
