from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)
LOGGER_NAME = "hybrid_retrieval_lab"
DEFAULT_LOG_FILE = Path("logs/hybrid-retrieval-lab.log")


class HybridRotatingFileHandler(RotatingFileHandler):
    pass


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        request_id = request_id_context.get()
        if request_id is not None:
            payload["request_id"] = request_id
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception_type"] = record.exc_info[0].__name__
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    log_file: Path | None = None,
    level: str | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> Path:
    path = log_file if log_file is not None else Path(os.environ.get("LOG_FILE", str(DEFAULT_LOG_FILE)))
    configured_level = (level if level is not None else os.environ.get("LOG_LEVEL", "INFO")).upper()
    if configured_level not in ("INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError("LOG_LEVEL must be INFO, WARNING, ERROR, or CRITICAL")
    size = max_bytes if max_bytes is not None else int(os.environ.get("LOG_MAX_BYTES", "10485760"))
    backups = backup_count if backup_count is not None else int(os.environ.get("LOG_BACKUP_COUNT", "5"))
    if size <= 0 or backups < 0:
        raise ValueError("LOG_MAX_BYTES must be positive and LOG_BACKUP_COUNT must not be negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    package_logger = logging.getLogger(LOGGER_NAME)
    for handler in package_logger.handlers[:]:
        if isinstance(handler, HybridRotatingFileHandler):
            package_logger.removeHandler(handler)
            handler.close()
    handler = HybridRotatingFileHandler(path, maxBytes=size, backupCount=backups, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    package_logger.addHandler(handler)
    package_logger.setLevel(configured_level)
    package_logger.propagate = False
    return path
