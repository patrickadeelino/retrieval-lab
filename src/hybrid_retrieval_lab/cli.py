from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from qdrant_client import QdrantClient

from hybrid_retrieval_lab.evaluation.report import render_html
from hybrid_retrieval_lab.evaluation.runner import evaluate
from hybrid_retrieval_lab.ingestion.identity import read_manifest
from hybrid_retrieval_lab.ingestion.indexer import index_corpus
from hybrid_retrieval_lab.logger import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["index", "evaluate"],
        help=(
            "index builds and validates a versioned collection before atomically switching the active alias; "
            "evaluate reads the existing index"
        ),
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/chunks.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries/pilot-queries.jsonl"))
    parser.add_argument("--qrels", type=Path, default=Path("data/qrels/pilot-proposed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/baseline"))
    parser.add_argument("--reviewed-qrels", action="store_true")
    args = parser.parse_args()
    configure_logging()
    logger.info("cli.command.started", extra={"fields": {"command": args.command}})
    if args.command == "evaluate":
        result = evaluate(args.corpus, args.queries, args.qrels, reviewed_qrels=args.reviewed_qrels)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "pilot.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.output / "index.html").write_text(render_html(result), encoding="utf-8")
        print(f"Report: {args.output / 'index.html'}")
        print(f"Data: {args.output / 'pilot.json'}")
        logger.info("cli.command.completed", extra={"fields": {"command": args.command, "output": str(args.output)}})
    else:
        client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        for attempt in range(30):
            try:
                client.get_collections()
                break
            except Exception:
                if attempt == 29:
                    raise
                time.sleep(1)
        collection = os.getenv("QDRANT_COLLECTION", "github_docs_pilot_active")
        legacy_collection = os.getenv("QDRANT_LEGACY_COLLECTION", "github_docs_pilot")
        count = index_corpus(client, collection, args.corpus, legacy_collection=legacy_collection)
        print(f"Indexed {count} chunks")
        manifest = read_manifest(client, collection)
        skipped_ids = ", ".join(chunk.id for chunk in manifest.skipped_chunks) or "none"
        print(f"Skipped {len(manifest.skipped_chunks)} chunks: {skipped_ids}")
        logger.info("cli.command.completed", extra={"fields": {"command": args.command, "point_count": count}})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("cli.command.failed")
        raise
