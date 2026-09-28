from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from hybrid_retrieval_lab.ingestion.indexer import index_corpus
from hybrid_retrieval_lab.ingestion.loader import load_chunks


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "/integration/" in str(item.path):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def indexed_collection() -> tuple[QdrantClient, str]:
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=120)
    collection = f"test_hrl_{uuid.uuid4().hex}"
    assert collection != os.getenv("QDRANT_COLLECTION", "github_docs_pilot_active")
    corpus = Path("data/corpus/chunks.jsonl")
    expected_count = len(load_chunks(corpus))
    try:
        assert index_corpus(client, collection, corpus) == expected_count
        yield client, collection
    finally:
        aliases = [item for item in client.get_aliases().aliases if item.alias_name == collection]
        physical_collections = {item.collection_name for item in aliases}
        if aliases:
            client.update_collection_aliases(
                change_aliases_operations=[
                    models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=collection))
                ]
            )
        elif client.collection_exists(collection):
            physical_collections.add(collection)
        for physical_collection in physical_collections:
            if client.collection_exists(physical_collection):
                client.delete_collection(physical_collection)
        client.close()
