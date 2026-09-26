from __future__ import annotations

from qdrant_client import models

from hybrid_retrieval_lab.ingestion.indexer import _publish_alias


class AliasClient:
    def __init__(self) -> None:
        self.operations: list[models.DeleteAliasOperation | models.CreateAliasOperation] = []

    def update_collection_aliases(
        self,
        *,
        change_aliases_operations: list[models.DeleteAliasOperation | models.CreateAliasOperation],
    ) -> None:
        self.operations = change_aliases_operations


def test_alias_replacement_deletes_and_creates_in_one_qdrant_operation() -> None:
    client = AliasClient()

    _publish_alias(client, "active", "generation_2", previous="generation_1")  # type: ignore[arg-type]

    assert len(client.operations) == 2
    assert isinstance(client.operations[0], models.DeleteAliasOperation)
    assert client.operations[0].delete_alias.alias_name == "active"
    assert isinstance(client.operations[1], models.CreateAliasOperation)
    assert client.operations[1].create_alias.alias_name == "active"
    assert client.operations[1].create_alias.collection_name == "generation_2"


def test_initial_alias_creation_uses_one_create_operation() -> None:
    client = AliasClient()

    _publish_alias(client, "active", "generation_1", previous=None)  # type: ignore[arg-type]

    assert len(client.operations) == 1
    assert isinstance(client.operations[0], models.CreateAliasOperation)
    assert client.operations[0].create_alias.collection_name == "generation_1"
