from __future__ import annotations

import hashlib
from pathlib import Path

from hybrid_retrieval_lab.evaluation.runner import _runtime_metadata


def test_runtime_metadata_records_source_revision_and_dependency_lock(tmp_path: Path, monkeypatch) -> None:
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("frozen dependencies\n", encoding="utf-8")
    monkeypatch.setenv("GIT_COMMIT", "a1b2c3d4")
    monkeypatch.setenv("GIT_TREE_STATE", "clean")
    monkeypatch.setenv("UV_VERSION", "uv 0.11.3")

    metadata = _runtime_metadata(tmp_path, "1.19.1")

    assert metadata["git_commit"] == "a1b2c3d4"
    assert metadata["git_tree_state"] == "clean"
    assert metadata["uv"] == "uv 0.11.3"
    assert metadata["uv_lock_sha256"] == hashlib.sha256(b"frozen dependencies\n").hexdigest()
    assert metadata["qdrant"] == "1.19.1"
