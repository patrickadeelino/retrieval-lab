from pathlib import Path


def snapshot_directory(model: object) -> Path:
    """Isolate access to the snapshot resolved by the pinned FastEmbed version."""
    directory = getattr(model, "_model_dir", None)
    if not isinstance(directory, Path) or directory.parent.name != "snapshots":
        raise ValueError("Encoder must use a versioned Hugging Face snapshot")
    return directory
