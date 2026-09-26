from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def load_script():
    def load(name):
        path = Path(__file__).resolve().parents[3] / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load
