from __future__ import annotations

from pathlib import Path

import pytest

from logpoc.data.generate import generate_all
from logpoc.data.prepare import prepare

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "data_v1.yaml"


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("v1")
    generate_all(CONFIG, out)
    return out


@pytest.fixture(scope="session")
def prepared_dir(data_dir) -> Path:
    return prepare(data_dir)


@pytest.fixture
def ml_root(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("ML_ROOT", str(tmp_path / "ml"))
    for var in ["RUN_ID", "FAIL_AFTER_STEPS", "GIT_SHA", "GIT_DIRTY"]:
        monkeypatch.delenv(var, raising=False)
    return tmp_path / "ml"
