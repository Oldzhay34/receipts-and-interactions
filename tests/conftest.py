"""Shared pytest fixtures.

Two local data sources:
* tests/fixtures/  hand-written edge cases (always available).
* data/sample/     small real sample exported from Colab (outputs/sample); tests that need it
                   are skipped with a clear reason until it exists.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from recsys.config import REPO_ROOT, Config, load_config
from recsys.data.load import load_interactions, load_recipes
from recsys.io import ArtifactStore

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_DIR = REPO_ROOT / "data" / "sample"


@pytest.fixture(scope="session")
def fixture_recipes() -> pd.DataFrame:
    return load_recipes(FIXTURES / "RAW_recipes.csv")


@pytest.fixture(scope="session")
def fixture_interactions() -> pd.DataFrame:
    return load_interactions(FIXTURES / "RAW_interactions.csv")


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    """A throw-away DATA_ROOT with the fixture CSVs placed in raw/."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("RAW_recipes.csv", "RAW_interactions.csv"):
        shutil.copy(FIXTURES / name, raw / name)
    return tmp_path


@pytest.fixture()
def cfg(data_root: Path) -> Config:
    return load_config(data_root=data_root)


@pytest.fixture()
def store(cfg: Config) -> ArtifactStore:
    return ArtifactStore.from_config(cfg)


def require_sample() -> Path:
    """Skip the calling test unless data/sample holds an exported sample artefact."""
    if not (SAMPLE_DIR / "manifest.json").is_file():
        pytest.skip(
            "data/sample/ not populated yet: run notebooks/00_setup.ipynb on Colab, then "
            "ArtifactStore.export('sample', 'data/sample') (see notebooks/00_setup.ipynb section 5)"
        )
    return SAMPLE_DIR
