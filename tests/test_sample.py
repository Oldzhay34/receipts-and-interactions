"""Checks on the real data sample in data/sample (skipped until it has been exported from Colab)."""

from __future__ import annotations

from recsys.config import load_config
from recsys.contracts import LOADED_INTERACTIONS_SCHEMA, LOADED_RECIPES_SCHEMA, validate
from recsys.io import read_artifact_dir

from .conftest import require_sample


def test_sample_matches_contracts(tmp_path) -> None:
    art = read_artifact_dir(require_sample())  # verifies sha256 of every file
    recipes, interactions = art["recipes"], art["interactions"]
    validate(recipes, LOADED_RECIPES_SCHEMA)
    validate(interactions, LOADED_INTERACTIONS_SCHEMA)
    n = load_config(data_root=tmp_path).section("sample")["n_recipes"]
    assert len(recipes) == n
    assert set(interactions["recipe_id"]) <= set(recipes["id"])
    assert art.manifest["stage"] == "sample"
    assert art.manifest["inputs"][0]["stage"] == "loaded"
