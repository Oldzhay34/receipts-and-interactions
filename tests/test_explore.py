"""Tests for recsys.analysis.explore on the hand-written fixtures (expected values derived by hand
from tests/fixtures/build_fixtures.py) and, when available, on data/sample."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from recsys.analysis.explore import (
    EXPLORATION_SCHEMAS,
    basic_counts,
    column_type_table,
    exploratory_targets,
    per_entity_counts,
    rating_counts,
    top_items,
)
from recsys.config import REPO_ROOT, Config
from recsys.contracts import validate
from recsys.io import ArtifactStore, read_artifact_dir

from .conftest import require_sample

NOTEBOOK = REPO_ROOT / "notebooks" / "01_exploration.ipynb"


@pytest.fixture(scope="module")
def tables(fixture_recipes, fixture_interactions) -> dict[str, pd.DataFrame]:
    return {
        "column_types": column_type_table(
            {"recipes": fixture_recipes, "interactions": fixture_interactions}
        ),
        "basic_counts": basic_counts(fixture_recipes, fixture_interactions),
        "per_entity_counts": per_entity_counts(fixture_interactions, fixture_recipes),
        "rating_counts": rating_counts(fixture_interactions),
        "top_items": top_items(fixture_recipes),
        "exploratory_targets": exploratory_targets(fixture_recipes, fixture_interactions),
    }


def test_all_outputs_match_their_schemas(tables) -> None:
    assert set(tables) == set(EXPLORATION_SCHEMAS)
    for name, df in tables.items():
        validate(df, EXPLORATION_SCHEMAS[name])


def test_functions_do_not_mutate_inputs(fixture_recipes, fixture_interactions) -> None:
    r, i = fixture_recipes.copy(deep=True), fixture_interactions.copy(deep=True)
    column_type_table({"recipes": fixture_recipes, "interactions": fixture_interactions})
    basic_counts(fixture_recipes, fixture_interactions)
    per_entity_counts(fixture_interactions, fixture_recipes)
    rating_counts(fixture_interactions)
    top_items(fixture_recipes)
    exploratory_targets(fixture_recipes, fixture_interactions)
    pd.testing.assert_frame_equal(fixture_recipes, r)
    pd.testing.assert_frame_equal(fixture_interactions, i)


# ----------------------------------------------------------------- column types


def test_column_types_declared(tables) -> None:
    t = tables["column_types"].set_index(["table", "column"])["type_class"]
    assert t[("recipes", "id")] == "categorical"
    assert t[("interactions", "rating")] == "categorical"
    assert t[("recipes", "minutes")] == "continuous"
    assert t[("recipes", "n_steps")] == "continuous"  # declared, not inferred from 5 values
    assert t[("recipes", "calories")] == "continuous"
    assert t[("recipes", "ingredients")] == "list"
    assert t[("recipes", "submitted")] == "datetime"
    assert t[("interactions", "review")] == "text"
    ct = tables["column_types"]
    assert len(ct) == 18 + 5 and (ct["source"] == "declared").all()
    row = ct[(ct["table"] == "recipes") & (ct["column"] == "description")].iloc[0]
    assert row["n_missing"] == 1
    assert pd.isna(ct[(ct["column"] == "tags")]["n_unique"].iloc[0])


def test_column_types_inferred_for_unknown_columns() -> None:
    df = pd.DataFrame(
        {
            "flag": [True, False, True],
            "small_int": [1, 2, 1],
            "big_float": [0.1, 0.2, 0.3],
            "words": ["a", "b", None],
            "when": pd.to_datetime(["2020-01-01"] * 3),
            "lst": [["x"], [], None],
        }
    )
    t = column_type_table({"t": df}, max_categories=2).set_index("column")
    assert t["type_class"].to_dict() == {
        "flag": "categorical",
        "small_int": "categorical",
        "big_float": "continuous",
        "words": "text",
        "when": "datetime",
        "lst": "list",
    }
    assert (t["source"] == "inferred").all()
    t1 = column_type_table({"t": df}, max_categories=1).set_index("column")
    assert t1.loc["small_int", "type_class"] == "continuous"


# ----------------------------------------------------------------- basic counts


def test_basic_counts(tables) -> None:
    b = tables["basic_counts"].iloc[0]
    assert b["n_recipes"] == 20
    assert b["n_interactions"] == 62
    assert b["n_users"] == 15
    assert b["n_contributors"] == 11
    assert b["n_recipes_with_interactions"] == 19
    assert b["n_recipes_without_interactions"] == 1  # recipe 111
    assert b["n_orphan_interactions"] == 0
    assert b["n_duplicate_interactions"] == 1  # 1002 x 101
    assert b["n_repeated_user_recipe_pairs"] == 1
    assert b["recipes_submitted_min"] == pd.Timestamp("2003-02-11")
    assert b["recipes_submitted_max"] == pd.Timestamp("2012-12-12")
    assert b["interactions_date_min"] < b["interactions_date_max"]


def test_basic_counts_orphans() -> None:
    recipes = pd.DataFrame(
        {"id": [1, 2], "contributor_id": [9, 9], "submitted": pd.to_datetime(["2001-01-01"] * 2)}
    )
    inter = pd.DataFrame(
        {
            "user_id": [5, 6],
            "recipe_id": [1, 99],
            "date": pd.to_datetime(["2002-01-01", "2003-01-01"]),
            "rating": [5, 4],
        }
    )
    b = basic_counts(recipes, inter).iloc[0]
    assert b["n_orphan_interactions"] == 1
    assert b["n_recipes_with_interactions"] == 1
    assert b["n_recipes_without_interactions"] == 1


# ----------------------------------------------------------------- per-entity counts


def test_per_entity_counts(tables) -> None:
    p = tables["per_entity_counts"].set_index("entity")
    assert p.loc["recipe", "n_entities"] == 20  # includes recipe 111 with 0 reviews
    assert p.loc["recipe", "n_zero"] == 1
    assert p.loc["recipe", "min"] == 0
    assert p.loc["user", "n_entities"] == 15
    assert p.loc["user", "n_one"] == 1  # user 1015
    assert p.loc["user", "min"] == 1
    assert p.loc["user", "max"] == 6  # user 1001
    assert p.loc["recipe", "max"] == 6  # recipe 101, duplicate row counted
    assert p.loc["user", "mean"] == pytest.approx(62 / 15)
    assert p.loc["recipe", "mean"] == pytest.approx(62 / 20)


def test_per_entity_counts_without_recipes_excludes_zero(fixture_interactions) -> None:
    p = per_entity_counts(fixture_interactions).set_index("entity")
    assert p.loc["recipe", "n_entities"] == 19
    assert p.loc["recipe", "n_zero"] == 0


# ----------------------------------------------------------------- rating counts


def test_rating_counts(tables) -> None:
    r = tables["rating_counts"].set_index("rating")
    assert list(r.index) == [0, 1, 2, 3, 4, 5]
    assert r["count"].sum() == 62
    assert r.loc[0, "count"] == 4
    assert bool(r.loc[0, "is_zero"]) and r.loc[0, "note"] != ""
    assert np.isnan(r.loc[0, "share_of_rated"])
    assert r["share"].sum() == pytest.approx(1.0)
    assert r.drop(0)["share_of_rated"].sum() == pytest.approx(1.0)
    assert r.loc[5, "share_of_rated"] == pytest.approx(r.loc[5, "count"] / 58)


def test_rating_counts_lists_absent_values() -> None:
    r = rating_counts(pd.DataFrame({"rating": [5, 5, 0]})).set_index("rating")
    assert r.loc[3, "count"] == 0
    assert r.loc[5, "share_of_rated"] == 1.0


# ----------------------------------------------------------------- top items


def test_top_items(tables) -> None:
    t = tables["top_items"]
    ing = t[t["kind"] == "ingredient"].set_index("rank")
    tag = t[t["kind"] == "tag"].set_index("rank")
    # garlic and salt both appear in 7 recipes; tie broken alphabetically
    assert ing.loc[1, ["item", "n_recipes"]].tolist() == ["garlic", 7]
    assert ing.loc[2, ["item", "n_recipes"]].tolist() == ["salt", 7]
    assert ing.loc[3, ["item", "n_recipes"]].tolist() == ["onion", 6]
    assert tag.loc[1, ["item", "n_recipes"]].tolist() == ["main-dish", 7]
    assert ing.loc[1, "share_of_recipes"] == pytest.approx(7 / 20)
    assert len(ing) == 20 and len(tag) == 20


def test_top_items_counts_once_per_recipe() -> None:
    recipes = pd.DataFrame(
        {"id": [1, 2], "ingredients": [["salt", "salt"], ["salt", None]], "tags": [["a"], []]}
    )
    t = top_items(recipes, n=5)
    row = t[(t["kind"] == "ingredient") & (t["item"] == "salt")].iloc[0]
    assert row["n_recipes"] == 2


# ----------------------------------------------------------------- exploratory targets


def test_rating_zero_excluded_from_average(tables) -> None:
    t = tables["exploratory_targets"].set_index("recipe_id")
    # recipe 104: ratings 0 (user 1003), 5, 4 -> mean of non-zero = 4.5, not 3.0
    assert t.loc[104, "n_interactions"] == 3
    assert t.loc[104, "n_ratings"] == 2
    assert t.loc[104, "avg_rating"] == pytest.approx(4.5)
    # recipe 119: 5, 4, 0 -> 4.5
    assert t.loc[119, "avg_rating"] == pytest.approx(4.5)
    # recipe 101: 5, 4, 4 (duplicate kept), 5, 3, 5 -> 26/6
    assert t.loc[101, "n_interactions"] == 6
    assert t.loc[101, "avg_rating"] == pytest.approx(26 / 6)


def test_recipe_without_interactions_has_no_rating(tables) -> None:
    t = tables["exploratory_targets"].set_index("recipe_id")
    assert len(t) == 20
    assert t.loc[111, "n_interactions"] == 0
    assert np.isnan(t.loc[111, "avg_rating"])
    assert not t.loc[111, "has_rating"]
    assert t["has_rating"].sum() == 19


def test_exploratory_targets_docstring_warns_about_leakage() -> None:
    doc = exploratory_targets.__doc__ or ""
    assert "Leakage" in doc and "1.9" in doc and "TRAIN" in doc


# ----------------------------------------------------------------- notebook


def _code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_starts_with_colab_cell_and_has_no_logic() -> None:
    cells = _code_cells()
    for needle in ("drive.mount", "git clone", "pip install -q -e .", "DATA_ROOT", "load_config()"):
        assert needle in cells[0]
    for src in cells[1:]:
        assert "def " not in src and "class " not in src


def test_notebook_runs_on_fixtures(cfg: Config, fixture_recipes, fixture_interactions) -> None:
    store = ArtifactStore.from_config(cfg)
    store.save("loaded", {"recipes": fixture_recipes, "interactions": fixture_interactions})
    ns: dict[str, object] = {"CFG": cfg, "display": lambda *a, **k: None}
    for src in _code_cells()[1:]:
        exec(compile(src, str(NOTEBOOK), "exec"), ns)  # noqa: S102 - our own notebook
    art = store.load("exploration")
    assert art.version == "v1"
    assert set(art.keys()) == set(EXPLORATION_SCHEMAS)
    assert art.manifest["inputs"][0]["stage"] == "loaded"
    assert {f.get("schema") for f in art.manifest["files"].values()} == {
        s.name for s in EXPLORATION_SCHEMAS.values()
    }


def test_notebook_comment_drafts_contain_no_literal_numbers() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    drafts = [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "markdown" and "Comment draft" in "".join(c["source"])
    ]
    assert len(drafts) == 6
    for d in drafts:
        # numbers must come from placeholders such as {basic_counts.n_recipes}, never typed in
        assert not _has_bare_number(d.split("Comment draft", 1)[1]), d


def _has_bare_number(text: str) -> bool:
    no_placeholders = re.sub(r"\{[^}]*\}", "", text)
    no_stage_ids = re.sub(r"\b1\.\d+\b", "", no_placeholders)  # stage ids like 1.3 are allowed
    no_ratings = re.sub(r"rating\s*=\s*0|\b0\s*=|0\s*-\s*5|1\s*-\s*5", "", no_stage_ids)
    return re.search(r"\d", no_ratings) is not None


# ----------------------------------------------------------------- real sample


def test_on_real_sample() -> None:
    art = read_artifact_dir(require_sample())
    recipes, inter = art["recipes"], art["interactions"]
    outs = {
        "column_types": column_type_table({"recipes": recipes, "interactions": inter}),
        "basic_counts": basic_counts(recipes, inter),
        "per_entity_counts": per_entity_counts(inter, recipes),
        "rating_counts": rating_counts(inter),
        "top_items": top_items(recipes),
        "exploratory_targets": exploratory_targets(recipes, inter),
    }
    for name, df in outs.items():
        validate(df, EXPLORATION_SCHEMAS[name])
    assert outs["basic_counts"].iloc[0]["n_orphan_interactions"] == 0
