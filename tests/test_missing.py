"""Tests for recsys.cleaning.missing (stage 1.3).

Expected values are derived by hand from tests/fixtures/build_fixtures.py.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from recsys import registry
from recsys.analysis.quality import detect_issues
from recsys.cleaning.missing import (
    DEFAULT_NO_COOK_TAGS,
    GroupMedianImputer,
    RatingZeroToNaN,
    ZeroAsMissing,
    assign_main_tag,
    main_tag_coverage,
    mark_has_rating,
    minutes_zero_examples,
    minutes_zero_n_steps,
    minutes_zero_summary,
    minutes_zero_tag_lift,
    missing_rates,
    rating_zero_examples,
    rating_zero_review_stats,
)
from recsys.config import REPO_ROOT, Config
from recsys.contracts import (
    INTERACTIONS_AFTER_MISSING_SCHEMA,
    MISSING_OUTPUT_SCHEMA,
    RECIPES_AFTER_MISSING_SCHEMA,
    validate,
)
from recsys.interfaces import Transformer
from recsys.io import ArtifactStore

NOTEBOOK = REPO_ROOT / "notebooks" / "03_missing.ipynb"


# ----------------------------------------------------------------- analysis tables


def test_missing_rates_on_fixtures(fixture_recipes, fixture_interactions) -> None:
    t = missing_rates(fixture_recipes, fixture_interactions)
    validate(t, MISSING_OUTPUT_SCHEMA["missing_rates"])
    r = t.set_index(["table", "column", "kind"])["n_missing"]
    assert r[("recipes", "name", "null")] == 1  # recipe 108
    assert r[("recipes", "description", "null")] == 1  # recipe 109
    assert r[("recipes", "steps", "empty_list")] == 1  # recipe 110
    assert r[("recipes", "tags", "empty_list")] == 0
    assert r[("interactions", "review", "null")] == 1  # 1011 x 116
    assert r[("interactions", "rating", "zero")] == 4
    assert r[("recipes", "minutes", "zero")] == 2  # 103, 105
    assert t.loc[t["kind"] == "zero", "share"].tolist() == pytest.approx([4 / 62, 2 / 20])


def test_rating_zero_rows_come_with_review_text(fixture_interactions) -> None:
    s = rating_zero_review_stats(fixture_interactions)
    validate(s, MISSING_OUTPUT_SCHEMA["rating_zero_review_stats"])
    row = s.iloc[0]
    assert row["n_zero"] == 4
    assert row["n_zero_with_review"] == 4
    assert row["share_zero_with_review"] == 1.0
    assert row["share_rated_with_review"] == pytest.approx(57 / 58)  # one rated row lacks review


def test_rating_zero_examples_are_reproducible(fixture_interactions) -> None:
    a = rating_zero_examples(fixture_interactions, n=3, random_state=0)
    b = rating_zero_examples(fixture_interactions, n=3, random_state=0)
    pd.testing.assert_frame_equal(a, b)
    validate(a, MISSING_OUTPUT_SCHEMA["rating_zero_examples"])
    assert len(a) == 3 and len(rating_zero_examples(fixture_interactions, n=99)) == 4


def test_minutes_zero_summary(fixture_recipes) -> None:
    s = minutes_zero_summary(fixture_recipes)
    validate(s, MISSING_OUTPUT_SCHEMA["minutes_zero_summary"])
    row = s.iloc[0]
    assert row["n_minutes_zero"] == 2
    assert row["n_zero_with_no_cook_tag"] == 1  # 103 (beverages, no-cook); 105 is a stew
    assert row["share_all_with_no_cook_tag"] == pytest.approx(4 / 20)  # 103, 104, 106, 115
    assert row["median_n_steps_zero"] == 2.0  # n_steps 1 and 3


def test_minutes_zero_tag_lift(fixture_recipes) -> None:
    t = minutes_zero_tag_lift(fixture_recipes)
    validate(t, MISSING_OUTPUT_SCHEMA["minutes_zero_tags"])
    assert t["tag"].tolist() == ["beef", "beverages", "easy", "main-dish", "no-cook", "stews"]
    bev = t.set_index("tag").loc["beverages"]
    assert bev["share_in_zero"] == 0.5 and bev["share_in_all"] == pytest.approx(0.1)
    assert bev["lift"] == pytest.approx(5.0)


def test_minutes_zero_tag_lift_without_zeros(fixture_recipes) -> None:
    no_zero = fixture_recipes[fixture_recipes["minutes"] != 0]
    t = minutes_zero_tag_lift(no_zero)
    assert t.empty and list(t.columns) == ["tag", "n_zero", "share_in_zero", "share_in_all", "lift"]


def test_minutes_zero_n_steps_and_examples(fixture_recipes) -> None:
    d = minutes_zero_n_steps(fixture_recipes)
    validate(d, MISSING_OUTPUT_SCHEMA["minutes_zero_n_steps"])
    assert d["n_zero"].sum() == 2 and d["n_all"].sum() == 20
    assert d.set_index("n_steps").loc[1, "n_zero"] == 1
    e = minutes_zero_examples(fixture_recipes)
    validate(e, MISSING_OUTPUT_SCHEMA["minutes_zero_examples"])
    assert set(e["id"]) == {103, 105}


def test_main_tag_assignment_and_coverage(fixture_recipes) -> None:
    tags = assign_main_tag(fixture_recipes["tags"]).set_axis(fixture_recipes["id"])
    assert tags[101] == "main-dish"
    assert tags[118] == "desserts"  # breads, baking, desserts -> desserts has priority
    assert tags[115] == "other"  # dips, mexican, no-cook
    c = main_tag_coverage(fixture_recipes)
    validate(c, MISSING_OUTPUT_SCHEMA["main_tag_coverage"])
    counts = c.set_index("main_tag")["n_recipes"]
    assert counts["main-dish"] == 7 and counts["desserts"] == 3 and counts["other"] == 1
    assert counts.sum() == 20


def test_mark_has_rating(fixture_recipes, fixture_interactions) -> None:
    flagged = mark_has_rating(fixture_recipes, fixture_interactions).set_index("id")
    assert not flagged.loc[111, "has_rating"]  # no interactions at all
    assert flagged["has_rating"].sum() == 19
    after = RatingZeroToNaN().fit_transform(fixture_interactions)
    again = mark_has_rating(fixture_recipes, after).set_index("id")
    pd.testing.assert_series_equal(again["has_rating"], flagged["has_rating"])
    assert "has_rating" not in fixture_recipes.columns  # input untouched


# ----------------------------------------------------------------- RatingZeroToNaN


def test_rating_zero_to_nan(fixture_interactions) -> None:
    t = RatingZeroToNaN()
    assert isinstance(t, Transformer)
    out = t.fit(fixture_interactions).transform(fixture_interactions)
    assert out["rating"].isna().sum() == 4
    assert out["rating"].dtype == "float64"
    assert (out["rating"].dropna() >= 1).all()
    assert (fixture_interactions["rating"] == 0).sum() == 4  # input untouched
    validate(out, INTERACTIONS_AFTER_MISSING_SCHEMA)


def test_rating_zero_to_nan_requires_fit(fixture_interactions) -> None:
    with pytest.raises(NotFittedError):
        RatingZeroToNaN().transform(fixture_interactions)


# ----------------------------------------------------------------- ZeroAsMissing


def test_zero_as_missing_keeps_no_cook_zeros(fixture_recipes) -> None:
    t = ZeroAsMissing(columns=("minutes",), keep_if_tags=DEFAULT_NO_COOK_TAGS)
    out = t.fit_transform(fixture_recipes).set_index("id")
    assert out.loc[103, "minutes"] == 0.0  # no-cook beverage: genuine zero
    assert np.isnan(out.loc[105, "minutes"])  # stew with 0 minutes: missing
    assert out["minutes"].isna().sum() == 1


def test_zero_as_missing_without_keep_tags(fixture_recipes) -> None:
    out = ZeroAsMissing(columns=("minutes",)).fit_transform(fixture_recipes)
    assert out["minutes"].isna().sum() == 2


# ----------------------------------------------------------------- GroupMedianImputer


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(1, 9),
            "tags": [["main-dish"]] * 4 + [["desserts"]] * 4,
            "n_steps": [3] * 8,
            "minutes": [10.0, 20.0, 30.0, np.nan, 100.0, 200.0, 300.0, 400.0],
        }
    )


def _test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [11, 12, 13, 14],
            "tags": [["main-dish"], ["desserts"], ["salads"], ["main-dish"]],
            "n_steps": [3, 3, 3, 3],
            "minutes": [np.nan, np.nan, np.nan, 999.0],
        }
    )


def test_group_median_learned_on_fit_data_only() -> None:
    imp = GroupMedianImputer(n_steps_bins=1, min_group_size=3).fit(_train())
    assert imp.group_medians_ == {"main-dish|0": 20.0, "desserts|0": 250.0}
    assert imp.global_median_ == 100.0  # median of the 7 observed train values
    before = (dict(imp.group_medians_), imp.global_median_, imp.n_steps_edges_.copy())
    out = imp.transform(_test())
    assert out["minutes"].tolist() == [20.0, 250.0, 100.0, 999.0]  # salads unseen -> global
    assert out["minutes_was_missing"].tolist() == [True, True, True, False]
    after = (dict(imp.group_medians_), imp.global_median_, imp.n_steps_edges_)
    assert before[0] == after[0] and before[1] == after[1]
    np.testing.assert_array_equal(before[2], after[2])  # transform never re-learns


def test_small_groups_fall_back_to_global_median() -> None:
    imp = GroupMedianImputer(n_steps_bins=1, min_group_size=4).fit(_train())
    assert imp.group_medians_ == {"desserts|0": 250.0}  # main-dish has only 3 observed values
    out = imp.transform(_test())
    assert out["minutes"].iloc[0] == 100.0
    assert imp.fallback_share(_test()) == pytest.approx(2 / 3)
    table = imp.group_table_.set_index("group")
    assert not table.loc["main-dish|0", "used"] and table.loc["desserts|0", "used"]


def test_n_steps_edges_come_from_fit_data() -> None:
    train = _train().assign(n_steps=[1, 2, 3, 4, 5, 6, 7, 8])
    imp = GroupMedianImputer(n_steps_bins=2, min_group_size=1).fit(train)
    np.testing.assert_array_equal(imp.n_steps_edges_, [4.5])
    shifted = _test().assign(n_steps=[100, 100, 100, 100])
    imp.transform(shifted)
    np.testing.assert_array_equal(imp.n_steps_edges_, [4.5])


def test_imputer_is_sklearn_compatible() -> None:
    imp = GroupMedianImputer(n_steps_bins=1, min_group_size=3, add_indicator=False)
    assert isinstance(imp, Transformer)
    with pytest.raises(NotFittedError):
        imp.transform(_test())
    copy = clone(imp)
    assert copy.get_params() == imp.get_params()
    out = imp.fit(_train()).transform(_test())
    assert "minutes_was_missing" not in out.columns
    assert _test()["minutes"].isna().sum() == 3  # input untouched


def test_imputer_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="no observed values"):
        GroupMedianImputer().fit(_train().assign(minutes=np.nan))
    with pytest.raises(KeyError, match="missing required columns"):
        GroupMedianImputer().fit(_train().drop(columns="tags"))


def test_transformers_available_through_registry() -> None:
    assert isinstance(registry.create("missing", "rating_zero_to_nan"), RatingZeroToNaN)
    assert isinstance(registry.create("missing", "zero_as_missing"), ZeroAsMissing)
    imp = registry.create("imputer", "group_median", n_steps_bins=2)
    assert isinstance(imp, GroupMedianImputer) and imp.n_steps_bins == 2


def test_full_chain_on_fixtures_matches_contract(fixture_recipes, fixture_interactions) -> None:
    """ZeroAsMissing -> GroupMedianImputer -> mark_has_rating gives RECIPES_AFTER_MISSING_SCHEMA."""
    pipe = Pipeline(
        [
            ("zero", ZeroAsMissing(columns=("minutes",), keep_if_tags=DEFAULT_NO_COOK_TAGS)),
            ("impute", GroupMedianImputer(n_steps_bins=2, min_group_size=1)),
        ]
    )
    recipes = pipe.fit_transform(fixture_recipes)
    recipes = mark_has_rating(recipes, RatingZeroToNaN().fit_transform(fixture_interactions))
    validate(recipes, RECIPES_AFTER_MISSING_SCHEMA)
    assert recipes.loc[recipes["minutes_was_missing"], "id"].tolist() == [105]
    assert recipes["minutes"].notna().all()


# ----------------------------------------------------------------- notebook


def _code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_starts_with_colab_cell_and_has_no_logic() -> None:
    cells = _code_cells()
    for needle in ("drive.mount", "git clone", "pip install -q -e .", "DATA_ROOT", "load_config()"):
        assert needle in cells[0]
    assert "SecretNotFoundError" in cells[0]  # works without a GH_TOKEN secret
    for src in cells[1:]:
        assert "def " not in src and "class " not in src


def test_notebook_runs_on_fixtures(cfg: Config, fixture_recipes, fixture_interactions) -> None:
    store = ArtifactStore.from_config(cfg)
    store.save("loaded", {"recipes": fixture_recipes, "interactions": fixture_interactions})
    store.save("quality", {"issues": detect_issues(fixture_recipes, fixture_interactions)})
    ns: dict[str, object] = {"CFG": cfg, "display": lambda *a, **k: None}
    for src in _code_cells()[1:]:
        exec(compile(src, str(NOTEBOOK), "exec"), ns)  # noqa: S102 - our own notebook
    art = store.load("missing")
    assert art.version == "v1"
    assert set(art.keys()) == set(MISSING_OUTPUT_SCHEMA)
    assert [i["stage"] for i in art.manifest["inputs"]] == ["loaded", "quality"]
    assert {f.get("schema") for f in art.manifest["files"].values()} == {
        s.name for s in MISSING_OUTPUT_SCHEMA.values()
    }
