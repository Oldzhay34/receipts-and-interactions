from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas.api import types as ptypes

from recsys.contracts import (
    LOADED_INTERACTIONS_SCHEMA,
    LOADED_RECIPES_SCHEMA,
    NUTRITION_COLUMNS,
    SchemaValidationError,
    validate,
)
from recsys.data.load import (
    load_interactions,
    load_recipes,
    make_stratified_sample,
    parse_list_column,
    parse_nutrition,
    read_raw_recipes,
    stratification_table,
)

from .conftest import FIXTURES

# ----------------------------------------------------------------- structure


def test_recipes_match_loaded_schema(fixture_recipes: pd.DataFrame) -> None:
    validate(fixture_recipes, LOADED_RECIPES_SCHEMA)
    assert fixture_recipes.shape == (20, len(LOADED_RECIPES_SCHEMA.columns))
    assert list(fixture_recipes.columns) == list(LOADED_RECIPES_SCHEMA.columns)
    assert "nutrition" not in fixture_recipes.columns


def test_interactions_match_loaded_schema(fixture_interactions: pd.DataFrame) -> None:
    validate(fixture_interactions, LOADED_INTERACTIONS_SCHEMA)
    assert len(fixture_interactions) == 62
    assert fixture_interactions["rating"].dtype == "int8"


def test_list_columns_are_python_lists(fixture_recipes: pd.DataFrame) -> None:
    r = fixture_recipes.set_index("id")
    assert r.loc[101, "ingredients"] == [
        "spaghetti",
        "tomatoes",
        "garlic",
        "olive oil",
        "basil",
        "salt",
    ]
    assert r.loc[103, "tags"] == ["beverages", "no-cook", "easy"]
    assert r.loc[110, "steps"] == []
    for col in ("tags", "steps", "ingredients"):
        assert fixture_recipes[col].map(type).eq(list).all()


def test_nutrition_split_into_seven_floats(fixture_recipes: pd.DataFrame) -> None:
    r = fixture_recipes.set_index("id")
    assert tuple(NUTRITION_COLUMNS) == (
        "calories",
        "total_fat_pdv",
        "sugar_pdv",
        "sodium_pdv",
        "protein_pdv",
        "saturated_fat_pdv",
        "carbohydrates_pdv",
    )
    assert r.loc[101, list(NUTRITION_COLUMNS)].tolist() == [
        410.2,
        18.0,
        30.0,
        22.0,
        25.0,
        9.0,
        21.0,
    ]
    assert all(ptypes.is_float_dtype(fixture_recipes[c]) for c in NUTRITION_COLUMNS)


def test_dates_are_datetimes(fixture_recipes: pd.DataFrame, fixture_interactions) -> None:
    assert ptypes.is_datetime64_any_dtype(fixture_recipes["submitted"])
    assert ptypes.is_datetime64_any_dtype(fixture_interactions["date"])
    r = fixture_recipes.set_index("id")
    assert r.loc[101, "submitted"] == pd.Timestamp("2003-02-11")


def test_integers_downcast_without_overflow(fixture_recipes: pd.DataFrame) -> None:
    # 2147483647 (int32 max) must survive downcasting exactly.
    assert fixture_recipes["minutes"].max() == 2147483647
    assert fixture_recipes["minutes"].dtype.itemsize <= 4
    assert fixture_recipes["id"].dtype.itemsize <= 2


# ----------------------------------------------------------------- edge cases are present
# (loading is faithful: none of these are dropped or "fixed" here)


def test_edge_cases_survive_loading(fixture_recipes: pd.DataFrame, fixture_interactions) -> None:
    r = fixture_recipes.set_index("id")
    i = fixture_interactions
    assert set(r.index[r["minutes"] == 0]) == {103, 105}
    assert r.loc[106, "minutes"] == 2147483647 and r.loc[107, "minutes"] == 1051200
    assert r.index[r["n_steps"] == 0].tolist() == [110]
    assert r["name"].isna().sum() == 1 and pd.isna(r.loc[108, "name"])
    assert r["description"].isna().sum() == 1 and pd.isna(r.loc[109, "description"])
    assert (i["rating"] == 0).sum() == 4
    assert i.duplicated().sum() == 1
    assert i[i.duplicated(keep=False)][["user_id", "recipe_id"]].iloc[0].tolist() == [1002, 101]
    assert 111 not in set(i["recipe_id"])
    per_user = i.groupby("user_id").size()
    assert per_user.loc[1015] == 1
    assert i["review"].isna().sum() == 1


def test_only_empty_fields_are_missing(tmp_path: Path) -> None:
    csv = (FIXTURES / "RAW_recipes.csv").read_text(encoding="utf-8")
    csv = csv.replace("classic spaghetti marinara", "None", 1)
    p = tmp_path / "r.csv"
    p.write_text(csv, encoding="utf-8")
    r = load_recipes(p).set_index("id")
    assert r.loc[101, "name"] == "None"  # literal text is kept, not turned into NaN


# ----------------------------------------------------------------- failures are loud


def test_raw_schema_rejects_missing_column(tmp_path: Path) -> None:
    df = pd.read_csv(FIXTURES / "RAW_recipes.csv").drop(columns="n_steps")
    p = tmp_path / "r.csv"
    df.to_csv(p, index=False)
    with pytest.raises(SchemaValidationError, match="missing columns: \\['n_steps'\\]"):
        read_raw_recipes(p)


def test_malformed_list_names_the_row() -> None:
    s = pd.Series(["['a', 'b']", "['oops'"], name="ingredients")
    with pytest.raises(ValueError, match="row id 7"):
        parse_list_column(s, ids=pd.Series([6, 7]))


def test_nutrition_wrong_length_names_the_row() -> None:
    s = pd.Series(["[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]", "[1.0, 2.0]"])
    with pytest.raises(ValueError, match="id 9"):
        parse_nutrition(s, ids=pd.Series([8, 9]))


def test_missing_nutrition_gives_nan_row() -> None:
    out = parse_nutrition(pd.Series(["[1, 2, 3, 4, 5, 6, 7]", None]))
    assert out.iloc[0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert out.iloc[1].isna().all()


def test_unparseable_date_raises(tmp_path: Path) -> None:
    csv = (FIXTURES / "RAW_interactions.csv").read_text(encoding="utf-8")
    first_date = csv.splitlines()[1].split(",")[2]
    p = tmp_path / "i.csv"
    p.write_text(csv.replace(first_date, "13/45/2002", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="unparseable dates"):
        load_interactions(p)


def test_out_of_domain_rating_rejected(tmp_path: Path) -> None:
    df = pd.read_csv(FIXTURES / "RAW_interactions.csv")
    df.loc[0, "rating"] = 7
    p = tmp_path / "i.csv"
    df.to_csv(p, index=False)
    with pytest.raises(SchemaValidationError, match="rating: 1 values outside allowed set"):
        load_interactions(p)


# ----------------------------------------------------------------- stratified sample


def test_sample_exact_size_deterministic_and_consistent(
    fixture_recipes: pd.DataFrame, fixture_interactions: pd.DataFrame
) -> None:
    r1, i1 = make_stratified_sample(fixture_recipes, fixture_interactions, 10, random_state=42)
    r2, i2 = make_stratified_sample(fixture_recipes, fixture_interactions, 10, random_state=42)
    assert len(r1) == 10
    pd.testing.assert_frame_equal(r1, r2)
    pd.testing.assert_frame_equal(i1, i2)
    assert set(i1["recipe_id"]) <= set(r1["id"])
    # every interaction of a sampled recipe is kept
    expected = fixture_interactions["recipe_id"].isin(r1["id"]).sum()
    assert len(i1) == expected


def test_sample_preserves_strata_within_one(fixture_recipes, fixture_interactions) -> None:
    n = 10
    rs, _ = make_stratified_sample(fixture_recipes, fixture_interactions, n, random_state=0)
    table = stratification_table(fixture_recipes, rs, "n_ingredients")
    exact = table["full_count"] * n / table["full_count"].sum()
    assert (table["sample_count"] - exact).abs().max() < 1
    assert table["sample_count"].sum() == n


def test_sample_larger_than_population_returns_all(fixture_recipes, fixture_interactions) -> None:
    rs, is_ = make_stratified_sample(fixture_recipes, fixture_interactions, 5000)
    assert len(rs) == len(fixture_recipes)
    assert len(is_) == len(fixture_interactions)
