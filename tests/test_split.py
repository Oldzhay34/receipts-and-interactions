"""Tests for recsys.data.split, recsys.data.targets and the 1.9 split contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys import registry
from recsys.contracts import (
    SPLITS_SCHEMA,
    STRATIFICATION_LABELS_SCHEMA,
    SchemaValidationError,
    assert_class_proportions,
    assert_disjoint,
    assert_rows_preserved,
    assert_temporal_order,
    validate,
)
from recsys.data.split import (
    GroupStratifiedSplitter,
    TemporalUserSplitter,
    apply_assignments,
    assignments_table,
    class_balance_table,
    make_dev_sample,
    split_summary,
)
from recsys.data.targets import (
    RecipeTargetBuilder,
    compute_recipe_targets,
    recipe_rating_stats,
    stratification_labels,
)
from recsys.interfaces import Splitter
from recsys.io.artifact_store import read_artifact_dir

from .conftest import require_sample

EDGES = [0.0, 3.5, 4.5, 5.0]
LABELS = ["low", "medium", "high"]


def _binner(name: str = "domain"):
    params = {"column": "avg_rating", "labels": LABELS, "output_col": "rating_class"}
    if name == "domain":
        params["edges"] = EDGES
    else:
        params["n_bins"] = 3
    return registry.create("binner", name, **params)


def _synthetic(n_recipes: int = 400, n_users: int = 80, seed: int = 0) -> tuple:
    """Interactions with skewed ratings (mostly 5), some 0s, dates and repeated users."""
    rng = np.random.default_rng(seed)
    rows = []
    for rid in range(1, n_recipes + 1):
        k = int(rng.integers(1, 8))
        users = rng.choice(n_users, size=k, replace=False) + 1000
        quality = rng.choice([1, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.4])
        for u in users:
            rating = 0 if rng.random() < 0.05 else int(np.clip(quality + rng.integers(-1, 2), 1, 5))
            day = pd.Timestamp("2010-01-01") + pd.Timedelta(days=int(rng.integers(0, 3000)))
            rows.append((int(u), rid, day, rating, "text"))
    inter = pd.DataFrame(rows, columns=["user_id", "recipe_id", "date", "rating", "review"])
    recipes = pd.DataFrame({"id": np.arange(1, n_recipes + 6)})  # 5 recipes never reviewed
    return inter, recipes


# ----------------------------------------------------------------- targets


def test_recipe_rating_stats_ignores_zero_and_nan() -> None:
    inter = pd.DataFrame({"recipe_id": [1, 1, 1, 2, 3], "rating": [5, 0, 3, np.nan, 0]})
    stats = recipe_rating_stats(inter)
    assert stats["recipe_id"].tolist() == [1]  # 2 and 3 have no rating > 0
    assert stats["avg_rating"].tolist() == [4.0] and stats["n_ratings"].tolist() == [2]


def test_class_boundaries_come_from_train_only() -> None:
    train = pd.DataFrame({"recipe_id": range(9), "rating": [1, 1, 2, 3, 3, 4, 4, 5, 5]})
    val = pd.DataFrame({"recipe_id": [100, 101], "rating": [5, 1]})
    targets, builder = compute_recipe_targets(train, _binner("equal_frequency"))
    edges = builder.binner_.edges_.copy()
    assert set(targets.columns) == {"recipe_id", "avg_rating", "n_ratings", "rating_class"}

    builder.transform(val)  # applying to val never refits
    np.testing.assert_array_equal(builder.binner_.edges_, edges)
    # changing val ratings does not move the boundaries; refitting on other data does
    other = RecipeTargetBuilder(_binner("equal_frequency")).fit(val)
    assert not np.array_equal(other.binner_.edges_, edges)
    val_targets = builder.transform(val).set_index("recipe_id")["rating_class"]
    assert val_targets.loc[100] == "high" and val_targets.loc[101] == "low"


def test_target_builder_needs_a_binner_and_rated_rows() -> None:
    with pytest.raises(ValueError, match="binner"):
        RecipeTargetBuilder().fit(pd.DataFrame({"recipe_id": [1], "rating": [5]}))
    with pytest.raises(ValueError, match="no rated"):
        RecipeTargetBuilder(_binner()).fit(pd.DataFrame({"recipe_id": [1], "rating": [0]}))


def test_stratification_labels_cover_unrated_recipes() -> None:
    inter = pd.DataFrame({"recipe_id": [1, 1, 2, 3], "rating": [5, 5, 0, 2]})
    labels = stratification_labels(inter, _binner(), unrated_label="unrated")
    assert labels.set_index("recipe_id")["label"].to_dict() == {1: "high", 2: "unrated", 3: "low"}


# ----------------------------------------------------------------- grouped split


def test_splitters_implement_the_protocol_and_are_registered() -> None:
    assert isinstance(GroupStratifiedSplitter(_binner()), Splitter)
    assert isinstance(TemporalUserSplitter(), Splitter)
    assert registry.get("splitter", "group_stratified") is GroupStratifiedSplitter
    assert registry.get("splitter", "temporal_user") is TemporalUserSplitter


def test_grouped_split_contracts() -> None:
    inter, recipes = _synthetic()
    splitter = GroupStratifiedSplitter(_binner(), random_state=42)
    res = splitter.split(inter, recipes)
    frames = res.as_dict()

    assert_rows_preserved(inter.reset_index(drop=True), frames)
    assert_disjoint(frames, "recipe_id")  # all interactions of a recipe in one set
    assert_class_proportions(splitter.labels_, tolerance=0.02)
    validate(splitter.labels_, STRATIFICATION_LABELS_SCHEMA)
    validate(assignments_table(res), SPLITS_SCHEMA)

    n = inter["recipe_id"].nunique()
    shares = splitter.labels_["split"].value_counts() / n
    assert shares["train"] == pytest.approx(0.70, abs=0.01)
    assert shares["val"] == pytest.approx(0.15, abs=0.01)
    assert shares["test"] == pytest.approx(0.15, abs=0.01)
    assert "unrated" in set(splitter.labels_["label"])
    assert res.params["n_recipes_without_interactions"] == 5


def test_grouped_split_is_reproducible() -> None:
    inter, recipes = _synthetic()
    a = assignments_table(GroupStratifiedSplitter(_binner(), random_state=42).split(inter, recipes))
    b = assignments_table(GroupStratifiedSplitter(_binner(), random_state=42).split(inter, recipes))
    c = assignments_table(GroupStratifiedSplitter(_binner(), random_state=7).split(inter, recipes))
    pd.testing.assert_frame_equal(a, b)
    assert not a["split"].equals(c["split"])


def test_grouped_split_rejects_bad_ratios() -> None:
    inter, recipes = _synthetic()
    with pytest.raises(ValueError, match="sum to 1"):
        GroupStratifiedSplitter(_binner(), train=0.8, val=0.15, test=0.15).split(inter, recipes)


# ----------------------------------------------------------------- temporal split


def test_temporal_split_contracts() -> None:
    inter, recipes = _synthetic()
    res = TemporalUserSplitter(min_interactions=5, n_val=1, n_test=1).split(inter, recipes)
    frames = res.as_dict()
    assert_rows_preserved(inter.reset_index(drop=True), frames)
    assert_temporal_order(frames)

    sizes = inter.groupby("user_id").size()
    eligible = sizes[sizes >= 5].index
    assert set(res.test["user_id"]) == set(eligible) == set(res.val["user_id"])
    assert res.test.groupby("user_id").size().eq(1).all()
    short = sizes[sizes < 5].index
    assert set(res.train["user_id"]) >= set(short)  # short histories stay in train
    # the test row is each user's latest interaction
    latest = inter.groupby("user_id")["date"].max()
    assert (res.test.set_index("user_id")["date"] == latest.loc[res.test["user_id"]]).all()


def test_temporal_split_breaks_date_ties_by_recipe_id() -> None:
    day = pd.Timestamp("2015-05-05")
    inter = pd.DataFrame(
        {"user_id": [1] * 5, "recipe_id": [50, 10, 40, 20, 30], "date": [day] * 5, "rating": 5}
    )
    res = TemporalUserSplitter(min_interactions=5).split(inter, pd.DataFrame({"id": []}))
    assert res.test["recipe_id"].tolist() == [50] and res.val["recipe_id"].tolist() == [40]


@pytest.mark.parametrize("kw", [{"n_val": 0}, {"min_interactions": 2}])
def test_temporal_split_rejects_bad_params(kw) -> None:
    inter, recipes = _synthetic()
    with pytest.raises(ValueError):
        TemporalUserSplitter(**kw).split(inter, recipes)


# ----------------------------------------------------------------- assertions catch violations


def test_assertions_detect_violations() -> None:
    inter, recipes = _synthetic()
    res = GroupStratifiedSplitter(_binner()).split(inter, recipes)
    full = inter.reset_index(drop=True)
    with pytest.raises(SchemaValidationError, match="in no set"):
        assert_rows_preserved(full, {"train": res.train.iloc[1:], "val": res.val, "test": res.test})
    with pytest.raises(SchemaValidationError, match="more than one set"):
        assert_rows_preserved(full, {"train": res.train, "val": res.val, "test": res.train})
    leaked = pd.concat([res.val, res.train.head(1)])
    with pytest.raises(SchemaValidationError, match="recipe_id"):
        assert_disjoint({"train": res.train, "val": leaked}, "recipe_id")
    with pytest.raises(SchemaValidationError, match="after their first"):
        assert_temporal_order(res.as_dict())  # a grouped split is not temporal
    skewed = pd.DataFrame(
        {"split": ["train"] * 4 + ["val"] * 4, "label": list("aaab") + list("abbb")}
    )
    with pytest.raises(SchemaValidationError, match="share differs"):
        assert_class_proportions(skewed, tolerance=0.1)


def test_assignments_round_trip() -> None:
    inter, recipes = _synthetic()
    res = TemporalUserSplitter().split(inter, recipes)
    back = apply_assignments(inter, assignments_table(res), "temporal_user")
    for name in ("train", "val", "test"):
        pd.testing.assert_frame_equal(back.as_dict()[name], res.as_dict()[name])
    with pytest.raises(ValueError):
        apply_assignments(inter.iloc[1:], assignments_table(res), "x")


def test_summary_tables() -> None:
    inter, recipes = _synthetic()
    splitter = GroupStratifiedSplitter(_binner())
    res = splitter.split(inter, recipes)
    summary = split_summary({"group_stratified": res})
    assert summary["n_interactions"].sum() == len(inter)
    assert summary["share_interactions"].sum() == pytest.approx(1.0)
    balance = class_balance_table(splitter.labels_)
    assert balance.groupby("split")["share"].sum().round(10).eq(1.0).all()


# ----------------------------------------------------------------- dev sample


def test_dev_sample_is_stratified_and_keeps_interactions() -> None:
    inter, recipes = _synthetic(n_recipes=600)
    labels = stratification_labels(inter, _binner())
    dev_r, dev_i = make_dev_sample(recipes, inter, labels, n_recipes=150)
    assert len(dev_r) == 150 and dev_r["id"].is_unique
    full_share = labels["label"].value_counts(normalize=True)
    dev_share = dev_r["label"].value_counts(normalize=True)
    assert (dev_share - full_share).abs().max() < 0.01
    assert set(dev_i["recipe_id"]) <= set(dev_r["id"])
    assert len(dev_i) == inter["recipe_id"].isin(dev_r["id"]).sum()


# ----------------------------------------------------------------- real sample


def test_both_splits_on_real_sample() -> None:
    art = read_artifact_dir(require_sample(), names=["recipes", "interactions"])
    recipes, inter = art["recipes"], art["interactions"]
    splitter = GroupStratifiedSplitter(_binner())
    grouped = splitter.split(inter, recipes)
    temporal = TemporalUserSplitter().split(inter, recipes)
    full = inter.reset_index(drop=True)
    assert_rows_preserved(full, grouped.as_dict())
    assert_rows_preserved(full, temporal.as_dict())
    assert_disjoint(grouped.as_dict(), "recipe_id")
    assert_temporal_order(temporal.as_dict())
    assert_class_proportions(splitter.labels_, tolerance=0.01)
