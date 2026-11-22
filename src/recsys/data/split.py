"""Interaction splits for stage 1.9 (RAW_recipes is never split: it is the catalogue).

Two :class:`~recsys.interfaces.Splitter` implementations, registered under kind ``"splitter"``:

* :class:`GroupStratifiedSplitter` (``"group_stratified"``): every recipe goes to exactly one of
  train / val / test with ALL its interactions; recipes are stratified by their rating class.
  Used for the recipe-level rating models (no recipe is seen in training and evaluated later).
* :class:`TemporalUserSplitter` (``"temporal_user"``): per user, the latest interactions are
  test and the ones before them validation. Used for personalisation.

Both return a :class:`~recsys.interfaces.SplitResult` whose frames keep the POSITIONAL row
number of the input interactions as index (input index is reset first), so a split can be
stored compactly as an assignment table (:func:`assignments_table`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from recsys.data.load import make_stratified_sample
from recsys.data.targets import stratification_labels
from recsys.interfaces import SplitResult
from recsys.registry import register

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


def _check_ratios(train: float, val: float, test: float) -> None:
    ratios = (train, val, test)
    if any(r <= 0 for r in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"train/val/test ratios must be > 0 and sum to 1, got {ratios}")


@register("splitter", "group_stratified")
class GroupStratifiedSplitter:
    """Split interactions by recipe (groups), stratified by the recipe rating class.

    Args:
        label_binner: Unfitted binner turning ``avg_rating`` into the class (same config as the
            target). Fitted here on ALL interactions, only to stratify (see
            :func:`recsys.data.targets.stratification_labels`).
        train / val / test: Shares of RECIPES per set (interaction shares follow from them).
        random_state: Seed of both shuffles.
        unrated_label: Class of recipes without any rating > 0.

    After :meth:`split`, ``labels_`` holds the stratification label of every recipe.
    """

    def __init__(
        self,
        label_binner: Any,
        train: float = 0.70,
        val: float = 0.15,
        test: float = 0.15,
        random_state: int = 42,
        unrated_label: str = "unrated",
    ) -> None:
        self.label_binner = label_binner
        self.train = train
        self.val = val
        self.test = test
        self.random_state = random_state
        self.unrated_label = unrated_label

    def split(self, interactions: pd.DataFrame, recipes: pd.DataFrame) -> SplitResult:
        _check_ratios(self.train, self.val, self.test)
        df = interactions.reset_index(drop=True)
        labels = stratification_labels(df, self.label_binner, self.unrated_label)
        ids, y = labels["recipe_id"].to_numpy(), labels["label"].to_numpy()
        rest_ids, test_ids, rest_y, _ = train_test_split(
            ids, y, test_size=self.test, stratify=y, random_state=self.random_state
        )
        train_ids, val_ids = train_test_split(
            rest_ids,
            test_size=self.val / (self.train + self.val),
            stratify=rest_y,
            random_state=self.random_state,
        )
        where = pd.Series("train", index=ids)
        where.loc[val_ids] = "val"
        where.loc[test_ids] = "test"
        labels["split"] = labels["recipe_id"].map(where)
        self.labels_ = labels
        assigned = df["recipe_id"].map(where)
        return SplitResult(
            train=df[assigned == "train"],
            val=df[assigned == "val"],
            test=df[assigned == "test"],
            strategy="group_stratified",
            params={
                "train": self.train,
                "val": self.val,
                "test": self.test,
                "random_state": self.random_state,
                "unrated_label": self.unrated_label,
                "unit": "recipe",
                "n_catalogue_recipes": int(len(recipes)),
                "n_recipes_without_interactions": int(
                    (~recipes["id"].isin(labels["recipe_id"])).sum()
                ),
            },
        )


@register("splitter", "temporal_user")
class TemporalUserSplitter:
    """Per-user temporal split: the latest ``n_test`` interactions -> test, the ``n_val`` before
    them -> val, the rest -> train.

    Users with fewer than ``min_interactions`` interactions stay entirely in train (too short a
    history to evaluate). Order: ``date``, then ``recipe_id`` (deterministic on equal dates).
    """

    def __init__(
        self,
        min_interactions: int = 5,
        n_val: int = 1,
        n_test: int = 1,
        user_col: str = "user_id",
        date_col: str = "date",
    ) -> None:
        self.min_interactions = min_interactions
        self.n_val = n_val
        self.n_test = n_test
        self.user_col = user_col
        self.date_col = date_col

    def split(self, interactions: pd.DataFrame, recipes: pd.DataFrame) -> SplitResult:
        if self.n_val < 1 or self.n_test < 1:
            raise ValueError("n_val and n_test must be >= 1")
        if self.min_interactions < self.n_val + self.n_test + 1:
            raise ValueError("min_interactions must leave at least one train interaction")
        df = interactions.reset_index(drop=True)
        order = df.sort_values([self.user_col, self.date_col, "recipe_id"], kind="mergesort")
        grp = order.groupby(self.user_col, sort=False)
        from_end = grp.cumcount(ascending=False)  # 0 = latest interaction of the user
        eligible = grp[self.user_col].transform("size") >= self.min_interactions
        where = pd.Series("train", index=order.index)
        where[eligible & (from_end < self.n_test)] = "test"
        where[eligible & (from_end >= self.n_test) & (from_end < self.n_test + self.n_val)] = "val"
        where = where.reindex(df.index)
        return SplitResult(
            train=df[where == "train"],
            val=df[where == "val"],
            test=df[where == "test"],
            strategy="temporal_user",
            params={
                "min_interactions": self.min_interactions,
                "n_val": self.n_val,
                "n_test": self.n_test,
                "n_eligible_users": int(order.loc[eligible, self.user_col].nunique()),
                "n_users": int(df[self.user_col].nunique()),
            },
        )


def assignments_table(result: SplitResult) -> pd.DataFrame:
    """Compact form of a split: one row per interaction with its set.

    Returns:
        Columns row (positional row of the split input), user_id, recipe_id, split; sorted by row.
    """
    parts = [
        pd.DataFrame(
            {
                "row": frame.index.to_numpy(dtype="int64"),
                "user_id": frame["user_id"].to_numpy(dtype="int64"),
                "recipe_id": frame["recipe_id"].to_numpy(dtype="int64"),
                "split": name,
            }
        )
        for name, frame in result.as_dict().items()
    ]
    return pd.concat(parts, ignore_index=True).sort_values("row").reset_index(drop=True)


def apply_assignments(
    interactions: pd.DataFrame, assignments: pd.DataFrame, strategy: str
) -> SplitResult:
    """Rebuild a :class:`SplitResult` from a stored assignment table (inverse of
    :func:`assignments_table`). ``interactions`` must be the same table the split was made on.
    """
    df = interactions.reset_index(drop=True)
    if len(df) != len(assignments):
        raise ValueError(f"{len(df)} interactions but {len(assignments)} assignments")
    a = assignments.set_index("row")
    if not (a.loc[df.index, "recipe_id"].to_numpy() == df["recipe_id"].to_numpy()).all():
        raise ValueError("assignments do not match these interactions (recipe_id differs)")
    where = a.loc[df.index, "split"]
    frames = {name: df[where.to_numpy() == name] for name in SPLIT_NAMES}
    return SplitResult(**frames, strategy=strategy)


def split_summary(results: Mapping[str, SplitResult]) -> pd.DataFrame:
    """Rows, recipes and users per set of each strategy.

    Returns:
        Columns strategy, split, n_interactions, share_interactions, n_recipes, n_users,
        n_rated (rating > 0).
    """
    rows = []
    for strategy, res in results.items():
        total = sum(len(f) for f in res.as_dict().values())
        for name, f in res.as_dict().items():
            r = f["rating"].astype("float64")
            rows.append(
                {
                    "strategy": strategy,
                    "split": name,
                    "n_interactions": int(len(f)),
                    "share_interactions": len(f) / total if total else np.nan,
                    "n_recipes": int(f["recipe_id"].nunique()),
                    "n_users": int(f["user_id"].nunique()),
                    "n_rated": int((r.notna() & (r > 0)).sum()),
                }
            )
    return pd.DataFrame(rows)


def class_balance_table(labels: pd.DataFrame) -> pd.DataFrame:
    """Share of each stratification label per set (labels with a ``split`` column).

    Returns:
        Columns split, label, n_recipes, share; one row per (split, label).
    """
    counts = labels.groupby(["split", "label"]).size().rename("n_recipes").reset_index()
    counts["share"] = counts["n_recipes"] / counts.groupby("split")["n_recipes"].transform("sum")
    counts["n_recipes"] = counts["n_recipes"].astype("int64")
    return counts.sort_values(["split", "label"]).reset_index(drop=True)


def make_dev_sample(
    recipes: pd.DataFrame,
    interactions: pd.DataFrame,
    labels: pd.DataFrame,
    n_recipes: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Development sample: ``n_recipes`` rated-or-not recipes stratified on their split label.

    Only recipes that have interactions (i.e. a label) are sampled; interactions are all those of
    the sampled recipes. Because the grouped split assigns whole recipes, restricting a grouped
    split to the sample gives a valid split of the sample.

    Returns:
        ``(dev_recipes with a 'label' column, dev_interactions)``.
    """
    labelled = recipes.merge(labels[["recipe_id", "label"]], left_on="id", right_on="recipe_id")
    labelled = labelled.drop(columns="recipe_id")
    return make_stratified_sample(
        labelled, interactions, n_recipes, stratify_col="label", random_state=random_state
    )
