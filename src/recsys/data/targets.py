"""Recipe-level targets for stage 1.9: ``avg_rating``, ``n_ratings`` and ``rating_class``.

Leakage rule: targets used for training are computed from TRAIN interactions only. Validation and
test targets are computed from their own interactions, but the class boundaries (the fitted
binner) always come from train (:class:`RecipeTargetBuilder`).

The only exception is :func:`stratification_labels`, which uses ALL interactions to give every
recipe a class before the split exists; those labels are used for stratifying the split and
nothing else.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from recsys.registry import register

TARGET_COLUMNS: tuple[str, ...] = ("recipe_id", "avg_rating", "n_ratings", "rating_class")


def recipe_rating_stats(interactions: pd.DataFrame, rating_col: str = "rating") -> pd.DataFrame:
    """Mean and count of the ratings > 0 of each recipe (rating 0 / NaN = no stars, ignored).

    Returns:
        Columns recipe_id, avg_rating, n_ratings; only recipes with at least one rating > 0,
        sorted by recipe_id.
    """
    r = interactions[rating_col].astype("float64")
    rated = interactions.loc[r.notna() & (r > 0), ["recipe_id", rating_col]]
    stats = (
        rated.groupby("recipe_id")[rating_col]
        .agg(avg_rating="mean", n_ratings="count")
        .reset_index()
        .sort_values("recipe_id", kind="mergesort")
        .reset_index(drop=True)
    )
    stats["recipe_id"] = stats["recipe_id"].astype("int64")
    stats["avg_rating"] = stats["avg_rating"].astype("float64")
    stats["n_ratings"] = stats["n_ratings"].astype("int64")
    return stats


@register("targets", "recipe")
class RecipeTargetBuilder(BaseEstimator, TransformerMixin):
    """Build recipe targets; the class binner is fitted on TRAIN interactions only.

    Args:
        binner: An unfitted binning transformer (e.g. ``DomainBins``) working on column
            ``avg_rating`` and writing ``class_col``. It is cloned in ``fit``.
        rating_col: Rating column of the interactions.
        class_col: Name of the class column written by the binner.

    Fitted attributes:
        binner_: The binner fitted on the train ``avg_rating`` values.
        n_fit_recipes_: Number of rated recipes the binner was fitted on.
    """

    def __init__(
        self, binner: Any = None, rating_col: str = "rating", class_col: str = "rating_class"
    ) -> None:
        self.binner = binner
        self.rating_col = rating_col
        self.class_col = class_col

    def fit(self, X: pd.DataFrame, y: object = None) -> RecipeTargetBuilder:
        if self.binner is None:
            raise ValueError("RecipeTargetBuilder needs a binner")
        stats = recipe_rating_stats(X, self.rating_col)
        if stats.empty:
            raise ValueError("no rated interactions to fit the target classes on")
        self.binner_ = clone(self.binner).fit(stats)
        self.n_fit_recipes_ = len(stats)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Targets of the recipes rated in ``X`` (classes from the train-fitted binner).

        Returns:
            Columns recipe_id, avg_rating, n_ratings, rating_class (str).
        """
        check_is_fitted(self, "binner_")
        stats = recipe_rating_stats(X, self.rating_col)
        out = self.binner_.transform(stats)
        out[self.class_col] = out[self.class_col].astype(str)
        return out[["recipe_id", "avg_rating", "n_ratings", self.class_col]].reset_index(drop=True)


def compute_recipe_targets(
    train_interactions: pd.DataFrame, binner: Any, rating_col: str = "rating"
) -> tuple[pd.DataFrame, RecipeTargetBuilder]:
    """Targets of the train recipes plus the fitted builder (reuse it for val / test).

    Returns:
        ``(train_targets, builder)``; ``builder.transform(val_interactions)`` gives the
        validation targets with the train class boundaries.
    """
    builder = RecipeTargetBuilder(binner=binner, rating_col=rating_col).fit(train_interactions)
    return builder.transform(train_interactions), builder


def stratification_labels(
    interactions: pd.DataFrame,
    binner: Any,
    unrated_label: str = "unrated",
    rating_col: str = "rating",
) -> pd.DataFrame:
    """One class per recipe of ``interactions``, computed from ALL of them (split stratification).

    Recipes without any rating > 0 get ``unrated_label`` so they are spread over the sets too.
    These labels are only used to stratify the split, never as a training target.

    Returns:
        Columns recipe_id, label; one row per recipe in ``interactions``, sorted by recipe_id.
    """
    builder = RecipeTargetBuilder(binner=binner, rating_col=rating_col).fit(interactions)
    rated = builder.transform(interactions).set_index("recipe_id")[builder.class_col]
    ids = np.sort(interactions["recipe_id"].unique())
    labels = pd.Series(ids, name="recipe_id").map(rated).fillna(unrated_label).astype(str)
    return pd.DataFrame({"recipe_id": ids.astype("int64"), "label": labels.to_numpy()})
