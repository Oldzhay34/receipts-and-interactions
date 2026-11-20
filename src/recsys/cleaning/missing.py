"""Missing-value handling for stage 1.3.

Two kinds of public objects:

* **Analysis functions** (pure: DataFrame in, DataFrame out, no plotting) producing the evidence
  tables of the ``missing`` stage: missing rates and the rating = 0 and minutes = 0 profiles.
* **Transformers** (sklearn ``BaseEstimator`` + ``TransformerMixin``, registered by name):
  ``RatingZeroToNaN``, ``ZeroAsMissing`` and ``GroupMedianImputer``. Anything they learn is learned
  in ``fit`` and stored in trailing-underscore attributes. The pipeline (1.9) fits them on TRAIN
  data only and applies them unchanged to validation and test.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from recsys.registry import register

#: Course-like Food.com tags, in priority order, used to give each recipe one "main tag" for the
#: GroupMedianImputer groups. A recipe gets the first of these it carries, otherwise "other".
#: Coverage on the full data is reported by :func:`main_tag_coverage`.
DEFAULT_MAIN_TAGS: tuple[str, ...] = (
    "beverages",
    "desserts",
    "breakfast",
    "appetizers",
    "salads",
    "soups-stews",
    "breads",
    "side-dishes",
    "sauces",
    "snacks",
    "main-dish",
)

#: Tags suggesting that a recorded total time of 0 minutes can be genuine.
DEFAULT_NO_COOK_TAGS: tuple[str, ...] = ("no-cook", "beverages")

OTHER_TAG = "other"


# =========================================================================== analysis


def missing_rates(recipes: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """Missing-value counts per column (data for the missing-rate bar plot).

    ``kind`` is ``"null"`` for NaN/None, ``"empty_list"`` for list columns holding ``[]`` and
    ``"zero"`` for the two zero codes that may mean "missing": ``interactions.rating == 0`` (no
    stars given) and ``recipes.minutes == 0`` (decided with :func:`minutes_zero_summary`).

    Returns:
        Columns table, column, kind, n_rows, n_missing, share (n_missing / n_rows).
    """
    rows: list[dict[str, object]] = []
    for table, df in (("recipes", recipes), ("interactions", interactions)):
        n = len(df)
        for col in df.columns:
            s = df[col]
            rows.append(_rate_row(table, col, "null", n, int(s.isna().sum())))
            if _is_list_column(s):
                n_empty = int(s.dropna().map(len).eq(0).sum())
                rows.append(_rate_row(table, col, "empty_list", n, n_empty))
    rows.append(
        _rate_row(
            "interactions",
            "rating",
            "zero",
            len(interactions),
            int((interactions["rating"] == 0).sum()),
        )
    )
    rows.append(
        _rate_row("recipes", "minutes", "zero", len(recipes), int((recipes["minutes"] == 0).sum()))
    )
    return pd.DataFrame(rows)


def rating_zero_review_stats(interactions: pd.DataFrame) -> pd.DataFrame:
    """Do rating = 0 rows come with a written review? (single-row table)

    If most zero-rated rows carry review text, the 0 is "no stars given" rather than a failed
    import, which supports treating it as missing (not as the worst score).

    Returns:
        Columns n_rows, n_zero, share_zero, n_zero_with_review, share_zero_with_review,
        share_rated_with_review (same share among rating > 0 rows, for comparison).
    """
    has_review = _has_text(interactions["review"])
    zero = interactions["rating"] == 0
    rated = interactions["rating"] > 0
    n_zero = int(zero.sum())
    n_rated = int(rated.sum())
    row = {
        "n_rows": len(interactions),
        "n_zero": n_zero,
        "share_zero": n_zero / len(interactions) if len(interactions) else np.nan,
        "n_zero_with_review": int((zero & has_review).sum()),
        "share_zero_with_review": float((zero & has_review).sum() / n_zero) if n_zero else np.nan,
        "share_rated_with_review": (
            float((rated & has_review).sum() / n_rated) if n_rated else np.nan
        ),
    }
    return pd.DataFrame([row])


def rating_zero_examples(
    interactions: pd.DataFrame, n: int = 10, random_state: int = 42
) -> pd.DataFrame:
    """A reproducible random sample of rating = 0 interactions with their review text."""
    zero = interactions.loc[interactions["rating"] == 0, ["user_id", "recipe_id", "date", "review"]]
    k = min(n, len(zero))
    return zero.sample(n=k, random_state=random_state).reset_index(drop=True)


def minutes_zero_summary(
    recipes: pd.DataFrame, no_cook_tags: Sequence[str] = DEFAULT_NO_COOK_TAGS
) -> pd.DataFrame:
    """Profile of recipes with minutes = 0 versus all recipes (single-row table).

    Returns:
        Columns n_recipes, n_minutes_zero, share_minutes_zero, n_zero_with_no_cook_tag,
        share_zero_with_no_cook_tag, share_all_with_no_cook_tag, median_n_steps_zero,
        median_n_steps_all, median_n_ingredients_zero, median_n_ingredients_all.
    """
    zero = recipes["minutes"] == 0
    tagged = _has_any_tag(recipes["tags"], no_cook_tags)
    n_zero = int(zero.sum())
    row = {
        "n_recipes": len(recipes),
        "n_minutes_zero": n_zero,
        "share_minutes_zero": n_zero / len(recipes) if len(recipes) else np.nan,
        "n_zero_with_no_cook_tag": int((zero & tagged).sum()),
        "share_zero_with_no_cook_tag": float((zero & tagged).sum() / n_zero) if n_zero else np.nan,
        "share_all_with_no_cook_tag": float(tagged.mean()) if len(recipes) else np.nan,
        "median_n_steps_zero": float(recipes.loc[zero, "n_steps"].median()) if n_zero else np.nan,
        "median_n_steps_all": float(recipes["n_steps"].median()),
        "median_n_ingredients_zero": (
            float(recipes.loc[zero, "n_ingredients"].median()) if n_zero else np.nan
        ),
        "median_n_ingredients_all": float(recipes["n_ingredients"].median()),
    }
    return pd.DataFrame([row])


def minutes_zero_tag_lift(recipes: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Most frequent tags among minutes = 0 recipes, with their share in the whole catalogue.

    ``lift`` > 1 means the tag is over-represented among minutes = 0 recipes (e.g. "no-cook").

    Returns:
        Columns tag, n_zero, share_in_zero, share_in_all, lift; ``top_n`` rows by n_zero.
    """
    zero = recipes["minutes"] == 0
    n_zero = int(zero.sum())
    cols = ["tag", "n_zero", "share_in_zero", "share_in_all", "lift"]
    if n_zero == 0:
        return pd.DataFrame(columns=cols).astype(
            {
                "tag": object,
                "n_zero": "int64",
                "share_in_zero": "float64",
                "share_in_all": "float64",
                "lift": "float64",
            }
        )
    all_counts = _tag_recipe_counts(recipes)
    zero_counts = _tag_recipe_counts(recipes.loc[zero])
    out = pd.DataFrame({"tag": zero_counts.index.astype(str), "n_zero": zero_counts.to_numpy()})
    out["share_in_zero"] = out["n_zero"] / n_zero
    out["share_in_all"] = out["tag"].map(all_counts).to_numpy() / len(recipes)
    out["lift"] = out["share_in_zero"] / out["share_in_all"]
    out = out.sort_values(["n_zero", "tag"], ascending=[False, True], kind="mergesort")
    out["n_zero"] = out["n_zero"].astype("int64")
    return out.head(top_n).reset_index(drop=True)[cols]


def minutes_zero_n_steps(recipes: pd.DataFrame) -> pd.DataFrame:
    """Distribution of n_steps for minutes = 0 recipes versus all recipes.

    Returns:
        Columns n_steps, n_zero, share_zero, n_all, share_all (one row per n_steps value).
    """
    zero = recipes["minutes"] == 0
    all_counts = recipes["n_steps"].value_counts()
    zero_counts = recipes.loc[zero, "n_steps"].value_counts()
    out = pd.DataFrame({"n_all": all_counts}).join(pd.DataFrame({"n_zero": zero_counts}))
    out = out.fillna(0).astype("int64").sort_index()
    out.index.name = "n_steps"
    out = out.reset_index()
    out["n_steps"] = out["n_steps"].astype("int64")
    n_zero = int(zero.sum())
    out["share_zero"] = out["n_zero"] / n_zero if n_zero else np.nan
    out["share_all"] = out["n_all"] / len(recipes)
    return out[["n_steps", "n_zero", "share_zero", "n_all", "share_all"]]


def minutes_zero_examples(
    recipes: pd.DataFrame, n: int = 10, random_state: int = 42
) -> pd.DataFrame:
    """A reproducible random sample of minutes = 0 recipes (tags joined, first step shown)."""
    zero = recipes.loc[recipes["minutes"] == 0]
    k = min(n, len(zero))
    s = zero.sample(n=k, random_state=random_state)
    return pd.DataFrame(
        {
            "id": s["id"].astype("int64").to_numpy(),
            "name": s["name"].to_numpy(),
            "n_steps": s["n_steps"].astype("int64").to_numpy(),
            "n_ingredients": s["n_ingredients"].astype("int64").to_numpy(),
            "tags": s["tags"].map(lambda t: ", ".join(t) if isinstance(t, list) else "").to_numpy(),
            "first_step": s["steps"]
            .map(lambda t: t[0] if isinstance(t, list) and t else "")
            .to_numpy(),
        }
    )


def main_tag_coverage(
    recipes: pd.DataFrame, main_tags: Sequence[str] = DEFAULT_MAIN_TAGS
) -> pd.DataFrame:
    """How many recipes each main tag (GroupMedianImputer grouping) receives, including "other".

    Returns:
        Columns main_tag, n_recipes, share; in the priority order of ``main_tags`` then "other".
    """
    assigned = assign_main_tag(recipes["tags"], main_tags)
    counts = assigned.value_counts().reindex([*main_tags, OTHER_TAG], fill_value=0)
    return pd.DataFrame(
        {
            "main_tag": counts.index.astype(str),
            "n_recipes": counts.to_numpy(dtype="int64"),
            "share": counts.to_numpy() / len(recipes) if len(recipes) else np.nan,
        }
    )


def mark_has_rating(recipes: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``recipes`` with ``has_rating`` (at least one rating > 0).

    Recipes without a rating stay in the recommendation pool but are excluded from training.
    This flag uses all interactions because it describes the catalogue; training targets are
    recomputed from TRAIN interactions in 1.9. Works whether rating = 0 has already been turned
    into NaN or not (NaN > 0 is False).
    """
    rated_ids = interactions.loc[interactions["rating"] > 0, "recipe_id"].unique()
    out = recipes.copy()
    out["has_rating"] = out["id"].isin(rated_ids)
    return out


def assign_main_tag(tags: pd.Series, main_tags: Sequence[str] = DEFAULT_MAIN_TAGS) -> pd.Series:
    """First tag of ``main_tags`` contained in each recipe's tag list, else ``"other"``."""
    priority = list(main_tags)

    def pick(t: object) -> str:
        if not isinstance(t, list):
            return OTHER_TAG
        present = set(t)
        for tag in priority:
            if tag in present:
                return tag
        return OTHER_TAG

    return tags.map(pick).astype(object)


# =========================================================================== transformers


@register("missing", "rating_zero_to_nan")
class RatingZeroToNaN(BaseEstimator, TransformerMixin):
    """Turn rating = 0 ("no stars given") into NaN. Stateless; output column is float64."""

    def __init__(self, column: str = "rating") -> None:
        self.column = column

    def fit(self, X: pd.DataFrame, y: object = None) -> RatingZeroToNaN:
        _require_columns(X, [self.column])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "n_features_in_")
        _require_columns(X, [self.column])
        out = X.copy()
        col = out[self.column].astype("float64")
        out[self.column] = col.mask(col == 0)
        return out


@register("missing", "zero_as_missing")
class ZeroAsMissing(BaseEstimator, TransformerMixin):
    """Turn 0 into NaN in ``columns``, except for rows carrying one of ``keep_if_tags``.

    Example: ``ZeroAsMissing(columns=("minutes",), keep_if_tags=("no-cook", "beverages"))`` treats
    a 0-minute stew as missing but keeps 0 minutes for a no-cook drink. Stateless; the changed
    columns become float64.
    """

    def __init__(
        self,
        columns: Sequence[str] = ("minutes",),
        keep_if_tags: Sequence[str] = (),
        tag_col: str = "tags",
    ) -> None:
        self.columns = columns
        self.keep_if_tags = keep_if_tags
        self.tag_col = tag_col

    def fit(self, X: pd.DataFrame, y: object = None) -> ZeroAsMissing:
        _require_columns(X, [*self.columns, *([self.tag_col] if self.keep_if_tags else [])])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "n_features_in_")
        out = X.copy()
        keep = (
            _has_any_tag(out[self.tag_col], self.keep_if_tags)
            if self.keep_if_tags
            else pd.Series(False, index=out.index)
        )
        for col in self.columns:
            values = out[col].astype("float64")
            out[col] = values.mask((values == 0) & ~keep)
        return out


@register("imputer", "group_median")
class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Fill NaN in ``target_col`` with the median of its (main tag, n_steps range) group.

    Everything is learned in ``fit`` from the data it is given (TRAIN only in the pipeline):

    * ``n_steps_edges_``: inner quantile edges of ``steps_col`` (``n_steps_bins`` equal-frequency
      ranges);
    * ``group_medians_``: ``"main_tag|bin" -> median`` for groups with at least
      ``min_group_size`` observed values;
    * ``global_median_``: fallback for unseen or too-small groups;
    * ``group_table_``: the same information as a DataFrame, for the report.

    ``transform`` never changes these attributes. With ``add_indicator`` a boolean column
    ``<target_col>_was_missing`` records which values were imputed.
    """

    def __init__(
        self,
        target_col: str = "minutes",
        tag_col: str = "tags",
        steps_col: str = "n_steps",
        main_tags: Sequence[str] = DEFAULT_MAIN_TAGS,
        n_steps_bins: int = 4,
        min_group_size: int = 5,
        add_indicator: bool = True,
    ) -> None:
        self.target_col = target_col
        self.tag_col = tag_col
        self.steps_col = steps_col
        self.main_tags = main_tags
        self.n_steps_bins = n_steps_bins
        self.min_group_size = min_group_size
        self.add_indicator = add_indicator

    def fit(self, X: pd.DataFrame, y: object = None) -> GroupMedianImputer:
        _require_columns(X, [self.target_col, self.tag_col, self.steps_col])
        if self.n_steps_bins < 1:
            raise ValueError("n_steps_bins must be >= 1")
        observed = X[self.target_col].notna()
        if not observed.any():
            raise ValueError(f"{self.target_col} has no observed values to learn medians from")

        steps = X[self.steps_col].astype("float64")
        qs = np.linspace(0, 1, self.n_steps_bins + 1)[1:-1]
        self.n_steps_edges_ = np.unique(np.nanquantile(steps, qs)) if len(qs) else np.array([])
        keys = self._group_keys(X)

        target = X.loc[observed, self.target_col].astype("float64")
        stats = target.groupby(keys[observed]).agg(["median", "size"])
        used = stats["size"] >= self.min_group_size
        self.group_medians_ = {str(k): float(v) for k, v in stats.loc[used, "median"].items()}
        self.global_median_ = float(target.median())

        parts = stats.index.to_series().str.split("|", n=1, expand=True)
        self.group_table_ = pd.DataFrame(
            {
                "group": stats.index.astype(str),
                "main_tag": parts[0].to_numpy(),
                "n_steps_bin": parts[1].astype("int64").to_numpy(),
                "n_observed": stats["size"].to_numpy(dtype="int64"),
                "median": stats["median"].to_numpy(dtype="float64"),
                "used": used.to_numpy(),
            }
        ).reset_index(drop=True)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["group_medians_", "global_median_", "n_steps_edges_"])
        _require_columns(X, [self.target_col, self.tag_col, self.steps_col])
        out = X.copy()
        values = out[self.target_col].astype("float64")
        missing = values.isna()
        if missing.any():
            keys = self._group_keys(out.loc[missing])
            fill = keys.map(self.group_medians_).astype("float64").fillna(self.global_median_)
            values.loc[missing] = fill.to_numpy()
        out[self.target_col] = values
        if self.add_indicator:
            out[f"{self.target_col}_was_missing"] = missing.to_numpy()
        return out

    def fallback_share(self, X: pd.DataFrame) -> float:
        """Share of the missing values in ``X`` that would get the global (not group) median."""
        check_is_fitted(self, "group_medians_")
        missing = X[self.target_col].isna()
        if not missing.any():
            return 0.0
        keys = self._group_keys(X.loc[missing])
        return float((~keys.isin(list(self.group_medians_))).mean())

    def _group_keys(self, X: pd.DataFrame) -> pd.Series:
        tag = assign_main_tag(X[self.tag_col], self.main_tags)
        steps = X[self.steps_col].astype("float64").to_numpy()
        bins = np.searchsorted(self.n_steps_edges_, steps, side="right")
        return tag.astype(str) + "|" + pd.Series(bins, index=X.index).astype(str)


# =========================================================================== helpers


def _rate_row(table: str, column: str, kind: str, n: int, n_missing: int) -> dict[str, object]:
    return {
        "table": table,
        "column": str(column),
        "kind": kind,
        "n_rows": n,
        "n_missing": n_missing,
        "share": n_missing / n if n else np.nan,
    }


def _is_list_column(s: pd.Series) -> bool:
    if not pd.api.types.is_object_dtype(s):
        return False
    present = s.dropna()
    return len(present) > 0 and bool(present.map(lambda v: isinstance(v, list)).all())


def _has_text(s: pd.Series) -> pd.Series:
    return s.map(lambda v: isinstance(v, str) and v.strip() != "")


def _has_any_tag(tags: pd.Series, wanted: Sequence[str]) -> pd.Series:
    wanted_set = set(wanted)
    return tags.map(lambda t: isinstance(t, list) and not wanted_set.isdisjoint(t)).astype(bool)


def _tag_recipe_counts(recipes: pd.DataFrame) -> pd.Series:
    exploded = recipes[["id", "tags"]].explode("tags").dropna(subset=["tags"])
    return exploded.drop_duplicates(["id", "tags"])["tags"].value_counts()


def _require_columns(X: pd.DataFrame, cols: Sequence[str]) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(X).__name__}")
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")
