"""Exploratory tables for stage 1.1 (descriptive statistics).

Every public function is pure: it takes DataFrames, returns a new DataFrame and draws nothing.
Plotting and narrative belong to the notebook (notebooks/01_exploration.ipynb).

Inputs are the ``loaded`` stage tables (LOADED_RECIPES_SCHEMA / LOADED_INTERACTIONS_SCHEMA).
Loading is faithful, so these tables still contain rating = 0 rows and duplicate interactions;
each function states how it treats them.

Output schemas for the ``exploration`` stage live in :mod:`recsys.contracts`
(:data:`~recsys.contracts.EXPLORATION_SCHEMAS`, re-exported here) and are passed to
``ArtifactStore.save``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from recsys.contracts import EXPLORATION_SCHEMAS, RATING_VALUES
from recsys.contracts import EXPLORATION_TYPE_CLASSES as TYPE_CLASSES

__all__ = [
    "DECLARED_COLUMN_TYPES",
    "EXPLORATION_SCHEMAS",
    "TYPE_CLASSES",
    "basic_counts",
    "column_type_table",
    "exploratory_targets",
    "per_entity_counts",
    "rating_counts",
    "top_items",
]

#: Analyst-declared type of every Food.com column. Declared types win over inference, because
#: inference from a small sample can misclassify (e.g. n_steps with few distinct values).
DECLARED_COLUMN_TYPES: Mapping[str, tuple[str, str]] = {
    # column: (type_class, note)
    "id": ("categorical", "identifier"),
    "recipe_id": ("categorical", "identifier"),
    "user_id": ("categorical", "identifier"),
    "contributor_id": ("categorical", "identifier"),
    "rating": ("categorical", "ordinal 0-5; 0 = no stars given"),
    "minutes": ("continuous", ""),
    "n_steps": ("continuous", "count"),
    "n_ingredients": ("continuous", "count"),
    "calories": ("continuous", ""),
    "total_fat_pdv": ("continuous", "% daily value"),
    "sugar_pdv": ("continuous", "% daily value"),
    "sodium_pdv": ("continuous", "% daily value"),
    "protein_pdv": ("continuous", "% daily value"),
    "saturated_fat_pdv": ("continuous", "% daily value"),
    "carbohydrates_pdv": ("continuous", "% daily value"),
    "submitted": ("datetime", ""),
    "date": ("datetime", ""),
    "tags": ("list", ""),
    "steps": ("list", ""),
    "ingredients": ("list", ""),
    "name": ("text", ""),
    "description": ("text", ""),
    "review": ("text", ""),
}

_NO_RATING_NOTE = "no stars given (becomes NaN in 1.3)"


# --------------------------------------------------------------------------- 1. column types


def column_type_table(tables: Mapping[str, pd.DataFrame], max_categories: int = 10) -> pd.DataFrame:
    """Classify every column of every table as continuous / categorical / text / list / datetime.

    Known Food.com columns use :data:`DECLARED_COLUMN_TYPES` (``source = "declared"``). Unknown
    columns are inferred from dtype (``source = "inferred"``): datetime -> datetime; Python lists
    -> list; bool/category -> categorical; integers with at most ``max_categories`` distinct values
    -> categorical, other numbers -> continuous; strings -> text.

    Args:
        tables: table name -> DataFrame (e.g. ``{"recipes": ..., "interactions": ...}``).
        max_categories: distinct-value threshold for inferring an integer column as categorical.

    Returns:
        One row per column: table, column, dtype, type_class, source, note, n_missing, n_unique
        (``<NA>`` for list columns, whose values are unhashable).
    """
    rows = []
    for table, df in tables.items():
        for col in df.columns:
            s = df[col]
            is_list = _is_list_column(s)
            if col in DECLARED_COLUMN_TYPES:
                type_class, note = DECLARED_COLUMN_TYPES[col]
                source = "declared"
            else:
                type_class, note = _infer_type_class(s, is_list, max_categories), ""
                source = "inferred"
            rows.append(
                {
                    "table": table,
                    "column": str(col),
                    "dtype": str(s.dtype),
                    "type_class": type_class,
                    "source": source,
                    "note": note,
                    "n_missing": int(s.isna().sum()),
                    "n_unique": pd.NA if is_list else int(s.nunique(dropna=True)),
                }
            )
    out = pd.DataFrame(rows)
    out["n_unique"] = out["n_unique"].astype("Int64")
    return out


# --------------------------------------------------------------------------- 2. basic counts


def basic_counts(recipes: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """Headline sizes and date ranges, as a single-row table (one typed column per metric).

    Columns:
        n_recipes, n_interactions, n_users (distinct user_id), n_contributors,
        n_recipes_with_interactions (distinct recipe_id present in ``recipes``),
        n_recipes_without_interactions, n_orphan_interactions (recipe_id not in ``recipes``),
        n_duplicate_interactions (rows identical to an earlier row in every column),
        n_repeated_user_recipe_pairs (rows whose (user_id, recipe_id) appeared earlier),
        interactions_date_min/max, recipes_submitted_min/max.
    """
    recipe_ids = pd.Index(recipes["id"].unique())
    rated_ids = pd.Index(interactions["recipe_id"].unique())
    in_catalog = rated_ids.isin(recipe_ids)
    row = {
        "n_recipes": len(recipes),
        "n_interactions": len(interactions),
        "n_users": int(interactions["user_id"].nunique()),
        "n_contributors": int(recipes["contributor_id"].nunique()),
        "n_recipes_with_interactions": int(in_catalog.sum()),
        "n_recipes_without_interactions": int((~recipe_ids.isin(rated_ids)).sum()),
        "n_orphan_interactions": int((~interactions["recipe_id"].isin(recipe_ids)).sum()),
        "n_duplicate_interactions": int(interactions.duplicated().sum()),
        "n_repeated_user_recipe_pairs": int(
            interactions.duplicated(subset=["user_id", "recipe_id"]).sum()
        ),
        "interactions_date_min": interactions["date"].min(),
        "interactions_date_max": interactions["date"].max(),
        "recipes_submitted_min": recipes["submitted"].min(),
        "recipes_submitted_max": recipes["submitted"].max(),
    }
    out = pd.DataFrame([row])
    int_cols = [c for c in out.columns if c.startswith("n_")]
    out[int_cols] = out[int_cols].astype("int64")
    return out


# --------------------------------------------------------------------------- 3. per-entity counts


def per_entity_counts(
    interactions: pd.DataFrame, recipes: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Distribution of the number of interactions (reviews) per recipe and per user.

    Every interaction row counts, including rating = 0 rows and duplicates (this describes the
    raw activity; cleaning happens in 1.3). If ``recipes`` is given, recipes without any
    interaction are included with a count of 0, so the recipe row describes the whole catalogue.
    Users exist only through interactions, so the user minimum is at least 1.

    Returns:
        Rows ``entity`` in (recipe, user); columns n_entities, n_zero, n_one, pct_one, mean, std,
        min, q25, median, q75, max.
    """
    per_recipe = interactions.groupby("recipe_id").size()
    if recipes is not None:
        per_recipe = per_recipe.reindex(recipes["id"].unique(), fill_value=0)
    per_user = interactions.groupby("user_id").size()
    rows = [_count_stats("recipe", per_recipe), _count_stats("user", per_user)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 4. rating counts


def rating_counts(interactions: pd.DataFrame) -> pd.DataFrame:
    """Count of each rating value 0..5 (all values listed, even if absent).

    rating = 0 means "no stars given" on Food.com and is flagged separately (``is_zero``);
    ``share_of_rated`` excludes those rows from the denominator.

    Returns:
        Columns rating, count, share (of all rows), share_of_rated (NaN for 0), is_zero, note.
    """
    counts = interactions["rating"].value_counts().reindex(RATING_VALUES, fill_value=0)
    total = int(counts.sum())
    rated = int(counts.drop(0).sum())
    out = pd.DataFrame({"rating": list(RATING_VALUES), "count": counts.to_numpy(dtype="int64")})
    out["share"] = out["count"] / total if total else np.nan
    out["share_of_rated"] = np.where(
        out["rating"] == 0, np.nan, out["count"] / rated if rated else np.nan
    )
    out["is_zero"] = out["rating"] == 0
    out["note"] = np.where(out["is_zero"], _NO_RATING_NOTE, "")
    return out


# --------------------------------------------------------------------------- 5. top items


def top_items(recipes: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """The ``n`` most frequent ingredients and tags, counted as number of recipes containing them.

    An item repeated inside one recipe counts once. Items are the raw strings (normalisation of
    ingredient variants such as "egg" / "eggs" is done in 1.5). Ties are broken alphabetically so
    the table is deterministic.

    Returns:
        Columns kind (ingredient / tag), rank (1..n), item, n_recipes, share_of_recipes.
    """
    parts = []
    for kind, col in (("ingredient", "ingredients"), ("tag", "tags")):
        exploded = (
            recipes[["id", col]].explode(col).dropna(subset=[col]).drop_duplicates(["id", col])
        )
        counts = exploded[col].value_counts()
        table = (
            pd.DataFrame({"item": counts.index.astype(str), "n_recipes": counts.to_numpy()})
            .sort_values(["n_recipes", "item"], ascending=[False, True], kind="mergesort")
            .head(n)
            .reset_index(drop=True)
        )
        table.insert(0, "kind", kind)
        table.insert(1, "rank", np.arange(1, len(table) + 1))
        table["share_of_recipes"] = table["n_recipes"] / len(recipes)
        parts.append(table)
    out = pd.concat(parts, ignore_index=True)
    out["rank"] = out["rank"].astype("int64")
    out["n_recipes"] = out["n_recipes"].astype("int64")
    return out


# --------------------------------------------------------------------------- 6. exploratory targets


def exploratory_targets(recipes: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """Per-recipe rating summary computed on ALL interactions — FOR EXPLORATION ONLY.

    .. warning:: Leakage
        These values use every interaction, including those that will end up in validation and
        test. They must NOT be used as model features or targets. The target used by the model
        is recomputed in 1.9 from TRAIN interactions only.

    rating = 0 rows ("no stars given") are excluded from ``avg_rating`` and ``n_ratings`` but
    counted in ``n_interactions``. Duplicate rows are counted as they are (removed in 1.3).
    Every recipe of ``recipes`` appears, including recipes without any interaction.

    Returns:
        Columns recipe_id, n_interactions, n_ratings (rating > 0), avg_rating (NaN if
        n_ratings == 0), has_rating (n_ratings > 0); sorted by recipe_id.
    """
    ids = pd.Index(np.sort(recipes["id"].unique()), name="recipe_id")
    grouped = interactions.groupby("recipe_id")["rating"]
    n_inter = grouped.size().reindex(ids, fill_value=0)
    rated = interactions.loc[interactions["rating"] > 0].groupby("recipe_id")["rating"]
    n_ratings = rated.size().reindex(ids, fill_value=0)
    avg = rated.mean().reindex(ids)
    out = pd.DataFrame(
        {
            "recipe_id": ids.to_numpy(),
            "n_interactions": n_inter.to_numpy(dtype="int64"),
            "n_ratings": n_ratings.to_numpy(dtype="int64"),
            "avg_rating": avg.to_numpy(dtype="float64"),
        }
    )
    out["has_rating"] = out["n_ratings"] > 0
    return out


# --------------------------------------------------------------------------- internals


def _is_list_column(s: pd.Series) -> bool:
    if not ptypes.is_object_dtype(s):
        return False
    present = s.dropna()
    return len(present) > 0 and bool(present.map(lambda v: isinstance(v, list)).all())


def _infer_type_class(s: pd.Series, is_list: bool, max_categories: int) -> str:
    if ptypes.is_datetime64_any_dtype(s):
        return "datetime"
    if is_list:
        return "list"
    if ptypes.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return "categorical"
    if ptypes.is_integer_dtype(s):
        return "categorical" if s.nunique(dropna=True) <= max_categories else "continuous"
    if ptypes.is_numeric_dtype(s):
        return "continuous"
    return "text"


def _count_stats(entity: str, counts: pd.Series) -> dict[str, object]:
    c = counts.astype("int64")
    n = len(c)
    return {
        "entity": entity,
        "n_entities": n,
        "n_zero": int((c == 0).sum()),
        "n_one": int((c == 1).sum()),
        "pct_one": float((c == 1).mean()) if n else np.nan,
        "mean": float(c.mean()) if n else np.nan,
        "std": float(c.std()) if n > 1 else np.nan,
        "min": int(c.min()) if n else 0,
        "q25": float(c.quantile(0.25)) if n else np.nan,
        "median": float(c.median()) if n else np.nan,
        "q75": float(c.quantile(0.75)) if n else np.nan,
        "max": int(c.max()) if n else 0,
    }
