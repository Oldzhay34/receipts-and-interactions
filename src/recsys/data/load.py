"""Load RAW_recipes.csv / RAW_interactions.csv into typed DataFrames.

Each loader validates the raw table on the way in and the parsed table on the way out
. Loading is faithful: it does not drop, impute or clip anything; those
decisions belong to 1.2-1.4.

Parsing:
* ``tags``, ``steps``, ``ingredients``: string repr -> Python list (``ast.literal_eval``).
* ``nutrition``: "[a, b, ...]" -> the 7 float columns in :data:`NUTRITION_COLUMNS`; the raw string
  column is dropped.
* ``submitted`` / ``date``: -> datetime64.
* Integer columns are downcast to the smallest safe integer dtype (memory).
"""

from __future__ import annotations

import ast
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

from recsys.contracts import (
    LOADED_INTERACTIONS_SCHEMA,
    LOADED_RECIPES_SCHEMA,
    NUTRITION_COLUMNS,
    RAW_INTERACTIONS_SCHEMA,
    RAW_RECIPES_SCHEMA,
    validate,
)

PathLike = str | os.PathLike[str]

DEFAULT_NA_VALUES: tuple[str, ...] = ("",)
LIST_COLUMNS: tuple[str, ...] = ("tags", "steps", "ingredients")
DATE_FORMAT = "%Y-%m-%d"

_RECIPE_INT_COLS = ("id", "minutes", "contributor_id", "n_steps", "n_ingredients")
_INTERACTION_INT_COLS = ("user_id", "recipe_id")


# --------------------------------------------------------------------------- read (raw)


def read_raw_recipes(path: PathLike, na_values: Sequence[str] = DEFAULT_NA_VALUES) -> pd.DataFrame:
    """Read RAW_recipes.csv as-is and validate it against RAW_RECIPES_SCHEMA."""
    df = pd.read_csv(path, keep_default_na=False, na_values=list(na_values))
    return validate(df, RAW_RECIPES_SCHEMA)


def read_raw_interactions(
    path: PathLike, na_values: Sequence[str] = DEFAULT_NA_VALUES
) -> pd.DataFrame:
    """Read RAW_interactions.csv as-is and validate it against RAW_INTERACTIONS_SCHEMA."""
    df = pd.read_csv(path, keep_default_na=False, na_values=list(na_values))
    return validate(df, RAW_INTERACTIONS_SCHEMA)


# --------------------------------------------------------------------------- parse


def parse_recipes(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a validated raw recipes table into the LOADED_RECIPES_SCHEMA table."""
    validate(raw, RAW_RECIPES_SCHEMA)
    df = raw.copy()
    for col in LIST_COLUMNS:
        df[col] = parse_list_column(df[col], ids=df["id"])
    nutrition = parse_nutrition(df["nutrition"], ids=df["id"])
    df = df.drop(columns="nutrition")
    df = pd.concat([df, nutrition], axis=1)
    df["submitted"] = parse_dates(df["submitted"], ids=df["id"])
    df = _downcast_ints(df, _RECIPE_INT_COLS)
    df = df[list(LOADED_RECIPES_SCHEMA.columns)]
    return validate(df, LOADED_RECIPES_SCHEMA)


def parse_interactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a validated raw interactions table into the LOADED_INTERACTIONS_SCHEMA table."""
    validate(raw, RAW_INTERACTIONS_SCHEMA)
    df = raw.copy()
    df["date"] = parse_dates(df["date"], ids=df.index.to_series())
    rating = df["rating"]
    if not np.all(np.equal(np.mod(rating, 1), 0)):
        raise ValueError("rating contains non-integer values")
    df["rating"] = rating.astype("int8")
    df = _downcast_ints(df, _INTERACTION_INT_COLS)
    df = df[list(LOADED_INTERACTIONS_SCHEMA.columns)]
    return validate(df, LOADED_INTERACTIONS_SCHEMA)


def load_recipes(path: PathLike, na_values: Sequence[str] = DEFAULT_NA_VALUES) -> pd.DataFrame:
    """Read + parse RAW_recipes.csv (validated in and out)."""
    return parse_recipes(read_raw_recipes(path, na_values))


def load_interactions(path: PathLike, na_values: Sequence[str] = DEFAULT_NA_VALUES) -> pd.DataFrame:
    """Read + parse RAW_interactions.csv (validated in and out)."""
    return parse_interactions(read_raw_interactions(path, na_values))


# --------------------------------------------------------------------------- column parsers


def parse_list_column(s: pd.Series, ids: pd.Series | None = None) -> pd.Series:
    """Parse "['a', 'b']" strings into lists with ``ast.literal_eval``; missing stays None.

    Raises:
        ValueError: naming the first offending row id if a value is not a valid list literal.
    """
    ids = ids if ids is not None else s.index.to_series()

    def one(value: object, rid: object) -> list | None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        try:
            parsed = ast.literal_eval(value)  # type: ignore[arg-type]
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"{s.name}: row id {rid} is not a list literal: {value!r:.80}"
            ) from exc
        if not isinstance(parsed, list):
            raise ValueError(f"{s.name}: row id {rid} parsed to {type(parsed).__name__}, not list")
        return parsed

    return pd.Series(
        [one(v, r) for v, r in zip(s.to_numpy(), ids.to_numpy(), strict=True)],
        index=s.index,
        name=s.name,
        dtype=object,
    )


def parse_nutrition(s: pd.Series, ids: pd.Series | None = None) -> pd.DataFrame:
    """Split "[cal, fat, sugar, sodium, protein, sat_fat, carbs]" into 7 float columns.

    Missing strings give a row of NaN. A non-missing string with a count other than 7 or a
    non-numeric element raises ValueError naming the row id.
    """
    ids = ids if ids is not None else s.index.to_series()
    present = s.notna()
    parts = s[present].str.strip().str.strip("[]").str.split(",")
    lengths = parts.str.len()
    bad = lengths != len(NUTRITION_COLUMNS)
    if bad.any():
        first = bad[bad].index[0]
        raise ValueError(
            f"nutrition: {int(bad.sum())} rows without exactly {len(NUTRITION_COLUMNS)} values "
            f"(first: id {ids.loc[first]}, value {s.loc[first]!r:.80})"
        )
    out = pd.DataFrame(np.nan, index=s.index, columns=list(NUTRITION_COLUMNS), dtype="float64")
    if present.any():
        mat = pd.DataFrame(parts.tolist(), index=parts.index, columns=list(NUTRITION_COLUMNS))
        for col in NUTRITION_COLUMNS:
            stripped = mat[col].str.strip()
            num = pd.to_numeric(stripped, errors="coerce")
            if num.isna().any():
                first = num[num.isna()].index[0]
                raise ValueError(
                    f"nutrition.{col}: non-numeric value {stripped.loc[first]!r} "
                    f"at id {ids.loc[first]}"
                )
            out.loc[mat.index, col] = num.astype("float64")
    return out


def parse_dates(s: pd.Series, ids: pd.Series | None = None) -> pd.Series:
    """Parse YYYY-MM-DD strings. Missing stays NaT; an unparseable non-missing value raises."""
    ids = ids if ids is not None else s.index.to_series()
    parsed = pd.to_datetime(s, format=DATE_FORMAT, errors="coerce")
    bad = parsed.isna() & s.notna()
    if bad.any():
        first = bad[bad].index[0]
        raise ValueError(
            f"{s.name}: {int(bad.sum())} unparseable dates (first: id {ids.loc[first]}, "
            f"value {s.loc[first]!r})"
        )
    return parsed


def _downcast_ints(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


# --------------------------------------------------------------------------- sampling


def make_stratified_sample(
    recipes: pd.DataFrame,
    interactions: pd.DataFrame,
    n_recipes: int,
    stratify_col: str = "n_ingredients",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample ``n_recipes`` recipes preserving the distribution of ``stratify_col``.

    Allocation per stratum is proportional with the largest-remainder method, so the total is
    exactly ``n_recipes`` and each stratum's count is within 1 of its exact proportional share.
    Interactions are ALL interactions of the sampled recipes (not an independent sample), so the
    two tables stay consistent.

    Returns:
        (recipes_sample sorted by id, interactions_sample) with reset indices.
    """
    if n_recipes <= 0:
        raise ValueError("n_recipes must be positive")
    if n_recipes >= len(recipes):
        chosen = recipes.copy()
    else:
        counts = recipes[stratify_col].value_counts(dropna=False).sort_index()
        exact = counts * (n_recipes / len(recipes))
        quota = np.floor(exact).astype(int)
        remainder = int(n_recipes - quota.sum())
        if remainder:
            # Ties broken by stratum value so the result is deterministic.
            order = (exact - quota).sort_values(ascending=False, kind="mergesort").index
            quota.loc[order[:remainder]] += 1
        parts = [
            grp.sample(n=int(quota.loc[key]), random_state=random_state)
            for key, grp in recipes.groupby(stratify_col, dropna=False, sort=True)
            if quota.loc[key] > 0
        ]
        chosen = pd.concat(parts)
    recipes_s = chosen.sort_values("id").reset_index(drop=True)
    inter_s = interactions[interactions["recipe_id"].isin(recipes_s["id"])].reset_index(drop=True)
    return recipes_s, inter_s


def stratification_table(
    full: pd.DataFrame, sample: pd.DataFrame, col: str = "n_ingredients"
) -> pd.DataFrame:
    """Per-stratum count and share in the full table vs the sample (evidence for the sample)."""
    out = pd.DataFrame(
        {
            "full_count": full[col].value_counts(dropna=False),
            "sample_count": sample[col].value_counts(dropna=False),
        }
    ).fillna(0)
    out = out.astype("int64").sort_index()
    out["full_share"] = out["full_count"] / out["full_count"].sum()
    out["sample_share"] = out["sample_count"] / out["sample_count"].sum()
    out["share_diff"] = out["sample_share"] - out["full_share"]
    out.index.name = col
    return out.reset_index()
