"""Rule-based distribution classification for numeric features (Phase 1.6)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from recsys.contracts import NUTRITION_COLUMNS
from recsys.data.targets import recipe_rating_stats


@dataclass(frozen=True)
class DistributionClassification:
    """Result of classifying a single numeric feature's distribution."""

    feature: str
    distribution_type: str
    skewness: float
    kurtosis: float
    p99_to_median_ratio: float


def _count_histogram_modes(values: np.ndarray, bins: int = 20) -> int:
    """Count prominent local peaks separated by a genuine valley (< 60% of peak height)."""
    if len(values) < 10 or np.allclose(values, values[0]):
        return 1
    counts, _ = np.histogram(values, bins=bins)
    max_count = counts.max()
    if max_count == 0:
        return 1

    # Flat/uniform histograms have no deep valley between bins
    if counts.min() >= 0.6 * max_count:
        return 1

    threshold = 0.25 * max_count
    peaks: list[int] = []
    for i in range(len(counts)):
        left = counts[i - 1] if i > 0 else -1
        right = counts[i + 1] if i < len(counts) - 1 else -1
        if counts[i] >= threshold and counts[i] >= left and counts[i] > right:
            peaks.append(i)

    if len(peaks) <= 1:
        return 1

    # Keep only peaks separated by a valley below 60% of the smaller peak
    distinct_peaks = [peaks[0]]
    for p in peaks[1:]:
        prev = distinct_peaks[-1]
        valley = counts[prev : p + 1].min()
        if valley < 0.6 * min(counts[prev], counts[p]):
            distinct_peaks.append(p)
        elif counts[p] > counts[prev]:
            distinct_peaks[-1] = p
    return len(distinct_peaks)


def classify_distribution(series: pd.Series) -> DistributionClassification:
    """Classify a numeric series into a standard distribution shape.

    Rule-based thresholds (evaluated in order):
    1. ``long-tail``: skewness > 2.0 and (kurtosis > 7.0 or p99/median > 5.0 with median > 0).
    2. ``right-skewed``: skewness > 0.5.
    3. ``left-skewed``: skewness < -0.5.
    4. ``multimodal``: |skewness| <= 0.5 and >= 2 prominent histogram peaks (>= 25% of max bin).
    5. ``uniform``: |skewness| <= 0.5 and excess kurtosis < -1.0.
    6. ``normal``: otherwise (|skewness| <= 0.5 and unimodal, near-mesokurtic).
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    name = str(series.name or "feature")
    if len(clean) < 3 or clean.nunique() <= 1:
        return DistributionClassification(
            feature=name,
            distribution_type="constant",
            skewness=0.0,
            kurtosis=0.0,
            p99_to_median_ratio=1.0,
        )

    skew = float(clean.skew())
    kurt = float(clean.kurt())  # Fisher excess kurtosis (normal = 0, uniform = -1.2)
    median = float(clean.median())
    p99 = float(clean.quantile(0.99))
    ratio = float(p99 / median) if median > 0 else float("inf")

    if skew > 2.0 and (kurt > 7.0 or (median > 0 and ratio > 5.0)):
        dist_type = "long-tail"
    elif skew > 0.5:
        dist_type = "right-skewed"
    elif skew < -0.5:
        dist_type = "left-skewed"
    elif _count_histogram_modes(clean.to_numpy()) >= 2:
        dist_type = "multimodal"
    elif kurt < -1.0:
        dist_type = "uniform"
    else:
        dist_type = "normal"

    return DistributionClassification(
        feature=name,
        distribution_type=dist_type,
        skewness=round(skew, 4),
        kurtosis=round(kurt, 4),
        p99_to_median_ratio=round(ratio, 4) if np.isfinite(ratio) else float("inf"),
    )


def distribution_summary_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Build a summary DataFrame classifying all requested numeric columns."""
    rows = [classify_distribution(df[col]).__dict__ for col in cols if col in df.columns]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ notebook 06 / 07 helpers
# Moved out of the notebooks (1.12-A O4: notebooks call src/recsys, they hold no logic).

#: Numeric recipe features analysed in 1.6 and scaled in 1.8.
NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "minutes",
    "n_steps",
    "n_ingredients",
    *NUTRITION_COLUMNS,
)


def present_columns(df: pd.DataFrame, cols: Iterable[str] = NUMERIC_FEATURE_COLUMNS) -> list[str]:
    """``cols`` that exist in ``df``, in the given order."""
    return [c for c in cols if c in df.columns]


def list_items(lists: pd.Series, name: str) -> pd.Series:
    """One row per element of a list column (e.g. every ingredient of every recipe)."""
    return lists.explode().rename(name)


def reviews_per_entity(interactions: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Interactions per user and per recipe (``reviews_per_user``, ``reviews_per_recipe``)."""
    user_counts = interactions.groupby("user_id").size().rename("reviews_per_user")
    recipe_counts = interactions.groupby("recipe_id").size().rename("reviews_per_recipe")
    return user_counts, recipe_counts


def cold_start_summary(user_counts: pd.Series, recipe_counts: pd.Series) -> pd.DataFrame:
    """Share of users / recipes with a single interaction, plus median and maximum counts."""
    rows = []
    for entity, counts in (("user", user_counts), ("recipe", recipe_counts)):
        rows.append(
            {
                "entity": entity,
                "total_unique": int(len(counts)),
                "single_review_count": int((counts == 1).sum()),
                "single_review_pct": round(float((counts == 1).mean() * 100), 2),
                "median_reviews": float(counts.median()),
                "max_reviews": int(counts.max()),
            }
        )
    return pd.DataFrame(rows)


def with_rating_stats(
    recipes: pd.DataFrame, interactions: pd.DataFrame, how: str = "left"
) -> pd.DataFrame:
    """Recipes with exploratory ``avg_rating`` / ``n_ratings`` (ratings > 0 only).

    For figures and binning comparisons only: the model targets are computed from TRAIN
    interactions in 1.9 (:mod:`recsys.data.targets`). ``how="inner"`` keeps rated recipes only.
    """
    return recipes.merge(
        recipe_rating_stats(interactions), left_on="id", right_on="recipe_id", how=how
    )


def rating_by_list_item(
    df: pd.DataFrame, list_col: str = "tags", value: str = "avg_rating"
) -> pd.DataFrame:
    """Long table ``(item, value)``: one row per element of ``list_col`` of every rated recipe."""
    return df[[list_col, value]].dropna().explode(list_col)
