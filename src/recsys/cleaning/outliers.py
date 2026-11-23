"""Outlier analysis and treatment for stage 1.4.

Two kinds of public objects:

* **Analysis functions** (pure: DataFrame in, DataFrame out, no plotting): IQR tables (also on the
  log1p scale), quantile tables, threshold counts, extreme examples, invalid-value counts and a
  before/after comparison table.
* **Transformers** (sklearn ``BaseEstimator`` + ``TransformerMixin``, registered by name under
  kind ``"outlier"``): ``InvalidValueRule`` (impossible values -> NaN), ``QuantileClipper``
  (bounds learned in ``fit``) and ``Log1pTransformer``. Learned state lives in trailing-underscore
  attributes; the 1.9 pipeline fits them on TRAIN data only.

Which strategy applies to which column comes from the ``outliers`` block of ``config/config.yaml``
(:func:`outlier_steps_from_config`), so adding a column or a method does not change this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from recsys import registry
from recsys.contracts import NUTRITION_COLUMNS
from recsys.registry import register

#: Numeric recipe columns analysed for outliers.
OUTLIER_COLUMNS: tuple[str, ...] = ("minutes", "n_steps", "n_ingredients", *NUTRITION_COLUMNS)

#: Quantiles reported by :func:`quantile_table` to choose thresholds from the data.
DEFAULT_QUANTILES: tuple[float, ...] = (
    0.0,
    0.01,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.99,
    0.995,
    0.999,
    1.0,
)

#: Strategy names usable in config, in the order they are applied.
STEP_ORDER: tuple[str, ...] = ("invalid_rule", "quantile_clip", "log1p")

Bounds = Mapping[str, Sequence[float | None]]


# =========================================================================== analysis


def iqr_table(
    df: pd.DataFrame, cols: Sequence[str], k: float = 1.5, log1p: bool = False
) -> pd.DataFrame:
    """Tukey IQR fences, box-plot statistics and outlier counts per column.

    With ``log1p=True`` the statistics are computed on ``log1p(x)`` (negative values excluded);
    fences are reported back on the original scale so both tables are comparable.

    Returns:
        Columns column, scale ("raw" / "log1p"), n, q1, median, q3, iqr, lower_fence, upper_fence,
        whisker_low, whisker_high, n_below, n_above, share_outside, min, max. All values on the
        original scale; ``q1`` / ``median`` / ``q3`` / ``iqr`` on the scale named in ``scale``.
    """
    rows = []
    for col in cols:
        s = df[col].dropna().astype("float64")
        if log1p:
            s = s[s >= 0]
        z = np.log1p(s) if log1p else s
        q1, med, q3 = (float(v) for v in z.quantile([0.25, 0.5, 0.75]))
        iqr = q3 - q1
        lo_z, hi_z = q1 - k * iqr, q3 + k * iqr
        lo, hi = (float(np.expm1(lo_z)), float(np.expm1(hi_z))) if log1p else (lo_z, hi_z)
        inside = s[(s >= lo) & (s <= hi)]
        n = len(s)
        n_below, n_above = int((s < lo).sum()), int((s > hi).sum())
        rows.append(
            {
                "column": col,
                "scale": "log1p" if log1p else "raw",
                "n": n,
                "q1": q1,
                "median": med,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": lo,
                "upper_fence": hi,
                "whisker_low": float(inside.min()) if len(inside) else np.nan,
                "whisker_high": float(inside.max()) if len(inside) else np.nan,
                "n_below": n_below,
                "n_above": n_above,
                "share_outside": (n_below + n_above) / n if n else np.nan,
                "min": float(s.min()) if n else np.nan,
                "max": float(s.max()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def quantile_table(
    df: pd.DataFrame, cols: Sequence[str], quantiles: Sequence[float] = DEFAULT_QUANTILES
) -> pd.DataFrame:
    """Long table of quantiles per column (to read thresholds from the data).

    Returns:
        Columns column, quantile, value.
    """
    rows = []
    for col in cols:
        values = df[col].dropna().astype("float64").quantile(list(quantiles))
        rows.extend(
            {"column": col, "quantile": float(q), "value": float(v)} for q, v in values.items()
        )
    return pd.DataFrame(rows)


def threshold_counts(df: pd.DataFrame, col: str, thresholds: Sequence[float]) -> pd.DataFrame:
    """How many rows exceed each candidate threshold of ``col``.

    Returns:
        Columns column, threshold, n_above, share_above.
    """
    s = df[col].dropna()
    n = len(s)
    rows = [
        {
            "column": col,
            "threshold": float(t),
            "n_above": int((s > t).sum()),
            "share_above": float((s > t).mean()) if n else np.nan,
        }
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def extreme_examples(
    recipes: pd.DataFrame, col: str, n: int = 10, min_value: float | None = None
) -> pd.DataFrame:
    """The ``n`` recipes with the largest ``col`` (optionally only those above ``min_value``).

    Returns:
        Columns id, name, column, value, n_steps, n_ingredients; sorted by value descending.
        Recipes with a missing ``col`` are skipped.
    """
    data = recipes.loc[recipes[col].notna()]
    if min_value is not None:
        data = data.loc[data[col] > min_value]
    top = data.nlargest(n, col)
    return pd.DataFrame(
        {
            "id": top["id"].astype("int64").to_numpy(),
            "name": top["name"].to_numpy(),
            "column": col,
            "value": top[col].astype("float64").to_numpy(),
            "n_steps": top["n_steps"].astype("int64").to_numpy(),
            "n_ingredients": top["n_ingredients"].astype("int64").to_numpy(),
        }
    )


def invalid_counts(df: pd.DataFrame, bounds: Bounds) -> pd.DataFrame:
    """Rows outside the valid range ``[lower, upper]`` per column (``None`` = no bound).

    Returns:
        Columns column, lower, upper, n_below, n_above, n_invalid, share_invalid.
    """
    rows = []
    for col, (lower, upper) in bounds.items():
        s = df[col]
        below = int((s < lower).sum()) if lower is not None else 0
        above = int((s > upper).sum()) if upper is not None else 0
        n = int(s.notna().sum())
        rows.append(
            {
                "column": col,
                "lower": float(lower) if lower is not None else np.nan,
                "upper": float(upper) if upper is not None else np.nan,
                "n_below": below,
                "n_above": above,
                "n_invalid": below + above,
                "share_invalid": (below + above) / n if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def before_after_table(
    frames: Mapping[str, pd.DataFrame], cols: Sequence[str], k: float = 1.5
) -> pd.DataFrame:
    """Summary and box-plot statistics of ``cols`` at each treatment stage.

    Args:
        frames: stage label -> DataFrame, in display order (e.g. ``{"before": raw,
            "after_rules": clipped, "after_log1p": logged}``).
        cols: Columns to summarise; columns missing from a frame are skipped for that frame.
        k: Tukey fence multiplier used for the whiskers.

    Returns:
        Columns column, stage, n, n_missing, min, whisker_low, q1, median, q3, whisker_high, max,
        mean, skew, n_outside_fences. One row per (column, stage).
    """
    rows = []
    for col in cols:
        for stage, df in frames.items():
            if col not in df.columns:
                continue
            s = df[col].astype("float64")
            v = s.dropna()
            row: dict[str, Any] = {
                "column": col,
                "stage": stage,
                "n": len(s),
                "n_missing": int(s.isna().sum()),
            }
            if len(v):
                q1, med, q3 = (float(x) for x in v.quantile([0.25, 0.5, 0.75]))
                lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
                inside = v[(v >= lo) & (v <= hi)]
                row.update(
                    min=float(v.min()),
                    whisker_low=float(inside.min()),
                    q1=q1,
                    median=med,
                    q3=q3,
                    whisker_high=float(inside.max()),
                    max=float(v.max()),
                    mean=float(v.mean()),
                    skew=float(v.skew()) if len(v) > 2 else np.nan,
                    n_outside_fences=int(len(v) - len(inside)),
                )
            else:
                row.update(
                    dict.fromkeys(
                        ("min", "whisker_low", "q1", "median", "q3", "whisker_high", "max"),
                        np.nan,
                    ),
                    mean=np.nan,
                    skew=np.nan,
                    n_outside_fences=0,
                )
            rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================== transformers


@register("outlier", "invalid_rule")
class InvalidValueRule(BaseEstimator, TransformerMixin):
    """Turn values outside the valid range into NaN (``bounds``: column -> [lower, upper]).

    Bounds are domain rules (e.g. ``n_steps >= 1``, ``minutes <= 1 year``), not learned from data,
    so the transformer is stateless. Rows are never dropped; what happens to the NaN (imputation or
    exclusion) is decided by later steps.
    """

    def __init__(self, bounds: Bounds | None = None) -> None:
        self.bounds = bounds

    def fit(self, X: pd.DataFrame, y: object = None) -> InvalidValueRule:
        _require_columns(X, list(self._bounds()))
        for col, pair in self._bounds().items():
            if len(pair) != 2:
                raise ValueError(f"bounds[{col!r}] must be [lower, upper], got {pair!r}")
            lower, upper = pair
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"bounds[{col!r}]: lower {lower} > upper {upper}")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "n_features_in_")
        _require_columns(X, list(self._bounds()))
        out = X.copy()
        for col, (lower, upper) in self._bounds().items():
            values = out[col].astype("float64")
            invalid = pd.Series(False, index=out.index)
            if lower is not None:
                invalid |= values < lower
            if upper is not None:
                invalid |= values > upper
            out[col] = values.mask(invalid)
        return out

    def _bounds(self) -> Bounds:
        return self.bounds or {}


@register("outlier", "quantile_clip")
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip ``columns`` to the [``lower_q``, ``upper_q``] quantiles learned in ``fit``.

    ``bounds_`` (column -> (low, high)) is computed from the fit data only (NaN ignored) and is
    never changed by ``transform``.
    """

    def __init__(
        self, columns: Sequence[str] = (), lower_q: float = 0.0, upper_q: float = 0.99
    ) -> None:
        self.columns = columns
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X: pd.DataFrame, y: object = None) -> QuantileClipper:
        if not 0.0 <= self.lower_q < self.upper_q <= 1.0:
            raise ValueError(
                f"need 0 <= lower_q < upper_q <= 1, got {self.lower_q}, {self.upper_q}"
            )
        _require_columns(X, list(self.columns))
        self.bounds_: dict[str, tuple[float, float]] = {}
        for col in self.columns:
            v = X[col].astype("float64").dropna()
            if v.empty:
                raise ValueError(f"{col} has no observed values to learn clip bounds from")
            low, high = v.quantile([self.lower_q, self.upper_q])
            self.bounds_[col] = (float(low), float(high))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "bounds_")
        _require_columns(X, list(self.bounds_))
        out = X.copy()
        for col, (low, high) in self.bounds_.items():
            out[col] = out[col].astype("float64").clip(lower=low, upper=high)
        return out

    def bounds_table(self, X: pd.DataFrame | None = None) -> pd.DataFrame:
        """Learned bounds per column; with ``X`` also how many of its values would be clipped.

        Returns:
            Columns column, lower_q, upper_q, lower, upper, n_clipped_low, n_clipped_high
            (counts are 0 when ``X`` is None). Reading this never changes ``bounds_``.
        """
        check_is_fitted(self, "bounds_")
        rows = []
        for col, (low, high) in self.bounds_.items():
            values = X[col].astype("float64") if X is not None else pd.Series(dtype="float64")
            rows.append(
                {
                    "column": col,
                    "lower_q": float(self.lower_q),
                    "upper_q": float(self.upper_q),
                    "lower": low,
                    "upper": high,
                    "n_clipped_low": int((values < low).sum()),
                    "n_clipped_high": int((values > high).sum()),
                }
            )
        return pd.DataFrame(rows)


@register("outlier", "log1p")
class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Replace ``columns`` by ``log1p(x)``. Stateless; negative values raise (NaN stays NaN)."""

    def __init__(self, columns: Sequence[str] = ()) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object = None) -> Log1pTransformer:
        _require_columns(X, list(self.columns))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "n_features_in_")
        _require_columns(X, list(self.columns))
        out = X.copy()
        for col in self.columns:
            values = out[col].astype("float64")
            if (values < 0).any():
                raise ValueError(f"{col} has negative values; log1p needs x >= 0")
            out[col] = np.log1p(values)
        return out

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            out[col] = np.expm1(out[col].astype("float64"))
        return out


def outlier_steps_from_config(block: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Build the configured outlier transformers, in :data:`STEP_ORDER`.

    ``block`` is ``cfg.section("outliers")``::

        invalid_rule: {minutes: [0, 525600], n_steps: [1, null], ...}
        quantile_clip: {lower_q: 0.0, upper_q: 0.99}
        columns: {minutes: [invalid_rule, quantile_clip, log1p], ...}

    Returns:
        ``[(step_name, transformer), ...]`` for the steps used by at least one column. The 1.9
        pipeline can interleave other steps (e.g. imputation after ``invalid_rule``).
    """
    columns: Mapping[str, Sequence[str]] = block.get("columns", {})
    unknown = {s for steps in columns.values() for s in steps} - set(STEP_ORDER)
    if unknown:
        raise ValueError(f"unknown outlier steps in config: {sorted(unknown)}")
    steps: list[tuple[str, Any]] = []
    for step in STEP_ORDER:
        cols = [c for c, s in columns.items() if step in s]
        if not cols:
            continue
        if step == "invalid_rule":
            rules = block.get("invalid_rule", {})
            missing = [c for c in cols if c not in rules]
            if missing:
                raise ValueError(f"invalid_rule listed for {missing} but no bounds given")
            steps.append(
                (step, registry.create("outlier", step, bounds={c: rules[c] for c in cols}))
            )
        elif step == "quantile_clip":
            params = block.get("quantile_clip", {})
            steps.append((step, registry.create("outlier", step, columns=tuple(cols), **params)))
        else:
            steps.append((step, registry.create("outlier", step, columns=tuple(cols))))
    return steps


def _require_columns(X: pd.DataFrame, cols: Sequence[str]) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(X).__name__}")
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")
