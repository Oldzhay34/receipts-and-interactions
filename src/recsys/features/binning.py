"""Binning strategy transformers and comparison helpers (Phase 1.7)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from recsys.registry import register


def _to_series(X: pd.DataFrame | pd.Series, column: str | None) -> pd.Series:
    if isinstance(X, pd.DataFrame):
        if column is None:
            raise ValueError("`column` must be specified when transforming a DataFrame.")
        return pd.to_numeric(X[column], errors="coerce")
    return pd.to_numeric(X, errors="coerce")


def _apply_cut(
    series: pd.Series,
    edges: np.ndarray,
    labels: Sequence[str],
    right: bool = True,
) -> pd.Categorical:
    """Cut a numeric series using fixed bin edges; out-of-range values fall into outer bins.

    ``right=True`` gives bins ``(a, b]``; ``right=False`` gives ``[a, b)`` (e.g. "quick < 30").
    """
    extended = np.array(edges, dtype=float).copy()
    extended[0] = -np.inf
    extended[-1] = np.inf
    return pd.cut(series, bins=extended, labels=list(labels), include_lowest=True, right=right)


def _format_output(
    X: pd.DataFrame | pd.Series,
    binned: pd.Categorical,
    column: str | None,
    output_col: str | None,
) -> pd.DataFrame | pd.Series:
    if isinstance(X, pd.DataFrame):
        out = X.copy()
        target_name = output_col or f"{column}_bin"
        out[target_name] = binned
        return out
    return pd.Series(binned, index=X.index, name=output_col or f"{X.name}_bin")


@register("binner", "domain")
class DomainBins(BaseEstimator, TransformerMixin):
    """Domain-knowledge binning with explicit fixed edges and labels.

    ``right`` chooses which side of an edge is closed: ``True`` -> ``(a, b]`` (an edge value goes
    to the lower bin), ``False`` -> ``[a, b)`` (it goes to the upper bin, e.g. quick < 30).
    """

    def __init__(
        self,
        column: str,
        edges: Sequence[float],
        labels: Sequence[str],
        output_col: str | None = None,
        right: bool = True,
    ) -> None:
        self.column = column
        self.edges = edges
        self.labels = labels
        self.output_col = output_col
        self.right = right

    def fit(self, X: pd.DataFrame | pd.Series, y: pd.Series | None = None) -> DomainBins:
        if len(self.edges) != len(self.labels) + 1:
            raise ValueError("`edges` length must equal `len(labels) + 1`.")
        self.edges_ = np.asarray(self.edges, dtype=float)
        self.labels_ = list(self.labels)
        return self

    def transform(self, X: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        series = _to_series(X, self.column)
        binned = _apply_cut(series, self.edges_, self.labels_, right=self.right)
        return _format_output(X, binned, self.column, self.output_col)


@register("binner", "equal_width")
class EqualWidthBins(BaseEstimator, TransformerMixin):
    """Equal-width binning whose edges are learned from fit() data only."""

    def __init__(
        self,
        column: str,
        n_bins: int = 3,
        labels: Sequence[str] | None = None,
        output_col: str | None = None,
    ) -> None:
        self.column = column
        self.n_bins = n_bins
        self.labels = labels
        self.output_col = output_col

    def fit(self, X: pd.DataFrame | pd.Series, y: pd.Series | None = None) -> EqualWidthBins:
        series = _to_series(X, self.column).dropna()
        if series.empty:
            raise ValueError(f"Column '{self.column}' has no valid numeric values to fit.")
        min_val, max_val = float(series.min()), float(series.max())
        if np.isclose(min_val, max_val):
            max_val = min_val + 1.0
        self.edges_ = np.linspace(min_val, max_val, self.n_bins + 1)
        self.labels_ = (
            list(self.labels)
            if self.labels is not None
            else [f"bin_{i}" for i in range(self.n_bins)]
        )
        return self

    def transform(self, X: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        series = _to_series(X, self.column)
        binned = _apply_cut(series, self.edges_, self.labels_)
        return _format_output(X, binned, self.column, self.output_col)


@register("binner", "equal_frequency")
class EqualFrequencyBins(BaseEstimator, TransformerMixin):
    """Quantile (equal-frequency) binning learned from fit() data only."""

    def __init__(
        self,
        column: str,
        n_bins: int = 3,
        labels: Sequence[str] | None = None,
        output_col: str | None = None,
    ) -> None:
        self.column = column
        self.n_bins = n_bins
        self.labels = labels
        self.output_col = output_col

    def fit(self, X: pd.DataFrame | pd.Series, y: pd.Series | None = None) -> EqualFrequencyBins:
        series = _to_series(X, self.column).dropna()
        if series.empty:
            raise ValueError(f"Column '{self.column}' has no valid numeric values to fit.")
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        raw_edges = np.quantile(series.to_numpy(), quantiles)
        unique_edges = np.unique(raw_edges)
        if len(unique_edges) < 2:
            unique_edges = np.array([float(series.min()), float(series.max()) + 1.0])
        self.edges_ = unique_edges
        n_actual = len(self.edges_) - 1
        if self.labels is not None and len(self.labels) == n_actual:
            self.labels_ = list(self.labels)
        else:
            if self.labels is not None:
                warnings.warn(
                    f"{self.column}: only {n_actual} distinct quantile bins (tied values), "
                    f"so the {len(self.labels)} configured labels cannot be used; "
                    f"falling back to q_0..q_{n_actual - 1}",
                    stacklevel=2,
                )
            self.labels_ = [f"q_{i}" for i in range(n_actual)]
        return self

    def transform(self, X: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        series = _to_series(X, self.column)
        binned = _apply_cut(series, self.edges_, self.labels_)
        return _format_output(X, binned, self.column, self.output_col)


def compare_binning_strategies(
    df: pd.DataFrame,
    column: str,
    strategies: dict[str, BaseEstimator],
) -> pd.DataFrame:
    """Fit and compare multiple binning strategies on a column, returning a summary table."""
    rows: list[dict[str, object]] = []
    for name, binner in strategies.items():
        transformed = binner.fit_transform(df)
        out_col = getattr(binner, "output_col", None) or f"{column}_bin"
        counts = transformed[out_col].value_counts(dropna=False)
        total = int(counts.sum())
        for label, cnt in counts.items():
            rows.append(
                {
                    "column": column,
                    "strategy": name,
                    "bin": str(label),
                    "count": int(cnt),
                    "pct": round(float(cnt / total * 100), 2) if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)
