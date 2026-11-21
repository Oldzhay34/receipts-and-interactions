"""Numeric scaling pipeline and combined distance metric (Phase 1.8)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from recsys.cleaning.outliers import Log1pTransformer
from recsys.interfaces import DistanceMetric
from recsys.registry import register


class _Log1pAdapter(BaseEstimator, TransformerMixin):
    """Apply 1.4's Log1pTransformer to EVERY column of a DataFrame or ndarray.

    ``Log1pTransformer`` only transforms the columns it is given, so the column list is taken
    from the fit data (``inner_``). An ndarray passed to ``transform`` must have the same number
    of columns as the fit data; it is mapped onto the fitted column names.
    """

    def fit(self, X: Any, y: Any = None) -> _Log1pAdapter:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=float))
        self.inner_ = Log1pTransformer(columns=tuple(df.columns)).fit(df)
        self.n_features_in_ = df.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "inner_")
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"expected {self.n_features_in_} columns, got array of shape {arr.shape}"
                )
            df = pd.DataFrame(arr, columns=list(self.inner_.columns))
        return np.asarray(self.inner_.transform(df), dtype=float)


class _ColumnScaler(BaseEstimator, TransformerMixin):
    """Scale only ``columns`` of a DataFrame; every other column is passed through unchanged.

    The scaling statistics are learned in ``fit`` (on TRAIN data in the 1.9 pipeline) and stored
    in ``scaler_``; ``transform`` only applies them, so a val / test value outside the train range
    is not refitted (Min-Max can give values < 0 or > 1 there). No log is taken here: the 1.9
    pipeline applies log1p as its own step before scaling. NaN is ignored in ``fit`` and kept.
    """

    def __init__(self, columns: tuple[str, ...] | list[str] = ()) -> None:
        self.columns = columns

    def _make_scaler(self) -> BaseEstimator:
        raise NotImplementedError

    def _require(self, X: Any) -> list[str]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"expected a DataFrame, got {type(X).__name__}")
        cols = list(self.columns)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise KeyError(f"missing required columns: {missing}")
        return cols

    def fit(self, X: pd.DataFrame, y: Any = None) -> _ColumnScaler:
        cols = self._require(X)
        self.scaler_ = self._make_scaler()
        if cols:
            self.scaler_.fit(X[cols].astype("float64"))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "scaler_")
        cols = self._require(X)
        out = X.copy()
        if cols:
            out[cols] = self.scaler_.transform(X[cols].astype("float64"))
        return out


@register("scaler", "minmax")
class MinMaxColumnScaler(_ColumnScaler):
    """Min-Max scaling of ``columns`` to the train range [0, 1] (registry: scaler / minmax)."""

    def _make_scaler(self) -> BaseEstimator:
        return MinMaxScaler()


@register("scaler", "zscore")
class ZScoreColumnScaler(_ColumnScaler):
    """Z-score scaling of ``columns`` with train mean / std (registry: scaler / zscore)."""

    def _make_scaler(self) -> BaseEstimator:
        return StandardScaler()


def _extract_scaling_cfg(cfg: Any) -> dict[str, Any]:
    """Read scaling settings from dict or config object with safe defaults."""
    if isinstance(cfg, Mapping):
        section = cfg.get("scaling", cfg)
    else:
        section = getattr(cfg, "scaling", {}) or {}
    if not isinstance(section, Mapping):
        return {"method": "minmax", "use_log1p": True}
    return {
        "method": str(section.get("method", "minmax")).lower(),
        "use_log1p": bool(section.get("use_log1p", True)),
    }


def build_numeric_scaler(cfg: Any = None) -> Pipeline:
    """Build a sklearn Pipeline: optional Log1p -> MinMaxScaler | StandardScaler.

    Note: In the production ML pipeline (1.9), this scaler is fit strictly on
    the training split and applied to validation/test splits without refitting.
    """
    settings = _extract_scaling_cfg(cfg or {})
    method = settings["method"]
    use_log1p = settings["use_log1p"]

    steps: list[tuple[str, BaseEstimator]] = []
    if use_log1p:
        steps.append(("log1p", _Log1pAdapter()))

    if method in ("minmax", "min_max"):
        steps.append(("scaler", MinMaxScaler()))
    elif method in ("zscore", "z_score", "standard"):
        steps.append(("scaler", StandardScaler()))
    else:
        raise ValueError(f"Unsupported scaling method: '{method}'. Use 'minmax' or 'zscore'.")

    return Pipeline(steps=steps)


def compare_scalers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Compute before/after summary statistics for Raw, Log1p+MinMax, and Log1p+ZScore."""
    valid_cols = [c for c in cols if c in df.columns]
    raw_df = df[valid_cols].apply(pd.to_numeric, errors="coerce").dropna()

    mm_arr = build_numeric_scaler({"method": "minmax", "use_log1p": True}).fit_transform(raw_df)
    zs_arr = build_numeric_scaler({"method": "zscore", "use_log1p": True}).fit_transform(raw_df)

    rows: list[dict[str, Any]] = []
    for idx, col in enumerate(valid_cols):
        for label, values in (
            ("raw", raw_df[col].to_numpy(dtype=float)),
            ("log1p_minmax", mm_arr[:, idx]),
            ("log1p_zscore", zs_arr[:, idx]),
        ):
            rows.append(
                {
                    "feature": col,
                    "transform": label,
                    "min": round(float(np.min(values)), 4),
                    "median": round(float(np.median(values)), 4),
                    "mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values)), 4),
                    "max": round(float(np.max(values)), 4),
                }
            )
    return pd.DataFrame(rows)


class WeightedCombinedDistance:
    """Combines an injected ingredient DistanceMetric with scaled numeric distance.

    Implements the DistanceMetric protocol: ``pairwise(A, B) -> np.ndarray``.
    Each input ``A`` and ``B`` is a tuple ``(X_ingredients, X_numeric_scaled)``.
    Combined distance = (1 - numeric_weight) * d_ing + numeric_weight * d_num_norm.
    """

    def __init__(
        self,
        ingredient_metric: DistanceMetric,
        numeric_weight: float = 0.2,
    ) -> None:
        if not 0.0 <= numeric_weight <= 1.0:
            raise ValueError("`numeric_weight` must be between 0.0 and 1.0.")
        self.ingredient_metric = ingredient_metric
        self.numeric_weight = float(numeric_weight)

    def pairwise(
        self,
        A: tuple[Any, np.ndarray],
        B: tuple[Any, np.ndarray],
    ) -> np.ndarray:
        A_ing, A_num = A
        B_ing, B_num = B

        d_ing = np.asarray(self.ingredient_metric.pairwise(A_ing, B_ing), dtype=float)
        A_num_arr = np.asarray(A_num, dtype=float)
        B_num_arr = np.asarray(B_num, dtype=float)

        if A_num_arr.ndim == 1:
            A_num_arr = A_num_arr.reshape(1, -1)
        if B_num_arr.ndim == 1:
            B_num_arr = B_num_arr.reshape(1, -1)

        # Normalize Euclidean distance by sqrt(n_features) so MinMax-scaled values stay in [0, 1]
        n_features = max(A_num_arr.shape[1], 1)
        d_num = cdist(A_num_arr, B_num_arr, metric="euclidean") / np.sqrt(n_features)

        return (1.0 - self.numeric_weight) * d_ing + self.numeric_weight * d_num
