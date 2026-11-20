"""Numeric scaling pipeline and combined distance metric (Phase 1.8)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from recsys.interfaces import DistanceMetric
from recsys.registry import register

try:
    from recsys.cleaning.outliers import Log1pTransformer  # type: ignore[import-not-found]
except ImportError:

    @register("transformer", "log1p_fallback")
    class Log1pTransformer(BaseEstimator, TransformerMixin):
        """Fallback log1p transformer when Phase 1.4 is not yet merged."""

        def fit(self, X: Any, y: Optional[Any] = None) -> "Log1pTransformer":
            return self

        def transform(self, X: Any) -> np.ndarray:
            arr = np.asarray(X, dtype=float)
            return np.log1p(np.clip(arr, a_min=0.0, a_max=None))


class _Log1pAdapter(BaseEstimator, TransformerMixin):
    """Wraps 1.4's Log1pTransformer so it accepts both DataFrame and ndarray."""

    def __init__(self) -> None:
        self._inner = Log1pTransformer()

    def fit(self, X: Any, y: Optional[Any] = None) -> "_Log1pAdapter":
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self._inner.fit(df, y)
        return self

    def transform(self, X: Any) -> np.ndarray:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        out = self._inner.transform(df)
        return np.asarray(out, dtype=float)


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
        A: Tuple[Any, np.ndarray],
        B: Tuple[Any, np.ndarray],
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