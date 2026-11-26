"""Numeric scaling pipeline and combined distance metric (Phase 1.8)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
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


class CombinedFeatures:
    """Ingredient matrix and scaled numeric features of the same recipes, as one matrix-like object.

    The kNN code (:func:`recsys.models.knn.nearest_neighbors`, the rating models,
    :func:`recsys.models.knn.recipe_rating_dataset`) only needs ``.shape`` and row indexing, so a
    ``CombinedFeatures`` can be passed wherever an ingredient matrix is expected, together with
    :class:`WeightedCombinedDistance` as the metric.

    Args:
        ingredients: ``(n, n_vocab)`` ingredient matrix (sparse or dense), rows = recipes.
        numeric: ``(n, n_numeric)`` numeric features, already scaled (fit on TRAIN only).

    ``shape`` is ``(n, n_vocab + n_numeric)``. Indexing with a slice, an integer array or a
    boolean mask returns a new ``CombinedFeatures`` with the selected rows; an integer keeps a
    single row (shape ``(1, ...)``).
    """

    __slots__ = ("ingredients", "numeric")

    def __init__(self, ingredients: Any, numeric: Any) -> None:
        num = np.asarray(numeric, dtype=np.float64)
        if num.ndim == 1:
            num = num.reshape(-1, 1)
        if num.ndim != 2:
            raise ValueError(f"numeric must be 2-D, got shape {num.shape}")
        ing = sparse.csr_matrix(ingredients) if sparse.issparse(ingredients) else ingredients
        if getattr(ing, "ndim", 2) != 2:
            raise ValueError("ingredients must be a 2-D matrix")
        if ing.shape[0] != num.shape[0]:
            raise ValueError(f"ingredients has {ing.shape[0]} rows but numeric has {num.shape[0]}")
        if np.isnan(num).any():
            raise ValueError("numeric features contain NaN: impute before combining")
        self.ingredients = ing
        self.numeric = num

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(self.ingredients.shape[0]),
            int(self.ingredients.shape[1] + self.numeric.shape[1]),
        )

    @property
    def n_ingredient_columns(self) -> int:
        return int(self.ingredients.shape[1])

    def __len__(self) -> int:
        return int(self.numeric.shape[0])

    def __getitem__(self, key: Any) -> CombinedFeatures:
        if isinstance(key, tuple):
            raise TypeError("CombinedFeatures supports row indexing only")
        if isinstance(key, (int, np.integer)):
            n = len(self)
            i = int(key) + n if key < 0 else int(key)
            if not 0 <= i < n:
                raise IndexError(f"row {key} out of range for {n} rows")
            key = slice(i, i + 1)
        return CombinedFeatures(self.ingredients[key], self.numeric[key])

    def __repr__(self) -> str:
        return f"CombinedFeatures(shape={self.shape}, n_numeric={self.numeric.shape[1]})"


class WeightedCombinedDistance:
    """Combines an injected ingredient DistanceMetric with scaled numeric distance.

    Implements the DistanceMetric protocol: ``pairwise(A, B) -> np.ndarray``. ``A`` and ``B`` are
    :class:`CombinedFeatures` (an ``(X_ingredients, X_numeric_scaled)`` tuple is still accepted).
    Combined distance = (1 - numeric_weight) * d_ing + numeric_weight * d_num_norm, where
    ``d_num_norm`` is the Euclidean distance of the scaled numeric features divided by
    ``sqrt(n_numeric)`` (in [0, 1] for Min-Max scaled features).

    Example (rating model with the combined distance)::

        X = CombinedFeatures(ingredient_matrix, scaled_numeric)   # rows = catalogue recipes
        model = RatingClassifierKNN(metric=WeightedCombinedDistance(JaccardDistance(), 0.2))
        X_tr, y_tr, ids_tr = recipe_rating_dataset(X, catalogue_ids, targets_train)
        model.fit(X_tr, y_tr)
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

    @staticmethod
    def _parts(X: Any, name: str) -> tuple[Any, np.ndarray]:
        if isinstance(X, CombinedFeatures):
            return X.ingredients, X.numeric
        if isinstance(X, tuple) and len(X) == 2:
            c = CombinedFeatures(*X)
            return c.ingredients, c.numeric
        raise TypeError(
            f"{name} must be CombinedFeatures (or an (ingredients, numeric) tuple), "
            f"got {type(X).__name__}"
        )

    def pairwise(self, A: Any, B: Any) -> np.ndarray:
        A_ing, A_num = self._parts(A, "A")
        B_ing, B_num = self._parts(B, "B")
        if A_num.shape[1] != B_num.shape[1]:
            raise ValueError(f"numeric feature counts differ: {A_num.shape[1]} vs {B_num.shape[1]}")

        d_ing = np.asarray(self.ingredient_metric.pairwise(A_ing, B_ing), dtype=float)
        if self.numeric_weight == 0.0:
            return d_ing
        # Normalize Euclidean distance by sqrt(n_features) so MinMax-scaled values stay in [0, 1]
        n_features = max(A_num.shape[1], 1)
        d_num = cdist(A_num, B_num, metric="euclidean") / np.sqrt(n_features)

        return (1.0 - self.numeric_weight) * d_ing + self.numeric_weight * d_num
