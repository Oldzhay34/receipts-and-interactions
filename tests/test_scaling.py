"""Unit tests for numeric scaling and combined distance (Phase 1.8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial.distance import euclidean

from recsys.features.distance import JaccardDistance
from recsys.features.scaling import (
    WeightedCombinedDistance,
    build_numeric_scaler,
    compare_scalers,
)
from recsys.interfaces import DistanceMetric


def test_build_numeric_scaler_minmax_and_zscore() -> None:
    X_train = np.array([[10.0, 100.0], [30.0, 500.0], [60.0, 1200.0]])
    X_test = np.array([[20.0, 300.0]])

    mm = build_numeric_scaler({"scaling": {"method": "minmax", "use_log1p": True}})
    mm_train = mm.fit_transform(X_train)
    mm_test = mm.transform(X_test)
    assert np.isclose(mm_train.min(), 0.0)
    assert np.isclose(mm_train.max(), 1.0)
    assert mm_test.shape == (1, 2)

    zs = build_numeric_scaler({"scaling": {"method": "zscore", "use_log1p": True}})
    zs_train = zs.fit_transform(X_train)
    assert np.allclose(zs_train.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(zs_train.std(axis=0), 1.0, atol=1e-7)


def test_build_numeric_scaler_invalid_method_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported scaling method"):
        build_numeric_scaler({"method": "robust"})


def test_unscaled_calories_dominates_knn_distance() -> None:
    """Counterexample: without scaling, calories alone dictates Euclidean distance."""
    # Features: [minutes, n_ingredients, calories]
    query = np.array([15.0, 4.0, 300.0])
    # Recipe A: almost identical cooking time & ingredients, slightly higher calories
    recipe_a = np.array([16.0, 4.0, 450.0])
    # Recipe B: completely different cooking time & ingredients, identical calories
    recipe_b = np.array([120.0, 18.0, 300.0])

    # Unscaled: Recipe B appears closer than Recipe A purely because of calories scale
    assert euclidean(query, recipe_b) < euclidean(query, recipe_a)

    # Scaled (Log1p + MinMax): Recipe A is correctly closer than Recipe B
    scaler = build_numeric_scaler({"method": "minmax", "use_log1p": True})
    catalog = np.vstack([query, recipe_a, recipe_b])
    scaled = scaler.fit_transform(catalog)
    assert euclidean(scaled[0], scaled[1]) < euclidean(scaled[0], scaled[2])


def test_weighted_combined_distance_injects_metric() -> None:
    metric = WeightedCombinedDistance(ingredient_metric=JaccardDistance(), numeric_weight=0.25)
    assert isinstance(metric, DistanceMetric)

    A_ing = csr_matrix([[1, 1, 0], [0, 0, 1]])
    B_ing = csr_matrix([[1, 1, 0], [1, 0, 1]])
    A_num = np.array([[0.0, 0.0], [1.0, 1.0]])
    B_num = np.array([[0.0, 0.0], [0.5, 0.5]])

    D = metric.pairwise((A_ing, A_num), (B_ing, B_num))
    assert D.shape == (2, 2)
    # Identical recipe has zero distance
    assert np.isclose(D[0, 0], 0.0)
    # All distances are finite and non-negative
    assert np.all(D >= 0.0)


def test_compare_scalers_output_columns() -> None:
    df = pd.DataFrame({"minutes": [10, 30, 60, 120], "calories": [100, 250, 500, 1500]})
    summary = compare_scalers(df, ["minutes", "calories"])
    assert set(summary["transform"]) == {"raw", "log1p_minmax", "log1p_zscore"}
    assert {"feature", "transform", "min", "median", "mean", "std", "max"}.issubset(summary.columns)


def test_log1p_step_really_takes_the_log() -> None:
    X = np.array([[0.0, 9.0], [99.0, 999.0], [9.0, 0.0]])
    scaler = build_numeric_scaler({"method": "minmax", "use_log1p": True})
    out = scaler.fit_transform(X)
    logged = np.log1p(X)
    expected = (logged - logged.min(axis=0)) / (logged.max(axis=0) - logged.min(axis=0))
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(scaler.named_steps["log1p"].transform(X), logged)


def test_log1p_step_accepts_dataframe_then_ndarray() -> None:
    train = pd.DataFrame({"minutes": [10.0, 100.0], "calories": [50.0, 500.0]})
    step = build_numeric_scaler({"method": "zscore", "use_log1p": True}).named_steps["log1p"]
    step.fit(train)
    np.testing.assert_allclose(step.transform(train.to_numpy()), np.log1p(train.to_numpy()))
    with pytest.raises(ValueError, match="expected 2 columns"):
        step.transform(np.ones((1, 3)))
