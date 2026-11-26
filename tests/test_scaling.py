"""Unit tests for numeric scaling and combined distance (Phase 1.8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial.distance import euclidean

from recsys.features.distance import JaccardDistance
from recsys.features.scaling import (
    CombinedFeatures,
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


def test_column_scalers_learn_on_fit_data_only() -> None:
    from sklearn.base import clone

    from recsys import registry

    train = pd.DataFrame({"a": [0.0, 5.0, 10.0], "b": [1.0, 2.0, 3.0], "keep": ["x", "y", "z"]})
    val = pd.DataFrame({"a": [20.0, -10.0], "b": [2.0, 2.0], "keep": ["u", "v"]})

    mm = registry.create("scaler", "minmax", columns=["a", "b"]).fit(train)
    out = mm.transform(val)
    np.testing.assert_allclose(out["a"], [2.0, -1.0])  # train range, not refitted on val
    assert out["keep"].tolist() == ["u", "v"]  # other columns untouched
    np.testing.assert_allclose(mm.transform(train)[["a", "b"]].to_numpy().min(axis=0), 0.0)

    zs = registry.create("scaler", "zscore", columns=["a"]).fit(train)
    np.testing.assert_allclose(zs.transform(train)["a"].mean(), 0.0, atol=1e-12)
    assert zs.scaler_.mean_[0] == pytest.approx(5.0)
    zs.transform(val)
    assert zs.scaler_.mean_[0] == pytest.approx(5.0)

    assert clone(mm).get_params() == {"columns": ["a", "b"]}
    with pytest.raises(KeyError):
        mm.transform(val.drop(columns="a"))
    with pytest.raises(TypeError):
        registry.create("scaler", "minmax", columns=["a"]).fit(train.to_numpy())


# --------------------------------------------------------------------------- combined features


def _combined() -> tuple[CombinedFeatures, np.ndarray]:
    ing = csr_matrix(
        np.array(
            [[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0], [0, 1, 1, 0]]
        )
    )
    num = np.array([[0.1, 0.2], [0.9, 0.8], [0.15, 0.2], [0.5, 0.5], [0.55, 0.45], [0.0, 1.0]])
    return CombinedFeatures(ing, num), np.array(["high", "high", "high", "low", "low", "low"])


def test_combined_features_behave_like_a_matrix() -> None:
    X, _ = _combined()
    assert X.shape == (6, 6) and len(X) == 6 and X.n_ingredient_columns == 4
    sub = X[np.array([4, 0])]
    assert isinstance(sub, CombinedFeatures) and sub.shape == (2, 6)
    np.testing.assert_array_equal(sub.numeric, [[0.55, 0.45], [0.1, 0.2]])
    assert X[1:3].shape == (2, 6) and X[-1].shape == (1, 6)
    assert X[np.array([True, False, True, False, False, False])].shape == (2, 6)
    with pytest.raises(ValueError, match="rows"):
        CombinedFeatures(csr_matrix(np.eye(3)), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="NaN"):
        CombinedFeatures(csr_matrix(np.eye(2)), np.array([[np.nan], [0.0]]))
    with pytest.raises(IndexError):
        X[6]


def test_weighted_combined_distance_accepts_combined_features() -> None:
    X, _ = _combined()
    metric = WeightedCombinedDistance(ingredient_metric=JaccardDistance(), numeric_weight=0.25)
    D = metric.pairwise(X[:2], X)
    assert D.shape == (2, 6) and np.isclose(D[0, 0], 0.0)
    # same as the tuple form
    D_tuple = metric.pairwise((X.ingredients[:2], X.numeric[:2]), (X.ingredients, X.numeric))
    np.testing.assert_allclose(D, D_tuple)
    # weight 0 is the pure ingredient distance
    D0 = WeightedCombinedDistance(JaccardDistance(), numeric_weight=0.0).pairwise(X, X)
    np.testing.assert_allclose(D0, JaccardDistance().pairwise(X.ingredients, X.ingredients))
    with pytest.raises(TypeError, match="CombinedFeatures"):
        metric.pairwise(X.ingredients, X.ingredients)


def test_rating_models_run_with_the_combined_distance() -> None:
    """1.12-A O2: the combined distance must plug into the kNN code (P2 grid search)."""
    from recsys.models.knn import (
        RatingClassifierKNN,
        RatingRegressorKNN,
        nearest_neighbors,
        recipe_rating_dataset,
    )

    X, y = _combined()
    metric = WeightedCombinedDistance(JaccardDistance(), numeric_weight=0.5)
    idx, dist = nearest_neighbors(metric, X, X, k=2, batch_size=4)
    assert idx[:, 0].tolist() == list(range(6)) and np.allclose(dist[:, 0], 0.0)

    clf = RatingClassifierKNN(metric=metric, n_neighbors=3, batch_size=2).fit(X, y)
    assert clf.predict(X[[0, 3]]).tolist() == ["high", "low"]
    reg = RatingRegressorKNN(metric=metric, n_neighbors=2, weights="distance").fit(
        X, np.arange(6.0)
    )
    assert reg.predict(X).shape == (6,)

    targets = pd.DataFrame(
        {"recipe_id": [30, 10], "rating_class": ["high", "high"], "avg_rating": [4.0, 5.0]}
    )
    X_t, y_t, ids = recipe_rating_dataset(X, [10, 20, 30, 40, 50, 60], targets)
    assert isinstance(X_t, CombinedFeatures) and X_t.shape == (2, 6) and ids.tolist() == [10, 30]
    np.testing.assert_array_equal(X_t.numeric, X.numeric[[0, 2]])


def test_build_numeric_scaler_reads_the_project_config(tmp_path) -> None:  # noqa: ANN001
    """1.12-A O6: a Config object must not silently fall back to the default."""
    import yaml
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    from recsys.config import DEFAULT_CONFIG_PATH, load_config

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["scaling"] = {"method": "zscore", "use_log1p": False}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path, data_root=tmp_path)

    pipe = build_numeric_scaler(cfg)
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)
    assert "log1p" not in pipe.named_steps
    assert isinstance(build_numeric_scaler(None).named_steps["scaler"], MinMaxScaler)
    with pytest.raises(TypeError, match="Config, a mapping or None"):
        build_numeric_scaler("zscore")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="scaling block"):
        build_numeric_scaler({"scaling": "zscore"})


def test_notebook_08_helpers() -> None:
    """1.12-A O4: the scaled frames and the distance sample come from src, not the notebook."""
    from recsys.contracts import combined_distance_sample_schema, validate
    from recsys.features.scaling import combined_distance_sample, scaled_views

    df = pd.DataFrame(
        {
            "minutes": [10, 20, 30, np.nan, 40, 50],
            "n_ingredients": [3, 5, 7, 2, 9, 4],
            "calories": [100.0, 250.0, 400.0, 50.0, 800.0, 120.0],
            "ingredients": [["a", "b"], ["a"], ["b", "c"], ["d"], ["a", "c"], ["c"]],
        }
    )
    cols = ["minutes", "n_ingredients", "calories"]
    views = scaled_views(df, cols, {"method": "zscore", "use_log1p": True})
    assert set(views) == {"raw", "minmax", "zscore"} and len(views["raw"]) == 5  # NaN row dropped
    assert np.isclose(views["minmax"].min().min(), 0.0)
    assert np.isclose(views["minmax"].max().max(), 1.0)
    assert np.allclose(views["zscore"].mean(), 0.0)

    dist = combined_distance_sample(df, views["minmax"], numeric_weight=0.5, n=4)
    validate(dist, combined_distance_sample_schema(4))
    assert np.allclose(np.diag(dist), 0.0) and np.allclose(dist, dist.T)
    assert list(dist.columns) == ["recipe_0", "recipe_1", "recipe_2", "recipe_3"]
