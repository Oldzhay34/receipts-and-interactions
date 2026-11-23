"""Tests for recsys.models.knn and recsys.evaluation.metrics (stage 1.10)."""

from __future__ import annotations

import ast
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recsys import registry
from recsys.config import REPO_ROOT, Config
from recsys.contracts import KNN_PROTOTYPE_OUTPUT_SCHEMA, validate
from recsys.evaluation import metrics as M
from recsys.evaluation.metrics import (
    TestSetAccessError,
    check_split,
    compare_classifiers,
    compare_regressors,
    evaluate_classification,
    evaluate_regression,
)
from recsys.features.distance import CosineDistance, JaccardDistance
from recsys.features.ingredients import IngredientNormalizer, QueryVectorizer
from recsys.interfaces import Recommendation, Recommender
from recsys.io import ArtifactStore
from recsys.io.artifact_store import read_artifact_dir
from recsys.models.knn import (
    KNNRecipeRecommender,
    MajorityBaseline,
    MeanBaseline,
    RatingClassifierKNN,
    RatingRegressorKNN,
    RecipeRecommendation,
    catalogue_rows,
    catalogue_table,
    example_recommendations,
    nearest_neighbors,
    query_vectorizer_from_pipeline,
    recipe_rating_dataset,
)

from .conftest import require_sample

NOTEBOOK = REPO_ROOT / "notebooks" / "10_knn_prototype.ipynb"
NOTEBOOK_09 = REPO_ROOT / "notebooks" / "09_split.ipynb"

# --------------------------------------------------------------------------- toy catalogue

VOCAB = ["butter", "chicken", "egg", "flour", "garlic", "lemon", "rice", "sugar"]

CATALOGUE = pd.DataFrame(
    {
        "id": [10, 20, 30, 40, 50, 60],
        "name": ["lemon chicken", "garlic rice", "shortbread", "cake", "chicken rice", "plain"],
        "minutes": [40.0, 20.0, 45.0, np.nan, 25.0, 5.0],
        "minutes_bin": ["medium", "quick", "medium", "long", "quick", "quick"],
        "ingredients_filtered": [
            ["chicken", "garlic", "lemon"],
            ["garlic", "rice"],
            ["butter", "flour", "sugar"],
            ["butter", "egg", "flour", "sugar"],
            ["chicken", "rice"],
            [],
        ],
    }
)

# recipe 10 -> 5 and 3 (avg 4.0), 30 -> 4; 20 has only a 0 ("no stars"), 40 / 50 / 60 unrated
INTERACTIONS = pd.DataFrame(
    {
        "user_id": [1, 2, 3, 4],
        "recipe_id": [10, 10, 30, 20],
        "rating": [5, 3, 4, 0],
    }
)


def _vectorizer() -> QueryVectorizer:
    return QueryVectorizer(IngredientNormalizer().fit(CATALOGUE), VOCAB)


@pytest.fixture()
def rec() -> KNNRecipeRecommender:
    return KNNRecipeRecommender(_vectorizer(), rating_neighbors=1).fit(CATALOGUE, INTERACTIONS)


# --------------------------------------------------------------------------- neighbours


def test_nearest_neighbors_order_ties_and_batches() -> None:
    ref = sparse.csr_matrix(np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]]))
    X = sparse.csr_matrix(np.array([[1, 0, 0], [0, 0, 1]]))
    idx, dist = nearest_neighbors(JaccardDistance(), X, ref, k=3)
    assert idx[0].tolist() == [0, 1, 2]  # tie between rows 0 and 1: lower row first
    assert dist[0].tolist() == pytest.approx([0.0, 0.0, 0.5])
    assert idx[1, 0] == 3
    idx_b, dist_b = nearest_neighbors(JaccardDistance(), X, ref, k=3, batch_size=1)
    assert (idx_b == idx).all() and np.allclose(dist_b, dist)
    assert nearest_neighbors(JaccardDistance(), X, ref, k=99)[0].shape == (2, 4)  # k capped
    with pytest.raises(ValueError):
        nearest_neighbors(JaccardDistance(), X, ref, k=0)


# --------------------------------------------------------------------------- rating models


def _toy_xy() -> tuple[sparse.csr_matrix, np.ndarray]:
    X = sparse.csr_matrix(
        np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]])
    )
    return X, np.array(["high", "high", "low", "low", "low"])


def test_majority_baseline_predicts_train_mode() -> None:
    X, y = _toy_xy()
    pred = MajorityBaseline().fit(X, y).predict(X[:2])
    assert pred.tolist() == ["low", "low"]


def test_uniform_and_distance_vote_differ_as_expected() -> None:
    X, y = _toy_xy()
    q = sparse.csr_matrix(np.array([[1, 1, 0, 0]]))
    # k=5: uniform -> 3 low vs 2 high; distance-weighted -> the two exact matches dominate
    uniform = RatingClassifierKNN(n_neighbors=5, weights="uniform").fit(X, y)
    weighted = RatingClassifierKNN(n_neighbors=5, weights="distance").fit(X, y)
    assert uniform.predict(q).tolist() == ["low"]
    assert weighted.predict(q).tolist() == ["high"]
    proba = uniform.predict_proba(q)
    assert proba.sum() == pytest.approx(1.0)
    assert uniform.classes_.tolist() == ["high", "low"]
    with pytest.raises(ValueError):
        RatingClassifierKNN(weights="nope").fit(X, y)
    with pytest.raises(RuntimeError):
        RatingClassifierKNN().predict(q)


def test_metric_is_injected_not_hardcoded() -> None:
    calls: list[tuple[int, int]] = []

    class Spy:
        def pairwise(self, A, B):
            calls.append((A.shape[0], B.shape[0]))
            return JaccardDistance().pairwise(A, B)

    X, y = _toy_xy()
    RatingClassifierKNN(metric=Spy(), n_neighbors=1).fit(X, y).predict(X[:2])
    assert calls == [(2, 5)]
    # the default is Jaccard (4.5: ingredient distance is never Euclidean)
    assert isinstance(RatingClassifierKNN()._metric(), JaccardDistance)


def test_regressor_and_mean_baseline() -> None:
    X, _ = _toy_xy()
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    q = sparse.csr_matrix(np.array([[1, 1, 0, 0]]))
    assert RatingRegressorKNN(n_neighbors=2).fit(X, y).predict(q)[0] == pytest.approx(4.5)
    assert MeanBaseline().fit(X, y).predict(q)[0] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        RatingRegressorKNN().fit(X, np.array([1.0, np.nan, 1, 1, 1]))


def test_models_are_registered() -> None:
    assert registry.get("model", "knn_recommender") is KNNRecipeRecommender
    assert registry.get("model", "knn_classifier") is RatingClassifierKNN
    assert registry.get("model", "knn_regressor") is RatingRegressorKNN
    assert registry.get("model", "majority") is MajorityBaseline
    assert registry.get("model", "mean") is MeanBaseline


# --------------------------------------------------------------------------- datasets


def test_recipe_rating_dataset_aligns_rows_and_restricts() -> None:
    matrix = sparse.identity(4, format="csr")
    ids = [7, 5, 9, 3]
    targets = pd.DataFrame(
        {"recipe_id": [9, 3, 5], "avg_rating": [4.0, 2.0, 5.0], "rating_class": ["m", "l", "h"]}
    )
    X, y, out_ids = recipe_rating_dataset(matrix, ids, targets, restrict_to=[3, 9])
    assert out_ids.tolist() == [3, 9]  # sorted by recipe_id
    assert y.tolist() == ["l", "m"]
    assert X.toarray().argmax(axis=1).tolist() == [3, 2]  # rows of ids 3 and 9
    _, y_reg, _ = recipe_rating_dataset(matrix, ids, targets, target_col="avg_rating")
    assert y_reg.tolist() == [2.0, 5.0, 4.0]
    with pytest.raises(KeyError):
        catalogue_rows(ids, [42])


def test_catalogue_table_uses_raw_minutes() -> None:
    feats = CATALOGUE.assign(minutes=[0.1] * 6)  # scaled feature column
    out = catalogue_table(feats, CATALOGUE[["id", "minutes"]])
    assert out["minutes"].tolist()[:3] == [40.0, 20.0, 45.0]
    assert out["id"].tolist() == CATALOGUE["id"].tolist()


def test_catalogue_table_hides_minutes_the_pipeline_treated_as_missing() -> None:
    raw = pd.DataFrame({"id": [10, 20, 30], "minutes": [2147483647.0, 0.0, 45.0]})
    feats = CATALOGUE.iloc[:3].assign(
        minutes=[0.4, 0.4, 0.5], minutes_was_missing=[True, True, False]
    )
    out = catalogue_table(feats, raw)
    assert out["minutes"].isna().tolist() == [True, True, False]  # invalid / 0 -> unknown
    assert out["minutes"].iloc[2] == 45.0
    assert out["minutes_bin"].tolist() == ["medium", "quick", "medium"]  # filter still works


def test_invalid_minutes_never_reach_a_recommendation(fixture_recipes, fixture_interactions):
    """End to end on the fixtures: 106 (int32 max), 107 (two years), 103 / 105 (zero)."""
    from recsys.config import load_config
    from recsys.pipeline.build import PreprocessingPipeline

    cfg = load_config(data_root=REPO_ROOT)
    cfg.raw["ingredients"]["min_count"] = 1  # tiny fixture vocab
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = PreprocessingPipeline.from_config(cfg).fit(fixture_recipes, fixture_interactions)
    feats = pipe.transform(fixture_recipes)
    cat = catalogue_table(feats, fixture_recipes)
    rec = KNNRecipeRecommender(query_vectorizer_from_pipeline(pipe)).fit(
        cat, fixture_interactions, ingredient_matrix=pipe.encode(feats)
    )
    shown = {}
    for q in (["salt"], ["garlic", "salt"], ["flour", "sugar"]):
        shown.update({r.recipe_id: r.minutes for r in rec.recommend(q, k=20)})
    bad = {rid: m for rid, m in shown.items() if m is not None and not 0 < m <= 43200}
    assert not bad, bad
    assert {103, 105, 106, 107} & set(shown)  # the invalid recipes were actually recommended


# --------------------------------------------------------------------------- recommender


def test_recommender_implements_interface_and_returns_plain_types(rec) -> None:
    assert isinstance(rec, Recommender)
    recs = rec.recommend(["Chicken", "fresh garlic"], k=3)
    assert recs and all(isinstance(r, Recommendation) for r in recs)
    assert all(isinstance(r, RecipeRecommendation) for r in recs)
    payload = json.dumps([r.to_dict() for r in recs])  # no pandas / numpy types
    assert "lemon chicken" in payload
    for r in recs:
        for v in r.to_dict().values():
            assert not isinstance(v, (np.generic, pd.Series, pd.DataFrame))


def test_recommendations_sorted_by_similarity_with_matches(rec) -> None:
    recs = rec.recommend(["chicken", "garlic"], k=10)
    sims = [r.similarity for r in recs]
    assert sims == sorted(sims, reverse=True)
    assert recs[0].recipe_id == 10  # 2 of 3 shared
    assert recs[0].matched_ingredients == ("chicken", "garlic")
    assert recs[0].missing_ingredients == ("lemon",)
    assert recs[0].score == pytest.approx(recs[0].similarity) == pytest.approx(2 / 3)
    assert recs[0].distance == pytest.approx(1 / 3)
    # recipes sharing no ingredient (shortbread, cake, plain) are never recommended
    assert {r.recipe_id for r in recs} == {10, 20, 50}


def test_known_and_estimated_ratings(rec) -> None:
    by_id = {r.recipe_id: r for r in rec.recommend(["flour", "sugar", "butter", "egg"], k=5)}
    assert by_id[30].predicted_rating == pytest.approx(4.0)
    assert by_id[30].rating_is_estimated is False
    # 40 has no rating -> estimated from its nearest RATED recipe (30, rating_neighbors=1)
    assert by_id[40].rating_is_estimated is True
    assert by_id[40].predicted_rating == pytest.approx(4.0)
    # rating 0 is "no stars": recipe 20 counts as unrated
    by_id = {r.recipe_id: r for r in rec.recommend(["garlic", "rice"], k=5)}
    assert by_id[20].rating_is_estimated is True
    assert rec.n_rated_ == 2


def test_ratings_come_only_from_the_given_interactions() -> None:
    """Leakage: a recipe rated only outside the fit interactions must be estimated."""
    val_only = pd.DataFrame({"user_id": [9], "recipe_id": [40], "rating": [1]})
    fit_on = INTERACTIONS  # "train"
    rec = KNNRecipeRecommender(_vectorizer()).fit(CATALOGUE, fit_on)
    r40 = next(r for r in rec.recommend(["egg", "butter"], k=5) if r.recipe_id == 40)
    assert r40.rating_is_estimated is True
    rec_all = KNNRecipeRecommender(_vectorizer()).fit(
        CATALOGUE, pd.concat([fit_on, val_only], ignore_index=True)
    )
    r40 = next(r for r in rec_all.recommend(["egg", "butter"], k=5) if r.recipe_id == 40)
    assert r40.rating_is_estimated is False and r40.predicted_rating == 1.0


def test_minutes_bin_filter(rec) -> None:
    recs = rec.recommend(["chicken", "garlic", "rice"], k=10, filters={"minutes_bin": "quick"})
    assert recs and {r.minutes_bin for r in recs} == {"quick"}
    both = rec.recommend(
        ["chicken", "garlic", "rice"], k=10, filters={"minutes_bin": ["quick", "medium"]}
    )
    assert len(both) > len(recs)
    assert rec.recommend(["chicken"], k=5, filters={"minutes_bin": None})  # None = no filter
    with pytest.raises(KeyError):
        rec.recommend(["chicken"], filters={"calories": 1})


def test_unknown_and_empty_queries(rec) -> None:
    res = rec.recommend_with_details(["chicken", "unobtainium"], k=2)
    assert res.unknown_ingredients == ["unobtainium"]
    assert res.recommendations
    none = rec.recommend_with_details(["unobtainium"], k=2)
    assert none.recommendations == [] and none.unknown_ingredients == ["unobtainium"]
    assert rec.recommend([], k=3) == []
    with pytest.raises(ValueError):
        rec.recommend(["chicken"], k=0)


def test_missing_minutes_is_none_and_user_id_is_accepted(rec) -> None:
    r40 = next(r for r in rec.recommend(["egg"], k=5, user_id=123) if r.recipe_id == 40)
    assert r40.minutes is None


def test_fit_validates_inputs() -> None:
    v = _vectorizer()
    with pytest.raises(KeyError):
        KNNRecipeRecommender(v).fit(CATALOGUE.drop(columns="minutes_bin"), INTERACTIONS)
    with pytest.raises(ValueError):
        KNNRecipeRecommender(v).fit(CATALOGUE, None)
    with pytest.raises(ValueError):  # matrix rows do not match the catalogue
        KNNRecipeRecommender(v).fit(
            CATALOGUE, INTERACTIONS, ingredient_matrix=sparse.csr_matrix((2, len(VOCAB)))
        )
    with pytest.raises(ValueError):  # nothing rated
        KNNRecipeRecommender(v).fit(CATALOGUE, INTERACTIONS.assign(rating=0))
    with pytest.raises(RuntimeError):
        KNNRecipeRecommender(v).recommend(["chicken"])


def test_precomputed_matrix_gives_same_results(rec) -> None:
    from recsys.models.knn import _encode_lists

    m = _encode_lists(CATALOGUE["ingredients_filtered"], _vectorizer().vocab_to_idx)
    other = KNNRecipeRecommender(_vectorizer(), rating_neighbors=1).fit(
        CATALOGUE, INTERACTIONS, ingredient_matrix=m
    )
    q = ["butter", "sugar", "chicken"]
    assert other.recommend(q, k=4) == rec.recommend(q, k=4)


def test_cosine_metric_can_be_injected() -> None:
    rec = KNNRecipeRecommender(_vectorizer(), metric=CosineDistance()).fit(CATALOGUE, INTERACTIONS)
    top = rec.recommend(["chicken", "garlic"], k=1)[0]
    assert top.recipe_id == 10
    assert top.similarity == pytest.approx(2 / np.sqrt(6), abs=1e-6)


def test_example_recommendations_table_meets_schema(rec) -> None:
    queries = [
        {"id": "a", "ingredients": ["chicken", "garlic"]},
        {"id": "b", "ingredients": ["rice", "xyz"], "filters": {"minutes_bin": "quick"}},
    ]
    table = example_recommendations(rec, queries, k=2)
    validate(table, KNN_PROTOTYPE_OUTPUT_SCHEMA["example_recommendations"])
    b = table[table["query_id"] == "b"]
    assert (b["filters"] == "minutes_bin=quick").all()
    assert (b["unknown_ingredients"] == "xyz").all()
    assert table.groupby("query_id")["rank"].apply(list).to_dict() == {"a": [1, 2], "b": [1, 2]}


def test_save_load_roundtrip_is_one_atomic_version(rec, store: ArtifactStore) -> None:
    extra = {"note": {"k": 1}}
    art = rec.save(store, params={"x": 1}, extra_objects=extra)
    assert (art.stage, art.version) == ("knn_prototype", "v1")
    assert {"recommender", "note"} <= set(art.manifest["files"])
    loaded = KNNRecipeRecommender.load(store)
    q = ["chicken", "garlic"]
    assert loaded.recommend(q, k=3) == rec.recommend(q, k=3)
    rec.save(store)
    assert store.versions("knn_prototype") == ["v1", "v2"]  # never overwritten


# --------------------------------------------------------------------------- metrics + guard


def test_classification_metrics_on_known_values() -> None:
    y_true = ["a", "a", "a", "b"]
    y_pred = ["a", "a", "a", "a"]  # majority-style prediction
    res = evaluate_classification(y_true, y_pred, labels=["a", "b"], split="val")
    assert res["accuracy"] == pytest.approx(0.75)
    assert res["balanced_accuracy"] == pytest.approx(0.5)  # accuracy is misleading
    assert res["macro_f1"] == pytest.approx((6 / 7 + 0) / 2)
    cm = M.confusion_matrix(y_true, y_pred, ["b", "a"])
    assert cm.index.tolist() == ["b", "a"] and cm.loc["b", "a"] == 1 and cm.loc["a", "a"] == 3


def test_regression_metrics_on_known_values() -> None:
    res = evaluate_regression([1.0, 3.0], [2.0, 5.0], split="val")
    assert res["mae"] == pytest.approx(1.5)
    assert res["rmse"] == pytest.approx(np.sqrt(2.5))


def test_test_set_is_refused_and_override_locked() -> None:
    assert M.TEST_EVALUATION_UNLOCKED is False  # P2 unlocks it, not Phase 1
    check_split("val")
    check_split("train")
    with pytest.raises(TestSetAccessError):
        check_split("test")
    with pytest.raises(TestSetAccessError):
        check_split("test", allow_test=True)
    with pytest.raises(ValueError):
        check_split("validation")
    with pytest.raises(TestSetAccessError):
        evaluate_classification(["a"], ["a"], ["a"], split="test")
    with pytest.raises(TestSetAccessError):
        evaluate_regression([1.0], [1.0], split="test", allow_test=True)


def test_compare_refuses_test_before_fitting_anything() -> None:
    fitted: list[str] = []

    class Spy(MajorityBaseline):
        def fit(self, X, y):
            fitted.append("x")
            return super().fit(X, y)

    X, y = _toy_xy()
    with pytest.raises(TestSetAccessError):
        compare_classifiers({"m": Spy()}, X, y, X, y, ["high", "low"], split="test")
    with pytest.raises(TestSetAccessError):
        compare_regressors({"m": MeanBaseline()}, X, np.ones(5), X, np.ones(5), split="test")
    assert fitted == []


def test_compare_tables_meet_schemas() -> None:
    X, y = _toy_xy()
    cls, cm = compare_classifiers(
        {"majority": MajorityBaseline(), "knn": RatingClassifierKNN(n_neighbors=3)},
        X,
        y,
        X,
        y,
        ["high", "low"],
    )
    validate(cls, KNN_PROTOTYPE_OUTPUT_SCHEMA["classification_metrics"])
    validate(cm, KNN_PROTOTYPE_OUTPUT_SCHEMA["confusion_matrices"])
    assert cm.groupby("model")["count"].sum().tolist() == [5, 5]
    reg = compare_regressors(
        {"mean": MeanBaseline(), "knn": RatingRegressorKNN(n_neighbors=2)},
        X,
        np.arange(5.0),
        X,
        np.arange(5.0),
    )
    validate(reg, KNN_PROTOTYPE_OUTPUT_SCHEMA["regression_metrics"])


# --------------------------------------------------------------------------- notebook


def _code_cells(path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_starts_with_colab_cell_and_has_no_logic() -> None:
    cells = _code_cells(NOTEBOOK)
    for needle in ("drive.mount", "git clone", "pip install -q -e .", "DATA_ROOT", "load_config()"):
        assert needle in cells[0]
    for src in cells[1:]:
        defs = [
            n
            for n in ast.walk(ast.parse(src))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
        ]
        assert not defs, src[:80]
    assert all('"test"' not in src or "check_split" in src for src in cells[1:])


def test_notebook_runs_on_sample(cfg: Config) -> None:
    """Run 09 (to create `splits`) and then 10 on the local sample."""
    import matplotlib

    matplotlib.use("Agg")
    art = read_artifact_dir(require_sample(), names=["recipes", "interactions"])
    cfg.raw["splits"]["dev_sample"]["n_recipes"] = 1000
    store = ArtifactStore.from_config(cfg)
    store.save("loaded", {"recipes": art["recipes"], "interactions": art["interactions"]})
    ns: dict[str, object] = {"CFG": cfg, "display": lambda *a, **k: None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for path in (NOTEBOOK_09, NOTEBOOK):
            for src in _code_cells(path)[1:]:
                exec(compile(src, str(path), "exec"), ns)  # noqa: S102 - our own notebooks
    out = store.load("knn_prototype")
    assert out.version == "v1"
    assert [i["stage"] for i in out.manifest["inputs"]] == ["splits", "loaded"]
    for name, schema in KNN_PROTOTYPE_OUTPUT_SCHEMA.items():
        assert out.manifest["files"][name]["schema"] == schema.name
    assert set(out["classification_metrics"]["split"]) == {"val"}
    assert set(out["classification_metrics"]["model"]) == {
        "majority",
        "knn_uniform",
        "knn_distance",
    }
    ex = out["example_recommendations"]
    assert set(ex["query_id"]) == {"q1", "q2", "q3"}
    assert (ex.loc[ex["query_id"] == "q3", "minutes_bin"] == "quick").all()
    assert isinstance(out["recommender"], KNNRecipeRecommender)
