"""kNN recipe recommender and kNN rating models (stage 1.10, preliminary prototype).

Everything here works on the ingredient matrix of stage 1.9 (``splits/ingredient_matrix``: one
row per catalogue recipe, columns = vocab) and depends on the
:class:`~recsys.interfaces.DistanceMetric` interface, never on a concrete metric (Jaccard is only
the default). Ingredient distance is Jaccard or cosine, never Euclidean (4.5).

* :class:`KNNRecipeRecommender` (:class:`~recsys.interfaces.Recommender`): query ingredients ->
  the k most similar catalogue recipes; recipes without a known rating get one estimated from
  their rated ingredient neighbours (``rating_is_estimated=True``).
* :class:`RatingClassifierKNN` (uniform / distance-weighted vote) and :class:`MajorityBaseline`
  predict the recipe rating class; :class:`RatingRegressorKNN` and :class:`MeanBaseline`
  predict the average rating.

Return values of :meth:`KNNRecipeRecommender.recommend` are plain Python types (no pandas), so
the Phase 4 API can serialise them directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.data.targets import recipe_rating_stats
from recsys.features.distance import JaccardDistance
from recsys.features.ingredients import QueryVectorizer
from recsys.interfaces import DistanceMetric, Recommendation
from recsys.registry import register

Weights = Literal["uniform", "distance"]

#: Stage the recommender artefact is written to (config ``stages``).
STAGE = "knn_prototype"
#: Object name of the fitted recommender inside a ``knn_prototype`` version.
RECOMMENDER_OBJECT = "recommender"


# --------------------------------------------------------------------------- neighbours


def nearest_neighbors(
    metric: DistanceMetric,
    X: Any,
    reference: Any,
    k: int,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` nearest ``reference`` rows of every ``X`` row, computed in row batches.

    Only a ``(batch_size, n_reference)`` distance block is in memory at a time. Ties are broken
    by the lower reference row, so results are deterministic.

    Returns:
        ``(indices, distances)``, both of shape ``(n_X, min(k, n_reference))``, nearest first.
    """
    n_ref = reference.shape[0]
    if n_ref == 0:
        raise ValueError("reference set is empty")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k = min(k, n_ref)
    n = X.shape[0]
    idx = np.empty((n, k), dtype=np.int64)
    dist = np.empty((n, k), dtype=np.float64)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        d = np.asarray(metric.pairwise(X[start:stop], reference), dtype=np.float64)
        # stable sort on (distance, column) -> lower column wins ties
        order = np.argsort(d, axis=1, kind="stable")[:, :k]
        idx[start:stop] = order
        dist[start:stop] = np.take_along_axis(d, order, axis=1)
    return idx, dist


def _vote_weights(distances: np.ndarray, weights: Weights, eps: float) -> np.ndarray:
    if weights == "uniform":
        return np.ones_like(distances)
    if weights == "distance":
        return 1.0 / (distances + eps)
    raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}")


# --------------------------------------------------------------------------- rating models


class _KNNBase:
    def __init__(
        self,
        metric: DistanceMetric | None = None,
        n_neighbors: int = 15,
        weights: Weights = "uniform",
        batch_size: int = 512,
        eps: float = 1e-6,
    ) -> None:
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.batch_size = batch_size
        self.eps = eps

    def _metric(self) -> DistanceMetric:
        return self.metric if self.metric is not None else JaccardDistance()

    def _fit_X(self, X: Any, y: Any) -> np.ndarray:
        y_arr = np.asarray(y)
        if X.shape[0] != len(y_arr):
            raise ValueError(f"X has {X.shape[0]} rows but y has {len(y_arr)}")
        if X.shape[0] == 0:
            raise ValueError("cannot fit on an empty training set")
        _vote_weights(np.zeros(1), self.weights, self.eps)  # validate early
        self.X_fit_ = sparse.csr_matrix(X) if sparse.issparse(X) else np.asarray(X)
        return y_arr

    def _neighbours(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "X_fit_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        return nearest_neighbors(self._metric(), X, self.X_fit_, self.n_neighbors, self.batch_size)


@register("model", "knn_classifier")
class RatingClassifierKNN(_KNNBase):
    """kNN vote over the rating class of the most similar TRAIN recipes.

    Args:
        metric: Injected ingredient distance (default Jaccard).
        n_neighbors: k.
        weights: ``"uniform"`` (one vote each) or ``"distance"`` (vote ``1 / (d + eps)``).
        batch_size: Rows per distance block.
        eps: Keeps distance weights finite for identical ingredient sets (d = 0).

    Ties in the vote go to the first class in sorted order (``classes_``).
    """

    def fit(self, X: Any, y: Sequence[str]) -> RatingClassifierKNN:
        y_arr = self._fit_X(X, y).astype(str)
        self.classes_ = np.unique(y_arr)
        self.y_codes_ = np.searchsorted(self.classes_, y_arr)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Vote shares per class (columns ordered as ``classes_``)."""
        idx, dist = self._neighbours(X)
        w = _vote_weights(dist, self.weights, self.eps)
        votes = np.zeros((idx.shape[0], len(self.classes_)))
        codes = self.y_codes_[idx]
        for c in range(len(self.classes_)):
            votes[:, c] = (w * (codes == c)).sum(axis=1)
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)  # raises first if not fitted
        return self.classes_[np.argmax(proba, axis=1)]


@register("model", "knn_regressor")
class RatingRegressorKNN(_KNNBase):
    """kNN (weighted) mean of the average rating of the most similar TRAIN recipes."""

    def fit(self, X: Any, y: Sequence[float]) -> RatingRegressorKNN:
        y_arr = self._fit_X(X, y).astype(np.float64)
        if np.isnan(y_arr).any():
            raise ValueError("y contains NaN: fit only on rated recipes")
        self.y_fit_ = y_arr
        return self

    def predict(self, X: Any) -> np.ndarray:
        idx, dist = self._neighbours(X)
        w = _vote_weights(dist, self.weights, self.eps)
        return (w * self.y_fit_[idx]).sum(axis=1) / w.sum(axis=1)


@register("model", "majority")
class MajorityBaseline:
    """Always predicts the most frequent TRAIN class (ties: first in sorted order)."""

    def fit(self, X: Any, y: Sequence[str]) -> MajorityBaseline:
        values, counts = np.unique(np.asarray(y).astype(str), return_counts=True)
        if len(values) == 0:
            raise ValueError("cannot fit on an empty training set")
        self.classes_ = values
        self.majority_ = str(values[np.argmax(counts)])
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(X.shape[0], self.majority_, dtype=object)


@register("model", "mean")
class MeanBaseline:
    """Always predicts the mean TRAIN average rating (regression baseline)."""

    def fit(self, X: Any, y: Sequence[float]) -> MeanBaseline:
        self.mean_ = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(X.shape[0], self.mean_, dtype=np.float64)


# --------------------------------------------------------------------------- datasets


def catalogue_rows(catalogue_ids: Sequence[int], ids: Sequence[int]) -> np.ndarray:
    """Row positions of ``ids`` in the catalogue (the row order of the ingredient matrix)."""
    pos = pd.Series(np.arange(len(catalogue_ids)), index=pd.Index(catalogue_ids, name="id"))
    if pos.index.has_duplicates:
        raise ValueError("catalogue ids are not unique")
    missing = pd.Index(ids).difference(pos.index)
    if len(missing):
        raise KeyError(f"{len(missing)} ids are not in the catalogue (e.g. {list(missing[:5])})")
    return pos.loc[list(ids)].to_numpy()


def recipe_rating_dataset(
    matrix: Any,
    catalogue_ids: Sequence[int],
    targets: pd.DataFrame,
    restrict_to: Sequence[int] | None = None,
    target_col: str = "rating_class",
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Recipe-level ``(X, y, recipe_ids)`` for one split.

    Args:
        matrix: Catalogue ingredient matrix (rows aligned with ``catalogue_ids``).
        catalogue_ids: Recipe id of every matrix row.
        targets: A ``splits/targets_<split>`` table (recipe_id, avg_rating, rating_class, ...).
        restrict_to: Optional recipe ids to keep (e.g. the development sample).
        target_col: ``"rating_class"`` (classification) or ``"avg_rating"`` (regression).

    Rows are sorted by recipe_id so the result does not depend on the input order.
    """
    t = targets
    if restrict_to is not None:
        t = t[t["recipe_id"].isin(pd.Index(restrict_to))]
    t = t.sort_values("recipe_id", kind="mergesort")
    ids = t["recipe_id"].to_numpy(dtype=np.int64)
    rows = catalogue_rows(catalogue_ids, ids)
    return matrix[rows], t[target_col].to_numpy(), ids


# --------------------------------------------------------------------------- recommender


@dataclass(frozen=True)
class RecipeRecommendation(Recommendation):
    """A :class:`~recsys.interfaces.Recommendation` with the fields the app shows.

    Subclass, so it can be used wherever a ``Recommendation`` is expected (Liskov). ``score`` is
    the ingredient similarity ``1 - distance``; ``predicted_rating`` is the known train average
    or, if ``rating_is_estimated``, the kNN estimate.
    """

    name: str = ""
    similarity: float = 0.0
    rating_is_estimated: bool = False
    minutes: int | None = None
    minutes_bin: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python dict (tuples become lists), ready for JSON."""
        d = asdict(self)
        d["matched_ingredients"] = list(self.matched_ingredients)
        d["missing_ingredients"] = list(self.missing_ingredients)
        return d


@dataclass(frozen=True)
class RecommendationResult:
    """:meth:`KNNRecipeRecommender.recommend_with_details` output."""

    recommendations: list[RecipeRecommendation]
    unknown_ingredients: list[str]
    query_ingredients: list[str]


_META_COLUMNS = ("id", "name", "minutes", "minutes_bin")


@register("model", "knn_recommender")
class KNNRecipeRecommender:
    """Ingredient-based kNN over the WHOLE catalogue (recipes without ratings included).

    Args:
        vectorizer: A query vectorizer with ``vectorize(ingredients) -> (1 x V csr, unknown)``
            and ``normalizer`` / ``vocab_to_idx`` attributes, built with the SAME normaliser and
            vocab as the catalogue matrix (:class:`recsys.features.ingredients.QueryVectorizer`;
            see :func:`query_vectorizer_from_pipeline`).
        metric: Injected ingredient distance (default Jaccard). It must accept two sparse
            ingredient matrices (Jaccard / cosine); a combined metric that needs numeric
            features cannot score a free-text query.
        rating_neighbors: k of the rating estimate for recipes without a known rating.
        rating_weights: Neighbour weighting of that estimate.
        min_similarity: Recipes with similarity <= this value (no shared ingredient by default)
            are never recommended.
        batch_size: Rows per distance block when estimating ratings.

    ``user_id`` is accepted for interface compatibility and ignored here: personalisation is P3
    (a ``PersonalizedRecommender`` will wrap this class).
    """

    def __init__(
        self,
        vectorizer: Any,
        metric: DistanceMetric | None = None,
        rating_neighbors: int = 10,
        rating_weights: Weights = "distance",
        min_similarity: float = 0.0,
        batch_size: int = 512,
    ) -> None:
        self.vectorizer = vectorizer
        self.metric = metric
        self.rating_neighbors = rating_neighbors
        self.rating_weights = rating_weights
        self.min_similarity = min_similarity
        self.batch_size = batch_size

    def _metric(self) -> DistanceMetric:
        return self.metric if self.metric is not None else JaccardDistance()

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        recipes: pd.DataFrame,
        interactions: pd.DataFrame | None = None,
        *,
        ingredient_matrix: Any = None,
        ingredient_col: str = "ingredients_filtered",
    ) -> KNNRecipeRecommender:
        """Index the catalogue.

        Args:
            recipes: The catalogue, one row per recipe: ``id``, ``name``, ``minutes`` (the raw
                minutes shown to users, not the scaled feature), ``minutes_bin`` and
                ``ingredient_col`` (normalised, filtered ingredient lists).
            interactions: Interactions whose ratings (> 0) give the known average ratings
                (pass TRAIN interactions when evaluating). Required: at least one catalogue
                recipe must be rated to estimate the others.
            ingredient_matrix: Precomputed catalogue matrix (``splits/ingredient_matrix``),
                rows aligned with ``recipes``; built from ``ingredient_col`` if omitted.
            ingredient_col: Column with the ingredient lists.
        """
        missing = [c for c in (*_META_COLUMNS, ingredient_col) if c not in recipes.columns]
        if missing:
            raise KeyError(f"catalogue is missing columns {missing}")
        if recipes["id"].duplicated().any():
            raise ValueError("catalogue recipe ids are not unique")
        if interactions is None:
            raise ValueError("interactions are needed for the known ratings")

        vocab_to_idx = self.vectorizer.vocab_to_idx
        if ingredient_matrix is None:
            ingredient_matrix = _encode_lists(recipes[ingredient_col], vocab_to_idx)
        if ingredient_matrix.shape != (len(recipes), len(vocab_to_idx)):
            raise ValueError(
                f"ingredient matrix {ingredient_matrix.shape} does not match "
                f"{len(recipes)} recipes x {len(vocab_to_idx)} vocab"
            )
        self.matrix_ = sparse.csr_matrix(ingredient_matrix, dtype=np.float32)

        meta = recipes[list(_META_COLUMNS)].reset_index(drop=True).copy()
        meta["id"] = meta["id"].astype(np.int64)
        meta["minutes_bin"] = meta["minutes_bin"].astype(str)
        meta["ingredients"] = [
            tuple(x) if isinstance(x, (list, tuple, np.ndarray)) else ()
            for x in recipes[ingredient_col]
        ]
        stats = recipe_rating_stats(interactions).set_index("recipe_id")["avg_rating"]
        meta["rating"] = meta["id"].map(stats).astype("float64")
        rated = np.flatnonzero(meta["rating"].notna().to_numpy())
        if len(rated) == 0:
            raise ValueError("no catalogue recipe has a rating in these interactions")
        self.meta_ = meta
        self.rating_model_ = RatingRegressorKNN(
            metric=self.metric,
            n_neighbors=self.rating_neighbors,
            weights=self.rating_weights,
            batch_size=self.batch_size,
        ).fit(self.matrix_[rated], meta["rating"].to_numpy()[rated])
        self.n_rated_ = int(len(rated))
        return self

    # ------------------------------------------------------------------ query
    def _check_fitted(self) -> None:
        if not hasattr(self, "matrix_"):
            raise RuntimeError("KNNRecipeRecommender is not fitted")

    def _filter_mask(self, filters: Mapping[str, Any] | None) -> np.ndarray:
        mask = np.ones(len(self.meta_), dtype=bool)
        for key, value in (filters or {}).items():
            if value is None:
                continue
            if key not in self.meta_.columns or key == "ingredients":
                raise KeyError(f"unknown filter {key!r}; filterable: {list(_META_COLUMNS[1:])}")
            allowed = list(value) if isinstance(value, (list, tuple, set)) else [value]
            mask &= self.meta_[key].isin(allowed).to_numpy()
        return mask

    def recommend_with_details(
        self,
        ingredients: Sequence[str],
        k: int = 10,
        user_id: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> RecommendationResult:
        """Like :meth:`recommend`, plus the query ingredients that are not in the vocab."""
        self._check_fitted()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        query_vec, unknown = self.vectorizer.vectorize(list(ingredients))
        query = self.vectorizer.normalizer.normalize_list(list(ingredients))
        if query_vec.nnz == 0:
            return RecommendationResult([], list(unknown), query)

        sim = 1.0 - np.asarray(self._metric().pairwise(query_vec, self.matrix_), dtype=np.float64)
        sim = sim.ravel()
        candidates = np.flatnonzero(self._filter_mask(filters) & (sim > self.min_similarity))
        if len(candidates) == 0:
            return RecommendationResult([], list(unknown), query)
        # highest similarity first, then lowest recipe_id (deterministic)
        order = np.lexsort((self.meta_["id"].to_numpy()[candidates], -sim[candidates]))
        top = candidates[order[:k]]

        ratings = self.meta_["rating"].to_numpy()[top]
        estimated = np.isnan(ratings)
        if estimated.any():
            ratings = ratings.copy()
            ratings[estimated] = self.rating_model_.predict(self.matrix_[top[estimated]])

        query_set = set(query)
        recs = []
        for pos, row in enumerate(top):
            m = self.meta_.iloc[row]
            recipe_ings = m["ingredients"]
            minutes = m["minutes"]
            recs.append(
                RecipeRecommendation(
                    recipe_id=int(m["id"]),
                    score=float(sim[row]),
                    distance=float(1.0 - sim[row]),
                    predicted_rating=float(ratings[pos]),
                    matched_ingredients=tuple(i for i in recipe_ings if i in query_set),
                    missing_ingredients=tuple(i for i in recipe_ings if i not in query_set),
                    name=str(m["name"]) if pd.notna(m["name"]) else "",
                    similarity=float(sim[row]),
                    rating_is_estimated=bool(estimated[pos]),
                    minutes=None if pd.isna(minutes) else int(minutes),
                    minutes_bin=str(m["minutes_bin"]),
                )
            )
        return RecommendationResult(recs, list(unknown), query)

    def recommend(
        self,
        ingredients: Sequence[str],
        k: int = 10,
        user_id: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[Recommendation]:
        """The ``k`` catalogue recipes most similar to ``ingredients``.

        Args:
            ingredients: Free-text ingredient names (normalised like the catalogue).
            k: Number of results (fewer if fewer recipes share an ingredient).
            user_id: Ignored (P3).
            filters: Column -> value or list of values, e.g. ``{"minutes_bin": "quick"}``.

        Returns:
            :class:`RecipeRecommendation` objects, best first; empty if no query ingredient is in
            the vocab.
        """
        return list(self.recommend_with_details(ingredients, k, user_id, filters).recommendations)

    # ------------------------------------------------------------------ persistence
    def save(
        self,
        store: Any,
        inputs: Sequence[Any] = (),
        params: Mapping[str, Any] | None = None,
        extra_objects: Mapping[str, Any] | None = None,
        schemas: Mapping[str, Any] | None = None,
    ) -> Any:
        """Write the fitted recommender (and ``extra_objects``) as ONE new ``knn_prototype``
        version through the ArtifactStore (atomic, with manifest)."""
        self._check_fitted()
        objects = {RECOMMENDER_OBJECT: self, **dict(extra_objects or {})}
        return store.save(STAGE, objects, inputs=inputs, params=params, schemas=schemas)

    @classmethod
    def load(cls, store: Any, version: str = "LATEST") -> KNNRecipeRecommender:
        """Load a saved recommender (sha256-verified by the ArtifactStore)."""
        obj = store.load(STAGE, version, names=[RECOMMENDER_OBJECT])[RECOMMENDER_OBJECT]
        if not isinstance(obj, cls):
            raise TypeError(f"{STAGE}/{RECOMMENDER_OBJECT} is a {type(obj).__name__}")
        return obj


def _encode_lists(lists: pd.Series, vocab_to_idx: Mapping[str, int]) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for r, items in enumerate(lists):
        if isinstance(items, (list, tuple, np.ndarray)):
            for c in sorted({vocab_to_idx[i] for i in items if i in vocab_to_idx}):
                rows.append(r)
                cols.append(c)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(lists), len(vocab_to_idx)))


def query_vectorizer_from_pipeline(pipeline: Any) -> Any:
    """A :class:`~recsys.features.ingredients.QueryVectorizer` with the fitted 1.9 pipeline's
    ingredient normaliser and encoder (same vocab, strategy and idf as the catalogue matrix)."""
    normalizers = [s for _, s in pipeline.ingredient_steps_ if hasattr(s, "normalize_list")]
    if len(normalizers) != 1:
        raise ValueError(f"expected one ingredient normaliser, found {len(normalizers)}")
    enc = pipeline.encoder_
    return QueryVectorizer(
        normalizer=normalizers[0], vocab=enc.vocab_, idf=enc.idf_, strategy=enc.strategy
    )


def example_recommendations(
    recommender: KNNRecipeRecommender,
    queries: Sequence[Mapping[str, Any]],
    k: int,
) -> pd.DataFrame:
    """Run config example queries (``{id, ingredients, filters?}``) and tabulate the results.

    Returns:
        One row per (query, rank): query_id, query, filters, rank, recipe_id, name, similarity,
        predicted_rating, rating_is_estimated, minutes, minutes_bin, matched_ingredients,
        unknown_ingredients (lists joined with ", " so the table stores as parquet).
    """
    rows: list[dict[str, Any]] = []
    for q in queries:
        filters = dict(q.get("filters") or {})
        res = recommender.recommend_with_details(q["ingredients"], k=k, filters=filters)
        for rank, rec in enumerate(res.recommendations, start=1):
            rows.append(
                {
                    "query_id": str(q["id"]),
                    "query": ", ".join(q["ingredients"]),
                    "filters": ", ".join(f"{a}={b}" for a, b in sorted(filters.items())),
                    "rank": rank,
                    "recipe_id": rec.recipe_id,
                    "name": rec.name,
                    "similarity": rec.similarity,
                    "predicted_rating": rec.predicted_rating,
                    "rating_is_estimated": rec.rating_is_estimated,
                    "minutes": None if rec.minutes is None else float(rec.minutes),
                    "minutes_bin": rec.minutes_bin or "",
                    "matched_ingredients": ", ".join(rec.matched_ingredients),
                    "unknown_ingredients": ", ".join(res.unknown_ingredients),
                }
            )
    out = pd.DataFrame(rows, columns=_EXAMPLE_COLUMNS)
    out["rank"] = out["rank"].astype("int64")
    out["recipe_id"] = out["recipe_id"].astype("int64")
    out["minutes"] = out["minutes"].astype("float64")
    out["rating_is_estimated"] = out["rating_is_estimated"].astype(bool)
    return out


_EXAMPLE_COLUMNS = (
    "query_id",
    "query",
    "filters",
    "rank",
    "recipe_id",
    "name",
    "similarity",
    "predicted_rating",
    "rating_is_estimated",
    "minutes",
    "minutes_bin",
    "matched_ingredients",
    "unknown_ingredients",
)


def catalogue_table(
    features: pd.DataFrame,
    raw_recipes: pd.DataFrame,
    ingredient_col: str = "ingredients_filtered",
) -> pd.DataFrame:
    """Recommender catalogue: the 1.9 features (row order = ingredient matrix rows) with the RAW
    ``minutes`` of the ``loaded`` recipes, because the feature column is log-scaled.

    Returns:
        Columns id, name, minutes (raw, may be NaN), minutes_bin, ``ingredient_col``; same rows
        and order as ``features``.
    """
    raw = raw_recipes.set_index("id")["minutes"]
    out = features[["id", "name", "minutes_bin", ingredient_col]].copy()
    out["minutes"] = out["id"].map(raw).astype("float64")
    return out.reset_index(drop=True)
