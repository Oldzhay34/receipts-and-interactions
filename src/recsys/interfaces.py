"""Small structural interfaces shared by both team members.

High-level code (pipeline, API) depends on these Protocols, never on concrete classes.
They are ``runtime_checkable`` so tests can assert conformance with ``isinstance``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Transformer(Protocol):
    """sklearn-compatible transformer.

    Concrete classes subclass ``sklearn.base.BaseEstimator`` + ``TransformerMixin`` and learn
    state ONLY in ``fit`` (stored in trailing-underscore attributes), on train data (4.1).
    """

    def fit(self, X: Any, y: Any = None) -> Transformer: ...

    def transform(self, X: Any) -> Any: ...

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SplitResult:
    """Output of a :class:`Splitter`.

    Attributes:
        train / val / test: Interaction subsets (RAW_recipes is never split).
        strategy: Name of the splitter that produced it (e.g. ``"grouped_stratified"``).
        params: Parameters used, recorded into the artefact manifest.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    strategy: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "test": self.test}


@runtime_checkable
class Splitter(Protocol):
    def split(self, interactions: pd.DataFrame, recipes: pd.DataFrame) -> SplitResult: ...


@runtime_checkable
class DistanceMetric(Protocol):
    """Pairwise distance between row sets. Ingredient vectors use Jaccard or cosine (4.5)."""

    def pairwise(self, A: Any, B: Any) -> np.ndarray:
        """Return an ``(n_A, n_B)`` dense distance matrix (A, B may be scipy.sparse)."""
        ...


@dataclass(frozen=True)
class Recommendation:
    """One recommended recipe.

    Attributes:
        recipe_id: Food.com recipe id.
        score: Ranking score, higher is better.
        distance: Ingredient distance to the query, if the recommender is distance-based.
        predicted_rating: Predicted rating on the 1-5 scale, if available.
        matched_ingredients: Query ingredients present in the recipe.
        missing_ingredients: Recipe ingredients absent from the query.
    """

    recipe_id: int
    score: float
    distance: float | None = None
    predicted_rating: float | None = None
    matched_ingredients: tuple[str, ...] = ()
    missing_ingredients: tuple[str, ...] = ()


@runtime_checkable
class Recommender(Protocol):
    def fit(
        self, recipes: pd.DataFrame, interactions: pd.DataFrame | None = None
    ) -> Recommender: ...

    def recommend(
        self,
        ingredients: Sequence[str],
        k: int = 10,
        user_id: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[Recommendation]: ...
