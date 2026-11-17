from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from recsys.interfaces import (
    DistanceMetric,
    Recommendation,
    Recommender,
    SplitResult,
    Splitter,
    Transformer,
)


class Identity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class Dummies:
    def pairwise(self, A, B):
        return np.zeros((len(A), len(B)))

    def split(self, interactions, recipes):
        e = interactions.iloc[0:0]
        return SplitResult(interactions, e, e, strategy="dummy")

    def fit(self, recipes, interactions=None):
        return self

    def recommend(self, ingredients, k=10, user_id=None, filters=None):
        return [Recommendation(recipe_id=1, score=1.0)]


def test_sklearn_transformer_satisfies_protocol() -> None:
    assert isinstance(Identity(), Transformer)


def test_other_protocols() -> None:
    d = Dummies()
    assert isinstance(d, DistanceMetric)
    assert isinstance(d, Splitter)
    assert isinstance(d, Recommender)
    assert not isinstance(object(), Recommender)


def test_dataclasses_are_frozen_and_split_as_dict() -> None:
    rec = Recommendation(recipe_id=5, score=0.9, distance=0.1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.score = 1.0  # type: ignore[misc]
    df = pd.DataFrame({"a": [1]})
    res = Dummies().split(df, df)
    assert set(res.as_dict()) == {"train", "val", "test"}
    assert res.strategy == "dummy"
