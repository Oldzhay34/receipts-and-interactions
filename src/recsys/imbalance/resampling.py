"""Class-imbalance handling for stage 1.9: random under / over sampling and class weights.

Resampling is applied to the TRAIN set only (:func:`resample_train`): validation and test keep
the real class distribution, otherwise their metrics would describe a dataset that does not
exist. SMOTE is deliberately not offered here: it interpolates between neighbours, which on sparse
binary ingredient vectors produces fractional "half-ingredients"; see the 1.9 decisions.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from recsys.interfaces import SplitResult
from recsys.registry import register


def class_weights(y: pd.Series) -> dict[str, float]:
    """Balanced class weights ``n / (n_classes * count_c)`` (sklearn's "balanced" formula).

    Returns:
        class label (as str) -> weight; the weighted class counts are all equal.
    """
    counts = pd.Series(y).dropna().astype(str).value_counts().sort_index()
    if counts.empty:
        raise ValueError("class_weights needs at least one label")
    n, k = int(counts.sum()), len(counts)
    return {str(c): n / (k * int(m)) for c, m in counts.items()}


class _RandomSampler:
    """Shared logic: draw, per class, a target number of rows with a fixed seed."""

    replace: bool = False

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def _target(self, counts: pd.Series) -> int:
        raise NotImplementedError

    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Return the resampled ``(X, y)``; rows keep their original index (duplicates for
        over-sampling)."""
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        y = pd.Series(np.asarray(y), index=X.index, name=getattr(y, "name", None))
        counts = y.value_counts()
        target = self._target(counts)
        rng = np.random.default_rng(self.random_state)
        chosen = []
        for label in sorted(counts.index, key=str):
            idx = np.flatnonzero((y == label).to_numpy())
            if self.replace and len(idx) < target:
                extra = rng.choice(idx, size=target - len(idx), replace=True)
                idx = np.concatenate([idx, extra])
            elif len(idx) > target:
                idx = rng.choice(idx, size=target, replace=False)
            chosen.append(np.sort(idx))
        pos = np.concatenate(chosen)
        return X.iloc[pos], y.iloc[pos]


@register("resampler", "random_under")
class RandomUnderSampler(_RandomSampler):
    """Down-sample every class to the size of the smallest class (no duplicates)."""

    replace = False

    def _target(self, counts: pd.Series) -> int:
        return int(counts.min())


@register("resampler", "random_over")
class RandomOverSampler(_RandomSampler):
    """Up-sample every class to the size of the largest class (by drawing with replacement)."""

    replace = True

    def _target(self, counts: pd.Series) -> int:
        return int(counts.max())


def resample_train(
    result: SplitResult, sampler: RandomUnderSampler | RandomOverSampler, label_col: str
) -> SplitResult:
    """Resample ONLY ``result.train`` on ``label_col``; val and test are returned unchanged."""
    train, _ = sampler.fit_resample(result.train, result.train[label_col])
    params: Mapping[str, object] = {
        **result.params,
        "resampled": type(sampler).__name__,
        "resample_label": label_col,
    }
    return SplitResult(
        train=train, val=result.val, test=result.test, strategy=result.strategy, params=params
    )


def resampling_summary(
    y: pd.Series, samplers: Mapping[str, RandomUnderSampler | RandomOverSampler]
) -> pd.DataFrame:
    """Class counts of ``y`` as is ("none") and after each sampler, plus the class weights.

    Returns:
        Columns method, label, n_rows, class_weight (weight only on the "none" rows).
    """
    y = pd.Series(np.asarray(y)).astype(str)
    weights = class_weights(y)
    rows = [
        {"method": "none", "label": c, "n_rows": int(n), "class_weight": weights[c]}
        for c, n in y.value_counts().sort_index().items()
    ]
    frame = pd.DataFrame({"y": y})
    for name, sampler in samplers.items():
        _, y_res = sampler.fit_resample(frame, y)
        rows.extend(
            {"method": name, "label": c, "n_rows": int(n), "class_weight": np.nan}
            for c, n in y_res.value_counts().sort_index().items()
        )
    return pd.DataFrame(rows)
