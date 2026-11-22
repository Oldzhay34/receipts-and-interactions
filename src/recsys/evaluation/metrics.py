"""Evaluation metrics and the test-set guard (stage 1.10).

Classification: accuracy, balanced accuracy, macro-F1, confusion matrix. Regression: MAE, RMSE.
The majority class is only a baseline (4.4); accuracy alone is misleading on the imbalanced
rating classes, so every table reports balanced accuracy and macro-F1 next to it.

Test-set guard (4.4: the test set is used once, in the final evaluation): every ``evaluate_*``
function takes the name of the split it scores and refuses ``"test"``. The override
``allow_test=True`` is LOCKED in Phase 1 (:data:`TEST_EVALUATION_UNLOCKED` is ``False``); P2
unlocks it explicitly, in its own reviewed change, for the single logged final evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as skm

#: Phase 1 lock of the test-set override. Only P2 may change this (one final, logged run).
TEST_EVALUATION_UNLOCKED: bool = False

EVALUABLE_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})


class TestSetAccessError(PermissionError):
    """The test split was passed to an evaluation function without an unlocked override."""

    __test__ = False  # not a pytest test class


def check_split(split: str, allow_test: bool = False) -> None:
    """Raise unless ``split`` may be evaluated now.

    Raises:
        ValueError: unknown split name.
        TestSetAccessError: ``split == "test"`` and the override is not given or still locked.
    """
    if split not in EVALUABLE_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(EVALUABLE_SPLITS)}")
    if split != "test":
        return
    if not allow_test:
        raise TestSetAccessError(
            "the test set is reserved for the single final evaluation; use the validation set"
        )
    if not TEST_EVALUATION_UNLOCKED:
        raise TestSetAccessError(
            "test evaluation is locked until Phase 2 (TEST_EVALUATION_UNLOCKED is False)"
        )


# --------------------------------------------------------------------------- metrics


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    return float(skm.accuracy_score(y_true, y_pred))


def balanced_accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Mean recall over the classes present in ``y_true``."""
    return float(skm.balanced_accuracy_score(y_true, y_pred))


def macro_f1(
    y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any] | None = None
) -> float:
    """Unweighted mean F1 over ``labels`` (a class never predicted scores 0)."""
    return float(skm.f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def confusion_matrix(
    y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any]
) -> pd.DataFrame:
    """Counts with true classes as rows and predicted classes as columns, in ``labels`` order."""
    cm = skm.confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(
        cm,
        index=pd.Index(list(labels), name="true_label"),
        columns=pd.Index(list(labels), name="pred_label"),
    )


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(skm.mean_absolute_error(y_true, y_pred))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))


# --------------------------------------------------------------------------- evaluation


def evaluate_classification(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any],
    split: str,
    allow_test: bool = False,
) -> dict[str, Any]:
    """Accuracy, balanced accuracy and macro-F1 of one prediction (guarded split)."""
    check_split(split, allow_test)
    return {
        "split": split,
        "n_recipes": int(len(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, labels),
    }


def evaluate_regression(
    y_true: Sequence[float], y_pred: Sequence[float], split: str, allow_test: bool = False
) -> dict[str, Any]:
    """MAE and RMSE of one prediction (guarded split)."""
    check_split(split, allow_test)
    return {
        "split": split,
        "n_recipes": int(len(y_true)),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


def compare_classifiers(
    models: Mapping[str, Any],
    X_train: Any,
    y_train: Sequence[Any],
    X_eval: Any,
    y_eval: Sequence[Any],
    labels: Sequence[Any],
    split: str = "val",
    allow_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each model on train, score it on ``split``.

    The guard runs BEFORE any model is fitted, so a refused split costs nothing.

    Returns:
        ``(metrics, confusion)``: one metrics row per model (model, split, n_recipes, accuracy,
        balanced_accuracy, macro_f1) and the confusion matrices in long form (model,
        true_label, pred_label, count).
    """
    check_split(split, allow_test)
    rows, cms = [], []
    for name, model in models.items():
        y_pred = model.fit(X_train, y_train).predict(X_eval)
        rows.append(
            {"model": name, **evaluate_classification(y_eval, y_pred, labels, split, allow_test)}
        )
        cm = confusion_matrix(y_eval, y_pred, labels).stack().rename("count").reset_index()
        cm.insert(0, "model", name)
        cms.append(cm)
    confusion = pd.concat(cms, ignore_index=True)
    confusion["count"] = confusion["count"].astype("int64")
    return pd.DataFrame(rows), confusion


def compare_regressors(
    models: Mapping[str, Any],
    X_train: Any,
    y_train: Sequence[float],
    X_eval: Any,
    y_eval: Sequence[float],
    split: str = "val",
    allow_test: bool = False,
) -> pd.DataFrame:
    """Fit each model on train, score MAE / RMSE on ``split`` (one row per model)."""
    check_split(split, allow_test)
    rows = []
    for name, model in models.items():
        y_pred = model.fit(X_train, y_train).predict(X_eval)
        rows.append({"model": name, **evaluate_regression(y_eval, y_pred, split, allow_test)})
    return pd.DataFrame(rows)
