"""Evaluation metrics and the test-set guard (stage 1.10).

Classification: accuracy, balanced accuracy, macro-F1, confusion matrix. Regression: MAE, RMSE.
The majority class is only a baseline (4.4); accuracy alone is misleading on the imbalanced
rating classes, so every table reports balanced accuracy and macro-F1 next to it.

Test-set guard (4.4: the test set is used once, in the final evaluation). Two checks:

* **Label**: every ``evaluate_*`` / ``compare_*`` function takes the name of the split it scores
  and refuses ``"test"``. The override ``allow_test=True`` is LOCKED in Phase 1
  (:data:`TEST_EVALUATION_UNLOCKED` is ``False``); P2 unlocks it explicitly, in its own reviewed
  change, for the single logged final evaluation.
* **Data**: a label can be wrong (test data passed as ``split="val"``). :class:`HeldOutRecipes`
  holds the recipe ids of the test split; given the ids of the evaluated recipes, the guard
  refuses any overlap with them unless the split really is ``"test"``, and a ``"test"``
  evaluation must consist of test recipes only. ``compare_*`` always require it; the test path
  (``split="test"``) cannot run without it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class HeldOutRecipes:
    """Recipe ids of the held-out test split, for the data check of the test-set guard.

    Build it from the split artefact, e.g. ``HeldOutRecipes.from_targets(splits["targets_test"])``.
    """

    test_ids: frozenset[int]

    @classmethod
    def from_ids(cls, ids: Iterable[Any]) -> HeldOutRecipes:
        return cls(frozenset(int(i) for i in ids))

    @classmethod
    def from_targets(cls, targets_test: pd.DataFrame, id_col: str = "recipe_id") -> HeldOutRecipes:
        """From a ``splits/targets_test`` table (or any table with the test recipe ids)."""
        if id_col not in targets_test.columns:
            raise KeyError(f"targets table has no {id_col!r} column")
        return cls.from_ids(targets_test[id_col].to_numpy())

    def overlap(self, recipe_ids: Iterable[Any]) -> int:
        """Number of distinct ``recipe_ids`` that belong to the test split."""
        return len(self.test_ids.intersection(int(i) for i in recipe_ids))


def check_split(
    split: str,
    allow_test: bool = False,
    *,
    recipe_ids: Iterable[Any] | None = None,
    held_out: HeldOutRecipes | None = None,
) -> None:
    """Raise unless ``split`` may be evaluated now.

    Args:
        split: Name of the evaluated split (``"train"``, ``"val"`` or ``"test"``).
        allow_test: Override for the single final test evaluation (locked in Phase 1).
        recipe_ids: Recipe ids of the evaluated rows (the third value of
            :func:`recsys.models.knn.recipe_rating_dataset`). Needed with ``held_out``.
        held_out: Test recipe ids. When given, the data is checked as well as the label.

    Raises:
        ValueError: unknown split name, or ``held_out`` without ``recipe_ids``.
        TestSetAccessError: ``split == "test"`` and the override is not given or still locked;
            ``split == "test"`` without the data check; test recipes evaluated under another
            split name; a ``"test"`` evaluation that contains non-test recipes.
    """
    if split not in EVALUABLE_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(EVALUABLE_SPLITS)}")
    if split == "test":
        if not allow_test:
            raise TestSetAccessError(
                "the test set is reserved for the single final evaluation; use the validation set"
            )
        if not TEST_EVALUATION_UNLOCKED:
            raise TestSetAccessError(
                "test evaluation is locked until Phase 2 (TEST_EVALUATION_UNLOCKED is False)"
            )
        if held_out is None or recipe_ids is None:
            raise TestSetAccessError(
                "a test evaluation must pass held_out= and recipe_ids= so the guard can check that "
                "the data really is the test split"
            )
    if held_out is None:
        return
    if recipe_ids is None:
        raise ValueError("held_out= needs recipe_ids= (the ids of the evaluated recipes)")
    ids = np.unique(np.fromiter(recipe_ids, dtype=np.int64))
    n_test = held_out.overlap(ids)
    if split != "test" and n_test:
        raise TestSetAccessError(
            f"{n_test} of the {len(ids)} evaluated recipes belong to the test split, but "
            f"split={split!r}: the test set is reserved for the single final evaluation"
        )
    if split == "test" and n_test != len(ids):
        raise TestSetAccessError(
            f"split='test' but {len(ids) - n_test} of the {len(ids)} evaluated recipes are not "
            "test recipes"
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
    *,
    recipe_ids: Iterable[Any] | None = None,
    held_out: HeldOutRecipes | None = None,
) -> dict[str, Any]:
    """Accuracy, balanced accuracy and macro-F1 of one prediction (guarded split, see
    :func:`check_split`)."""
    check_split(split, allow_test, recipe_ids=recipe_ids, held_out=held_out)
    return {
        "split": split,
        "n_recipes": int(len(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, labels),
    }


def evaluate_regression(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    split: str,
    allow_test: bool = False,
    *,
    recipe_ids: Iterable[Any] | None = None,
    held_out: HeldOutRecipes | None = None,
) -> dict[str, Any]:
    """MAE and RMSE of one prediction (guarded split, see :func:`check_split`)."""
    check_split(split, allow_test, recipe_ids=recipe_ids, held_out=held_out)
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
    *,
    eval_recipe_ids: Sequence[Any],
    held_out: HeldOutRecipes,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each model on train, score it on ``split``.

    The guard runs BEFORE any model is fitted, so a refused split costs nothing. It checks the
    label and the data: ``eval_recipe_ids`` (ids of the ``X_eval`` rows) must not contain test
    recipes unless ``split == "test"`` (see :func:`check_split`).

    Returns:
        ``(metrics, confusion)``: one metrics row per model (model, split, n_recipes, accuracy,
        balanced_accuracy, macro_f1) and the confusion matrices in long form (model,
        true_label, pred_label, count).
    """
    _check_eval_rows(X_eval, y_eval, eval_recipe_ids)
    guard = {"recipe_ids": eval_recipe_ids, "held_out": held_out}
    check_split(split, allow_test, **guard)
    rows, cms = [], []
    for name, model in models.items():
        y_pred = model.fit(X_train, y_train).predict(X_eval)
        rows.append(
            {
                "model": name,
                **evaluate_classification(y_eval, y_pred, labels, split, allow_test, **guard),
            }
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
    *,
    eval_recipe_ids: Sequence[Any],
    held_out: HeldOutRecipes,
) -> pd.DataFrame:
    """Fit each model on train, score MAE / RMSE on ``split`` (one row per model).

    Guarded like :func:`compare_classifiers` (label and data, before any fit).
    """
    _check_eval_rows(X_eval, y_eval, eval_recipe_ids)
    guard = {"recipe_ids": eval_recipe_ids, "held_out": held_out}
    check_split(split, allow_test, **guard)
    rows = []
    for name, model in models.items():
        y_pred = model.fit(X_train, y_train).predict(X_eval)
        rows.append(
            {"model": name, **evaluate_regression(y_eval, y_pred, split, allow_test, **guard)}
        )
    return pd.DataFrame(rows)


def _check_eval_rows(X_eval: Any, y_eval: Sequence[Any], eval_recipe_ids: Sequence[Any]) -> None:
    """The ids must describe the evaluated rows, or the data check would check something else."""
    n_x, n_y, n_ids = X_eval.shape[0], len(y_eval), len(eval_recipe_ids)
    if not n_x == n_y == n_ids:
        raise ValueError(
            f"X_eval has {n_x} rows, y_eval {n_y} and eval_recipe_ids {n_ids}: they must match"
        )
