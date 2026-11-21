"""Preprocessing pipeline of stage 1.9, assembled ONLY from ``config.yaml`` and the registry.

This module imports no concrete transformer: every step is ``registry.create(kind, name,
**params)`` with the kind / name / params read from the ``pipeline`` block of the config, so the
1.3-1.8 components can change (or be added) without touching this file.

Config conventions (see ``config/config.yaml: pipeline``):

* a step is ``{kind, name, params, optional}``;
* any value ``{ref: "a.b.c"}`` is replaced by that path of the config (reuse of other blocks);
* a params value that is itself ``{kind, name, ...}`` is built as a nested step (e.g. the binner
  of the target builder);
* ``optional: true`` skips a step whose ``(kind, name)`` is not registered yet and records it in
  ``skipped_steps_`` instead of failing.

Leakage (4.1): :meth:`PreprocessingPipeline.fit` learns everything from the TRAIN interactions and
the recipes that occur in them; :meth:`transform` / :meth:`targets` / :meth:`encode` only apply
what was learned.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from recsys import registry
from recsys.config import Config
from recsys.registry import RegistryError

_STEP_KEYS = frozenset({"kind", "name", "params", "optional"})


def resolve_refs(value: Any, config: Mapping[str, Any]) -> Any:
    """Replace every ``{ref: "a.b.c"}`` in ``value`` (recursively) by ``config["a"]["b"]["c"]``."""
    if isinstance(value, Mapping):
        if set(value) == {"ref"}:
            node: Any = config
            for part in str(value["ref"]).split("."):
                if not isinstance(node, Mapping) or part not in node:
                    raise KeyError(f"config reference {value['ref']!r}: no key {part!r}")
                node = node[part]
            return deepcopy(node)
        return {k: resolve_refs(v, config) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_refs(v, config) for v in value]
    return value


def _is_step_spec(value: Any) -> bool:
    return isinstance(value, Mapping) and {"kind", "name"} <= set(value) <= _STEP_KEYS


def build_step(spec: Mapping[str, Any], config: Mapping[str, Any]) -> Any | None:
    """Instantiate one step from its spec; ``None`` if it is optional and not registered."""
    spec = resolve_refs(dict(spec), config)
    unknown = set(spec) - _STEP_KEYS
    if unknown:
        raise ValueError(f"unknown keys {sorted(unknown)} in pipeline step {spec}")
    params = {
        k: (build_step(v, config) if _is_step_spec(v) else v)
        for k, v in (spec.get("params") or {}).items()
    }
    try:
        cls = registry.get(spec["kind"], spec["name"])
    except RegistryError:
        if spec.get("optional"):
            return None
        raise
    return cls(**params)


def _step_label(i: int, spec: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    s = resolve_refs({k: spec[k] for k in ("kind", "name")}, config)
    target = (spec.get("params") or {}).get("target_col") or (spec.get("params") or {}).get(
        "column"
    )
    suffix = f"[{target}]" if isinstance(target, str) else ""
    return f"{i:02d}_{s['kind']}.{s['name']}{suffix}"


class PreprocessingPipeline:
    """Interaction cleaning, recipe targets, recipe features and the ingredient encoding.

    Args:
        spec: The ``pipeline`` block (``interaction_steps``, ``target``, ``recipe_steps``,
            ``ingredient_steps``, ``ingredient_encoder``).
        config: The whole parsed config, to resolve ``{ref: ...}`` values.

    Fitted attributes (set by :meth:`fit`): ``interaction_steps_``, ``target_``,
    ``recipe_steps_``, ``ingredient_steps_``, ``encoder_`` (lists of ``(label, step)`` or a step),
    ``skipped_steps_`` (labels of optional steps that are not registered),
    ``n_fit_interactions_``, ``n_fit_recipes_``.
    """

    def __init__(self, spec: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self.spec = spec
        self.config = config

    @classmethod
    def from_config(cls, cfg: Config) -> PreprocessingPipeline:
        return cls(cfg.section("pipeline"), cfg.raw)

    # ------------------------------------------------------------------ fit
    def _build_list(self, key: str) -> list[tuple[str, Any]]:
        steps = []
        for i, spec in enumerate(self.spec.get(key) or []):
            label = _step_label(i, spec, self.config)
            step = build_step(spec, self.config)
            if step is None:
                self.skipped_steps_.append(f"{key}/{label}")
                continue
            steps.append((label, step))
        return steps

    @staticmethod
    def _fit_chain(steps: Sequence[tuple[str, Any]], X: pd.DataFrame) -> pd.DataFrame:
        for _, step in steps:
            X = step.fit(X).transform(X)
        return X

    @staticmethod
    def _apply_chain(steps: Sequence[tuple[str, Any]], X: pd.DataFrame) -> pd.DataFrame:
        for _, step in steps:
            X = step.transform(X)
        return X

    def fit(self, recipes: pd.DataFrame, interactions: pd.DataFrame) -> PreprocessingPipeline:
        """Fit every step on TRAIN data.

        Args:
            recipes: The catalogue; only recipes occurring in ``interactions`` are used to fit.
            interactions: TRAIN interactions only.
        """
        self.skipped_steps_: list[str] = []
        self.interaction_steps_ = self._build_list("interaction_steps")
        self.recipe_steps_ = self._build_list("recipe_steps")
        self.ingredient_steps_ = self._build_list("ingredient_steps")
        self.target_ = build_step(self.spec["target"], self.config)
        self.encoder_ = build_step(self.spec["ingredient_encoder"], self.config)

        inter = self._fit_chain(self.interaction_steps_, interactions)
        self.target_.fit(inter)
        train_recipes = recipes[recipes["id"].isin(inter["recipe_id"].unique())]
        if train_recipes.empty:
            raise ValueError("no catalogue recipe occurs in the train interactions")
        feats = self._fit_chain(self.recipe_steps_, train_recipes)
        feats = self._fit_chain(self.ingredient_steps_, feats)
        self.encoder_.fit(feats)
        self.n_fit_interactions_ = int(len(interactions))
        self.n_fit_recipes_ = int(len(train_recipes))
        return self

    # ------------------------------------------------------------------ apply
    def _check_fitted(self) -> None:
        if not hasattr(self, "encoder_"):
            raise RuntimeError("PreprocessingPipeline is not fitted; call fit(train) first")

    def transform_interactions(self, interactions: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        return self._apply_chain(self.interaction_steps_, interactions)

    def targets(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Recipe targets of ``interactions`` with the TRAIN class boundaries."""
        return self.target_.transform(self.transform_interactions(interactions))

    def transform(self, recipes: pd.DataFrame) -> pd.DataFrame:
        """Recipe features (cleaned, clipped, binned, logged / scaled, ingredient lists)."""
        self._check_fitted()
        feats = self._apply_chain(self.recipe_steps_, recipes)
        return self._apply_chain(self.ingredient_steps_, feats)

    def encode(self, features: pd.DataFrame) -> Any:
        """Ingredient matrix (``scipy.sparse``) of :meth:`transform` output; columns = vocab."""
        self._check_fitted()
        return self.encoder_.transform(features)

    @property
    def vocab_(self) -> list[str]:
        self._check_fitted()
        return list(self.encoder_.vocab_)

    # ------------------------------------------------------------------ inspection
    def step_labels(self) -> dict[str, list[str]]:
        self._check_fitted()
        return {
            "interaction_steps": [n for n, _ in self.interaction_steps_],
            "recipe_steps": [n for n, _ in self.recipe_steps_],
            "ingredient_steps": [n for n, _ in self.ingredient_steps_],
            "skipped": list(self.skipped_steps_),
        }

    def fitted_state(self) -> dict[str, Any]:
        """Everything learned in :meth:`fit` (trailing-underscore attributes), JSON-friendly.

        Two pipelines fitted on the same train data have equal states; the leakage tests compare
        states to prove that val / test data never reach ``fit``.
        """
        self._check_fitted()
        steps: list[tuple[str, Any]] = [
            *self.interaction_steps_,
            ("target", self.target_),
            *self.recipe_steps_,
            *self.ingredient_steps_,
            ("encoder", self.encoder_),
        ]
        return {name: _learned(step) for name, step in steps}

    def fitted_state_json(self) -> str:
        """:meth:`fitted_state` as canonical JSON text (sorted keys), for exact comparison."""
        return json.dumps(self.fitted_state(), sort_keys=True, default=str)


def _learned(obj: Any) -> dict[str, Any]:
    return {
        k: _jsonable(v)
        for k, v in sorted(vars(obj).items())
        if k.endswith("_") and not k.startswith("_")
    }


def _jsonable(v: Any) -> Any:
    if isinstance(v, BaseEstimator):
        return {"class": type(v).__name__, "learned": _learned(v)}
    if isinstance(v, pd.DataFrame):
        return {str(c): _jsonable(v[c].tolist()) for c in v.columns}
    if isinstance(v, pd.Series):
        return _jsonable(v.to_dict())
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (set, frozenset)):
        return sorted(_jsonable(x) for x in v)
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v
