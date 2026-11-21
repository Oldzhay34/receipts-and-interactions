"""Tests for recsys.pipeline.build (config + registry only), leakage, and recsys.imbalance."""

from __future__ import annotations

import ast
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from recsys import registry
from recsys.config import REPO_ROOT, Config, load_config
from recsys.contracts import (
    RECIPE_FEATURES_SCHEMA,
    RECIPE_TARGETS_SCHEMA,
    SPLITS_OUTPUT_SCHEMA,
    validate,
)
from recsys.data.split import GroupStratifiedSplitter
from recsys.imbalance.resampling import (
    RandomOverSampler,
    RandomUnderSampler,
    class_weights,
    resample_train,
    resampling_summary,
)
from recsys.interfaces import SplitResult
from recsys.io import ArtifactStore
from recsys.io.artifact_store import read_artifact_dir
from recsys.pipeline.build import PreprocessingPipeline, build_step, resolve_refs
from recsys.registry import RegistryError

from .conftest import require_sample

BUILD_PY = REPO_ROOT / "src" / "recsys" / "pipeline" / "build.py"
NOTEBOOK = REPO_ROOT / "notebooks" / "09_split.ipynb"


@pytest.fixture(scope="module")
def repo_cfg(tmp_path_factory: pytest.TempPathFactory) -> Config:
    return load_config(data_root=tmp_path_factory.mktemp("root"))


@pytest.fixture(scope="module")
def sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    art = read_artifact_dir(require_sample(), names=["recipes", "interactions"])
    return art["recipes"], art["interactions"]


@pytest.fixture(scope="module")
def sample_split(repo_cfg: Config, sample) -> SplitResult:
    recipes, inter = sample
    binner = build_step(repo_cfg.section("pipeline")["target"]["params"]["binner"], repo_cfg.raw)
    return GroupStratifiedSplitter(binner).split(inter, recipes)


def _fit(cfg: Config, recipes: pd.DataFrame, train: pd.DataFrame) -> PreprocessingPipeline:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PreprocessingPipeline.from_config(cfg).fit(recipes, train)


# ----------------------------------------------------------------- config + registry only


def test_build_module_imports_no_concrete_component() -> None:
    tree = ast.parse(BUILD_PY.read_text(encoding="utf-8"))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    recsys_modules = {m for m in modules if m.startswith("recsys")}
    assert recsys_modules <= {"recsys", "recsys.registry", "recsys.config"}, recsys_modules


def test_resolve_refs() -> None:
    cfg = {"a": {"b": [1, 2], "c": {"d": "x"}}}
    spec = {"p": {"ref": "a.b"}, "q": [{"ref": "a.c.d"}, 3], "r": {"s": {"ref": "a.c"}}}
    assert resolve_refs(spec, cfg) == {"p": [1, 2], "q": ["x", 3], "r": {"s": {"d": "x"}}}
    with pytest.raises(KeyError, match="no key"):
        resolve_refs({"ref": "a.zz"}, cfg)


def test_build_step_nested_optional_and_unknown(repo_cfg: Config) -> None:
    target = build_step(repo_cfg.section("pipeline")["target"], repo_cfg.raw)
    assert type(target).__name__ == "RecipeTargetBuilder"
    assert type(target.binner).__name__ == "DomainBins"  # nested step built from the registry
    assert build_step({"kind": "nope", "name": "x", "optional": True}, {}) is None
    with pytest.raises(RegistryError):
        build_step({"kind": "nope", "name": "x"}, {})
    with pytest.raises(ValueError, match="unknown keys"):
        build_step({"kind": "outlier", "name": "log1p", "colums": []}, {})


def test_pipeline_steps_follow_config_order(
    repo_cfg: Config, fixture_recipes, fixture_interactions
):
    pipe = _fit(repo_cfg, fixture_recipes, fixture_interactions)
    labels = pipe.step_labels()
    kinds = [lbl.split("_", 1)[1].split(".")[0] for lbl in labels["recipe_steps"]]
    assert kinds == [
        "missing",
        "outlier",
        "imputer",
        "imputer",
        "imputer",
        "outlier",
        "binner",
        "binner",
        "outlier",
    ]
    assert labels["ingredient_steps"] == ["00_ingredients.normalizer", "01_ingredients.rare_filter"]


def test_pipeline_columns_match_outlier_config(repo_cfg: Config) -> None:
    steps = {
        (s["kind"], s["name"] if isinstance(s["name"], str) else None): s
        for s in repo_cfg.section("pipeline")["recipe_steps"]
    }
    per_col = repo_cfg.section("outliers")["columns"]
    clip = steps[("outlier", "quantile_clip")]["params"]["columns"]
    log = steps[("outlier", "log1p")]["params"]["columns"]
    assert set(clip) == {c for c, s in per_col.items() if "quantile_clip" in s}
    assert set(log) == {c for c, s in per_col.items() if "log1p" in s}


def test_pipeline_on_fixtures_meets_contracts(repo_cfg, fixture_recipes, fixture_interactions):
    pipe = _fit(repo_cfg, fixture_recipes, fixture_interactions)
    feats = validate(pipe.transform(fixture_recipes), RECIPE_FEATURES_SCHEMA)
    validate(pipe.targets(fixture_interactions), RECIPE_TARGETS_SCHEMA)
    # 106 = INT_MAX minutes and 107 = two years are invalid -> NaN -> imputed, flagged
    flagged = feats.set_index("id").loc[[106, 107], "minutes_was_missing"]
    assert flagged.all()
    assert pipe.encode(feats).shape == (len(fixture_recipes), len(pipe.vocab_))
    with pytest.raises(RuntimeError, match="not fitted"):
        PreprocessingPipeline.from_config(repo_cfg).transform(fixture_recipes)


@pytest.mark.xfail(
    strict=True,
    reason="1.8 has no scaler registered under kind 'scaler' yet; the pipeline "
    "skips that optional step. Remove this mark once B registers it.",
)
def test_full_pipeline_has_no_skipped_component(repo_cfg, fixture_recipes, fixture_interactions):
    pipe = _fit(repo_cfg, fixture_recipes, fixture_interactions)
    assert pipe.skipped_steps_ == []


# ----------------------------------------------------------------- leakage


def test_val_and_test_data_never_reach_fit(repo_cfg, sample, sample_split) -> None:
    recipes, _ = sample
    split = sample_split
    base = _fit(repo_cfg, recipes, split.train)

    held_out = set(split.val["recipe_id"]) | set(split.test["recipe_id"])
    poisoned = recipes.copy()
    mask = poisoned["id"].isin(held_out)
    poisoned.loc[mask, "calories"] = 1e6
    poisoned.loc[mask, "minutes"] = 0
    poisoned.loc[mask, "n_steps"] = 999
    poisoned.loc[mask, "ingredients"] = pd.Series(
        [["unicorn dust"]] * int(mask.sum()), index=poisoned.index[mask]
    )
    refit = _fit(repo_cfg, poisoned, split.train)
    assert refit.fitted_state_json() == base.fitted_state_json()
    assert refit.n_fit_recipes_ == split.train["recipe_id"].nunique()


def test_changing_train_changes_fit_and_val_output(repo_cfg, sample, sample_split) -> None:
    recipes, _ = sample
    split = sample_split
    base = _fit(repo_cfg, recipes, split.train)

    train_ids = set(split.train["recipe_id"])
    changed = recipes.copy()
    mask = changed["id"].isin(train_ids)
    changed.loc[mask, "calories"] = changed.loc[mask, "calories"] * 10
    refit = _fit(repo_cfg, changed, split.train)
    assert refit.fitted_state_json() != base.fitted_state_json()

    val_recipes = recipes[recipes["id"].isin(set(split.val["recipe_id"]))]
    before = base.transform(val_recipes)
    after = refit.transform(val_recipes)
    assert not before["calories"].equals(after["calories"])  # train clip bound moved
    assert not before["calories_bin"].astype(str).equals(after["calories_bin"].astype(str))


def test_val_targets_use_train_boundaries(repo_cfg, sample, sample_split) -> None:
    recipes, _ = sample
    split = sample_split
    pipe = _fit(repo_cfg, recipes, split.train)
    state = pipe.fitted_state_json()
    val_targets = validate(pipe.targets(split.val), RECIPE_TARGETS_SCHEMA)
    assert set(val_targets["recipe_id"]) <= set(split.val["recipe_id"])
    assert pipe.fitted_state_json() == state  # computing val targets does not refit


# ----------------------------------------------------------------- resampling


def _labelled(counts: dict[str, int]) -> pd.DataFrame:
    labels = [c for c, n in counts.items() for _ in range(n)]
    return pd.DataFrame({"x": np.arange(len(labels)), "label": labels})


def test_class_weights_balance_counts() -> None:
    y = pd.Series(["a"] * 6 + ["b"] * 3 + ["c"] * 1)
    w = class_weights(y)
    assert w == pytest.approx({"a": 10 / 18, "b": 10 / 9, "c": 10 / 3})
    weighted = y.value_counts() * pd.Series(w)
    assert weighted.round(10).nunique() == 1


@pytest.mark.parametrize(("sampler", "target"), [(RandomUnderSampler, 2), (RandomOverSampler, 7)])
def test_random_samplers_equalise_classes(sampler, target) -> None:
    df = _labelled({"a": 7, "b": 4, "c": 2})
    X, y = sampler(random_state=0).fit_resample(df, df["label"])
    assert y.value_counts().eq(target).all()
    X2, _ = sampler(random_state=0).fit_resample(df, df["label"])
    pd.testing.assert_frame_equal(X, X2)  # reproducible
    if sampler is RandomUnderSampler:
        assert X.index.is_unique


def test_resampling_touches_train_only() -> None:
    split = SplitResult(
        train=_labelled({"a": 8, "b": 2}),
        val=_labelled({"a": 5, "b": 1}),
        test=_labelled({"a": 4, "b": 1}),
        strategy="x",
    )
    out = resample_train(split, RandomOverSampler(), "label")
    assert out.train["label"].value_counts().to_dict() == {"a": 8, "b": 8}
    assert out.val is split.val and out.test is split.test
    assert out.params["resampled"] == "RandomOverSampler"


def test_resampling_summary_and_registry() -> None:
    y = pd.Series(["a"] * 5 + ["b"] * 2)
    summary = resampling_summary(
        y, {"random_under": RandomUnderSampler(), "random_over": RandomOverSampler()}
    )
    counts = summary.set_index(["method", "label"])["n_rows"]
    assert counts[("none", "a")] == 5 and counts[("random_under", "a")] == 2
    assert counts[("random_over", "b")] == 5
    assert registry.get("resampler", "random_under") is RandomUnderSampler
    assert registry.get("resampler", "random_over") is RandomOverSampler


# ----------------------------------------------------------------- notebook


def _code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_starts_with_colab_cell_and_has_no_logic() -> None:
    cells = _code_cells()
    for needle in ("drive.mount", "git clone", "pip install -q -e .", "DATA_ROOT", "load_config()"):
        assert needle in cells[0]
    assert "SecretNotFoundError" in cells[0]
    for src in cells[1:]:  # no business logic: no function / class definitions
        defs = [
            n
            for n in ast.walk(ast.parse(src))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
        ]
        assert not defs, src[:80]


def test_notebook_runs_on_sample(cfg: Config, sample) -> None:
    import matplotlib

    matplotlib.use("Agg")
    recipes, inter = sample
    cfg.raw["splits"]["dev_sample"]["n_recipes"] = 1000  # the sample has 5000 recipes
    store = ArtifactStore.from_config(cfg)
    store.save("loaded", {"recipes": recipes, "interactions": inter})
    ns: dict[str, object] = {"CFG": cfg, "display": lambda *a, **k: None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for src in _code_cells()[1:]:
            exec(compile(src, str(NOTEBOOK), "exec"), ns)  # noqa: S102 - our own notebook
    art = store.load("splits")
    assert art.version == "v1"
    assert set(SPLITS_OUTPUT_SCHEMA) <= set(art.keys())
    assert {"pipeline", "pipeline_state", "ingredient_matrix", "vocab", "class_weights"} <= set(
        art.keys()
    )
    assert [i["stage"] for i in art.manifest["inputs"]] == ["loaded"]
    files = art.manifest["files"]
    for name, schema in SPLITS_OUTPUT_SCHEMA.items():
        assert files[name]["schema"] == schema.name
    assert len(art["dev_recipes"]) == 1000
    # the stored pipeline is the fitted one: same state as the json next to it
    assert art["pipeline"].fitted_state() == art["pipeline_state"]
