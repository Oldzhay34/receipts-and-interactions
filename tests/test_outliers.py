"""Tests for recsys.cleaning.outliers (fixtures, small frames and, if available, data/sample)."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from recsys import registry
from recsys.analysis.quality import detect_issues
from recsys.cleaning.missing import ZeroAsMissing
from recsys.cleaning.outliers import (
    OUTLIER_COLUMNS,
    STEP_ORDER,
    InvalidValueRule,
    Log1pTransformer,
    QuantileClipper,
    before_after_table,
    extreme_examples,
    invalid_counts,
    iqr_table,
    outlier_steps_from_config,
    quantile_table,
    threshold_counts,
)
from recsys.config import REPO_ROOT, Config, load_config
from recsys.contracts import (
    MINUTES_VALID_MAX,
    OUTLIER_BEFORE_AFTER_SCHEMA,
    OUTLIER_IQR_SCHEMA,
    OUTLIERS_OUTPUT_SCHEMA,
    RECIPES_AFTER_OUTLIERS_SCHEMA,
    SchemaValidationError,
    validate,
)
from recsys.interfaces import Transformer
from recsys.io import ArtifactStore
from recsys.io.artifact_store import read_artifact_dir

from .conftest import require_sample

matplotlib.use("Agg")

NOTEBOOK = REPO_ROOT / "notebooks" / "04_outliers.ipynb"


def _frame(values: list[float], col: str = "x") -> pd.DataFrame:
    return pd.DataFrame({col: values})


@pytest.fixture(scope="module")
def outliers_block(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return load_config(data_root=tmp_path_factory.mktemp("root")).section("outliers")


# ----------------------------------------------------------------- analysis


def test_iqr_table_matches_hand_computation() -> None:
    df = _frame([1, 2, 3, 4, 5, 6, 7, 8, 100])
    row = iqr_table(df, ["x"]).iloc[0]
    q1, q3 = 3.0, 7.0  # pandas linear quantiles of 1..8, 100
    assert (row.q1, row.q3, row.iqr) == (q1, q3, 4.0)
    assert (row.lower_fence, row.upper_fence) == (-3.0, 13.0)
    assert (row.n_below, row.n_above, row.whisker_high, row["max"]) == (0, 1, 8.0, 100.0)
    assert row.share_outside == pytest.approx(1 / 9)


def test_iqr_table_log_scale_flags_fewer_values(fixture_recipes) -> None:
    recipes = ZeroAsMissing(columns=("minutes",)).fit_transform(fixture_recipes)
    table = pd.concat(
        [iqr_table(recipes, OUTLIER_COLUMNS), iqr_table(recipes, OUTLIER_COLUMNS, log1p=True)],
        ignore_index=True,
    )
    validate(table, OUTLIER_IQR_SCHEMA)
    raw = table[table.scale == "raw"].set_index("column")
    log = table[table.scale == "log1p"].set_index("column")
    assert raw.loc["minutes", "n_above"] >= 2  # 106 (INT_MAX) and 107 (two years)
    assert log.loc["minutes", "upper_fence"] > raw.loc["minutes", "upper_fence"]


def test_quantile_and_threshold_tables(fixture_recipes) -> None:
    q = quantile_table(fixture_recipes, ["minutes"], quantiles=(0.0, 1.0))
    assert q["value"].tolist() == [0.0, 2147483647.0]
    t = threshold_counts(fixture_recipes, "minutes", (43200, 2_000_000_000))
    assert t["n_above"].tolist() == [2, 1]
    assert t["share_above"].iloc[0] == pytest.approx(2 / len(fixture_recipes))


def test_extreme_examples(fixture_recipes) -> None:
    ex = extreme_examples(fixture_recipes, "minutes", n=2)
    assert ex["id"].tolist() == [106, 107]
    assert (ex["column"] == "minutes").all()
    assert extreme_examples(fixture_recipes, "minutes", n=5, min_value=MINUTES_VALID_MAX)[
        "id"
    ].tolist() == [106, 107]


def test_invalid_counts_on_fixtures(fixture_recipes, outliers_block) -> None:
    counts = invalid_counts(fixture_recipes, outliers_block["invalid_rule"]).set_index("column")
    assert counts.loc["minutes", "n_above"] == 2  # 106, 107
    assert counts.loc["n_steps", "n_below"] == 1  # 110
    assert counts.loc["calories", "n_invalid"] == 0


def test_before_after_table() -> None:
    before = _frame([1.0, 2.0, 3.0, 100.0])
    after = _frame([1.0, 2.0, 3.0, np.nan])
    table = before_after_table({"before": before, "after": after, "empty": _frame([np.nan])}, ["x"])
    validate(table, OUTLIER_BEFORE_AFTER_SCHEMA)
    rows = table.set_index("stage")
    assert rows.loc["before", "max"] == 100.0 and rows.loc["after", "max"] == 3.0
    assert rows.loc["after", "n_missing"] == 1
    assert rows.loc["before", "n_outside_fences"] == 1
    assert np.isnan(rows.loc["empty", "median"])


def test_analysis_functions_do_not_mutate_inputs(fixture_recipes) -> None:
    copy = fixture_recipes.copy()
    iqr_table(fixture_recipes, OUTLIER_COLUMNS, log1p=True)
    quantile_table(fixture_recipes, OUTLIER_COLUMNS)
    extreme_examples(fixture_recipes, "calories")
    pd.testing.assert_frame_equal(fixture_recipes, copy)


# ----------------------------------------------------------------- InvalidValueRule


def test_invalid_rule_sets_nan_and_keeps_rows() -> None:
    df = pd.DataFrame({"minutes": [10, 600000, 0], "n_steps": [0, 3, 5], "other": [1, 2, 3]})
    out = InvalidValueRule({"minutes": [0, 525600], "n_steps": [1, None]}).fit_transform(df)
    assert len(out) == 3
    assert out["minutes"].tolist()[:1] == [10.0] and np.isnan(out["minutes"].iloc[1])
    assert out["minutes"].iloc[2] == 0.0  # bounds are inclusive
    assert np.isnan(out["n_steps"].iloc[0]) and out["n_steps"].iloc[1] == 3.0
    pd.testing.assert_series_equal(out["other"], df["other"])
    assert df["minutes"].tolist() == [10, 600000, 0]  # input not modified


def test_configured_minutes_bound_keeps_long_infusions(outliers_block) -> None:
    """Months-long liqueurs / extracts are real; only placeholders and joke values are invalid."""
    df = pd.DataFrame({"minutes": [129610.0, 288000.0, 1051200.0, 2147483647.0]})
    out = InvalidValueRule({"minutes": outliers_block["invalid_rule"]["minutes"]}).fit_transform(df)
    assert out["minutes"].notna().tolist() == [True, True, False, False]


@pytest.mark.parametrize(
    ("bounds", "error"),
    [({"x": [5, 1]}, ValueError), ({"x": [1]}, ValueError), ({"missing": [0, 1]}, KeyError)],
)
def test_invalid_rule_rejects_bad_bounds(bounds, error) -> None:
    with pytest.raises(error):
        InvalidValueRule(bounds).fit(_frame([1.0]))


# ----------------------------------------------------------------- QuantileClipper


def test_clip_bounds_learned_on_fit_data_only() -> None:
    train = _frame(list(range(1, 101)))  # 1..100
    test = _frame([-50.0, 50.0, 10_000.0])
    clipper = QuantileClipper(columns=("x",), lower_q=0.0, upper_q=0.9).fit(train)
    expected = (1.0, float(train["x"].quantile(0.9)))
    assert clipper.bounds_["x"] == expected

    out = clipper.transform(test)
    assert out["x"].tolist() == [1.0, 50.0, expected[1]]
    assert clipper.bounds_["x"] == expected  # transform does not change the bounds
    # fitting on other data gives other bounds: the bounds come from the fit data
    assert QuantileClipper(columns=("x",), upper_q=0.9).fit(test).bounds_["x"] != expected


def test_transform_never_refits() -> None:
    clipper = QuantileClipper(columns=("x",), upper_q=0.5).fit(_frame([1.0, 2.0, 3.0]))
    before = dict(clipper.bounds_)
    for data in ([100.0, 200.0], [-5.0], [np.nan, 2.5]):
        clipper.transform(_frame(data))
        clipper.bounds_table(_frame(data))
    assert clipper.bounds_ == before


def test_clip_ignores_nan_when_learning() -> None:
    clipper = QuantileClipper(columns=("x",), upper_q=1.0).fit(_frame([1.0, np.nan, 3.0]))
    assert clipper.bounds_["x"] == (1.0, 3.0)
    assert np.isnan(clipper.transform(_frame([np.nan]))["x"].iloc[0])


def test_bounds_table_counts() -> None:
    clipper = QuantileClipper(columns=("x",), lower_q=0.1, upper_q=0.9).fit(_frame(list(range(11))))
    table = clipper.bounds_table(_frame([-1.0, 5.0, 20.0, 30.0]))
    row = table.iloc[0]
    assert (row.lower, row.upper) == (1.0, 9.0)
    assert (row.n_clipped_low, row.n_clipped_high) == (1, 2)
    assert clipper.bounds_table()["n_clipped_high"].iloc[0] == 0


@pytest.mark.parametrize(("lower_q", "upper_q"), [(0.5, 0.5), (-0.1, 0.9), (0.0, 1.5)])
def test_clipper_rejects_bad_quantiles(lower_q, upper_q) -> None:
    with pytest.raises(ValueError):
        QuantileClipper(columns=("x",), lower_q=lower_q, upper_q=upper_q).fit(_frame([1.0]))


def test_clipper_rejects_all_nan_column() -> None:
    with pytest.raises(ValueError, match="no observed values"):
        QuantileClipper(columns=("x",)).fit(_frame([np.nan, np.nan]))


# ----------------------------------------------------------------- Log1pTransformer


def test_log1p_and_inverse() -> None:
    df = _frame([0.0, 1.0, np.e - 1, np.nan])
    tr = Log1pTransformer(columns=("x",)).fit(df)
    out = tr.transform(df)
    assert out["x"].iloc[:3].tolist() == pytest.approx([0.0, np.log(2), 1.0])
    assert np.isnan(out["x"].iloc[3])
    np.testing.assert_allclose(tr.inverse_transform(out)["x"].iloc[:3], df["x"].iloc[:3])


def test_log1p_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="negative"):
        Log1pTransformer(columns=("x",)).fit_transform(_frame([-1.0, 2.0]))


# ----------------------------------------------------------------- shared interface


@pytest.mark.parametrize(
    "transformer",
    [
        InvalidValueRule({"x": [0, 10]}),
        QuantileClipper(columns=("x",)),
        Log1pTransformer(columns=("x",)),
    ],
)
def test_transformers_share_the_sklearn_interface(transformer) -> None:
    df = _frame([1.0, 2.0, 30.0])
    assert isinstance(transformer, Transformer)
    with pytest.raises(NotFittedError):
        clone(transformer).transform(df)
    out = clone(transformer).fit(df).transform(df)
    assert isinstance(out, pd.DataFrame) and out.shape == df.shape
    assert clone(transformer).get_params() == transformer.get_params()
    with pytest.raises(TypeError):
        clone(transformer).fit(df.to_numpy())


def test_registered_under_outlier_kind() -> None:
    assert {"invalid_rule", "quantile_clip", "log1p"} <= set(
        registry.available("outlier")["outlier"]
    )
    assert isinstance(registry.create("outlier", "quantile_clip", columns=("x",)), QuantileClipper)
    assert registry.get("outlier", "invalid_rule") is InvalidValueRule
    assert registry.get("outlier", "log1p") is Log1pTransformer


# ----------------------------------------------------------------- config


def test_config_block_covers_all_columns(outliers_block) -> None:
    assert set(outliers_block["columns"]) == set(OUTLIER_COLUMNS)
    assert outliers_block["invalid_rule"]["minutes"] == [0, MINUTES_VALID_MAX]
    for col, steps in outliers_block["columns"].items():
        assert steps == [s for s in STEP_ORDER if s in steps], col  # listed in apply order


def test_steps_from_config(outliers_block) -> None:
    steps = outlier_steps_from_config(outliers_block)
    assert [name for name, _ in steps] == list(STEP_ORDER)
    by_name = dict(steps)
    assert set(by_name["invalid_rule"].bounds) == set(OUTLIER_COLUMNS)
    assert by_name["quantile_clip"].upper_q == outliers_block["quantile_clip"]["upper_q"]
    assert "n_ingredients" not in by_name["log1p"].columns


def test_steps_from_config_skips_unused_and_rejects_unknown() -> None:
    steps = outlier_steps_from_config({"columns": {"x": ["log1p"]}})
    assert [n for n, _ in steps] == ["log1p"]
    with pytest.raises(ValueError, match="unknown"):
        outlier_steps_from_config({"columns": {"x": ["winsorize"]}})
    with pytest.raises(ValueError, match="no bounds"):
        outlier_steps_from_config({"columns": {"x": ["invalid_rule"]}})


def test_config_chain_on_fixtures_matches_contract(fixture_recipes, outliers_block) -> None:
    recipes = ZeroAsMissing(columns=("minutes",)).fit_transform(fixture_recipes)
    steps = outlier_steps_from_config(outliers_block)
    rules = Pipeline([s for s in steps if s[0] != "log1p"]).fit(recipes)
    after = rules.transform(recipes)
    validate(after, RECIPES_AFTER_OUTLIERS_SCHEMA)
    assert (
        after.set_index("id").loc[[106, 107, 110], ["minutes", "n_steps"]].isna().any(axis=1).all()
    )
    logged = dict(steps)["log1p"].fit_transform(after)
    assert (logged["minutes"].dropna() <= np.log1p(MINUTES_VALID_MAX)).all()


def test_contract_rejects_untreated_values(fixture_recipes) -> None:
    with pytest.raises(SchemaValidationError):
        validate(fixture_recipes, RECIPES_AFTER_OUTLIERS_SCHEMA)  # INT_MAX minutes, n_steps 0


# ----------------------------------------------------------------- notebook


def _code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_starts_with_colab_cell_and_has_no_logic() -> None:
    cells = _code_cells()
    for needle in ("drive.mount", "git clone", "pip install -q -e .", "DATA_ROOT", "load_config()"):
        assert needle in cells[0]
    assert "SecretNotFoundError" in cells[0]
    for src in cells[1:]:
        assert "def " not in src and "class " not in src


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_notebook_runs_on_fixtures(cfg: Config, fixture_recipes, fixture_interactions) -> None:
    import matplotlib.pyplot as plt

    store = ArtifactStore.from_config(cfg)
    store.save("loaded", {"recipes": fixture_recipes, "interactions": fixture_interactions})
    store.save("quality", {"issues": detect_issues(fixture_recipes, fixture_interactions)})
    ns: dict[str, object] = {"CFG": cfg, "display": lambda *a, **k: None}
    try:
        for src in _code_cells()[1:]:
            exec(compile(src, str(NOTEBOOK), "exec"), ns)  # noqa: S102 - our own notebook
    finally:
        plt.close("all")
    art = store.load("outliers")
    assert art.version == "v1"
    assert set(art.keys()) == set(OUTLIERS_OUTPUT_SCHEMA)
    assert [i["stage"] for i in art.manifest["inputs"]] == ["loaded", "quality"]
    assert {f.get("schema") for f in art.manifest["files"].values()} == {
        s.name for s in OUTLIERS_OUTPUT_SCHEMA.values()
    }
    assert art.manifest["params"]["outliers_config"]["quantile_clip"]["upper_q"] == 0.99


# ----------------------------------------------------------------- real sample


def test_on_real_sample(outliers_block) -> None:
    recipes = read_artifact_dir(require_sample(), names=["recipes"])["recipes"]
    recipes = ZeroAsMissing(columns=("minutes",)).fit_transform(recipes)
    train = recipes.sample(frac=0.7, random_state=0)
    steps = dict(outlier_steps_from_config(outliers_block))
    rule, clipper = steps["invalid_rule"], steps["quantile_clip"]
    clipper.fit(rule.fit_transform(train))
    after = validate(clipper.transform(rule.transform(recipes)), RECIPES_AFTER_OUTLIERS_SCHEMA)
    table = before_after_table({"before": recipes, "after": after}, OUTLIER_COLUMNS)
    skew = table.pivot(index="column", columns="stage", values="skew")
    assert (skew["after"] < skew["before"]).all()
    # the clip bounds equal the train quantile, not the full-data one
    assert clipper.bounds_["calories"][1] == pytest.approx(
        rule.transform(train)["calories"].quantile(0.99)
    )
