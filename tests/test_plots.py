"""Unit tests for visualization style, plotting functions, and distribution classifier (1.6)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from recsys.analysis.distribution import classify_distribution, distribution_summary_table
from recsys.viz.plots import (
    bar_counts,
    box_by_group,
    corr_heatmap,
    hist_box,
    long_tail_hist,
    scatter_sample,
)
from recsys.viz.style import DPI, PALETTE, apply_style


def test_apply_style_sets_dpi() -> None:
    apply_style()
    assert plt.rcParams["figure.dpi"] == DPI
    assert plt.rcParams["savefig.dpi"] == DPI
    assert "primary" in PALETTE


def test_plot_functions_return_figure_and_do_not_save(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "minutes": rng.exponential(scale=30, size=200),
            "n_ingredients": rng.integers(2, 20, size=200),
            "calories": rng.lognormal(mean=5.5, sigma=0.6, size=200),
            "avg_rating": rng.uniform(3.0, 5.0, size=200),
            "tag": rng.choice(["quick", "dessert", "healthy", "dinner"], size=200),
        }
    )
    counts = pd.Series(rng.zipf(2.0, size=300), name="user_interactions")

    figs = [
        hist_box(df["minutes"], log1p=False),
        hist_box(df["minutes"], log1p=True),
        bar_counts(df["tag"], top_n=4),
        scatter_sample(df, "n_ingredients", "minutes", n=100),
        corr_heatmap(df, cols=["minutes", "n_ingredients", "calories", "avg_rating"]),
        box_by_group(df, value="avg_rating", group="tag", top_n_groups=4),
        long_tail_hist(counts, log=True),
    ]

    for fig in figs:
        assert isinstance(fig, Figure)
        plt.close(fig)

    # Plotting functions must never write files directly to disk (saving is ArtifactStore's job)
    assert list(tmp_path.iterdir()) == []


def test_classify_distribution_types() -> None:
    rng = np.random.default_rng(42)

    normal_s = pd.Series(rng.normal(0.0, 1.0, size=5000), name="normal_feat")
    uniform_s = pd.Series(rng.uniform(0.0, 10.0, size=5000), name="uniform_feat")
    long_tail_s = pd.Series(rng.lognormal(mean=1.0, sigma=1.6, size=5000), name="long_tail_feat")
    left_skew_s = pd.Series(5.0 - rng.exponential(scale=1.0, size=5000), name="left_skew_feat")
    bimodal_s = pd.Series(
        np.concatenate([rng.normal(-4.0, 0.6, 2500), rng.normal(4.0, 0.6, 2500)]),
        name="bimodal_feat",
    )

    assert classify_distribution(normal_s).distribution_type == "normal"
    assert classify_distribution(uniform_s).distribution_type == "uniform"
    assert classify_distribution(long_tail_s).distribution_type == "long-tail"
    assert classify_distribution(left_skew_s).distribution_type == "left-skewed"
    assert classify_distribution(bimodal_s).distribution_type == "multimodal"


def test_distribution_summary_table() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": rng.normal(10, 2, size=500),
            "b": rng.lognormal(1, 1.5, size=500),
        }
    )
    summary = distribution_summary_table(df, ["a", "b"])
    assert list(summary["feature"]) == ["a", "b"]
    assert {"feature", "distribution_type", "skewness", "kurtosis", "p99_to_median_ratio"}.issubset(
        summary.columns
    )


# ------------------------------------------------------------------ notebook 06 helpers (1.12-A O4)


def test_notebook_06_helpers() -> None:
    from recsys.analysis.distribution import (
        NUMERIC_FEATURE_COLUMNS,
        cold_start_summary,
        list_items,
        present_columns,
        rating_by_list_item,
        reviews_per_entity,
        with_rating_stats,
    )

    recipes = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "minutes": [10, 20, 30],
            "calories": [100.0, 200.0, 300.0],
            "tags": [["a", "b"], ["b"], ["c"]],
        }
    )
    inter = pd.DataFrame(
        {"user_id": [7, 7, 8, 9], "recipe_id": [1, 2, 1, 1], "rating": [5, 0, 3, 4]}
    )
    assert present_columns(recipes) == ["minutes", "calories"]
    assert len(NUMERIC_FEATURE_COLUMNS) == 10
    assert list_items(recipes["tags"], "tag").tolist() == ["a", "b", "b", "c"]

    users, recs = reviews_per_entity(inter)
    cs = cold_start_summary(users, recs).set_index("entity")
    assert (
        cs.loc["user", "single_review_count"] == 2 and cs.loc["user", "single_review_pct"] == 66.67
    )
    assert cs.loc["recipe", "max_reviews"] == 3

    viz = with_rating_stats(recipes, inter)
    assert viz["avg_rating"].tolist()[0] == 4.0  # rating 0 ignored
    assert pd.isna(viz.loc[viz["id"] == 2, "avg_rating"]).all()  # only a 0 rating
    assert len(with_rating_stats(recipes, inter, how="inner")) == 1
    long = rating_by_list_item(viz)
    assert long["tags"].tolist() == ["a", "b"] and (long["avg_rating"] == 4.0).all()
