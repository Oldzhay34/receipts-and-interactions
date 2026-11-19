"""Unit tests for Stage 1.2 Data Quality Report (recsys.analysis.quality)."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from recsys.analysis.quality import (
    ISSUE_RULES,
    IssueRule,
    categorical_dqr,
    continuous_dqr,
    detect_issues,
    list_column_dqr,
    render_table_png,
)
from recsys.contracts import QUALITY_ISSUES_SCHEMA, validate

matplotlib.use("Agg")


@pytest.fixture()
def sample_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic recipes and interactions covering all 8 Stage 1.2 edge cases."""
    recipes = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["soup", None, "cake", "salad", "tea"],
            "minutes": [30, 0, 20000, 15, 5],
            "n_steps": [4, 2, 5, 0, 1],
            "n_ingredients": [3, 2, 4, 2, 1],
            "description": ["good", "ok", None, "fresh", "hot"],
            "ingredients": [
                ["water", "salt", "carrot"],
                ["salt", "pepper"],
                ["flour", "sugar", "egg", "milk"],
                ["lettuce", "salt"],
                ["water"],
            ],
            "tags": [
                ["easy", "dinner"],
                ["no-cook"],
                ["dessert", "easy"],
                ["salad"],
                ["beverages"],
            ],
            "calories": [120.0, 50.0, 450.0, 80.0, 5.0],
        }
    )

    interactions = pd.DataFrame(
        {
            "user_id": [10, 11, 11, 12, 13],
            "recipe_id": [1, 2, 2, 3, 4],
            "rating": [5, 0, 0, 4, 3],
            "review": ["great", "nice", "nice", None, "ok"],
        }
    )
    return recipes, interactions


def test_continuous_dqr_columns_and_stats(sample_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    recipes, _ = sample_frames
    dqr = continuous_dqr(recipes, ["minutes", "calories"])
    expected_cols = [
        "Feature",
        "Count",
        "% Missing",
        "Cardinality",
        "Min",
        "Q1",
        "Mean",
        "Median",
        "Q3",
        "Max",
        "Std Dev",
        "Skewness",
        "Kurtosis",
    ]
    assert list(dqr.columns) == expected_cols
    assert len(dqr) == 2
    assert dqr.loc[dqr["Feature"] == "minutes", "Min"].item() == 0.0
    assert dqr.loc[dqr["Feature"] == "minutes", "Max"].item() == 20000.0


def test_categorical_dqr_modes(sample_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, interactions = sample_frames
    dqr = categorical_dqr(interactions, ["rating"])
    expected_cols = [
        "Feature",
        "Count",
        "% Missing",
        "Cardinality",
        "Mode",
        "Mode Freq",
        "Mode %",
        "2nd Mode",
        "2nd Mode Freq",
        "2nd Mode %",
    ]
    assert list(dqr.columns) == expected_cols
    assert dqr.loc[0, "Mode"] == "0"
    assert dqr.loc[0, "Mode Freq"] == 2


def test_list_column_dqr(sample_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    recipes, _ = sample_frames
    dqr = list_column_dqr(recipes, ["ingredients", "tags"])
    assert len(dqr) == 2
    ing_row = dqr.loc[dqr["Feature"] == "ingredients"].iloc[0]
    assert ing_row["Min Len"] == 1
    assert ing_row["Max Len"] == 4
    assert ing_row["Unique Items"] == 9


def test_detect_issues_all_8_rules_and_schema(
    sample_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    recipes, interactions = sample_frames
    assert len(ISSUE_RULES) == 8

    issues = detect_issues(recipes, interactions)
    validate(issues, QUALITY_ISSUES_SCHEMA)

    counts = dict(zip(issues["issue_id"], issues["affected_rows"]))
    assert counts["rating_zero"] == 2
    assert counts["minutes_zero"] == 1
    assert counts["minutes_extreme"] == 1
    assert counts["n_steps_zero"] == 1
    assert counts["missing_text_fields"] == 3
    assert counts["duplicate_interactions"] == 1
    assert counts["recipes_without_rating"] == 2  # recipe 2 (only 0 rating) and recipe 5 (no rating)
    assert counts["missing_recipe_name"] == 1


def test_detect_issues_extensible_rules(sample_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    recipes, interactions = sample_frames
    custom_rule = IssueRule(
        issue_id="low_calories",
        column="recipes.calories",
        rule="calories < 10",
        suggested_action="Review low-calorie beverages",
        finder=lambda r, _: r.loc[r["calories"] < 10, "id"].astype(int).tolist(),
    )
    issues = detect_issues(recipes, interactions, rules=[custom_rule])
    validate(issues, QUALITY_ISSUES_SCHEMA)
    assert len(issues) == 1
    assert issues.loc[0, "affected_rows"] == 1


def test_render_table_png_returns_figure(sample_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    recipes, _ = sample_frames
    dqr = continuous_dqr(recipes, ["minutes", "calories"])
    fig = render_table_png(dqr, title="Continuous DQR")
    assert isinstance(fig, Figure)
    plt.close(fig)