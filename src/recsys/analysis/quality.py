"""Data Quality Report (DQR) tables and issue detection registry (Stage 1.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def continuous_dqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Build a Kelleher-format Data Quality Report for continuous columns."""
    rows: list[dict[str, object]] = []
    n_total = len(df)

    for col in cols:
        s = df[col]
        n_missing = int(s.isna().sum())
        pct_missing = (n_missing / n_total * 100.0) if n_total > 0 else 0.0
        valid = s.dropna()

        rows.append(
            {
                "Feature": col,
                "Count": int(valid.count()),
                "% Missing": round(pct_missing, 4),
                "Cardinality": int(valid.nunique()),
                "Min": float(valid.min()) if not valid.empty else np.nan,
                "Q1": float(valid.quantile(0.25)) if not valid.empty else np.nan,
                "Mean": float(valid.mean()) if not valid.empty else np.nan,
                "Median": float(valid.median()) if not valid.empty else np.nan,
                "Q3": float(valid.quantile(0.75)) if not valid.empty else np.nan,
                "Max": float(valid.max()) if not valid.empty else np.nan,
                "Std Dev": float(valid.std()) if len(valid) > 1 else np.nan,
                "Skewness": float(valid.skew()) if len(valid) > 2 else np.nan,
                "Kurtosis": float(valid.kurtosis()) if len(valid) > 3 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def categorical_dqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Build a Kelleher-format Data Quality Report for categorical columns."""
    rows: list[dict[str, object]] = []
    n_total = len(df)

    for col in cols:
        s = df[col]
        n_missing = int(s.isna().sum())
        pct_missing = (n_missing / n_total * 100.0) if n_total > 0 else 0.0
        valid = s.dropna()
        n_valid = len(valid)
        vc = valid.value_counts()

        mode_val = vc.index[0] if len(vc) >= 1 else None
        mode_freq = int(vc.iloc[0]) if len(vc) >= 1 else 0
        mode_pct = (mode_freq / n_valid * 100.0) if n_valid > 0 else 0.0

        second_val = vc.index[1] if len(vc) >= 2 else None
        second_freq = int(vc.iloc[1]) if len(vc) >= 2 else 0
        second_pct = (second_freq / n_valid * 100.0) if n_valid > 0 else 0.0

        rows.append(
            {
                "Feature": col,
                "Count": n_valid,
                "% Missing": round(pct_missing, 4),
                "Cardinality": int(valid.nunique()),
                "Mode": str(mode_val) if mode_val is not None else "",
                "Mode Freq": mode_freq,
                "Mode %": round(mode_pct, 4),
                "2nd Mode": str(second_val) if second_val is not None else "",
                "2nd Mode Freq": second_freq,
                "2nd Mode %": round(second_pct, 4),
            }
        )

    return pd.DataFrame(rows)


def list_column_dqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Build a Data Quality Report for list-valued columns (e.g., ingredients, tags)."""
    rows: list[dict[str, object]] = []
    n_total = len(df)

    for col in cols:
        s = df[col]
        n_missing = int(s.isna().sum())
        pct_missing = (n_missing / n_total * 100.0) if n_total > 0 else 0.0
        valid = s.dropna()
        lengths = valid.apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 0)
        unique_items: set[object] = set()
        for items in valid:
            if isinstance(items, (list, tuple)):
                unique_items.update(items)

        rows.append(
            {
                "Feature": col,
                "Count": int(valid.count()),
                "% Missing": round(pct_missing, 4),
                "Unique Items": len(unique_items),
                "Min Len": int(lengths.min()) if not lengths.empty else 0,
                "Median Len": float(lengths.median()) if not lengths.empty else np.nan,
                "Mean Len": float(lengths.mean()) if not lengths.empty else np.nan,
                "Max Len": int(lengths.max()) if not lengths.empty else 0,
            }
        )

    return pd.DataFrame(rows)


@dataclass(frozen=True)
class IssueRule:
    """Declarative data quality rule registered for detect_issues."""

    issue_id: str
    column: str
    rule: str
    suggested_action: str
    finder: Callable[[pd.DataFrame, pd.DataFrame], list[int]]


ISSUE_RULES: list[IssueRule] = []


def register_issue_rule(
    issue_id: str,
    column: str,
    rule: str,
    suggested_action: str,
) -> Callable[[Callable[[pd.DataFrame, pd.DataFrame], list[int]]], Callable[[pd.DataFrame, pd.DataFrame], list[int]]]:
    """Register a new data quality issue rule without modifying detect_issues."""

    def decorator(
        fn: Callable[[pd.DataFrame, pd.DataFrame], list[int]],
    ) -> Callable[[pd.DataFrame, pd.DataFrame], list[int]]:
        ISSUE_RULES.append(
            IssueRule(
                issue_id=issue_id,
                column=column,
                rule=rule,
                suggested_action=suggested_action,
                finder=fn,
            )
        )
        return fn

    return decorator


@register_issue_rule(
    issue_id="rating_zero",
    column="interactions.rating",
    rule="rating == 0 (unrated interaction on Food.com)",
    suggested_action="Convert 0 to NaN in Stage 1.3; exclude from rating averages",
)
def _find_rating_zero(recipes: pd.DataFrame, interactions: pd.DataFrame) -> list[int]:
    mask = interactions["rating"] == 0
    return interactions.loc[mask, "recipe_id"].astype(int).tolist()


@register_issue_rule(
    issue_id="minutes_zero",
    column="recipes.minutes",
    rule="minutes == 0",
    suggested_action="Investigate no-cook vs missing in Stage 1.3; impute via GroupMedianImputer if missing",
)
def _find_minutes_zero(recipes: pd.DataFrame, interactions: pd.DataFrame) -> list[int]:
    mask = recipes["minutes"] == 0
    return recipes.loc[mask, "id"].astype(int).tolist()


@register_issue_rule(
    issue_id="minutes_extreme",
    column="recipes.minutes",
    rule="minutes > 10080 (more than 7 days)",
    suggested_action="Treat as invalid/extreme outlier in Stage 1.4 (InvalidValueRule / QuantileClipper)",
)
def _find_minutes_extreme(recipes: pd.DataFrame, interactions: pd.DataFrame) -> list[int]:
    mask = recipes["minutes"] > 10080
    return recipes.loc[mask, "id"].astype(int).tolist()


@register_issue_rule(
    issue_id="n_steps_zero",
    column="recipes.n_steps",
    rule="n_steps == 0",
    suggested_action="Flag as invalid value in Stage 1.4",
)
def _find_n_steps_zero(recipes: pd.DataFrame, interactions: pd.DataFrame) -> list[int]:
    mask = recipes["n_steps"] == 0
    return recipes.loc[mask, "id"].astype(int).tolist()


@register_issue_rule(
    issue_id="missing_text_fields",
    column="recipes.name, recipes.description, interactions.review",
    rule="Missing (NaN) values in name, description, or review",
    suggested_action="Handle per-column in Stage 1.3 (drop missing name, keep/fill description and review)",
)
def _find_missing_text(recipes: pd.DataFrame, interactions: pd.DataFrame) -> list[int]:
    r_miss = recipes.loc[recipes["name"].isna() | recipes["description"].isna(), "id"].astype(int).tolist()
    i_miss = interactions.loc[interactions["review"].isna(), "recipe_id"].astype(int).tolist()
    return r_miss + i_miss


@register_issue_rule(
    issue_id="duplicate_interactions",
    column="interactions.*",
    rule="Exact duplicate interaction rows",
    suggested_action="Drop exact duplicates in Stage 1.3",
)
def _find_duplicate_interactions(recipes: pd.DataFrame, interactions: pd.DataFrame) -> list[int]:
    mask = interactions.duplicated(keep="first")
    return interactions.loc[mask, "recipe_id"].astype(int).tolist()


@register_issue_rule(
    issue_id="recipes_without_rating",
    column="recipes.id",
    rule="Recipe has no non-zero rating in interactions",
    suggested_action="Mark has_rating=False in Stage 1.3; exclude from training, keep in recommendation pool",
)
def _find_recipes_without_rating(recipes: pd.DataFrame, interactions: pd.DataFrame) -> list[int]:
    rated_ids = set(interactions.loc[interactions["rating"] > 0, "recipe_id"].dropna().astype(int))
    mask = ~recipes["id"].astype(int).isin(rated_ids)
    return recipes.loc[mask, "id"].astype(int).tolist()


@register_issue_rule(
    issue_id="missing_recipe_name",
    column="recipes.name",
    rule="recipes.name is null",
    suggested_action="Drop rows missing critical identifier in Stage 1.3",
)
def _find_missing_name(recipes: pd.DataFrame, interactions: pd.DataFrame) -> list[int]:
    mask = recipes["name"].isna()
    return recipes.loc[mask, "id"].astype(int).tolist()


def detect_issues(
    recipes: pd.DataFrame,
    interactions: pd.DataFrame,
    rules: list[IssueRule] | None = None,
    max_examples: int = 5,
) -> pd.DataFrame:
    """Run all registered data quality rules and return a standardized issues table."""
    active_rules = ISSUE_RULES if rules is None else rules
    records: list[dict[str, object]] = []

    for rule in active_rules:
        ids = rule.finder(recipes, interactions)
        records.append(
            {
                "issue_id": rule.issue_id,
                "column": rule.column,
                "rule": rule.rule,
                "affected_rows": len(ids),
                "example_ids": ", ".join(str(x) for x in ids[:max_examples]),
                "suggested_action": rule.suggested_action,
            }
        )

    return pd.DataFrame(records)


def render_table_png(df: pd.DataFrame, title: str, max_col_width: int = 36) -> Figure:
    """Render a DataFrame as a clean matplotlib Figure for report inclusion."""
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        else:
            display_df[col] = display_df[col].astype(str).str.slice(0, max_col_width)

    n_rows, n_cols = display_df.shape
    fig_w = max(8.0, min(20.0, n_cols * 1.5))
    fig_h = max(2.2, (n_rows + 2) * 0.42)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, loc="left")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#e8e6df")
            cell.set_text_props(weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f9f8f5")

    fig.tight_layout()
    return fig