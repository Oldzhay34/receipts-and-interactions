"""Pure plotting functions returning matplotlib Figures without saving to disk."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from recsys.viz.style import FIGSIZE, PALETTE, apply_style


def hist_box(
    series: pd.Series,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    bins: int = 40,
    log1p: bool = False,
) -> Figure:
    """Plot a histogram and horizontal box plot side by side (no KDE)."""
    apply_style()
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if log1p:
        clean = clean[clean >= 0]
        values = np.log1p(clean)
        x_label = xlabel or f"log1p({series.name or 'value'})"
    else:
        values = clean
        x_label = xlabel or str(series.name or "value")

    fig, (ax_hist, ax_box) = plt.subplots(
        1, 2, figsize=FIGSIZE["hist_box"], gridspec_kw={"width_ratios": [2.2, 1.0]}
    )

    ax_hist.hist(values, bins=bins, color=PALETTE["primary"], alpha=0.85, edgecolor="white")
    ax_hist.set_xlabel(x_label)
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Histogram")

    ax_box.boxplot(
        values,
        orientation="horizontal",
        patch_artist=True,
        boxprops={"facecolor": PALETTE["secondary"], "alpha": 0.65},
        medianprops={"color": "#111827", "linewidth": 1.8},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4, "markerfacecolor": PALETTE["neutral"]},
    )
    ax_box.set_xlabel(x_label)
    ax_box.set_yticks([])
    ax_box.set_title("Box Plot")

    fig.suptitle(title or f"Distribution of {x_label}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def bar_counts(
    series: pd.Series,
    top_n: int = 20,
    title: Optional[str] = None,
    xlabel: str = "Count",
    ylabel: Optional[str] = None,
    horizontal: bool = True,
) -> Figure:
    """Plot value counts of a categorical/discrete series or pre-exploded items."""
    apply_style()
    counts = series.dropna().value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=FIGSIZE["bar"])
    if horizontal:
        counts_plot = counts.iloc[::-1]
        ax.barh(counts_plot.index.astype(str), counts_plot.values, color=PALETTE["primary"], alpha=0.88)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or str(series.name or "Category"))
    else:
        ax.bar(counts.index.astype(str), counts.values, color=PALETTE["primary"], alpha=0.88)
        ax.set_xlabel(ylabel or str(series.name or "Category"))
        ax.set_ylabel(xlabel)

    ax.set_title(title or f"Top {len(counts)} {series.name or 'Items'}")
    fig.tight_layout()
    return fig


def scatter_sample(
    df: pd.DataFrame,
    x: str,
    y: str,
    n: int = 20_000,
    alpha: float = 0.25,
    random_state: int = 42,
    title: Optional[str] = None,
    log_x: bool = False,
    log_y: bool = False,
) -> Figure:
    """Plot a sampled scatter plot between two numeric columns."""
    apply_style()
    sub = df[[x, y]].dropna()
    if len(sub) > n:
        sub = sub.sample(n=n, random_state=random_state)

    fig, ax = plt.subplots(figsize=FIGSIZE["scatter"])
    ax.scatter(sub[x], sub[y], s=10, alpha=alpha, color=PALETTE["accent"], edgecolors="none")
    if log_x:
        ax.set_xscale("symlog")
    if log_y:
        ax.set_yscale("symlog")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} vs {x} (n={len(sub):,})")
    fig.tight_layout()
    return fig


def corr_heatmap(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    method: str = "spearman",
    title: Optional[str] = None,
) -> Figure:
    """Plot a correlation matrix heatmap with numeric annotations."""
    apply_style()
    num_df = df[cols] if cols is not None else df.select_dtypes(include=[np.number])
    corr = num_df.corr(method=method)

    fig, ax = plt.subplots(figsize=FIGSIZE["heatmap"])
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"{method.title()} r")

    labels = list(corr.columns)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.6 else "#111827"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.grid(False)
    ax.set_title(title or f"Feature Correlation Heatmap ({method.title()})")
    fig.tight_layout()
    return fig


def box_by_group(
    df: pd.DataFrame,
    value: str,
    group: str,
    top_n_groups: int = 10,
    title: Optional[str] = None,
) -> Figure:
    """Plot side-by-side box plots of a numeric value grouped by top categories."""
    apply_style()
    sub = df[[value, group]].dropna()
    top_groups = sub[group].value_counts().head(top_n_groups).index.tolist()
    grouped_data = [sub.loc[sub[group] == g, value].values for g in top_groups]

    fig, ax = plt.subplots(figsize=FIGSIZE["box_group"])
    ax.boxplot(
        grouped_data,
        tick_labels=[str(g) for g in top_groups],
        patch_artist=True,
        boxprops={"facecolor": PALETTE["primary"], "alpha": 0.65},
        medianprops={"color": "#111827", "linewidth": 1.6},
        flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.35},
    )
    ax.set_xlabel(group)
    ax.set_ylabel(value)
    ax.tick_params(axis="x", rotation=30)
    ax.set_title(title or f"{value} by {group} (Top {len(top_groups)})")
    fig.tight_layout()
    return fig


def long_tail_hist(
    counts: pd.Series,
    log: bool = True,
    bins: int = 50,
    title: Optional[str] = None,
    xlabel: str = "Interaction Count",
) -> Figure:
    """Plot a long-tail interaction count histogram (optionally with log-scaled axes)."""
    apply_style()
    clean = pd.to_numeric(counts, errors="coerce").dropna()
    clean = clean[clean > 0]

    fig, ax = plt.subplots(figsize=FIGSIZE["long_tail"])
    if log and len(clean) > 0:
        max_val = max(float(clean.max()), 1.0)
        bin_edges = np.logspace(0, np.log10(max_val + 1.0), bins)
        ax.hist(clean, bins=bin_edges, color=PALETTE["highlight"], alpha=0.85, edgecolor="white")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{xlabel} (log scale)")
        ax.set_ylabel("Frequency (log scale)")
    else:
        ax.hist(clean, bins=bins, color=PALETTE["highlight"], alpha=0.85, edgecolor="white")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")

    ax.set_title(title or f"Long-Tail Distribution: {xlabel}")
    fig.tight_layout()
    return fig