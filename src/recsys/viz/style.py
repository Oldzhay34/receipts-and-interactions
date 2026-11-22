"""Unified matplotlib visual style for all project figures (single style source)."""

from __future__ import annotations

import matplotlib as mpl

DPI: int = 150

PALETTE: dict[str, str] = {
    "primary": "#1D4ED8",
    "secondary": "#C2410C",
    "accent": "#0F766E",
    "neutral": "#475569",
    "muted": "#94A3B8",
    "highlight": "#BE185D",
    "grid": "#E2E8F0",
}

FIGSIZE: dict[str, tuple[float, float]] = {
    "hist_box": (10.5, 4.2),
    "bar": (9.5, 5.5),
    "scatter": (7.5, 5.2),
    "heatmap": (8.5, 7.2),
    "box_group": (10.0, 5.0),
    "long_tail": (8.5, 4.6),
}


def apply_style() -> None:
    """Apply consistent rcParams across all figures."""
    mpl.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#CBD5E1",
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
        }
    )
