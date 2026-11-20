"""Unit tests for binning strategies (Phase 1.7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.features.binning import (
    DomainBins,
    EqualFrequencyBins,
    EqualWidthBins,
    compare_binning_strategies,
)
from recsys.registry import create


def test_domain_bins_preserves_numeric_and_handles_extremes() -> None:
    train = pd.DataFrame({"minutes": [10, 25, 45, 90]})
    test = pd.DataFrame({"minutes": [-5, 15, 60, 10_000]})

    binner = DomainBins(column="minutes", edges=[0, 30, 60, np.inf], labels=["quick", "medium", "long"])
    binner.fit(train)
    out = binner.transform(test)

    assert "minutes" in out.columns
    assert "minutes_bin" in out.columns
    assert list(out["minutes"]) == [-5, 15, 60, 10_000]
    assert list(out["minutes_bin"]) == ["quick", "quick", "medium", "long"]


def test_equal_width_edges_fixed_at_fit_and_out_of_range_clamped() -> None:
    train = pd.DataFrame({"calories": [100.0, 200.0, 300.0, 400.0]})
    test = pd.DataFrame({"calories": [0.0, 250.0, 99999.0]})

    binner = EqualWidthBins(column="calories", n_bins=3, labels=["low", "medium", "high"])
    binner.fit(train)
    edges_before = binner.edges_.copy()

    out = binner.transform(test)
    assert np.allclose(binner.edges_, edges_before)
    assert list(out["calories_bin"]) == ["low", "medium", "high"]


def test_equal_frequency_edges_fixed_at_fit() -> None:
    train = pd.DataFrame({"calories": [10, 20, 30, 40, 50, 60, 70, 80, 90]})
    test = pd.DataFrame({"calories": [-100, 50, 5000]})

    binner = EqualFrequencyBins(column="calories", n_bins=3, labels=["low", "medium", "high"])
    binner.fit(train)
    edges_before = binner.edges_.copy()

    out = binner.transform(test)
    assert np.allclose(binner.edges_, edges_before)
    assert list(out["calories_bin"]) == ["low", "medium", "high"]


def test_registry_creates_all_binners() -> None:
    d = create("binner", "domain", column="minutes", edges=[0, 30, 60, np.inf], labels=["q", "m", "l"])
    w = create("binner", "equal_width", column="calories", n_bins=3)
    f = create("binner", "equal_frequency", column="calories", n_bins=3)

    assert isinstance(d, DomainBins)
    assert isinstance(w, EqualWidthBins)
    assert isinstance(f, EqualFrequencyBins)


def test_compare_binning_strategies_returns_summary() -> None:
    df = pd.DataFrame({"calories": [100, 150, 200, 250, 300, 900]})
    summary = compare_binning_strategies(
        df,
        column="calories",
        strategies={
            "equal_width": EqualWidthBins(column="calories", n_bins=3, labels=["low", "medium", "high"]),
            "equal_frequency": EqualFrequencyBins(column="calories", n_bins=3, labels=["low", "medium", "high"]),
        },
    )
    assert {"column", "strategy", "bin", "count", "pct"}.issubset(summary.columns)
    assert set(summary["strategy"]) == {"equal_width", "equal_frequency"}