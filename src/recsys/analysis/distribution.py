"""Rule-based distribution classification for numeric features (Phase 1.6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DistributionClassification:
    """Result of classifying a single numeric feature's distribution."""

    feature: str
    distribution_type: str
    skewness: float
    kurtosis: float
    p99_to_median_ratio: float


def _count_histogram_modes(values: np.ndarray, bins: int = 20) -> int:
    """Count prominent local peaks separated by a genuine valley (< 60% of peak height)."""
    if len(values) < 10 or np.allclose(values, values[0]):
        return 1
    counts, _ = np.histogram(values, bins=bins)
    max_count = counts.max()
    if max_count == 0:
        return 1

    # Flat/uniform histograms have no deep valley between bins
    if counts.min() >= 0.6 * max_count:
        return 1

    threshold = 0.25 * max_count
    peaks: list[int] = []
    for i in range(len(counts)):
        left = counts[i - 1] if i > 0 else -1
        right = counts[i + 1] if i < len(counts) - 1 else -1
        if counts[i] >= threshold and counts[i] >= left and counts[i] > right:
            peaks.append(i)

    if len(peaks) <= 1:
        return 1

    # Keep only peaks separated by a valley below 60% of the smaller peak
    distinct_peaks = [peaks[0]]
    for p in peaks[1:]:
        prev = distinct_peaks[-1]
        valley = counts[prev : p + 1].min()
        if valley < 0.6 * min(counts[prev], counts[p]):
            distinct_peaks.append(p)
        elif counts[p] > counts[prev]:
            distinct_peaks[-1] = p
    return len(distinct_peaks)


def classify_distribution(series: pd.Series) -> DistributionClassification:
    """Classify a numeric series into a standard distribution shape.

    Rule-based thresholds (evaluated in order):
    1. ``long-tail``: skewness > 2.0 and (kurtosis > 7.0 or p99/median > 5.0 with median > 0).
    2. ``right-skewed``: skewness > 0.5.
    3. ``left-skewed``: skewness < -0.5.
    4. ``multimodal``: |skewness| <= 0.5 and >= 2 prominent histogram peaks (>= 25% of max bin).
    5. ``uniform``: |skewness| <= 0.5 and excess kurtosis < -1.0.
    6. ``normal``: otherwise (|skewness| <= 0.5 and unimodal, near-mesokurtic).
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    name = str(series.name or "feature")
    if len(clean) < 3 or clean.nunique() <= 1:
        return DistributionClassification(
            feature=name,
            distribution_type="constant",
            skewness=0.0,
            kurtosis=0.0,
            p99_to_median_ratio=1.0,
        )

    skew = float(clean.skew())
    kurt = float(clean.kurt())  # Fisher excess kurtosis (normal = 0, uniform = -1.2)
    median = float(clean.median())
    p99 = float(clean.quantile(0.99))
    ratio = float(p99 / median) if median > 0 else float("inf")

    if skew > 2.0 and (kurt > 7.0 or (median > 0 and ratio > 5.0)):
        dist_type = "long-tail"
    elif skew > 0.5:
        dist_type = "right-skewed"
    elif skew < -0.5:
        dist_type = "left-skewed"
    elif _count_histogram_modes(clean.to_numpy()) >= 2:
        dist_type = "multimodal"
    elif kurt < -1.0:
        dist_type = "uniform"
    else:
        dist_type = "normal"

    return DistributionClassification(
        feature=name,
        distribution_type=dist_type,
        skewness=round(skew, 4),
        kurtosis=round(kurt, 4),
        p99_to_median_ratio=round(ratio, 4) if np.isfinite(ratio) else float("inf"),
    )


def distribution_summary_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Build a summary DataFrame classifying all requested numeric columns."""
    rows = [classify_distribution(df[col]).__dict__ for col in cols if col in df.columns]
    return pd.DataFrame(rows)
