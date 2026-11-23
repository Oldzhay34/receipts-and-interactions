# 9. Binning Strategies

Source: `binning/v1` (input `loaded/v1`; `params`: minutes domain edges 0/30/60/inf, avg_rating domain edges 0.0/3.5/4.5/5.0, n_bins 3). Tables: `minutes_comparison.parquet`, `calories_comparison.parquet`, `avg_rating_comparison.parquet`; figures `fig_07_01_rating_domain.png`, `fig_07_02_rating_eq_freq.png` (not inspected; see the deliverable note).

Each strategy is compared on how evenly it spreads recipes across bins.

## 9.1 Minutes

| Strategy | Quick | Medium | Long |
|---|---|---|---|
| Domain edges (0–30, 30–60, 60+) | 99,053 (42.76%) | 70,077 (30.25%) | 62,507 (26.98%) |
| Equal frequency | 78,159 (33.74%) | 80,392 (34.71%) | 73,086 (31.55%) |

Domain bins are interpretable (quick, medium, long) but unbalanced: quick recipes make up 42.76%. Equal-frequency bins are balanced but have no direct meaning for users. The filter in the kNN example query q3 (`minutes_bin: quick`) uses the domain version.

## 9.2 Calories

| Strategy | Low | Medium | High |
|---|---|---|---|
| Equal width | 231,636 (100.00%) | 0 (0.00%) | 1 (0.00%) |
| Equal frequency | 77,236 (33.34%) | 77,207 (33.33%) | 77,194 (33.33%) |

Equal-width binning fails here. Because of the long tail (Section 8), almost all recipes fall into the lowest bin, and the remaining bins hold a single recipe. Equal frequency is the only usable option for calories.

## 9.3 Average rating

| Strategy | Low | Medium | High |
|---|---|---|---|
| Domain edges (0–3.5, 3.5–4.5, 4.5–5.0) | 14,322 (6.32%) | 64,096 (28.29%) | 148,172 (65.39%) |
| Equal width | 3,153 (1.39%) | 12,517 (5.52%) | 210,920 (93.08%) |

Equal-frequency quantiles for ratings produced only two groups in the output (`q_0`: 78,418 (34.61%), `q_1`: 148,172 (65.39%)). Because many recipes share the same average rating, the quantile edges collapse. This is a known consequence of ties, not a bug in the table, but it means equal-frequency rating bins are not usable as three-class labels.

The domain edges (3.5 and 4.5) give three non-empty classes and are the likely source of the rating labels (to be confirmed against the 1.10 notebook). Equal width is unusable for the same reason as calories: 93% of recipes land in the top bin.

## 9.4 Summary

- Minutes: domain bins for interpretability, with an imbalance of about 43/30/27.
- Calories: equal frequency only.
- Average rating: domain edges 3.5 and 4.5. The class imbalance (65% high) is the baseline problem for the classifiers in Section 12.

Sources: `binning/v1/minutes_comparison.parquet`, `binning/v1/calories_comparison.parquet`, `binning/v1/avg_rating_comparison.parquet`, `binning/v1/manifest.json`.
