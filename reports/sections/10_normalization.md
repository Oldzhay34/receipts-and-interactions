# 10. Normalization and Scaling

Source: `scaling/v1` (input `loaded/v1`; `params`: default_method minmax, use_log1p true, numeric_weight 0.2, n_numeric_cols 10). Tables: `scaler_summary.parquet`, `combined_distance_sample.parquet`.

## 10.1 Transform choice

Section 8 showed that nine of ten numeric features are long-tail. The scaling stage therefore applies log1p before min-max scaling (default) and compares it with z-score on log1p values. Raw values are kept only for reference.

## 10.2 Effect of the transforms

Selected rows from `scaler_summary.parquet`:

| Feature | Transform | Min | Median | Max | Std |
|---|---|---|---|---|---|
| calories | raw | 0.0 | 313.4 | 434,360.2 | 1,189.71 |
| calories | log1p_minmax | 0.0 | 0.4430 | 1.0 | 0.0742 |
| calories | log1p_zscore | −5.9091 | 0.0598 | 7.5652 | 1.0 |
| sugar_pdv | raw | 0.0 | 25.0 | 362,729.0 | 800.08 |
| sugar_pdv | log1p_minmax | 0.0 | 0.2545 | 1.0 | 0.1169 |
| minutes | log1p_minmax | 0.0 | 0.1728 | 1.0 | 0.0507 |
| n_ingredients | log1p_minmax | 0.0 | 0.5207 | 1.0 | 0.1242 |

For calories, the raw maximum (434,360) is about 1,000 times the median (313). After log1p and min-max scaling, the median is 0.443 and the max is 1.0, so the bulk of recipes sits in a usable range. The same holds for sugar and minutes.

Min-max keeps values in [0, 1]. The z-score variant has no fixed range, but its maximum is still a useful check on outliers (for example, calories z = 7.57).

## 10.3 Combined distance

The scaling stage combines numeric and ingredient distances with `numeric_weight` 0.2. The sample of the combined distance matrix (5 recipes, `combined_distance_sample.parquet`) is:

| | recipe_0 | recipe_1 | recipe_2 | recipe_3 | recipe_4 |
|---|---|---|---|---|---|
| recipe_0 | 0.0000 | 0.8122 | 0.7838 | 0.7231 | 0.7616 |
| recipe_1 | 0.8122 | 0.0000 | 0.8273 | 0.8213 | 0.8149 |
| recipe_2 | 0.7838 | 0.8273 | 0.0000 | 0.7737 | 0.7768 |
| recipe_3 | 0.7231 | 0.8213 | 0.7737 | 0.0000 | 0.7191 |
| recipe_4 | 0.7616 | 0.8149 | 0.7768 | 0.7191 | 0.0000 |

This is a five-recipe sample, not a full evaluation. It shows the matrix is symmetric with zero diagonal. The values lie between about 0.72 and 0.83, so in this sample the pairs are fairly similar overall.

Sources: `scaling/v1/scaler_summary.parquet`, `scaling/v1/combined_distance_sample.parquet`, `scaling/v1/manifest.json`.
