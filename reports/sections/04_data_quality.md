# 4. Data Quality

Source: `quality/v1` (input stage `loaded/v1`, manifest sha256 `c204225f…d60a7`). Tables: `dqr_continuous.parquet`, `dqr_categorical.parquet`, `issues.parquet`; cold-start figures from `figures/v1/cold_start_summary.parquet`.

## 4.1 Continuous features

All 10 continuous features have 231,637 non-null values (0.0% missing).

| Feature | Median | Q3 | Max | Skewness | Kurtosis |
|---|---|---|---|---|---|
| minutes | 40 | 65 | 2,147,484,000 | 481.29 | 231,636.80 |
| n_steps | 9 | 12 | 145 | 2.03 | 10.69 |
| n_ingredients | 9 | 11 | 43 | 0.77 | 1.18 |
| calories | 313.4 | 519.7 | 434,360.2 | 215.30 | 76,664.54 |
| sugar_pdv | 25 | 68 | 362,729 | 402.83 | 182,232.62 |
| carbohydrates_pdv | 9 | 16 | 36,098 | 371.33 | 163,270.81 |

The other continuous features (total_fat_pdv, sodium_pdv, protein_pdv, saturated_fat_pdv) follow the same pattern: 0% missing, strongly right-skewed, extreme maxima. The skewness and kurtosis values are the main reason the later stages apply log1p transforms (see Section 10).

## 4.2 Categorical feature

`rating` (Count 1,132,367; 0.0% missing; cardinality 6). The mode is 5 with 816,364 occurrences (72.09%). The second mode is 4 with 187,360 occurrences (16.55%). Rating is heavily concentrated at the top of the scale.

## 4.3 Rule-based issues

Eight rule checks were run. The table below reports the counts from `issues.parquet`.

| Issue | Rule | Affected rows | Suggested action |
|---|---|---|---|
| rating_zero | interactions.rating == 0 | 60,847 | Convert 0 to NaN in Stage 1.3; exclude from rating averages |
| minutes_zero | recipes.minutes == 0 | 1,094 | Investigate no-cook vs missing; impute via group median if missing |
| minutes_extreme | minutes > 10,080 (over 7 days) | 256 | Treat as invalid/extreme outlier in Stage 1.4 |
| n_steps_zero | n_steps == 0 | 1 | Flag as invalid in Stage 1.4 |
| missing_text_fields | NaN in name, description or review | 5,145 | Handle per column in Stage 1.3 |
| duplicate_interactions | Exact duplicate interaction rows | 0 | Drop in Stage 1.3 (none found) |
| recipes_without_rating | No non-zero rating in interactions | 5,047 | Mark has_rating=False; exclude from training, keep in recommendation pool |
| missing_recipe_name | recipes.name is null | 1 | Drop in Stage 1.3 |

Note: the `issues` table lists 8 rules; the `duplicate_interactions` rule finds zero rows, so no duplicates are dropped.

## 4.4 Cold-start profile

From `figures/v1/cold_start_summary.parquet`:
- Users: 226,570 unique; 166,256 (73.38%) have exactly one review; median reviews per user 1; maximum 7,671.
- Recipes: 231,637 unique; 91,953 (39.70%) have exactly one review; median reviews per recipe 2; maximum 1,613.

Most users and about 40% of recipes have very little interaction history. This is the main reason the kNN prototype (Section 12) is content-based (ingredients only).

Sources: `quality/v1/dqr_continuous.parquet`, `quality/v1/dqr_categorical.parquet`, `quality/v1/issues.parquet`, `figures/v1/cold_start_summary.parquet`.
