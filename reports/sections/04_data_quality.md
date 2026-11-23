# Data Quality Report

The data quality report (DQR) profiles every column of the loaded data and runs a fixed set of
rule checks. Each rule records how many rows it affects, example ids and the action that the
cleaning stages take. All tables in this section come from one report version
<!-- src: outputs/quality/v2/manifest.json -->.

## Continuous features

**Finding.** All ten numeric recipe columns are complete: 231,637 values each and 0 % missing
<!-- src: outputs/quality/v2/dqr_continuous.parquet (Count, % Missing) --> ({{tab:dqr_continuous}}).
Their medians are moderate, but the maxima, skewness and kurtosis are extreme:

| Feature | Median | Q3 | Max | Skewness | Kurtosis |
|---|---|---|---|---|---|
| minutes | 40 | 65 | 2,147,483,647 | 481.29 | 231,636.80 |
| n_steps | 9 | 12 | 145 | 2.03 | 10.69 |
| n_ingredients | 9 | 11 | 43 | 0.77 | 1.18 |
| calories | 313.4 | 519.7 | 434,360.2 | 215.30 | 76,664.54 |
| total_fat_pdv | 20 | 41 | 17,183 | 55.75 | 10,453.60 |
| sugar_pdv | 25 | 68 | 362,729 | 402.83 | 182,232.62 |
| sodium_pdv | 14 | 33 | 29,338 | 80.29 | 12,790.49 |
| protein_pdv | 18 | 51 | 6,552 | 30.57 | 2,302.00 |
| saturated_fat_pdv | 23 | 52 | 10,395 | 18.36 | 1,005.54 |
| carbohydrates_pdv | 9 | 16 | 36,098 | 371.33 | 163,270.81 |

<!-- src: outputs/quality/v2/dqr_continuous.parquet (Median, Q3, Max, Skewness, Kurtosis) -->

**Decision.** Zero values of `minutes` and `n_steps` and the extreme maxima are handled by
explicit rules (Sections {{sec:missing}} and {{sec:outliers}}); skewed columns are clipped and
log-transformed before scaling (Sections {{sec:outliers}} and {{sec:scaling}}). **Rationale.** "No
missing values" does not mean "no data problems": the minimum of `minutes` is 0 and its maximum is
the largest 32-bit integer <!-- src: outputs/quality/v2/dqr_continuous.parquet (Min, Max) -->, and
with this skewness a few recipes would dominate any mean-based statistic or unscaled distance.

## Categorical and list features

`rating` has 1,132,367 values, 0 % missing and 6 distinct values; the mode is 5 with 816,364
occurrences (72.09 %) and the second mode is 4 with 187,360 (16.55 %)
<!-- src: outputs/quality/v2/dqr_categorical.parquet --> ({{tab:dqr_categorical}}). The ingredient
lists hold 14,942 distinct strings (1 to 43 per recipe, mean 9.05) and the tag lists 552 distinct
tags (1 to 73 per recipe, mean 17.88) <!-- src: outputs/quality/v2/dqr_lists.parquet -->
({{tab:dqr_lists}}). The rating distribution and the ingredient vocabulary are analysed further in
Sections {{sec:exploration}} and {{sec:ingredients}}.

## Rule-based issues

Nine rules were checked <!-- src: outputs/quality/v2/issues.parquet --> ({{tab:issues}}):

| Issue | Rule | Affected rows | Handling |
|---|---|---|---|
| rating_zero | interactions.rating == 0 | 60,847 | Rating becomes missing ("no stars given"); the interaction is kept (Section {{sec:missing}}) |
| minutes_zero | recipes.minutes == 0 | 1,094 | Treated as missing and imputed by course-group median (Section {{sec:missing}}) |
| minutes_extreme | minutes > 10,080 (over 7 days) | 256 | Only values above one year are invalid; the rest are kept and clipped (Section {{sec:outliers}}) |
| n_steps_zero | n_steps == 0 | 1 | Invalid value, imputed (Section {{sec:outliers}}) |
| missing_name | recipes.name is null | 1 | Recipe kept; the id identifies it |
| missing_description | recipes.description is null | 4,979 | Recipe kept; description is not a model feature |
| missing_review | interactions.review is null | 165 | Interaction kept; its rating is still valid |
| duplicate_interactions | exact duplicate interaction rows | 0 | Nothing to remove |
| recipes_without_rating | no rating above 0 | 5,047 | Kept in the recommendation pool, no rating target (Section {{sec:sampling}}) |

<!-- src: outputs/quality/v2/issues.parquet (issue_id, rule, affected_rows) -->

**Finding.** The dataset has no structural defects — no duplicate interactions, and only one
recipe without a name — but two columns use 0 as a hidden "unknown" code (60,847 ratings and 1,094
preparation times) and 256 preparation times exceed one week
<!-- src: outputs/quality/v2/issues.parquet -->. **Decision.** No row is deleted at this stage; every
issue is routed to the stage that can decide it with more evidence. **Rationale.** A rule can
flag a value, but whether a 0 means "missing" or "really zero", and whether a 200-day preparation
time is an error or a liqueur, needs the targeted analyses of Sections {{sec:missing}} and
{{sec:outliers}}.
