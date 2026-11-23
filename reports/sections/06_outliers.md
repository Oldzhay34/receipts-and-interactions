# Outliers

Ten numeric recipe columns are analysed: `minutes`, `n_steps`, `n_ingredients` and the seven
nutrition columns. `minutes = 0` is treated as missing before the analysis (Section
{{sec:missing}}), so the `minutes` statistics below are based on 230,543 recipes
<!-- src: outputs/outliers/v2/iqr.parquet (minutes: n) -->. All tables come from one analysis
version that also records the configuration used <!-- src: outputs/outliers/v2/manifest.json -->.

## IQR analysis

**Finding.** Tukey's fences (Q1 − 1.5·IQR, Q3 + 1.5·IQR) on the raw scale flag between 1.6 %
(`n_ingredients`) and 10.3 % (`minutes`, `sugar_pdv`) of the recipes as outliers; the upper fence
of `minutes` is 132.5 minutes and that of `calories` 1,037.65 kcal
<!-- src: outputs/outliers/v2/iqr.parquet (scale = raw: share_outside, upper_fence) -->. On the log1p
scale the same rule flags between 0.03 % (`protein_pdv`) and 4.7 % (`minutes`)
<!-- src: outputs/outliers/v2/iqr.parquet (scale = log1p: share_outside) --> ({{tab:outliers_iqr}};
box plots in {{fig:fig_06_01_minutes_raw}} – {{fig:fig_06_20_carbohydrates_pdv_log1p}}).
**Decision.** IQR fences are reported as evidence but are not used to remove or change values.
**Rationale.** On columns this skewed the raw-scale fences mark ordinary recipes — a two-and-a-half
hour roast or a 1,100 kcal cake — as outliers. The question that matters is not "is the value
rare?" but "can the value be true?", which leads to two different treatments.

## Invalid values versus real extremes

* **Invalid (impossible) values** cannot describe a recipe. They become missing values
  (`InvalidValueRule`) and are imputed or excluded later; rows are never dropped.
* **Real but extreme values** are possible but rare. They are kept and only limited in their
  influence (clipping and log transform, below).

The valid ranges are domain rules, not learned from data
<!-- src: outputs/outliers/v2/manifest.json (params.outliers_config.invalid_rule) -->:

| Column | Valid range | Invalid recipes |
|---|---|---|
| `minutes` | 0 – 525,600 (1 year) | 2 (< 0.001 %) |
| `n_steps` | ≥ 1 | 1 |
| `n_ingredients` | ≥ 1 | 0 |
| nutrition (7 columns) | ≥ 0 | 0 |

<!-- src: outputs/outliers/v2/invalid_counts.parquet (lower, upper, n_invalid, share_invalid) -->

### The preparation-time bound

**Finding.** The number of recipes above candidate upper bounds falls from 2,000 above one day to
256 above seven days, 70 above 30 days, 29 above 60 days and 2 above one year
<!-- src: outputs/outliers/v2/minutes_thresholds.parquet --> ({{tab:outliers_minutes_thresholds}}).
The two values above one year are clearly not preparation times: 2,147,483,647 minutes — the
largest 32-bit integer, i.e. a placeholder — for "no bake granola balls", and 1,051,200 minutes
(two years) for the joke recipe "how to preserve a husband"
<!-- src: outputs/outliers/v2/extreme_examples.parquet -->. Of the next eighteen, between 129,610 and
288,000 minutes (about 90 to 200 days), all but one are infusions and fermentations — homemade
liqueurs, mead, peach brandy, vanilla and cinnamon extract, hard apple cider — whose stated time
includes months of steeping and is therefore plausible; the exception is another joke recipe
("recipe for elephant stew") <!-- src: outputs/outliers/v2/extreme_examples.parquet -->
({{tab:outliers_extreme_examples}}).

**Decision.** The valid range of `minutes` is 0 – 525,600 minutes (one year)
<!-- src: outputs/outliers/v2/invalid_counts.parquet (minutes: upper, n_invalid) -->: only the two
impossible values become missing and are imputed; the long infusions are kept and limited by the
99th-percentile clip below, so they end up in the "long" time bin.
**Rationale.** A one-year bound separates the two impossible values from every plausible recipe in
the top of the distribution, whereas any bound between 30 days and one year would also discard
real infusions.

**Revision of an earlier bound.** A first version of this rule used 30 days (43,200 minutes),
chosen on the 5,000-recipe sample, where the longest time was 30,300 minutes (21 days)
<!-- src: outputs/sample/v1/recipes.parquet (max of minutes) -->. On the full data it marked 70
recipes invalid <!-- src: outputs/outliers/v1/invalid_counts.parquet (minutes: n_invalid) -->,
mostly the genuine infusions above, which the imputer then gave the median time of their course
group — between 5 and 70 minutes <!-- src: outputs/missing/v2/imputer_groups.parquet (median) -->
— i.e. the time of an ordinary recipe. The bound was therefore raised to one year and the analysis
re-run; all numbers in this section come from that corrected version
<!-- src: outputs/outliers/v2/manifest.json (params.outliers_config.invalid_rule.minutes) -->.

### Extreme nutrition values

**Finding.** No nutrition value is negative <!-- src: outputs/outliers/v2/invalid_counts.parquet -->,
but the largest are implausible as per-serving values: 434,360 kcal for "tennessee moonshine" and
101,615 kcal for "deep fried prime rib", followed by cakes and rib recipes between about 30,000
and 45,000 kcal <!-- src: outputs/outliers/v2/extreme_examples.parquet (column = calories) -->.
**Decision.** No upper validity bound is set for nutrition; the values are limited by the
percentile clip. **Rationale.** These values are most likely whole-batch nutrition entered as per
serving, but there is no principled threshold between "a very rich dish" and "a whole batch"; the
clip removes their influence without having to decide which ones are errors.

## Clipping at the 99th percentile

**Decision.** Every analysed column is clipped to the [minimum, 99th percentile] range learned on
the training data (`QuantileClipper`, lower quantile 0, upper quantile 0.99)
<!-- src: outputs/outliers/v2/manifest.json (params.outliers_config.quantile_clip) -->. In the
demonstration fit on a 70 % random share of the recipes the upper bounds are 960 minutes, 30 steps,
20 ingredients and 3,551.7 kcal, and between 1,639 (`n_ingredients`) and 2,295 (`protein_pdv`)
recipes per column are clipped <!-- src: outputs/outliers/v2/clip_bounds.parquet -->
({{tab:outliers_clip_bounds}}). The clipper that the models use is refitted on the training split
inside the preprocessing pipeline (Section {{sec:sampling}}). **Rationale.** Clipping keeps every
recipe and its order while preventing a few extreme values from dominating distances and scalers;
the bounds come from training data only, so validation and test recipes are clipped with the same
values and never influence them.

## Log transform

**Finding.** After the invalid-value rule and the clip, every column except `n_ingredients` is still
right-skewed (skewness 1.20 for `n_steps`, 3.64 for `calories`, 4.76 for `minutes`, 4.83 for
`sugar_pdv`); after log1p the skewness is between −0.56 and 0.31. `n_ingredients` has a skewness of
0.57 after clipping <!-- src: outputs/outliers/v2/before_after.parquet (skew by stage) -->:

| Column | Skewness raw | After rules + clip | After log1p |
|---|---|---|---|
| `minutes` | 480.15 | 4.76 | 0.31 |
| `n_steps` | 2.03 | 1.20 | −0.26 |
| `n_ingredients` | 0.77 | 0.57 | (not applied) |
| `calories` | 215.30 | 3.64 | −0.56 |
| `sugar_pdv` | 402.83 | 4.83 | −0.03 |
| `sodium_pdv` | 80.29 | 3.05 | −0.34 |

<!-- src: outputs/outliers/v2/before_after.parquet (skew; stages before, after_rules, after_log1p) -->

**Decision.** `log1p` is applied to `minutes`, `n_steps` and the seven nutrition columns, but not to
`n_ingredients` <!-- src: outputs/outliers/v2/manifest.json (params.outliers_config.columns) -->
({{tab:outliers_before_after}}). **Rationale.** The rule is "still skewed after clipping (skewness
> 1)"; `n_ingredients` is already close to symmetric, and taking its log would only distort a
count that is meaningful on its own scale.

## Order of the steps

In the preprocessing pipeline the outlier steps run in this order: invalid-value rule → imputation
of the resulting missing values → 99th-percentile clip → binning (on the original, clipped scale) →
log1p → scaling. Impossible values are therefore removed before anything is learned from the data,
and bins are defined on interpretable units (minutes, kcal) rather than on the log scale.
