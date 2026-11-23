# Binning

Three numeric quantities are turned into ordered classes: preparation time (`minutes_bin`, used as
a filter in the application), calories (`calories_bin`) and the average rating (`rating_class`,
the classification target). Three strategies are compared, all with the same interface and with
their bin edges learned on the training data only: domain bins with fixed, meaningful edges,
equal-width bins and equal-frequency (quantile) bins. The comparison tables come from one analysis
version on the loaded data <!-- src: outputs/binning/v2/manifest.json -->. Values outside the
training range fall into the outermost bin.

## Preparation time

| Strategy | quick | medium | long |
|---|---|---|---|
| Domain (< 30, 30–59, ≥ 60 minutes) | 80,662 (34.82 %) | 78,166 (33.75 %) | 72,809 (31.43 %) |
| Equal frequency | 78,159 (33.74 %) | 80,392 (34.71 %) | 73,086 (31.55 %) |

<!-- src: outputs/binning/v2/minutes_comparison.parquet (count, pct) -->

**Finding.** The domain bins are almost as balanced as the equal-frequency bins (34.82 / 33.75 /
31.43 % against 33.74 / 34.71 / 31.55 %) <!-- src: outputs/binning/v2/minutes_comparison.parquet -->
({{tab:minutes_comparison}}). **Decision.** Domain bins with edges 30 and 60 minutes; a recipe of
exactly 30 minutes is "medium" and one of exactly 60 minutes is "long"
<!-- src: config/config.yaml (binning.minutes) -->. **Rationale.** The domain edges mean something to a
user ("quick" = under half an hour) at almost no cost in balance, whereas equal-frequency edges
would move whenever the training data changes. In the preprocessing pipeline the bins are computed
after invalid values are imputed and extremes clipped, so a recipe with a missing preparation time
is binned by its imputed value (Section {{sec:outliers}}).

## Calories

| Strategy | low | medium | high |
|---|---|---|---|
| Equal width | 231,636 (100.00 %) | 0 (0.00 %) | 1 (0.00 %) |
| Equal frequency | 77,236 (33.34 %) | 77,207 (33.33 %) | 77,194 (33.33 %) |

<!-- src: outputs/binning/v2/calories_comparison.parquet (count, pct) -->

**Finding.** Equal-width bins fail completely: because the range runs up to 434,360 kcal, all
recipes but one fall into the lowest bin <!-- src: outputs/binning/v2/calories_comparison.parquet -->
<!-- src: outputs/quality/v2/dqr_continuous.parquet (calories Max) --> ({{tab:calories_comparison}}).
**Decision.** Equal-frequency bins (terciles learned on the training recipes). **Rationale.** There
is no natural "low / high calorie" threshold for whole recipes, and the long tail (Section
{{sec:visualization}}) makes any equal-width split meaningless.

## Average rating (classification target)

| Strategy | low | medium | high |
|---|---|---|---|
| Domain (≤ 3.5, 3.5–4.5, > 4.5) | 14,322 (6.32 %) | 64,096 (28.29 %) | 148,172 (65.39 %) |
| Equal width | 3,153 (1.39 %) | 12,517 (5.52 %) | 210,920 (93.08 %) |
| Equal frequency | 78,418 (34.61 %) in one bin | – | 148,172 (65.39 %) |

<!-- src: outputs/binning/v2/avg_rating_comparison.parquet (count, pct) -->

**Finding.** Equal-width bins put 93.08 % of the rated recipes into the top class. Equal-frequency
binning cannot produce three classes at all: so many recipes share the maximum average of 5.0
that the two upper quantile edges coincide, and only two bins remain (34.61 % and 65.39 %)
<!-- src: outputs/binning/v2/avg_rating_comparison.parquet --> ({{tab:avg_rating_comparison}};
{{fig:fig_07_01_rating_domain}}, {{fig:fig_07_02_rating_eq_freq}}). **Decision.** Domain edges 3.5
and 4.5 define `rating_class` <!-- src: config/config.yaml (binning.avg_rating) -->; this is the
target used by the split stratification and by the models (Sections {{sec:sampling}} and
{{sec:knn}}). **Rationale.** Of the three strategies it is the only one that yields three
non-empty, interpretable classes; the remaining imbalance (65 % "high") is handled by class
weights and imbalance-aware metrics rather than by moving the edges (Section {{sec:sampling}}).
