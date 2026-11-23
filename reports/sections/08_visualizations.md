# Visualizations and Distribution Analysis

All figures use one shared style (font, palette, 150 dpi) and are produced by plotting functions
that return a figure without saving it; the figures are then written together as one versioned
set <!-- src: outputs/figures/v1/manifest.json (30 figure files, inputs loaded v1) -->. They are
drawn from the loaded data, before any cleaning, so they show the problems that the later stages
address. Histograms, box plots, bar plots, scatter plots and a heat map are used; kernel density
and Q–Q plots are not.

## Distribution type of every numeric feature

Each numeric feature is classified by a fixed rule: *long-tail* if skewness > 2 and (excess
kurtosis > 7 or 99th-percentile / median > 5); otherwise *right-skewed* if skewness > 0.5,
*left-skewed* if skewness < −0.5, *multimodal* if the histogram has at least two separated peaks,
*uniform* if excess kurtosis < −1, and *normal* otherwise.
<!-- src: src/recsys/analysis/distribution.py (classify_distribution thresholds) -->

| Feature | Distribution type | Skewness | Kurtosis | p99 / median |
|---|---|---|---|---|
| minutes | long-tail | 481.29 | 231,636.80 | 22.60 |
| n_steps | long-tail | 2.03 | 10.69 | 3.33 |
| n_ingredients | right-skewed | 0.77 | 1.18 | 2.22 |
| calories | long-tail | 215.30 | 76,664.54 | 11.22 |
| total_fat_pdv | long-tail | 55.75 | 10,453.60 | 15.10 |
| sugar_pdv | long-tail | 402.83 | 182,232.62 | 45.67 |
| sodium_pdv | long-tail | 80.29 | 12,790.49 | 15.64 |
| protein_pdv | long-tail | 30.57 | 2,302.00 | 10.44 |
| saturated_fat_pdv | long-tail | 18.36 | 1,005.54 | 17.57 |
| carbohydrates_pdv | long-tail | 371.33 | 163,270.81 | 17.11 |

<!-- src: outputs/figures/v1/distribution_summary.parquet -->

**Finding.** Nine of the ten features are long-tail; only `n_ingredients` is merely right-skewed
(skewness 0.77, below the long-tail threshold of 2). The ratio of the 99th percentile to the median
ranges from 2.22 (`n_ingredients`) to 45.67 (`sugar_pdv`)
<!-- src: outputs/figures/v1/distribution_summary.parquet -->. The raw-scale histograms are a single
bar next to zero with an invisible tail (for `calories`, one bin holds almost all 231,637 recipes
while the axis runs to 434,360 kcal <!-- src: outputs/figures/v1/fig_06_07_calories_raw.png -->), while the log1p-scale versions show the full shape
({{fig:fig_06_01_minutes_raw}} – {{fig:fig_06_20_carbohydrates_pdv_log1p}}). **Decision.**
Histograms and box plots are always shown on both scales, and the long-tail features are clipped
and log-transformed before scaling (Sections {{sec:outliers}} and {{sec:scaling}}). **Rationale.**
On the raw scale the plots and any distance computation are dominated by a handful of extreme
recipes; `n_ingredients` is already close to symmetric and is not log-transformed.

## Ratings, ingredients, tags and activity

The bar plot of ratings ({{fig:fig_06_21_rating_counts}}), the 20 most frequent ingredients and tags
({{fig:fig_06_22_top20_ingredients}}, {{fig:fig_06_23_top20_tags}}) and the long-tail plots of
reviews per user and per recipe ({{fig:fig_06_24_user_reviews_long_tail}},
{{fig:fig_06_25_recipe_reviews_long_tail}}) are discussed with the descriptive statistics in
Section {{sec:exploration}}: ratings are concentrated at 5 stars, the most frequent ingredients are
pantry staples written in several spellings, and most users have a single review.

## Relationships between features

**Finding.** The Spearman correlations ({{fig:fig_06_29_corr_heatmap}}) show three groups:

* Recipe size: `minutes`, `n_steps` and `n_ingredients` are moderately related — 0.40 between
  `minutes` and `n_steps`, 0.44 between `n_steps` and `n_ingredients`, 0.36 between `minutes` and
  `n_ingredients` <!-- src: outputs/figures/v1/fig_06_29_corr_heatmap.png -->
  ({{fig:fig_06_26_scatter_ingredients_minutes}}).
* Nutrition: the nutrition columns are strongly related to each other — 0.92 between total and
  saturated fat, 0.83 between calories and total fat, 0.75 between calories and saturated fat,
  0.72 between calories and protein, 0.63 between sugar and carbohydrates
  <!-- src: outputs/figures/v1/fig_06_29_corr_heatmap.png -->.
* Rating: the average rating is practically unrelated to every numeric feature (all correlations
  between −0.04 and 0.03) and only weakly, negatively related to the number of ratings (−0.11)
  <!-- src: outputs/figures/v1/fig_06_29_corr_heatmap.png -->
  ({{fig:fig_06_27_scatter_calories_rating}}, {{fig:fig_06_28_scatter_steps_rating}}).

The average rating also differs little between the ten most frequent tags: every box has its
median close to 5 and an interquartile range from about 4.3 (about 4.2 for main-dish) to 5
<!-- src: outputs/figures/v1/fig_06_30_box_rating_by_tag.png (read from the plot) --> ({{fig:fig_06_30_box_rating_by_tag}}).

**Decision.** Recipe similarity is based on ingredients; the numeric features enter the distance
only with a small explicit weight (Section {{sec:scaling}}), and the rating is not expected to be
predictable from the numeric recipe features alone. **Rationale.** Correlated nutrition columns
count the same information several times in an unweighted distance, and features that do not
correlate with the rating cannot explain it; whether ingredients do is tested by the kNN prototype
(Section {{sec:knn}}).

The scaling figures ({{fig:fig_08_01_calories_raw}} – {{fig:fig_08_05_minutes_log1p_minmax}}) are
discussed in Section {{sec:scaling}}.
