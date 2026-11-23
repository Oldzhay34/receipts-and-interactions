# Data Exploration and Descriptive Statistics

All statistics in this section are computed on the full loaded data (231,637 recipes, 1,132,367
interactions <!-- src: outputs/exploration/v1/basic_counts.parquet -->). They describe the data and
motivate the cleaning decisions; none of them is used as a model input, so computing them on all
rows does not leak information into the evaluation.

## Numeric recipe features

{{tab:dqr_continuous}} gives count, missing share, cardinality, quartiles, mean, standard
deviation, skewness and kurtosis of the ten numeric recipe columns.

**Finding.** The typical recipe takes 40 minutes, has 9 steps, 9 ingredients and 313.4 kcal
(medians) <!-- src: outputs/quality/v2/dqr_continuous.parquet -->, but the maxima are orders of
magnitude larger: 2,147,483,647 minutes, 145 steps, 434,360 kcal and a sugar value of 362,729 % of
the daily value <!-- src: outputs/quality/v2/dqr_continuous.parquet -->. Consequently the means are
pulled far above the medians (`minutes`: mean 9,398.55 vs median 40; `calories`: 473.94 vs 313.4)
and the skewness is extreme for every column except the counts: `minutes` 481.3, `sugar_pdv` 402.8,
`carbohydrates_pdv` 371.3, `calories` 215.3, while `n_steps` has 2.03 and `n_ingredients` 0.77
<!-- src: outputs/quality/v2/dqr_continuous.parquet -->. The histograms and box plots on the raw and
on the log1p scale ({{fig:fig_06_01_minutes_raw}} – {{fig:fig_06_20_carbohydrates_pdv_log1p}})
show the same picture: a dense body and a very long right tail, which the log1p scale compresses
(Section {{sec:outliers}} quantifies the skew before and after the treatment).

**Decision.** Means and standard deviations of the raw columns are not used to describe or to
scale the data; quartiles are reported instead, extreme values are analysed separately
(Section {{sec:outliers}}) and skewed columns are log-transformed before scaling (Sections
{{sec:outliers}} and {{sec:scaling}}). **Rationale.** With this skew a handful of recipes
determines the mean, and an unscaled kNN distance would be decided by a single feature.

## Ratings and class imbalance

| Rating | Interactions | Share of all | Share of rated |
|---|---|---|---|
| 0 (no stars) | 60,847 | 5.37 % | – |
| 1 | 12,818 | 1.13 % | 1.20 % |
| 2 | 14,123 | 1.25 % | 1.32 % |
| 3 | 40,855 | 3.61 % | 3.81 % |
| 4 | 187,360 | 16.55 % | 17.49 % |
| 5 | 816,364 | 72.09 % | 76.19 % |

<!-- src: outputs/exploration/v1/rating_counts.parquet (count, share, share_of_rated) -->

**Finding.** Ratings are highly imbalanced ({{fig:fig_06_21_rating_counts}}): 76.19 % of the rated
interactions give 5 stars and only 6.33 % give 1–3 stars
<!-- src: outputs/exploration/v1/rating_counts.parquet (sum of share_of_rated for ratings 1–3) -->.
The value 0 occurs 60,847 times (5.37 %) <!-- src: outputs/exploration/v1/rating_counts.parquet -->;
on Food.com it means that the reviewer gave no stars (Section {{sec:missing}}). **Decision.** Rating
0 is treated as missing, not as the worst score; the rating target is evaluated with macro-F1,
balanced accuracy and the confusion matrix, and a majority-class predictor is reported only as a
baseline. **Rationale.** A model that always predicts "5 stars" would already be right about three
times out of four, so accuracy alone says little about the minority classes that matter for a
recommendation.

## Activity per user and per recipe (cold start)

| Entity | Count | With exactly one review | Median reviews | Mean reviews | Max reviews |
|---|---|---|---|---|---|
| Users | 226,570 | 166,256 (73.38 %) | 1 | 5.00 | 7,671 |
| Recipes | 231,637 | 91,953 (39.70 %) | 2 | 4.89 | 1,613 |

<!-- src: outputs/exploration/v1/per_entity_counts.parquet (n_entities, n_one, pct_one, median, mean, max) -->

**Finding.** Both distributions are long-tailed ({{fig:fig_06_24_user_reviews_long_tail}},
{{fig:fig_06_25_recipe_reviews_long_tail}}): 73.38 % of the users wrote a single review and 39.70 %
of the recipes have a single review, while the most active user wrote 7,671 reviews
<!-- src: outputs/exploration/v1/per_entity_counts.parquet -->. **Decision.** The core recommender
is content-based (ingredient similarity) and personalisation is added on top for users with a
history; the user-level temporal split only evaluates users with at least five interactions
(Section {{sec:sampling}}). **Rationale.** For most users there is no rating history to learn from,
and for many recipes the average rating rests on one review, so ingredient similarity has to carry
the recommendation.

## Ingredients and tags

The most frequent ingredients are pantry staples: salt appears in 37.0 % of the recipes, butter in
23.7 % and sugar in 19.2 % <!-- src: outputs/exploration/v1/top_items.parquet --> (top 20 in
{{fig:fig_06_22_top20_ingredients}}). The top 20 already contain several spellings of the same
ingredient — "eggs" (14.6 %) and "egg" (7.5 %), "garlic cloves" (11.1 %) and "garlic" (7.8 %),
"flour" (11.3 %) and "all-purpose flour" (7.6 %) <!-- src: outputs/exploration/v1/top_items.parquet -->.
Some tags are attached to almost every recipe: "preparation" to 99.5 %, "time-to-make" to 97.3 %
and "course" to 94.2 % <!-- src: outputs/exploration/v1/top_items.parquet -->
({{fig:fig_06_23_top20_tags}}).

**Decision.** Ingredient strings are normalised (case, plural forms, preparation words such as
"chopped") and rare strings are filtered before encoding (Section {{sec:ingredients}}); near-universal
tags are not used as features, and only a fixed list of main course tags is used to group recipes
(Section {{sec:missing}}). **Rationale.** Without normalisation two recipes that both use eggs can
look unrelated, and a feature present in almost every recipe cannot separate recipes.

## Relationships between features

Scatter plots of `n_ingredients` vs `minutes`, `calories` vs average rating and `n_steps` vs
average rating, the Spearman correlation heat map and the average rating by tag
({{fig:fig_06_26_scatter_ingredients_minutes}} – {{fig:fig_06_30_box_rating_by_tag}}) are
discussed with the visualisations in Section {{sec:visualization}}.
