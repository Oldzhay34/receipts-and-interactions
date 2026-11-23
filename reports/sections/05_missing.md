# Missing Values

Missing values appear in two forms in this dataset: real nulls and values that look valid but
encode "unknown" — a rating of 0 and a preparation time of 0 minutes. Both forms are counted in
{{tab:missing_rates}} (`kind` = `null`, `empty_list` or `zero`).

## Overview

| Table.column | Kind | Rows affected | Share |
|---|---|---|---|
| interactions.rating | zero | 60,847 | 5.37 % |
| recipes.description | null | 4,979 | 2.15 % |
| recipes.minutes | zero | 1,094 | 0.47 % |
| interactions.review | null | 165 | 0.015 % |
| recipes.name | null | 1 | < 0.001 % |
| recipes.steps | empty list | 1 | < 0.001 % |

<!-- src: outputs/missing/v2/missing_rates.parquet (n_missing, share) -->

Every other column has no missing value <!-- src: outputs/missing/v2/missing_rates.parquet -->.
The same counts appear as issues in the data quality report (`rating_zero`, `missing_description`,
`minutes_zero`, `missing_review`, `missing_name`) <!-- src: outputs/quality/v2/issues.parquet -->
({{tab:issues}}).

## Rating 0: "no stars given"

**Finding.** All 60,847 interactions with rating 0 contain a written review (share 1.0), exactly
like the rated interactions (share 0.99984 with a review)
<!-- src: outputs/missing/v2/rating_zero_review_stats.parquet -->. Sampled reviews are ordinary
comments, for example "Awesome. I added 2 stalks of celery and a red bell pepper"; some say
explicitly that no rating was given ("I did not give this any stars since I messed it up")
<!-- src: outputs/missing/v2/rating_zero_examples.parquet -->. **Decision.** Rating 0 is converted
to a missing value (`RatingZeroToNaN`); the interaction itself is kept. Recipes whose only ratings
are 0 — 5,047 recipes <!-- src: outputs/quality/v2/issues.parquet (recipes_without_rating) --> —
stay in the recommendation pool but have no rating target (they do not appear in the target
tables), and the split gives them the stratification class `unrated` (Section {{sec:sampling}}).
**Rationale.** A review without stars is not a one-star review; counting it as the lowest score
would bias every recipe average downwards and invent a "low" class that the reviewer never
expressed.

## Preparation time 0: missing, not "no cooking"

`minutes` is the total preparation time. A value of 0 could mean a genuine no-cook recipe or a
missing value.

**Finding.** The evidence for the 1,094 zero-minute recipes
<!-- src: outputs/missing/v2/minutes_zero_summary.parquet -->:

| Evidence | minutes = 0 | All recipes |
|---|---|---|
| Tagged `no-cook` or `beverages` | 3.84 % (42 recipes) | 7.03 % |
| Median number of steps | 9 | 9 |
| Median number of ingredients | 8 | 9 |
| Tagged `oven` | 34.6 % | 13.5 % (lift 2.57) |

<!-- src: outputs/missing/v2/minutes_zero_summary.parquet (share_zero_with_no_cook_tag, share_all_with_no_cook_tag, median_n_steps_*, median_n_ingredients_*) -->
<!-- src: outputs/missing/v2/minutes_zero_tags.parquet (oven: share_in_zero, share_in_all, lift) -->

Zero-minute recipes are *less* often no-cook or beverage recipes than the catalogue, have as many
steps and nearly as many ingredients, and are 2.57 times over-represented among oven recipes
<!-- src: outputs/missing/v2/minutes_zero_tags.parquet -->. All of them carry the tag
"15-minutes-or-less" (lift 5.27) <!-- src: outputs/missing/v2/minutes_zero_tags.parquet -->, but
Food.com derives that tag from `minutes` itself, so it is a consequence of the 0, not evidence for
a short recipe. Examples include "hamburgers to freeze" and "pacific rim glazed flank steak"
(17 steps) <!-- src: outputs/missing/v2/minutes_zero_examples.parquet --> ({{tab:minutes_zero_tags}},
{{tab:minutes_zero_n_steps}}).

**Decision.** Every `minutes = 0` is a missing value (no tag-based exception). It is imputed by
`GroupMedianImputer`: the median `minutes` of recipes with the same main course tag and a similar
number of steps; a missing-value indicator `minutes_was_missing` is added. **Rationale.** The zeros
behave like ordinary cooked recipes, so a 0 would make them look like the fastest recipes in the
catalogue and put them in the "quick" bin; the group median uses the two properties that explain
preparation time best among the available columns (course type and recipe length).

### Imputation groups

The main tag is the first course tag, in a fixed priority order of eleven tags (beverages,
desserts, breakfast, …, main-dish), that occurs in the recipe's tags, or `other` if none does
<!-- src: outputs/missing/v2/manifest.json (params.main_tags) -->; coverage ranges from main-dish
(26.2 % of the recipes) and desserts (18.5 %) to snacks (1.0 %), with 9.5 % `other`
<!-- src: outputs/missing/v2/main_tag_coverage.parquet --> ({{tab:main_tag_coverage}}). The number of
steps is cut into four bins at quartiles learned from the fit data. In the demonstration fit (70 %
random share of the recipes) all 48 groups (12 main tags × 4 step bins) have at least 298 observed
recipes, far above the minimum group size of 5, so the global-median fallback is never needed; the
group medians range from 5 minutes (beverages, fewest steps) to 70 minutes (breads, most steps)
<!-- src: outputs/missing/v2/imputer_groups.parquet --> ({{tab:imputer_groups}}). The imputer that
is actually used is refitted inside the preprocessing pipeline on the training split only
(Section {{sec:sampling}}).

## Text fields and the remaining cases

**Finding.** 4,979 recipes have no description, 165 reviews are empty and 1 recipe has no name
<!-- src: outputs/missing/v2/missing_rates.parquet -->. **Decision.** These rows are kept and the
fields stay missing. **Rationale.** Descriptions, names and review texts are not model features;
the recipe id identifies the recipe, and the star rating of an interaction without review text is
still valid.

One recipe has an empty step list <!-- src: outputs/missing/v2/missing_rates.parquet (steps empty_list) -->
and one recipe (id 176767) has `n_steps = 0` <!-- src: outputs/quality/v2/issues.parquet (n_steps_zero) -->;
a recipe without steps is treated as an invalid value in Section {{sec:outliers}}.
