# Preliminary kNN Prototype

The prototype checks two things before the modelling phase: whether ingredient-based kNN returns
sensible recipes for a free-text query, and how well a recipe's rating class can be predicted from
the ratings of its most similar training recipes. It uses the fitted preprocessing pipeline and the
grouped split of Section {{sec:sampling}}; all results come from one prototype version
<!-- src: outputs/knn_prototype/v4/manifest.json (inputs splits v5, loaded v1) -->. Only validation
results are reported: the test split is not evaluated here and stays reserved for the single final
evaluation; the evaluation code refuses the test split.

## Setup

* Distance: Jaccard on the binary ingredient vectors; the metric is passed to the models as a
  parameter, so cosine distance can be used instead. Euclidean distance is not used for
  ingredients.
* Rating models: majority-class baseline, kNN with uniform votes and kNN with distance-weighted
  votes (weight 1 / (d + 10⁻⁶)), k = 15 neighbours <!-- src: outputs/knn_prototype/v4/manifest.json (params.knn_prototype) -->;
  for regression the mean training rating is the baseline.
* Data: the recipes of the development sample — 20,551 rated training recipes as the reference
  set and 4,418 rated validation recipes <!-- src: outputs/knn_prototype/v4/manifest.json (params.n_train_recipes, n_val_recipes) -->.
  All models are fitted on the training recipes only.
* Recommender: indexes the whole catalogue, including recipes without ratings. Known ratings are
  training averages; for any other recipe the rating is estimated from its 10 most similar rated
  recipes (distance-weighted) and flagged as estimated
  <!-- src: outputs/knn_prototype/v4/manifest.json (params.knn_prototype.rating_neighbors) -->.

## Example queries

Three queries were run with k = 5; the third also filters on preparation time
<!-- src: outputs/knn_prototype/v4/manifest.json (params.knn_prototype.example_queries, top_k) -->
({{tab:example_recommendations}}):

| Query | Top results (similarity) |
|---|---|
| chicken, garlic, lemon | "the miracle 1 minute garlic peeling trick" (0.67), "clay oven roasted chicken" (0.50), "forty clove garlic chicken" (0.43) |
| flour, sugar, butter, eggs | "ann s shortbread", "buttery pound cake", "mrs corbins butter cookies" (all 1.00) |
| tomatoes, basil, mozzarella, olive oil — quick only | "tasty bbq tomatoes" (0.75), "st bart s no fuss tomato mango salad" (0.75), "basil oil" (0.67) |

<!-- src: outputs/knn_prototype/v4/example_recommendations.parquet (query, rank, name, similarity) -->

**Finding.** The results share most of the query ingredients, and the filter works: all five
results of the third query are "quick" recipes (5 to 25 minutes). Two of the fifteen shown
results have an estimated rating in the first query and one in the second
<!-- src: outputs/knn_prototype/v4/example_recommendations.parquet (minutes, minutes_bin, rating_is_estimated) -->.
Two weaknesses are visible. Jaccard similarity divides by the size of the union, so a recipe with
very few ingredients can rank first on a single shared ingredient — a garlic-peeling "trick" tops
the chicken query. And "mozzarella" is reported as an unrecognised ingredient
<!-- src: outputs/knn_prototype/v4/example_recommendations.parquet (unknown_ingredients) -->, because
the catalogue uses "mozzarella cheese" (Section {{sec:ingredients}}). **Decision.** Both are
recorded for the modelling phase: a minimum number of recipe ingredients or a similarity that does
not penalise the query side (e.g. containment), and matching query terms to vocabulary entries
that contain them. **Rationale.** Neither problem shows in the validation metrics below, but both
directly affect what a user sees.

## Rating class on the validation set

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|
| majority | 0.6517 | 0.3333 | 0.2630 |
| kNN, uniform | 0.6335 | 0.3356 | 0.2917 |
| kNN, distance-weighted | 0.6292 | 0.3367 | 0.2981 |

<!-- src: outputs/knn_prototype/v4/classification_metrics.parquet (split = val) -->

**Finding.** The majority baseline has the highest accuracy (0.6517) because about two thirds of
the validation recipes are "high"; its balanced accuracy is exactly 1/3 and its macro-F1 0.2630
<!-- src: outputs/knn_prototype/v4/classification_metrics.parquet --> ({{tab:classification_metrics}}).
Both kNN variants are slightly better on macro-F1 (0.2917 and 0.2981) and balanced accuracy, but
barely above chance. The confusion matrices show why
<!-- src: outputs/knn_prototype/v4/confusion_matrices.parquet --> ({{tab:confusion_matrices}}): of the
284 "low" validation recipes, the distance-weighted kNN predicts 260 as "high", 24 as "medium" and
none as "low"; of the 1,255 "medium" recipes it puts 99 into "medium" and 1,155 into "high".

**Interpretation.** Accuracy is misleading here: a model that never predicts a minority class
scores best on it. On the metrics that weight the classes equally, ingredient similarity alone
carries very little information about a recipe's rating — consistent with the near-zero
correlations between the rating and the recipe features (Section {{sec:visualization}}).

## Average rating on the validation set

| Model | MAE | RMSE |
|---|---|---|
| mean of training ratings | 0.4620 | 0.6431 |
| kNN, uniform | 0.4658 | 0.6532 |
| kNN, distance-weighted | 0.4654 | 0.6533 |

<!-- src: outputs/knn_prototype/v4/regression_metrics.parquet (split = val) -->

**Finding.** Neither kNN variant beats the mean of the training ratings (MAE 0.4620, RMSE 0.6431)
<!-- src: outputs/knn_prototype/v4/regression_metrics.parquet --> ({{tab:regression_metrics}}).

## Conclusion

**Decision.** In this form the prototype is a retrieval tool, not a rating predictor: it finds
recipes that match the user's ingredients, while its rating estimates should be presented as
indicative. **Rationale.** The validation results do not beat simple baselines, k and the metric
are untuned, and no user history is used yet. The modelling phase tunes k, the metric, the weight
of the numeric features and the class weights on the validation set, and adds personalisation from
the user's own ratings; the test set is used once at the end.
