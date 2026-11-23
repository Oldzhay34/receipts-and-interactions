# Limitations and Next Steps

Phase 1 delivers a versioned, validated data pipeline, two leakage-free splits, a preprocessing
pipeline fitted on training data only and a first kNN prototype. This section lists what Phase 1
leaves open and how the following phases address it.

## Open points from Phase 1

* **Rating prediction is at baseline level.** On the validation set the kNN classifiers reach a
  macro-F1 of at most 0.2981 against 0.2630 for the majority baseline, and neither kNN regressor
  beats the mean rating (MAE 0.4620) <!-- src: outputs/knn_prototype/v4/classification_metrics.parquet, regression_metrics.parquet -->
  (Section {{sec:knn}}). k, the distance metric and the feature weights are still untuned.
* **Jaccard favours very small recipes.** Because Jaccard divides by the union, a recipe with one
  or two ingredients can rank first on a single shared ingredient (Section {{sec:knn}}).
* **Query terms must match the vocabulary exactly.** "mozzarella" is not recognised because the
  catalogue uses "mozzarella cheese" <!-- src: outputs/knn_prototype/v4/example_recommendations.parquet (unknown_ingredients) -->;
  related entries such as "chicken" and "chicken breast" also stay separate (Section
  {{sec:ingredients}}).
* **Ratings shown by the prototype come from training data only.** This is correct for evaluation,
  but a deployed recommender would show estimated ratings for the 67,978 validation and test
  recipes that do have real ratings <!-- src: outputs/splits/v5/manifest.json (files.targets_val.n_rows + files.targets_test.n_rows) -->.
* **Cold start dominates.** 73.38 % of the users have a single review
  <!-- src: outputs/exploration/v1/per_entity_counts.parquet (user: pct_one) -->; only 23,086 users
  have enough history for the per-user temporal evaluation
  <!-- src: outputs/splits/v5/split_summary.parquet (temporal_user, test: n_users) -->.

## Plan for the following phases

* **Phase 2 — modelling.** Grid search on the validation set over k, the distance (Jaccard, cosine,
  TF-IDF cosine, the weighted combined distance), uniform versus distance-weighted votes, the
  numeric-feature weight, preparation time as a number or as a bin, and the class-imbalance
  strategy; every run is stored as its own versioned artefact. The best configuration is evaluated
  once on the test set and becomes the recommender's default. Remedies for the small-recipe bias
  (a minimum recipe size or a containment-based similarity) are part of this search.
* **Phase 3 — scoring and personalisation.** A ranking score that combines ingredient similarity
  with a Bayesian-averaged rating (so that recipes with one review are not over-trusted), user
  profiles built only from interactions before the evaluated one, a personalised recommender that
  wraps the Phase 2 model, popularity as the cold-start fallback, and Precision@k, Recall@k and
  NDCG@k on the temporal split.
* **Phase 4 — backend.** A FastAPI service with a relational database (users, ratings, favourites,
  profiles), password hashing and transactional rating updates; it loads the recommender artefact,
  refitted with all interactions so that known ratings are shown as known, and offers ingredient
  auto-completion from the vocabulary, which removes most unrecognised query terms.
* **Phase 5 — mobile application.** A Flutter client following Apple's Human Interface
  Guidelines: sign-in, ingredient entry with auto-completion and a preparation-time filter,
  results with matched / missing ingredients and an "estimated" badge for predicted ratings,
  rating, favourites, history and account deletion.
* **Phase 6 — integration.** End-to-end tests from registration to personalised recommendations,
  an API contract test, the final report with the test-set results, and a demonstration.
