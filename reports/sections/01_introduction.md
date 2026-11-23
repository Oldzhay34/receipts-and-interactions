# Introduction and Problem Definition

## Motivation

People usually decide what to cook from what is already in the kitchen. Recipe sites are organised
the other way round: one searches by dish name or category and then checks whether the
ingredients are available. This project builds a recommender that starts from the ingredients:
the user enters the ingredients at hand and receives the most similar recipes, together with how
well each recipe is expected to be liked.

The data is the Food.com dataset (recipes and user reviews). It contains 231,637 recipes
<!-- src: outputs/exploration/v1/basic_counts.parquet --> and 1,132,367 user interactions
(review + star rating) written by 226,570 users between 2000-01-25 and 2018-12-20
<!-- src: outputs/exploration/v1/basic_counts.parquet -->.

## Problem definition

The project has one ranking task and two prediction targets:

* **Ranking (recommendation).** Given a free-text list of ingredients, return the *k* catalogue
  recipes whose ingredient sets are most similar to it. Every catalogue recipe can be recommended,
  including recipes that have never been rated.
* **Rating regression.** Predict a recipe's average star rating `avg_rating` (1–5, mean of its
  ratings greater than 0).
* **Rating classification.** Predict the recipe's rating class `rating_class` = `low` (≤ 3.5),
  `medium` (3.5, 4.5] or `high` (> 4.5) <!-- src: config/config.yaml (binning.avg_rating) -->.
  The class is shown to the user when a recipe has too few ratings and is the main evaluation
  target, because ratings are strongly concentrated at 5 stars (Section {{sec:exploration}}).

Both targets are defined per recipe, not per review: the recommender ranks recipes, so the
quantity of interest is how a recipe is rated on average.

## Approach

**Finding.** Recipes are described mainly by their ingredient lists; interactions are sparse — most
users and many recipes have a single review (Section {{sec:exploration}}). **Decision.** A
content-based k-nearest-neighbours (kNN) approach on ingredient vectors, followed by distance-weighted
kNN for rating estimation and, in a later phase, personalisation from the user's own ratings.
**Rationale.** Content-based similarity needs no rating history, so it works for new users and for
recipes nobody has reviewed yet (the cold-start case that dominates this dataset), and kNN keeps
every recommendation explainable ("shares garlic, lemon and chicken with your list").

The pipeline is organised in stages; each stage reads the versioned output of the previous ones
and writes its own version with a manifest (checksums, row counts, schema, code commit, inputs):

1. loading and parsing of the two raw files (Section {{sec:dataset}}),
2. exploration and data quality (Sections {{sec:exploration}} and {{sec:quality}}),
3. missing values and outliers (Sections {{sec:missing}} and {{sec:outliers}}),
4. ingredient normalisation and encoding, visualisation, binning and scaling (Sections
   {{sec:ingredients}} – {{sec:scaling}}),
5. sampling and splitting, with one preprocessing pipeline fitted on training data only
   (Section {{sec:sampling}}),
6. a first kNN prototype (Section {{sec:knn}}).

Methodological rules that hold throughout the report:

* Ingredient similarity uses Jaccard or cosine distance on binary ingredient vectors, never
  Euclidean distance, and the binary vectors are not rescaled.
* Everything that is learned from data — imputation values, clipping bounds, bin edges, scalers,
  recipe averages, user profiles — is learned on the training split only and then applied
  unchanged to validation and test data.
* The majority class is used only as a baseline. Classification is evaluated with macro-F1,
  balanced accuracy and the confusion matrix; regression with MAE and RMSE. The test split is
  used once, for the final evaluation.

## Final product

The end product is a mobile application backed by an API. A user signs in, enters ingredients,
optionally filters by preparation time, and receives recommended recipes with their known or
estimated rating, the matched and missing ingredients, and any query ingredients that were not
recognised. Ratings the user gives are stored and used to personalise later recommendations.
The recommender is exposed through the same small interface (`recommend(ingredients, k,
user_id, filters)`) in the prototype of this phase and in the API of the later phases.
