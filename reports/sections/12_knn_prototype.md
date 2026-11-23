# 12. kNN Prototype (Validation Only)

Source: `knn_prototype/v1` (inputs `splits/v1` and `loaded/v1`; manifest sha256 of inputs as recorded in `manifest.json`). Tables: `classification_metrics.parquet`, `regression_metrics.parquet`, `confusion_matrices.parquet`.

This section reports validation results only. No test-set results are reported here (Constitution 4.4); the test split was not evaluated.

## 12.1 Setup

- Metric: Jaccard distance on ingredient sets (Constitution 4.5).
- Neighbors: `n_neighbors` 15 for retrieval; `rating_neighbors` 10 for rating prediction.
- Rating weights: uniform and distance.
- Training recipes: 20,551. Validation recipes: 4,418 (`evaluation_split` = val, random_state 42).
- Fit uses the training split only (Constitution 4.1).

## 12.2 Classification (rating class, validation)

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---|---|
| majority | 0.6517 | 0.3333 | 0.2630 |
| knn_uniform | 0.6335 | 0.3356 | 0.2917 |
| knn_distance | 0.6292 | 0.3367 | 0.2981 |

The majority baseline has the highest accuracy, because 65% of validation recipes are in the high class (Section 9). Balanced accuracy is close to the chance level of 0.3333 for all three models. The kNN models gain a little on macro F1 (0.292 and 0.298 against 0.263), which reflects predicting the minority classes more often, not better ranking.

## 12.3 Confusion pattern

The confusion matrices (`confusion_matrices.parquet`) show the following for validation (4,418 recipes):

- majority: predicts "high" for every recipe (284 low, 1,255 medium and 2,879 high recipes all go to high).
- knn_uniform: 18 low→medium, 266 low→high, 1 medium→low, 77 medium→medium, 1,177 medium→high, 157 high→medium, 2,722 high→high. Predictions of low are almost absent (1 recipe).
- knn_distance: 24 low→medium, 260 low→high, 1 medium→low, 99 medium→medium, 1,155 medium→high, 198 high→medium, 2,681 high→high.

Both kNN variants rarely predict "low". Most of their gain on macro F1 comes from correctly placing a few medium recipes.

## 12.4 Regression (average rating, validation)

| Model | MAE | RMSE |
|---|---|---|
| mean | 0.4620 | 0.6431 |
| knn_uniform | 0.4658 | 0.6532 |
| knn_distance | 0.4654 | 0.6533 |

The mean baseline beats both kNN variants on MAE and RMSE. The kNN predictions do not add information beyond the mean on this split.

## 12.5 Interpretation

The ingredient-only kNN prototype does not beat simple baselines on validation. This matches the design: it uses no user history (Section 4.4, cold start) and the rating labels are highly imbalanced (Section 9.3). The prototype is useful as a retrieval tool (example recommendations are in `example_recommendations.parquet` and are not part of this validation section), not as a rating predictor.

Sources: `knn_prototype/v1/classification_metrics.parquet`, `knn_prototype/v1/regression_metrics.parquet`, `knn_prototype/v1/confusion_matrices.parquet`, `knn_prototype/v1/manifest.json`.
