# 7. Ingredient Normalization and Vocabulary

Source: `ingredients/v1` (input stage `loaded/v1`, `params`: min_count 10, encoding binary, n_modifiers 45). Tables: `normalization_summary.parquet`, `merged_variants.parquet`, `rare_threshold_coverage.parquet`; matrices: `binary_matrix.npz`, `tfidf_matrix.npz`; vocabulary: `vocab.json`.

## 7.1 Normalization

| Measure | Value |
|---|---|
| Raw unique ingredient strings | 14,942 |
| Clean unique ingredients | 12,402 |
| Reduction | 2,540 (17.0%) |

The cleaning step merges surface variants (casing, modifiers such as "fresh", "canned", "dried", and similar qualifiers) into canonical names. The 20 canonical names with the most merged variants are:

| Canonical | Variants merged |
|---|---|
| chicken breast | 22 |
| red pepper | 20 |
| potato | 18 |
| shrimp | 17 |
| tomato | 15 |
| pepper | 15 |
| mushroom | 15 |
| onion | 13 |
| chicken | 11 |
| apple | 10 |

(Remaining top-20 entries include peanuts, chicken breast half, carrot, chili pepper, almonds, mustard, coconut, red chili pepper, jalapeno pepper and green pepper, with 8–10 variants each.)

The merge is not perfect. For example, "pepper" and "red pepper" remain separate canonical groups, and "chicken" and "chicken breast" are separate. Some of this is intended (different ingredients), some is ambiguity. This should be stated as a limitation in the final report.

## 7.2 Rare-token threshold

The vocabulary is cut by minimum recipe count. The `rare_threshold_coverage` table:

| min_count | Vocab size | Vocab retained (%) | Occurrence coverage (%) |
|---|---|---|---|
| 1 | 12,402 | 100.00 | 100.00 |
| 2 | 9,277 | 74.80 | 99.85 |
| 3 | 7,797 | 62.87 | 99.71 |
| 5 | 6,380 | 51.44 | 99.48 |
| 10 | 4,709 | 37.97 | 98.95 |
| 20 | 3,458 | 27.88 | 98.13 |
| 50 | 2,164 | 17.45 | 96.17 |
| 100 | 1,507 | 12.15 | 93.96 |

The project uses min_count = 10. This keeps 4,709 ingredients (37.97% of the clean vocabulary) and retains 98.95% of ingredient occurrences.

## 7.3 Encoded matrices

- Binary matrix: 231,637 recipes × 4,709 ingredients, CSR, 2,068,932 non-zero entries (`binary_matrix.npz`, 3,160,920 bytes).
- TF-IDF matrix: same shape and non-zero count (`tfidf_matrix.npz`, 10,379,278 bytes).

Dividing non-zero entries by recipes gives about 8.93 retained ingredients per recipe. This is a derived value from the manifest figures, not a separate output. It is lower than the raw mean of 9.05 ingredients (Section 4), because rare ingredients are dropped.

Sources: `ingredients/v1/normalization_summary.parquet`, `ingredients/v1/merged_variants.parquet`, `ingredients/v1/rare_threshold_coverage.parquet`, `ingredients/v1/manifest.json`.
