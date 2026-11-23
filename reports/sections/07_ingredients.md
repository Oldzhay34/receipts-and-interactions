# Ingredient Processing

Ingredients are the core of the recommender: a query is a list of ingredients and recipes are
compared through their ingredient sets. This section describes how the free-text ingredient
strings are normalised, how rare strings are filtered and how the lists are encoded. The analysis
version reported here normalises the whole catalogue <!-- src: outputs/ingredients/v2/manifest.json -->;
the normaliser, filter and encoder that the models use are refitted on the training split inside
the preprocessing pipeline (Section {{sec:sampling}}).

## Normalisation

**Finding.** Contributors write ingredients freely, so the same ingredient appears in many forms:
the catalogue has 14,942 distinct ingredient strings
<!-- src: outputs/ingredients/v2/normalization_summary.parquet (raw_unique_ingredients) -->, and the
most frequent ones include "eggs" and "egg", "garlic cloves" and "garlic" (Section
{{sec:exploration}}). **Decision.** `IngredientNormalizer` lower-cases each string, removes
punctuation, removes 43 preparation, size and state words ("fresh", "chopped", "large", "dried",
"to taste", …) and turns plural words into their singular form
<!-- src: outputs/ingredients/v2/manifest.json (params.n_modifiers) --> <!-- src: config/config.yaml (ingredients.modifiers) -->.
Words that change the ingredient itself — "sweet" (sweet potato) and "hot" (hot sauce) — are
deliberately not removed, and words whose final "s" is not a plural (molasses, asparagus, hummus)
are protected. The original column is kept and the result is written to a new column.
**Rationale.** Two recipes that both use eggs must share the same vocabulary entry, otherwise their
Jaccard similarity is understated; at the same time, merging two different ingredients would make
unrelated recipes look similar.

| Measure | Value |
|---|---|
| Raw distinct ingredient strings | 14,942 |
| Distinct ingredients after normalisation | 12,458 |
| Reduction | 2,484 (16.62 %) |

<!-- src: outputs/ingredients/v2/normalization_summary.parquet -->

The canonical ingredients with the most merged variants are chicken breast (22 variants), shrimp
(17), tomato and mushroom (15 each), almond (12) and onion, peanut, potato and chicken (11 each)
<!-- src: outputs/ingredients/v2/merged_variants.parquet --> ({{tab:merged_variants}}); for example,
"canned tomatoes", "chopped tomato" and "canned whole tomatoes" all become "tomato"
<!-- src: outputs/ingredients/v2/merged_variants.parquet (examples) -->. "Sweet potato" remains a
separate ingredient, used 2,105 times <!-- src: outputs/ingredients/v2/recipes_ingredients.parquet (ingredients_clean) -->.

**Limitation.** The rules are deliberately conservative: some distinct-but-related entries stay
separate ("chicken" and "chicken breast", "red pepper" and "pepper"), and a query term only matches
if it normalises to the same string as in the catalogue — "mozzarella" does not match the catalogue
entry "mozzarella cheese" (Section {{sec:knn}}).

## Rare-ingredient threshold

**Finding.** Most normalised strings are rare: requiring an ingredient to appear in at least 10
recipes keeps 4,761 of the 12,458 entries (38.22 %) but still covers 98.95 % of all ingredient
occurrences <!-- src: outputs/ingredients/v2/rare_threshold_coverage.parquet --> ({{tab:rare_threshold_coverage}}):

| min_count | Vocabulary size | Vocabulary retained | Occurrences covered |
|---|---|---|---|
| 1 | 12,458 | 100.00 % | 100.00 % |
| 2 | 9,334 | 74.92 % | 99.85 % |
| 3 | 7,844 | 62.96 % | 99.71 % |
| 5 | 6,434 | 51.65 % | 99.48 % |
| **10** | **4,761** | **38.22 %** | **98.95 %** |
| 20 | 3,489 | 28.01 % | 98.12 % |
| 50 | 2,183 | 17.52 % | 96.13 % |
| 100 | 1,514 | 12.15 % | 93.88 % |

<!-- src: outputs/ingredients/v2/rare_threshold_coverage.parquet -->

**Decision.** `RareIngredientFilter(min_count = 10)` <!-- src: config/config.yaml (ingredients.min_count) -->,
with the counts learned on the training recipes only. **Rationale.** The removed strings are typos,
brand names and one-off spellings; dropping 62 % of the vocabulary loses about 1 % of the
ingredient occurrences while making the vectors shorter and the distances less noisy. A lower
threshold keeps many entries that no query will ever hit; a higher one starts to drop real
ingredients (at 100, 6.12 % of the occurrences are lost)
<!-- src: outputs/ingredients/v2/rare_threshold_coverage.parquet (min_count = 100) -->. Fitted on the
training split inside the pipeline, the vocabulary has 4,114 ingredients
<!-- src: outputs/splits/v5/vocab.json -->.

## Encoding

**Decision.** `IngredientEncoder` turns each recipe's filtered list into a row of a sparse CSR
matrix with one column per vocabulary entry: binary (present / absent, the default) or TF-IDF
(rare ingredients weighted higher, rows L2-normalised). The vocabulary and the matrix are written
in the same version and checked against each other (number of columns = vocabulary size).
**Rationale.** A binary vector is exactly what Jaccard distance compares; TF-IDF is kept as an
alternative for cosine distance in the modelling phase.

On the whole catalogue both matrices have 231,637 rows, 4,761 columns and 2,069,198 non-zero
entries <!-- src: outputs/ingredients/v2/manifest.json (files.binary_matrix, files.tfidf_matrix) -->,
i.e. about 8.93 retained ingredients per recipe (2,069,198 / 231,637), slightly fewer than the raw
mean of 9.05 <!-- src: outputs/quality/v2/dqr_lists.parquet (ingredients Mean Len) --> because rare
ingredients are dropped. The matrix is more than 99.8 % empty, which is why sparse storage and
sparse distance computations are used. A user query is encoded by `QueryVectorizer` with the same
normaliser and vocabulary; query terms outside the vocabulary are returned separately so the
application can show them as "not recognised".
