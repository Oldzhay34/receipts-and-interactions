# Sampling and Data Splitting

Three different subsets of the data are used, each for one purpose: a small local sample for
tests, a development sample for fast model experiments, and the train / validation / test splits
for model fitting and evaluation. The recipe table is never split: it is the catalogue from which
recommendations are drawn, and every recipe stays recommendable.

## Local sample (tests)

**Decision.** 5,000 recipes are drawn with stratification on `n_ingredients`, together with all
23,707 interactions of those recipes <!-- src: outputs/sample/v1/manifest.json (params, files.n_rows) -->.
**Rationale.** Unit tests and notebooks run locally on real data in seconds; stratifying on the
ingredient count keeps the recipe-size distribution, and taking *all* interactions of the sampled
recipes (instead of an independent interaction sample) keeps the two tables consistent.
**Check.** Across the 41 strata the largest difference between the sample and the full share is
0.00011 <!-- src: outputs/sample/v1/stratification.parquet (share_diff) -->.

## Train / validation / test splits

### Why not a random split of the interactions

A recipe's targets (`avg_rating`, `n_ratings`, `rating_class`) are aggregates of its reviews. In a
random split of the interactions, most reviews of a test recipe are in the training data, so its
target is effectively known during training and the model can score well by memorising recipes.
For users, a random split would let a model learn a user's taste from reviews written *after* the
review it is evaluated on. Both are leakage. Two split strategies remove it, one per use case.

### (a) Grouped by recipe, stratified by rating class

**Decision.** Recipes — not interactions — are assigned to train / validation / test in the ratio
70 / 15 / 15 <!-- src: outputs/splits/v5/manifest.json (params.split) -->, and every interaction goes
to the set of its recipe. The assignment is stratified on a per-recipe class: the `rating_class` of
the recipe's average rating, or `unrated` for recipes without any rating above 0. This split is
used for the recipe-level rating models and the kNN recommender.

| Set | Recipes | Interactions | Share of interactions | Users | Rated interactions |
|---|---|---|---|---|---|
| train | 162,145 | 797,681 | 0.704 | 175,773 | 754,976 |
| validation | 34,746 | 168,511 | 0.149 | 55,066 | 159,396 |
| test | 34,746 | 166,175 | 0.147 | 54,575 | 157,148 |

<!-- src: outputs/splits/v5/split_summary.parquet (strategy = group_stratified) -->

The three sets contain 231,637 recipes, i.e. every catalogue recipe exactly once, and all
1,132,367 interactions <!-- src: outputs/splits/v5/split_summary.parquet -->
({{tab:split_summary}}). The interaction shares (0.704 / 0.149 / 0.147) differ slightly from
70 / 15 / 15 because the ratio applies to recipes <!-- src: outputs/splits/v5/split_summary.parquet (share_interactions) -->.

**Class balance.** The class shares are practically identical in the three sets
<!-- src: outputs/splits/v5/class_balance.parquet --> ({{tab:class_balance}}):

| Class | train | validation | test |
|---|---|---|---|
| high | 0.639674 | 0.639671 | 0.639671 |
| medium | 0.276703 | 0.276723 | 0.276723 |
| low | 0.061834 | 0.061820 | 0.061820 |
| unrated | 0.021789 | 0.021787 | 0.021787 |

<!-- src: outputs/splits/v5/class_balance.parquet (share) -->

**The stratification class and leakage.** The class used for stratifying is computed from all
interactions, because the split does not exist yet when it is needed. It is used for nothing
else: the training targets are recomputed from the training interactions only, and validation and
test targets from their own interactions with the class boundaries of the training set. Since
the split assigns whole recipes, a recipe's interactions are all in one set, so its stratification
class and its target are computed from the same reviews and no information crosses between the
sets. The resulting target tables contain 158,612 training, 33,989 validation and 33,989 test
recipes <!-- src: outputs/splits/v5/manifest.json (files.targets_train/val/test.n_rows) -->; the
remaining recipes of each set are the `unrated` ones (3,533 / 757 / 757)
<!-- src: outputs/splits/v5/class_balance.parquet (label = unrated) -->.

### (b) Per user, by time

**Decision.** For every user with at least five interactions
<!-- src: outputs/splits/v5/manifest.json (params.splits_config.temporal) -->, the latest
interaction is the test interaction and the one before it the validation interaction (ties on
the same date are ordered by recipe id); everything earlier, and all interactions of users with
fewer than five, are training data. This split is used for personalisation, where a user profile
must be built from the past and evaluated on the future.

| Set | Interactions | Share | Users | Recipes |
|---|---|---|---|---|
| train | 1,086,195 | 0.959 | 226,570 | 228,372 |
| validation | 23,086 | 0.020 | 23,086 | 15,702 |
| test | 23,086 | 0.020 | 23,086 | 15,690 |

<!-- src: outputs/splits/v5/split_summary.parquet (strategy = temporal_user) -->

23,086 of the 226,570 users have enough history to be evaluated
<!-- src: outputs/splits/v5/split_summary.parquet -->; the others still contribute their
interactions to training. A recipe can occur in several sets in this split, which is intended:
the evaluated quantity is the user, not the recipe.

### Why 70 / 15 / 15

A validation set is needed because the number of neighbours, the distance metric, the weight of
the numeric features and the class-imbalance strategy are all tuned; tuning on the test set would
make the final score optimistic, and the test set is used once. A larger training share would
mainly enlarge the kNN reference set, whose prediction cost grows with its size, while 15 % still
gives 34,746 validation and 34,746 test recipes
<!-- src: outputs/splits/v5/split_summary.parquet (n_recipes) -->.

### Checks before anything is written

The split is saved only after four checks pass: every input row is in exactly one set (row
conservation), no recipe id is shared between the sets of split (a), every user's training
interactions precede the validation and test interactions in split (b), and the class shares of
split (a) differ by at most 0.01 between the sets <!-- src: outputs/splits/v5/manifest.json (params.splits_config.class_share_tolerance) -->.
Both split assignments, the targets, the fitted preprocessing pipeline and the encoded catalogue
are written as one version, so they cannot get out of sync.

## Class imbalance

**Finding.** In the training targets, 103,720 recipes are `high`, 44,866 `medium` and 10,026 `low`
<!-- src: outputs/splits/v5/resampling_summary.parquet (method = none) -->. Random under-sampling
would reduce every class to 10,026 recipes and discard most of the `high` and `medium` recipes;
random over-sampling would copy `low` recipes about ten times to reach 103,720
<!-- src: outputs/splits/v5/resampling_summary.parquet --> ({{tab:resampling_summary}}).
**Decision.** No resampling; balanced class weights (`high` 0.510, `medium` 1.178, `low` 5.273)
<!-- src: outputs/splits/v5/class_weights.json --> are provided to the models, and any resampling
that is tried later is applied to the training set only. **Rationale.** Under-sampling throws away
most of the data; in a kNN model over-sampling only multiplies the votes of duplicated recipes,
which class weights achieve directly. SMOTE is not used: it creates synthetic recipes by
interpolating between neighbours, which on binary ingredient vectors yields fractional
"half-ingredients" and ingredient combinations no recipe has, and its neighbour search is
Euclidean. If synthetic minority samples are ever needed, SMOTE-NC (nominal features set to the
neighbours' majority value) or SMOTE-N (all-nominal data, value difference metric) are the
appropriate variants, again on the training set only.

## Development sample

**Decision.** A development sample of 30,000 recipes, stratified on the same class as the split,
with all 149,952 of their interactions <!-- src: outputs/splits/v5/dev_recipes.parquet + grouped_assignments.parquet (interactions of the sampled recipes) -->.
Its class composition is 19,190 `high`, 8,301 `medium`, 1,855 `low` and 654 `unrated` recipes
<!-- src: outputs/splits/v5/dev_recipes.parquet (label) -->. **Rationale.** kNN experiments on the
full training set are slow; because split (a) assigns whole recipes, restricting it to the sampled
recipes gives a valid split of the sample (21,022 / 4,505 / 4,473 recipes)
<!-- src: outputs/splits/v5/grouped_assignments.parquet (recipes of dev_recipes per split) -->, so
model choices can be explored quickly and then confirmed on the full sets.
