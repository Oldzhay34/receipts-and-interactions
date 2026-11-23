# Normalization and Scaling

kNN compares recipes through distances, so every feature that enters a distance must be on a
comparable scale. This section compares the two scaling methods on the numeric features, explains
why the binary ingredient vector is not scaled, and shows how numeric and ingredient distances are
combined.

## Why scaling is required

**Finding.** The raw numeric features differ by orders of magnitude: the median of `calories` is
313.4 with a standard deviation of 1,189.71, while `n_ingredients` has a median of 9 and a standard
deviation of 3.73 <!-- src: outputs/scaling/v1/scaler_summary.parquet (transform = raw) -->. A
counter-example in the test suite makes the consequence concrete: with unscaled features, a recipe
with almost the same preparation time (16 vs 15 minutes) and the same number of ingredients but 150
more calories is *farther* in Euclidean distance than one with a completely different preparation
time (120 minutes) and ingredient count (18 vs 4) but identical calories
<!-- src: tests/test_scaling.py::test_unscaled_calories_dominates_knn_distance -->. **Decision.** The
numeric features are scaled before any distance is computed. **Rationale.** Without scaling, the
feature with the largest numbers decides the neighbours on its own.

## Min-max versus z-score

The comparison applies log1p and then either min-max or z-score scaling to the ten numeric columns
of the loaded data ({{tab:scaler_summary}}; {{fig:fig_08_01_calories_raw}} –
{{fig:fig_08_05_minutes_log1p_minmax}}):

| Feature | Transform | Min | Median | Max | Std |
|---|---|---|---|---|---|
| calories | raw | 0.0 | 313.4 | 434,360.2 | 1,189.71 |
| calories | log1p + min-max | 0.0 | 0.4430 | 1.0 | 0.0742 |
| calories | log1p + z-score | −5.9091 | 0.0598 | 7.5652 | 1.0 |
| sugar_pdv | raw | 0.0 | 25.0 | 362,729.0 | 800.08 |
| sugar_pdv | log1p + min-max | 0.0 | 0.2545 | 1.0 | 0.1169 |
| minutes | log1p + min-max | 0.0 | 0.1728 | 1.0 | 0.0507 |
| n_ingredients | log1p + min-max | 0.0 | 0.5207 | 1.0 | 0.1242 |

<!-- src: outputs/scaling/v1/scaler_summary.parquet (min, median, max, std) -->

**Finding.** After log1p and min-max scaling every feature lies in [0, 1] with its median in the
middle of the range for `calories` (0.443) and `n_ingredients` (0.521), but `minutes` is squeezed
into a narrow band (median 0.173, standard deviation 0.051)
<!-- src: outputs/scaling/v1/scaler_summary.parquet -->: this comparison runs on the uncleaned
loaded data, where the maximum of 2,147,483,647 minutes still defines the upper end of the range.
Z-score values are unbounded (calories from −5.91 to 7.57)
<!-- src: outputs/scaling/v1/scaler_summary.parquet (transform = log1p_zscore) -->.

**Decision.** Min-max scaling is used (`scaling.method: minmax`) <!-- src: config/config.yaml (scaling) -->,
applied after the outlier treatment: in the preprocessing pipeline the numeric features are first
cleaned of invalid values, imputed and clipped at the training 99th percentile, then
log-transformed (all except `n_ingredients`), and only then scaled with statistics learned on the
training split (Sections {{sec:outliers}} and {{sec:sampling}}). **Rationale.** Min-max gives
every feature the same fixed range, which makes the weights of a combined distance directly
interpretable; the comparison above shows that scaling uncleaned data lets a single extreme value
compress a whole feature, which is why clipping comes first.

## Why the ingredient vector is not scaled

The ingredient vector is binary (ingredient absent or present) and is compared with Jaccard or
cosine distance, which only count shared and non-shared ingredients. Its values are already on a
fixed [0, 1] scale, so min-max scaling would change nothing, and z-score scaling would turn every
0 into a negative number, destroying the sparsity of the matrix and making "absent" depend on how
common an ingredient is. Scaling is therefore applied to the numeric features only.

## Combined distance

`WeightedCombinedDistance` combines an ingredient distance (Jaccard by default; the metric is
injected, so cosine can be used instead) with the Euclidean distance of the scaled numeric
features, normalised by the square root of the number of features:
`d = (1 − w) · d_ingredients + w · d_numeric` with `w = 0.2`
<!-- src: outputs/scaling/v1/manifest.json (params.numeric_weight) -->. On a sample of five recipes
the combined distances lie between 0.72 and 0.83, with a zero diagonal and a symmetric matrix
<!-- src: outputs/scaling/v1/combined_distance_sample.parquet --> ({{tab:combined_distance_sample}}).
This sample only checks that the metric behaves as a distance; the weight is tuned on the
validation set in the modelling phase.
