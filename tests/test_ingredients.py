"""Unit tests for stage 1.5: ingredient normalization, filtering, encoding, and distances."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recsys.features.distance import CosineDistance, JaccardDistance
from recsys.features.ingredients import (
    IngredientEncoder,
    IngredientNormalizer,
    QueryVectorizer,
    RareIngredientFilter,
    normalization_summary,
    normalize_ingredient,
    rare_threshold_coverage_table,
)


def _sample_recipes_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "ingredients": [
                ["Fresh Garlic", "chopped onions", "Olive Oil", "salt"],
                ["minced garlic", "large onion", "extra virgin olive oil", "black pepper"],
                ["tomatoes", "fresh basil", "olive oil", "salt"],
                ["chicken breasts", "garlic", "lemon juice", "unicorn spice"],
            ],
        }
    )


def test_normalize_ingredient_rules() -> None:
    assert normalize_ingredient("  Freshly Chopped Tomatoes! ") == "tomato"
    assert normalize_ingredient("Extra Virgin Olive Oil") == "olive oil"
    assert normalize_ingredient("large red onions") == "red onion"
    # Keep-s words should not be mangled
    assert normalize_ingredient("fresh asparagus") == "asparagus"
    assert normalize_ingredient("hummus") == "hummus"


def test_ingredient_normalizer_preserves_original_column() -> None:
    df = _sample_recipes_df()
    norm = IngredientNormalizer().fit(df)
    out = norm.transform(df)

    assert "ingredients" in out.columns
    assert "ingredients_clean" in out.columns
    assert out["ingredients"].tolist() == df["ingredients"].tolist()
    assert "garlic" in out.loc[0, "ingredients_clean"]
    assert "onion" in out.loc[0, "ingredients_clean"]
    assert "olive oil" in out.loc[1, "ingredients_clean"]


def test_normalization_summary_and_variants() -> None:
    df = _sample_recipes_df()
    out = IngredientNormalizer().fit_transform(df)
    summary, variants = normalization_summary(out)

    assert summary.loc[0, "raw_unique_ingredients"] >= summary.loc[0, "clean_unique_ingredients"]
    assert summary.loc[0, "reduction_count"] > 0
    assert "garlic" in variants["canonical"].values


def test_rare_ingredient_filter_fit_only_on_train() -> None:
    train_df = IngredientNormalizer().fit_transform(_sample_recipes_df())
    flt = RareIngredientFilter(min_count=2).fit(train_df)

    # 'unicorn spice' appears once -> filtered out; 'garlic' appears 3 times -> kept
    assert "garlic" in flt.vocab_
    assert "unicorn spice" not in flt.vocab_

    # New unseen ingredient in validation/test must also be dropped
    val_df = pd.DataFrame({"ingredients_clean": [["garlic", "brand new spice"]]})
    val_out = flt.transform(val_df)
    assert val_out.loc[0, "ingredients_filtered"] == ["garlic"]


def test_rare_threshold_coverage_table() -> None:
    df = IngredientNormalizer().fit_transform(_sample_recipes_df())
    cov = rare_threshold_coverage_table(df["ingredients_clean"], thresholds=(1, 2, 3))
    assert list(cov["min_count"]) == [1, 2, 3]
    assert cov.loc[0, "occurrence_coverage_pct"] == 100.0
    assert cov["vocab_size"].is_monotonic_decreasing


@pytest.mark.parametrize("strategy", ["binary", "tfidf"])
def test_ingredient_encoder_consistency(strategy: str) -> None:
    df = IngredientNormalizer().fit_transform(_sample_recipes_df())
    df = RareIngredientFilter(min_count=1).fit_transform(df)

    enc = IngredientEncoder(strategy=strategy).fit(df)
    mat = enc.transform(df)

    assert sparse.issparse(mat)
    assert mat.shape == (len(df), len(enc.vocab_))
    if strategy == "binary":
        assert set(mat.data.tolist()).issubset({1.0})
    else:
        # L2-normalized rows for non-empty recipes
        row_norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
        np.testing.assert_allclose(row_norms, np.ones(len(df)), atol=1e-5)


def test_query_vectorizer_returns_unknown_ingredients() -> None:
    df = IngredientNormalizer().fit_transform(_sample_recipes_df())
    df = RareIngredientFilter(min_count=2).fit_transform(df)
    enc = IngredientEncoder(strategy="binary").fit(df)

    norm = IngredientNormalizer().fit(df)
    qv = QueryVectorizer(normalizer=norm, vocab=enc.vocab_, strategy="binary")

    vec, unknown = qv.vectorize(["Fresh Garlic", "Dragon Fruit", "Chopped Onions"])
    assert vec.shape == (1, len(enc.vocab_))
    assert vec.nnz == 2
    assert unknown == ["dragon fruit"]


def test_jaccard_and_cosine_distances() -> None:
    a = sparse.csr_matrix([[1, 1, 0, 0]], dtype=np.float32)
    b = sparse.csr_matrix(
        [
            [1, 1, 0, 0],  # identical
            [1, 0, 1, 0],  # 1 shared, union=3 -> Jaccard dist = 2/3
            [0, 0, 1, 1],  # disjoint
        ],
        dtype=np.float32,
    )

    j_dist = JaccardDistance().pairwise(a, b)[0]
    np.testing.assert_allclose(j_dist, [0.0, 2.0 / 3.0, 1.0], atol=1e-5)

    c_dist = CosineDistance().pairwise(a, b)[0]
    np.testing.assert_allclose(c_dist, [0.0, 0.5, 1.0], atol=1e-5)


def test_euclidean_counterexample_on_sparse_binary_ingredients() -> None:
    """Demonstrate why Euclidean distance is unsuitable for binary ingredient vectors.

    Query has 3 ingredients: {chicken, garlic, lemon}.
    - Recipe A (relevant): shares ALL 3 ingredients + 5 extra pantry items (8 total).
    - Recipe B (irrelevant): shares ZERO ingredients, only 1 unrelated item (water).

    Euclidean distance wrongly ranks irrelevant Recipe B closer than relevant Recipe A,
    whereas Jaccard and Cosine correctly rank Recipe A closer.
    """
    # 9-ingredient vocabulary
    query = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    recipe_a = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=np.float32)  # 3/3 match + 5 extra
    recipe_b = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)  # 0/3 match, 1 item

    euclid_a = float(np.linalg.norm(query - recipe_a))  # sqrt(5) ≈ 2.236
    euclid_b = float(np.linalg.norm(query - recipe_b))  # sqrt(4) = 2.000
    assert euclid_b < euclid_a, "Euclidean wrongly ranks irrelevant Recipe B closer!"

    jaccard = JaccardDistance()
    cosine = CosineDistance()
    assert jaccard.pairwise(query, recipe_a)[0, 0] < jaccard.pairwise(query, recipe_b)[0, 0]
    assert cosine.pairwise(query, recipe_a)[0, 0] < cosine.pairwise(query, recipe_b)[0, 0]

def test_ingredients_output_matches_contract() -> None:
    from recsys.contracts import INGREDIENTS_OUTPUT_SCHEMA, validate

    df = IngredientNormalizer().fit_transform(_sample_recipes_df())
    df = RareIngredientFilter(min_count=1).fit(df).transform(df)
    validate(df[["id", "ingredients", "ingredients_clean", "ingredients_filtered"]],
             INGREDIENTS_OUTPUT_SCHEMA)
