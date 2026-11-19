"""Ingredient text normalization, rare-term filtering, sparse encoding, and query vectorization."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_MODIFIERS: tuple[str, ...] = (
    "fresh",
    "freshly",
    "dried",
    "dry",
    "chopped",
    "finely",
    "coarsely",
    "minced",
    "sliced",
    "diced",
    "grated",
    "shredded",
    "crushed",
    "ground",
    "whole",
    "large",
    "medium",
    "small",
    "extra",
    "virgin",
    "cold",
    "warm",
    "hot",
    "frozen",
    "thawed",
    "canned",
    "drained",
    "rinsed",
    "peeled",
    "seeded",
    "pitted",
    "boneless",
    "skinless",
    "unsalted",
    "salted",
    "sweet",
    "raw",
    "cooked",
    "roasted",
    "toasted",
    "melted",
    "softened",
    "room temperature",
    "optional",
    "to taste",
)

# Irregular or sensitive plural endings kept safe from naive stripping
_KEEP_S_WORDS: frozenset[str] = frozenset(
    {
        "hummus",
        "couscous",
        "molasses",
        "asparagus",
        "oats",
        "grits",
        "greens",
        "lentils",
        "noodles",
        "beans",
        "peas",
        "chives",
        "capers",
        "cloves",
        "raisins",
        "almonds",
        "walnuts",
        "pecans",
        "peanuts",
        "breadcrumbs",
        "cornflakes",
        "citrus",
        "swiss",
        "bass",
        "watercress",
        "lemongrass",
        "applesauce",
    }
)

_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_SPACE_RE = re.compile(r"\s+")


def _singularize_token(token: str) -> str:
    """Apply simple rule-based plural-to-singular normalization for a single word."""
    if len(token) <= 3 or token in _KEEP_S_WORDS:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ves") and len(token) > 4:
        return token[:-3] + "f"
    if token.endswith(("ches", "shes", "xes", "zes", "oes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def normalize_ingredient(text: str, modifiers: Iterable[str] = DEFAULT_MODIFIERS) -> str:
    """Normalize a single ingredient string."""
    cleaned = str(text).lower().strip()
    cleaned = _PUNCT_RE.sub(" ", cleaned)
    cleaned = _SPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return ""

    # Remove multi-word modifiers first, then single-word modifiers
    multi_mods = [m.lower().strip() for m in modifiers if " " in m.strip()]
    single_mods = {m.lower().strip() for m in modifiers if " " not in m.strip()}

    for phrase in sorted(multi_mods, key=len, reverse=True):
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned)

    tokens = [t for t in _SPACE_RE.sub(" ", cleaned).strip().split(" ") if t]
    filtered = [t for t in tokens if t not in single_mods]
    if not filtered:
        filtered = tokens  # avoid emptying an ingredient made only of modifiers

    singularized = [_singularize_token(t) for t in filtered]
    return " ".join(singularized).strip()


class IngredientNormalizer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer normalizing recipe ingredient lists.

    Preserves the original `ingredients` column and writes normalized lists
    into `ingredients_clean`.
    """

    def __init__(
        self,
        input_col: str = "ingredients",
        output_col: str = "ingredients_clean",
        modifiers: Sequence[str] | None = None,
    ) -> None:
        self.input_col = input_col
        self.output_col = output_col
        self.modifiers = modifiers

    def fit(self, X: pd.DataFrame, y: object = None) -> IngredientNormalizer:
        self.modifiers_: tuple[str, ...] = (
            tuple(self.modifiers) if self.modifiers is not None else DEFAULT_MODIFIERS
        )
        return self

    def normalize_list(self, items: Iterable[str]) -> list[str]:
        mods = getattr(self, "modifiers_", self.modifiers or DEFAULT_MODIFIERS)
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            norm = normalize_ingredient(item, mods)
            if norm and norm not in seen:
                seen.add(norm)
                out.append(norm)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.input_col not in X.columns:
            raise KeyError(f"Missing required column: {self.input_col}")
        df = X.copy()
        df[self.output_col] = [
            self.normalize_list(items) if isinstance(items, (list, tuple, set)) else []
            for items in df[self.input_col]
        ]
        return df


def normalization_summary(
    df: pd.DataFrame,
    raw_col: str = "ingredients",
    clean_col: str = "ingredients_clean",
    modifiers: Sequence[str] = DEFAULT_MODIFIERS,
    top_n_examples: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute before/after unique ingredient counts and merged variant examples."""
    raw_unique: set[str] = set()
    clean_unique: set[str] = set()
    variant_map: dict[str, set[str]] = defaultdict(set)

    for raw_list in df[raw_col]:
        if not isinstance(raw_list, (list, tuple, set)):
            continue
        for item in raw_list:
            raw_str = str(item).strip()
            if not raw_str:
                continue
            raw_unique.add(raw_str)
            norm = normalize_ingredient(raw_str, modifiers)
            if norm:
                clean_unique.add(norm)
                variant_map[norm].add(raw_str)

    if clean_col in df.columns:
        clean_unique = {
            ing
            for lst in df[clean_col]
            if isinstance(lst, (list, tuple, set))
            for ing in lst
        }

    summary_df = pd.DataFrame(
        [
            {
                "raw_unique_ingredients": len(raw_unique),
                "clean_unique_ingredients": len(clean_unique),
                "reduction_count": len(raw_unique) - len(clean_unique),
                "reduction_pct": round(
                    100.0 * (len(raw_unique) - len(clean_unique)) / max(len(raw_unique), 1),
                    2,
                ),
            }
        ]
    )

    merged_rows = [
        {
            "canonical": canon,
            "variant_count": len(variants),
            "examples": ", ".join(sorted(variants)[:6]),
        }
        for canon, variants in variant_map.items()
        if len(variants) > 1
    ]
    merged_rows.sort(key=lambda r: r["variant_count"], reverse=True)
    variants_df = pd.DataFrame(merged_rows[:top_n_examples])
    return summary_df, variants_df


def rare_threshold_coverage_table(
    ingredient_lists: Iterable[Iterable[str]],
    thresholds: Sequence[int] = (1, 2, 3, 5, 10, 20, 50, 100),
) -> pd.DataFrame:
    """Build coverage table mapping min_count thresholds to vocab size and occurrence coverage."""
    counts: Counter[str] = Counter()
    total_occurrences = 0
    for items in ingredient_lists:
        if not isinstance(items, (list, tuple, set)):
            continue
        for ing in items:
            counts[ing] += 1
            total_occurrences += 1

    total_unique = len(counts)
    rows = []
    for thr in thresholds:
        kept = {k: v for k, v in counts.items() if v >= thr}
        kept_occ = sum(kept.values())
        rows.append(
            {
                "min_count": int(thr),
                "vocab_size": len(kept),
                "vocab_retained_pct": round(100.0 * len(kept) / max(total_unique, 1), 2),
                "occurrence_coverage_pct": round(
                    100.0 * kept_occ / max(total_occurrences, 1), 2
                ),
            }
        )
    return pd.DataFrame(rows)


class RareIngredientFilter(BaseEstimator, TransformerMixin):
    """Filter out rare ingredients whose frequency in training data is below `min_count`."""

    def __init__(
        self,
        min_count: int = 10,
        input_col: str = "ingredients_clean",
        output_col: str = "ingredients_filtered",
    ) -> None:
        self.min_count = min_count
        self.input_col = input_col
        self.output_col = output_col

    def fit(self, X: pd.DataFrame, y: object = None) -> RareIngredientFilter:
        if self.input_col not in X.columns:
            raise KeyError(f"Missing required column: {self.input_col}")
        counts: Counter[str] = Counter()
        for items in X[self.input_col]:
            if isinstance(items, (list, tuple, set)):
                counts.update(items)
        self.counts_: Counter[str] = counts
        self.vocab_: list[str] = sorted(
            ing for ing, cnt in counts.items() if cnt >= self.min_count
        )
        self._vocab_set_: frozenset[str] = frozenset(self.vocab_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "_vocab_set_"):
            raise RuntimeError("RareIngredientFilter must be fitted before transform.")
        df = X.copy()
        df[self.output_col] = [
            [ing for ing in items if ing in self._vocab_set_]
            if isinstance(items, (list, tuple, set))
            else []
            for items in df[self.input_col]
        ]
        return df


class IngredientEncoder(BaseEstimator, TransformerMixin):
    """Encode ingredient lists into a CSR sparse matrix (`binary` or `tfidf`)."""

    def __init__(
        self,
        strategy: Literal["binary", "tfidf"] = "binary",
        input_col: str = "ingredients_filtered",
        min_count: int = 1,
    ) -> None:
        if strategy not in ("binary", "tfidf"):
            raise ValueError(f"Unsupported strategy: {strategy!r}. Use 'binary' or 'tfidf'.")
        self.strategy = strategy
        self.input_col = input_col
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y: object = None) -> IngredientEncoder:
        if self.input_col not in X.columns:
            raise KeyError(f"Missing required column: {self.input_col}")
        df_counts: Counter[str] = Counter()
        n_docs = len(X)
        for items in X[self.input_col]:
            if isinstance(items, (list, tuple, set)):
                df_counts.update(set(items))

        self.vocab_: list[str] = sorted(
            ing for ing, cnt in df_counts.items() if cnt >= self.min_count
        )
        self.vocab_to_idx_: dict[str, int] = {
            ing: idx for idx, ing in enumerate(self.vocab_)
        }
        if self.strategy == "tfidf":
            doc_freq = np.array([df_counts[ing] for ing in self.vocab_], dtype=np.float64)
            # Smooth IDF: log((1 + n) / (1 + df)) + 1
            self.idf_: np.ndarray = np.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0
        else:
            self.idf_ = np.ones(len(self.vocab_), dtype=np.float64)
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        if not hasattr(self, "vocab_to_idx_"):
            raise RuntimeError("IngredientEncoder must be fitted before transform.")
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for row_idx, items in enumerate(X[self.input_col]):
            if not isinstance(items, (list, tuple, set)):
                continue
            unique_indices = sorted(
                {self.vocab_to_idx_[ing] for ing in items if ing in self.vocab_to_idx_}
            )
            for col_idx in unique_indices:
                rows.append(row_idx)
                cols.append(col_idx)
                data.append(float(self.idf_[col_idx]) if self.strategy == "tfidf" else 1.0)

        mat = sparse.csr_matrix(
            (np.array(data, dtype=np.float32), (rows, cols)),
            shape=(len(X), len(self.vocab_)),
            dtype=np.float32,
        )
        if self.strategy == "tfidf" and mat.nnz > 0:
            norms = sparse.linalg.norm(mat, axis=1)
            norms[norms == 0.0] = 1.0
            mat = sparse.diags(1.0 / norms).dot(mat).tocsr().astype(np.float32)

        assert len(self.vocab_) == mat.shape[1], (
            f"Vocab/matrix mismatch: {len(self.vocab_)} != {mat.shape[1]}"
        )
        return mat


class QueryVectorizer:
    """Vectorize a user's free-text ingredient list using the same normalizer and vocab."""

    def __init__(
        self,
        normalizer: IngredientNormalizer,
        vocab: Sequence[str],
        idf: np.ndarray | None = None,
        strategy: Literal["binary", "tfidf"] = "binary",
    ) -> None:
        self.normalizer = normalizer
        self.vocab = list(vocab)
        self.vocab_to_idx = {ing: idx for idx, ing in enumerate(self.vocab)}
        self.strategy = strategy
        self.idf = idf

    def vectorize(self, raw_ingredients: Sequence[str]) -> tuple[sparse.csr_matrix, list[str]]:
        """Return `(query_vector_csr, unknown_ingredients)`."""
        normalized = self.normalizer.normalize_list(raw_ingredients)
        known_indices: list[int] = []
        unknown: list[str] = []

        for ing in normalized:
            if ing in self.vocab_to_idx:
                known_indices.append(self.vocab_to_idx[ing])
            else:
                unknown.append(ing)

        known_indices = sorted(set(known_indices))
        if self.strategy == "tfidf" and self.idf is not None:
            vals = np.array([float(self.idf[i]) for i in known_indices], dtype=np.float32)
            norm = float(np.linalg.norm(vals))
            if norm > 0:
                vals = vals / norm
        else:
            vals = np.ones(len(known_indices), dtype=np.float32)

        vec = sparse.csr_matrix(
            (vals, ([0] * len(known_indices), known_indices)),
            shape=(1, len(self.vocab)),
            dtype=np.float32,
        )
        return vec, unknown