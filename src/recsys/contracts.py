"""Lightweight schema contracts.

Every stage validates its inputs and outputs against a :class:`Schema` defined here. A failed
validation collects ALL violations and raises one :class:`SchemaValidationError`; nothing is
written (ArtifactStore validates before writing).

NOTE FOR LATER STAGES
    Add your stage schema in the "Stage schemas added by later stages" section at the END of this
    file (e.g. QUALITY_ISSUES_SCHEMA in 1.2, MISSING_OUTPUT_SCHEMA in 1.3). Append only: do not
    edit existing schemas or the validator. If an existing schema must change, ask the
    module owner.
    ``Schema.derive`` builds a new schema from an existing one without modifying it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pandas.api import types as ptypes

DType = Literal["int", "float", "number", "str", "bool", "datetime", "list", "category", "any"]

_MAX_EXAMPLES = 5


@dataclass(frozen=True)
class Column:
    """Constraints for one column.

    Attributes:
        dtype: Logical type. ``"int"`` accepts any integer dtype (int8..int64, nullable Int);
            ``"number"`` accepts int or float; ``"list"`` requires Python lists in an object column.
        nullable: If False, any missing value is a violation.
        min / max: Inclusive bounds checked on non-missing values.
        unique: Non-missing values must be unique.
        allowed: Closed set of permitted non-missing values.
    """

    dtype: DType
    nullable: bool = True
    min: Any = None
    max: Any = None
    unique: bool = False
    allowed: frozenset[Any] | None = None

    def __post_init__(self) -> None:
        if self.allowed is not None and not isinstance(self.allowed, frozenset):
            object.__setattr__(self, "allowed", frozenset(self.allowed))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"dtype": self.dtype, "nullable": self.nullable}
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        if self.unique:
            out["unique"] = True
        if self.allowed is not None:
            out["allowed"] = sorted(self.allowed, key=str)
        return out


@dataclass(frozen=True)
class Schema:
    """A table contract.

    Attributes:
        name: Identifier recorded in artefact manifests.
        columns: Ordered column -> :class:`Column` mapping. Missing columns are violations.
        unique_together: Column groups whose combined values must be unique.
        allow_extra_columns: If False, columns not listed are violations.
        min_rows: Minimum number of rows.
    """

    name: str
    columns: Mapping[str, Column]
    unique_together: tuple[tuple[str, ...], ...] = ()
    allow_extra_columns: bool = False
    min_rows: int = 0
    description: str = field(default="", compare=False)

    def derive(
        self,
        name: str,
        add: Mapping[str, Column] | None = None,
        drop: Iterable[str] = (),
        **overrides: Any,
    ) -> Schema:
        """Return a NEW schema based on this one (this schema is not modified)."""
        drop_set = set(drop)
        unknown = drop_set - set(self.columns)
        if unknown:
            raise KeyError(f"{self.name}: cannot drop unknown columns {sorted(unknown)}")
        cols = {k: v for k, v in self.columns.items() if k not in drop_set}
        cols.update(add or {})
        params: dict[str, Any] = {
            "unique_together": self.unique_together,
            "allow_extra_columns": self.allow_extra_columns,
            "min_rows": self.min_rows,
            "description": self.description,
        }
        params.update(overrides)
        return Schema(name=name, columns=cols, **params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "unique_together": [list(g) for g in self.unique_together],
            "allow_extra_columns": self.allow_extra_columns,
            "min_rows": self.min_rows,
        }


class SchemaValidationError(ValueError):
    """Raised with every violation found, so one run shows all problems."""

    def __init__(self, schema_name: str, violations: list[str]) -> None:
        self.schema_name = schema_name
        self.violations = list(violations)
        lines = "\n".join(f"  - {v}" for v in self.violations)
        super().__init__(
            f"Schema {schema_name!r} failed with {len(self.violations)} violation(s):\n{lines}"
        )


def validate(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Validate ``df`` against ``schema``; return ``df`` unchanged on success.

    Raises:
        SchemaValidationError: listing every violation.
        TypeError: if ``df`` is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"validate expects a DataFrame, got {type(df).__name__}")

    violations: list[str] = []
    if len(df) < schema.min_rows:
        violations.append(f"expected at least {schema.min_rows} rows, got {len(df)}")

    missing = [c for c in schema.columns if c not in df.columns]
    if missing:
        violations.append(f"missing columns: {missing}")
    if not schema.allow_extra_columns:
        extra = [c for c in df.columns if c not in schema.columns]
        if extra:
            violations.append(f"unexpected columns: {extra}")
    if df.columns.duplicated().any():
        violations.append(f"duplicate column names: {list(df.columns[df.columns.duplicated()])}")

    for name, col in schema.columns.items():
        if name in df.columns and not df.columns.duplicated().any():
            violations.extend(_check_column(df[name], name, col))

    for group in schema.unique_together:
        absent = [c for c in group if c not in df.columns]
        if absent:
            continue  # already reported as missing
        dup = df.duplicated(subset=list(group), keep=False)
        if dup.any():
            violations.append(
                f"{list(group)}: {int(dup.sum())} rows share duplicate key values "
                f"(e.g. index {_examples(df.index[dup])})"
            )

    if violations:
        raise SchemaValidationError(schema.name, violations)
    return df


def assert_disjoint(frames: Mapping[str, pd.DataFrame], key: str) -> None:
    """Raise if any value of ``key`` appears in more than one frame (e.g. train/val/test recipe_id).

    Used by grouped splits.
    """
    violations: list[str] = []
    names = list(frames)
    values = {n: set(frames[n][key].dropna().unique()) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = values[a] & values[b]
            if shared:
                violations.append(
                    f"{key}: {len(shared)} values shared by {a!r} and {b!r} "
                    f"(e.g. {_examples(sorted(shared, key=str))})"
                )
    if violations:
        raise SchemaValidationError(f"disjoint[{key}]", violations)


# --------------------------------------------------------------------------- internals


def _check_column(s: pd.Series, name: str, col: Column) -> list[str]:
    out: list[str] = []
    type_error = _dtype_violation(s, col.dtype)
    if type_error:
        return [f"{name}: {type_error}"]  # range checks on the wrong type are meaningless

    na = s.isna()
    if not col.nullable and na.any():
        out.append(f"{name}: {int(na.sum())} missing values in non-nullable column")

    present = s[~na]
    if col.min is not None:
        bad = present < col.min
        if bad.any():
            out.append(
                f"{name}: {int(bad.sum())} values below min {col.min} "
                f"(e.g. {_examples(present[bad])})"
            )
    if col.max is not None:
        bad = present > col.max
        if bad.any():
            out.append(
                f"{name}: {int(bad.sum())} values above max {col.max} "
                f"(e.g. {_examples(present[bad])})"
            )
    if col.allowed is not None:
        bad = ~present.isin(col.allowed)
        if bad.any():
            out.append(
                f"{name}: {int(bad.sum())} values outside allowed set "
                f"(e.g. {_examples(present[bad].unique())})"
            )
    if col.unique:
        dup = present.duplicated(keep=False)
        if dup.any():
            out.append(
                f"{name}: {int(dup.sum())} rows with duplicate values "
                f"(e.g. {_examples(present[dup].unique())})"
            )
    return out


def _dtype_violation(s: pd.Series, dtype: DType) -> str | None:
    actual = str(s.dtype)
    if dtype == "any":
        return None
    if dtype == "int":
        ok = ptypes.is_integer_dtype(s) and not ptypes.is_bool_dtype(s)
    elif dtype == "float":
        ok = ptypes.is_float_dtype(s)
    elif dtype == "number":
        ok = ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s)
    elif dtype == "bool":
        ok = ptypes.is_bool_dtype(s)
    elif dtype == "datetime":
        ok = ptypes.is_datetime64_any_dtype(s)
    elif dtype == "category":
        ok = isinstance(s.dtype, pd.CategoricalDtype)
    elif dtype == "str":
        if not (ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s)):
            return f"expected str, got dtype {actual}"
        inferred = pd.api.types.infer_dtype(s, skipna=True)
        ok = inferred in ("string", "empty")
        if not ok:
            return f"expected str values, found inferred type {inferred!r}"
    elif dtype == "list":
        if not ptypes.is_object_dtype(s):
            return f"expected list (object dtype), got dtype {actual}"
        non_list = s.dropna().map(lambda v: not isinstance(v, list))
        if non_list.any():
            bad_types = sorted({type(v).__name__ for v in s.dropna()[non_list].head(50)})
            return f"{int(non_list.sum())} non-list values (types: {bad_types})"
        return None
    else:  # pragma: no cover - guarded by the Literal type
        return f"unknown schema dtype {dtype!r}"
    return None if ok else f"expected {dtype}, got dtype {actual}"


def _examples(values: Iterable[Any]) -> list[Any]:
    out = []
    for i, v in enumerate(values):
        if i >= _MAX_EXAMPLES:
            break
        out.append(v.item() if hasattr(v, "item") else v)
    return out


# =========================================================================== 1.0 schemas
# RAW_* : the CSV exactly as read (structure + key integrity only).
# LOADED_*: after recsys.data.load parsing (lists, nutrition columns, datetimes).
# Value-range rules (minutes=0, huge minutes, n_steps=0, ...) are deliberately NOT enforced here:
# loading must not reject data whose handling is decided in 1.2-1.4.

_RATING_DOMAIN = frozenset({0, 1, 2, 3, 4, 5})

RAW_RECIPES_SCHEMA = Schema(
    name="raw_recipes",
    columns={
        "name": Column("str"),
        "id": Column("int", nullable=False, min=0, unique=True),
        "minutes": Column("int", nullable=False),
        "contributor_id": Column("int", nullable=False),
        "submitted": Column("str"),
        "tags": Column("str"),
        "nutrition": Column("str"),
        "n_steps": Column("int", nullable=False),
        "steps": Column("str"),
        "description": Column("str"),
        "ingredients": Column("str"),
        "n_ingredients": Column("int", nullable=False),
    },
    min_rows=1,
)

RAW_INTERACTIONS_SCHEMA = Schema(
    name="raw_interactions",
    columns={
        "user_id": Column("int", nullable=False),
        "recipe_id": Column("int", nullable=False),
        "date": Column("str"),
        "rating": Column("number", nullable=False, allowed=_RATING_DOMAIN),
        "review": Column("str"),
    },
    min_rows=1,
)

NUTRITION_COLUMNS: tuple[str, ...] = (
    "calories",
    "total_fat_pdv",
    "sugar_pdv",
    "sodium_pdv",
    "protein_pdv",
    "saturated_fat_pdv",
    "carbohydrates_pdv",
)

LOADED_RECIPES_SCHEMA = Schema(
    name="loaded_recipes",
    columns={
        "name": Column("str"),
        "id": Column("int", nullable=False, min=0, unique=True),
        "minutes": Column("int", nullable=False),
        "contributor_id": Column("int", nullable=False),
        "submitted": Column("datetime"),
        "tags": Column("list"),
        "n_steps": Column("int", nullable=False),
        "steps": Column("list"),
        "description": Column("str"),
        "ingredients": Column("list"),
        "n_ingredients": Column("int", nullable=False),
        **{c: Column("float") for c in NUTRITION_COLUMNS},
    },
    min_rows=1,
)

LOADED_INTERACTIONS_SCHEMA = Schema(
    name="loaded_interactions",
    columns={
        "user_id": Column("int", nullable=False),
        "recipe_id": Column("int", nullable=False),
        "date": Column("datetime"),
        "rating": Column("int", nullable=False, allowed=_RATING_DOMAIN),
        "review": Column("str"),
    },
    min_rows=1,
)


# =========================================================================== later stages
# Stage schemas added by later stages
# (append below; one block per stage, headed with its stage id)

# --------------------------------------------------------------------------- 1.1 exploration
# Output tables of recsys.analysis.explore (stage `exploration`).

EXPLORATION_TYPE_CLASSES: tuple[str, ...] = (
    "continuous",
    "categorical",
    "text",
    "list",
    "datetime",
)
RATING_VALUES: tuple[int, ...] = tuple(sorted(_RATING_DOMAIN))

COLUMN_TYPES_SCHEMA = Schema(
    "exploration_column_types",
    {
        "table": Column("str", nullable=False),
        "column": Column("str", nullable=False),
        "dtype": Column("str", nullable=False),
        "type_class": Column("str", nullable=False, allowed=frozenset(EXPLORATION_TYPE_CLASSES)),
        "source": Column("str", nullable=False, allowed=frozenset({"declared", "inferred"})),
        "note": Column("str", nullable=False),
        "n_missing": Column("int", nullable=False, min=0),
        "n_unique": Column("int", min=0),
    },
    unique_together=(("table", "column"),),
    min_rows=1,
)

BASIC_COUNTS_SCHEMA = Schema(
    "exploration_basic_counts",
    {
        **{
            c: Column("int", nullable=False, min=0)
            for c in (
                "n_recipes",
                "n_interactions",
                "n_users",
                "n_contributors",
                "n_recipes_with_interactions",
                "n_recipes_without_interactions",
                "n_orphan_interactions",
                "n_duplicate_interactions",
                "n_repeated_user_recipe_pairs",
            )
        },
        "interactions_date_min": Column("datetime"),
        "interactions_date_max": Column("datetime"),
        "recipes_submitted_min": Column("datetime"),
        "recipes_submitted_max": Column("datetime"),
    },
    min_rows=1,
)

PER_ENTITY_COUNTS_SCHEMA = Schema(
    "exploration_per_entity_counts",
    {
        "entity": Column("str", nullable=False, unique=True, allowed=frozenset({"recipe", "user"})),
        "n_entities": Column("int", nullable=False, min=0),
        "n_zero": Column("int", nullable=False, min=0),
        "n_one": Column("int", nullable=False, min=0),
        "pct_one": Column("float", min=0.0, max=1.0),
        "mean": Column("float", min=0.0),
        "std": Column("float", min=0.0),
        "min": Column("int", nullable=False, min=0),
        "q25": Column("float", min=0.0),
        "median": Column("float", min=0.0),
        "q75": Column("float", min=0.0),
        "max": Column("int", nullable=False, min=0),
    },
    min_rows=2,
)

RATING_COUNTS_SCHEMA = Schema(
    "exploration_rating_counts",
    {
        "rating": Column("int", nullable=False, unique=True, allowed=frozenset(RATING_VALUES)),
        "count": Column("int", nullable=False, min=0),
        "share": Column("float", min=0.0, max=1.0),
        "share_of_rated": Column("float", min=0.0, max=1.0),
        "is_zero": Column("bool", nullable=False),
        "note": Column("str", nullable=False),
    },
    min_rows=len(RATING_VALUES),
)

TOP_ITEMS_SCHEMA = Schema(
    "exploration_top_items",
    {
        "kind": Column("str", nullable=False, allowed=frozenset({"ingredient", "tag"})),
        "rank": Column("int", nullable=False, min=1),
        "item": Column("str", nullable=False),
        "n_recipes": Column("int", nullable=False, min=1),
        "share_of_recipes": Column("float", nullable=False, min=0.0, max=1.0),
    },
    unique_together=(("kind", "rank"), ("kind", "item")),
    min_rows=1,
)

EXPLORATORY_TARGETS_SCHEMA = Schema(
    "exploration_targets",
    {
        "recipe_id": Column("int", nullable=False, unique=True),
        "n_interactions": Column("int", nullable=False, min=0),
        "n_ratings": Column("int", nullable=False, min=0),
        "avg_rating": Column("float", min=1.0, max=5.0),
        "has_rating": Column("bool", nullable=False),
    },
    min_rows=1,
)

#: object name in the ``exploration`` stage -> schema (pass as ``schemas=`` to ArtifactStore.save).
EXPLORATION_SCHEMAS: Mapping[str, Schema] = {
    "column_types": COLUMN_TYPES_SCHEMA,
    "basic_counts": BASIC_COUNTS_SCHEMA,
    "per_entity_counts": PER_ENTITY_COUNTS_SCHEMA,
    "rating_counts": RATING_COUNTS_SCHEMA,
    "top_items": TOP_ITEMS_SCHEMA,
    "exploratory_targets": EXPLORATORY_TARGETS_SCHEMA,
}


# --------------------------------------------------------------------------- 1.2 quality
# Issues table of recsys.analysis.quality.detect_issues (stage `quality`); read by 1.3 and 1.4.

QUALITY_ISSUES_SCHEMA = Schema(
    name="quality_issues",
    columns={
        "issue_id": Column("str", nullable=False, unique=True),
        "column": Column("str", nullable=False),
        "rule": Column("str", nullable=False),
        "affected_rows": Column("int", nullable=False, min=0),
        "example_ids": Column("str", nullable=False),
        "suggested_action": Column("str", nullable=False),
    },
    min_rows=1,
    description="Data quality issues produced in stage 1.2 for stages 1.3 and 1.4.",
)

INGREDIENTS_OUTPUT_SCHEMA = Schema(
    name="ingredients_output",
    columns={
        "id": Column("int", nullable=False, unique=True),
        "ingredients": Column("list", nullable=False),
        "ingredients_clean": Column("list", nullable=False),
        "ingredients_filtered": Column("list", nullable=False),
    },
    min_rows=1,
    description="Cleaned and filtered ingredient lists produced in stage 1.5.",
)
# --------------------------------------------------------------------------- 1.3 missing
# Tables of the `missing` stage (recsys.cleaning.missing analysis functions) and the contracts of
# the data after the 1.3 transformers (used by the 1.9 pipeline).

MISSING_RATES_SCHEMA = Schema(
    "missing_rates",
    {
        "table": Column("str", nullable=False, allowed=frozenset({"recipes", "interactions"})),
        "column": Column("str", nullable=False),
        "kind": Column("str", nullable=False, allowed=frozenset({"null", "empty_list", "zero"})),
        "n_rows": Column("int", nullable=False, min=0),
        "n_missing": Column("int", nullable=False, min=0),
        "share": Column("float", min=0.0, max=1.0),
    },
    unique_together=(("table", "column", "kind"),),
    min_rows=1,
)

RATING_ZERO_REVIEW_STATS_SCHEMA = Schema(
    "missing_rating_zero_review_stats",
    {
        "n_rows": Column("int", nullable=False, min=0),
        "n_zero": Column("int", nullable=False, min=0),
        "share_zero": Column("float", min=0.0, max=1.0),
        "n_zero_with_review": Column("int", nullable=False, min=0),
        "share_zero_with_review": Column("float", min=0.0, max=1.0),
        "share_rated_with_review": Column("float", min=0.0, max=1.0),
    },
    min_rows=1,
)

RATING_ZERO_EXAMPLES_SCHEMA = Schema(
    "missing_rating_zero_examples",
    {
        "user_id": Column("int", nullable=False),
        "recipe_id": Column("int", nullable=False),
        "date": Column("datetime"),
        "review": Column("str"),
    },
)

MINUTES_ZERO_SUMMARY_SCHEMA = Schema(
    "missing_minutes_zero_summary",
    {
        "n_recipes": Column("int", nullable=False, min=0),
        "n_minutes_zero": Column("int", nullable=False, min=0),
        "share_minutes_zero": Column("float", min=0.0, max=1.0),
        "n_zero_with_no_cook_tag": Column("int", nullable=False, min=0),
        "share_zero_with_no_cook_tag": Column("float", min=0.0, max=1.0),
        "share_all_with_no_cook_tag": Column("float", min=0.0, max=1.0),
        "median_n_steps_zero": Column("float", min=0.0),
        "median_n_steps_all": Column("float", min=0.0),
        "median_n_ingredients_zero": Column("float", min=0.0),
        "median_n_ingredients_all": Column("float", min=0.0),
    },
    min_rows=1,
)

MINUTES_ZERO_TAGS_SCHEMA = Schema(
    "missing_minutes_zero_tags",
    {
        "tag": Column("str", nullable=False, unique=True),
        "n_zero": Column("int", nullable=False, min=1),
        "share_in_zero": Column("float", nullable=False, min=0.0, max=1.0),
        "share_in_all": Column("float", nullable=False, min=0.0, max=1.0),
        "lift": Column("float", nullable=False, min=0.0),
    },
)

MINUTES_ZERO_N_STEPS_SCHEMA = Schema(
    "missing_minutes_zero_n_steps",
    {
        "n_steps": Column("int", nullable=False, unique=True, min=0),
        "n_zero": Column("int", nullable=False, min=0),
        "share_zero": Column("float", min=0.0, max=1.0),
        "n_all": Column("int", nullable=False, min=0),
        "share_all": Column("float", nullable=False, min=0.0, max=1.0),
    },
    min_rows=1,
)

MINUTES_ZERO_EXAMPLES_SCHEMA = Schema(
    "missing_minutes_zero_examples",
    {
        "id": Column("int", nullable=False, unique=True),
        "name": Column("str"),
        "n_steps": Column("int", nullable=False, min=0),
        "n_ingredients": Column("int", nullable=False, min=0),
        "tags": Column("str", nullable=False),
        "first_step": Column("str", nullable=False),
    },
)

MAIN_TAG_COVERAGE_SCHEMA = Schema(
    "missing_main_tag_coverage",
    {
        "main_tag": Column("str", nullable=False, unique=True),
        "n_recipes": Column("int", nullable=False, min=0),
        "share": Column("float", min=0.0, max=1.0),
    },
    min_rows=1,
)

IMPUTER_GROUPS_SCHEMA = Schema(
    "missing_imputer_groups",
    {
        "group": Column("str", nullable=False, unique=True),
        "main_tag": Column("str", nullable=False),
        "n_steps_bin": Column("int", nullable=False, min=0),
        "n_observed": Column("int", nullable=False, min=1),
        "median": Column("float", nullable=False),
        "used": Column("bool", nullable=False),
    },
    min_rows=1,
)

#: object name in the ``missing`` stage -> schema (pass as ``schemas=`` to ArtifactStore.save).
MISSING_OUTPUT_SCHEMA: Mapping[str, Schema] = {
    "missing_rates": MISSING_RATES_SCHEMA,
    "related_issues": QUALITY_ISSUES_SCHEMA,
    "rating_zero_review_stats": RATING_ZERO_REVIEW_STATS_SCHEMA,
    "rating_zero_examples": RATING_ZERO_EXAMPLES_SCHEMA,
    "minutes_zero_summary": MINUTES_ZERO_SUMMARY_SCHEMA,
    "minutes_zero_tags": MINUTES_ZERO_TAGS_SCHEMA,
    "minutes_zero_n_steps": MINUTES_ZERO_N_STEPS_SCHEMA,
    "minutes_zero_examples": MINUTES_ZERO_EXAMPLES_SCHEMA,
    "main_tag_coverage": MAIN_TAG_COVERAGE_SCHEMA,
    "imputer_groups": IMPUTER_GROUPS_SCHEMA,
}

#: Interactions after RatingZeroToNaN: rating is float, 0 no longer allowed.
INTERACTIONS_AFTER_MISSING_SCHEMA = LOADED_INTERACTIONS_SCHEMA.derive(
    "interactions_after_missing",
    add={"rating": Column("float", min=1.0, max=5.0)},
)

#: Recipes after ZeroAsMissing + GroupMedianImputer(minutes) + mark_has_rating.
RECIPES_AFTER_MISSING_SCHEMA = LOADED_RECIPES_SCHEMA.derive(
    "recipes_after_missing",
    add={
        "minutes": Column("float", nullable=False, min=0.0),
        "minutes_was_missing": Column("bool", nullable=False),
        "has_rating": Column("bool", nullable=False),
    },
)


# --------------------------------------------------------------------------- 1.4 outliers
# Tables of the `outliers` stage (recsys.cleaning.outliers analysis functions) and the contract of
# recipes after the outlier rules (used by the 1.9 pipeline).

OUTLIER_IQR_SCHEMA = Schema(
    "outliers_iqr",
    {
        "column": Column("str", nullable=False),
        "scale": Column("str", nullable=False, allowed=frozenset({"raw", "log1p"})),
        "n": Column("int", nullable=False, min=0),
        "q1": Column("float"),
        "median": Column("float"),
        "q3": Column("float"),
        "iqr": Column("float", min=0.0),
        "lower_fence": Column("float"),
        "upper_fence": Column("float"),
        "whisker_low": Column("float"),
        "whisker_high": Column("float"),
        "n_below": Column("int", nullable=False, min=0),
        "n_above": Column("int", nullable=False, min=0),
        "share_outside": Column("float", min=0.0, max=1.0),
        "min": Column("float"),
        "max": Column("float"),
    },
    unique_together=(("column", "scale"),),
    min_rows=1,
)

OUTLIER_QUANTILES_SCHEMA = Schema(
    "outliers_quantiles",
    {
        "column": Column("str", nullable=False),
        "quantile": Column("float", nullable=False, min=0.0, max=1.0),
        "value": Column("float"),
    },
    unique_together=(("column", "quantile"),),
    min_rows=1,
)

OUTLIER_THRESHOLDS_SCHEMA = Schema(
    "outliers_thresholds",
    {
        "column": Column("str", nullable=False),
        "threshold": Column("float", nullable=False),
        "n_above": Column("int", nullable=False, min=0),
        "share_above": Column("float", min=0.0, max=1.0),
    },
    unique_together=(("column", "threshold"),),
    min_rows=1,
)

OUTLIER_EXAMPLES_SCHEMA = Schema(
    "outliers_extreme_examples",
    {
        "id": Column("int", nullable=False),
        "name": Column("str"),
        "column": Column("str", nullable=False),
        "value": Column("float", nullable=False),
        "n_steps": Column("int", nullable=False, min=0),
        "n_ingredients": Column("int", nullable=False, min=0),
    },
    unique_together=(("column", "id"),),
)

OUTLIER_INVALID_COUNTS_SCHEMA = Schema(
    "outliers_invalid_counts",
    {
        "column": Column("str", nullable=False, unique=True),
        "lower": Column("float"),
        "upper": Column("float"),
        "n_below": Column("int", nullable=False, min=0),
        "n_above": Column("int", nullable=False, min=0),
        "n_invalid": Column("int", nullable=False, min=0),
        "share_invalid": Column("float", min=0.0, max=1.0),
    },
    min_rows=1,
)

OUTLIER_CLIP_BOUNDS_SCHEMA = Schema(
    "outliers_clip_bounds",
    {
        "column": Column("str", nullable=False, unique=True),
        "lower_q": Column("float", nullable=False, min=0.0, max=1.0),
        "upper_q": Column("float", nullable=False, min=0.0, max=1.0),
        "lower": Column("float", nullable=False),
        "upper": Column("float", nullable=False),
        "n_clipped_low": Column("int", nullable=False, min=0),
        "n_clipped_high": Column("int", nullable=False, min=0),
    },
    min_rows=1,
)

OUTLIER_BEFORE_AFTER_SCHEMA = Schema(
    "outliers_before_after",
    {
        "column": Column("str", nullable=False),
        "stage": Column("str", nullable=False),
        "n": Column("int", nullable=False, min=0),
        "n_missing": Column("int", nullable=False, min=0),
        "min": Column("float"),
        "whisker_low": Column("float"),
        "q1": Column("float"),
        "median": Column("float"),
        "q3": Column("float"),
        "whisker_high": Column("float"),
        "max": Column("float"),
        "mean": Column("float"),
        "skew": Column("float"),
        "n_outside_fences": Column("int", nullable=False, min=0),
    },
    unique_together=(("column", "stage"),),
    min_rows=1,
)

#: object name in the ``outliers`` stage -> schema (pass as ``schemas=`` to ArtifactStore.save).
OUTLIERS_OUTPUT_SCHEMA: Mapping[str, Schema] = {
    "iqr": OUTLIER_IQR_SCHEMA,
    "quantiles": OUTLIER_QUANTILES_SCHEMA,
    "minutes_thresholds": OUTLIER_THRESHOLDS_SCHEMA,
    "extreme_examples": OUTLIER_EXAMPLES_SCHEMA,
    "invalid_counts": OUTLIER_INVALID_COUNTS_SCHEMA,
    "clip_bounds": OUTLIER_CLIP_BOUNDS_SCHEMA,
    "before_after": OUTLIER_BEFORE_AFTER_SCHEMA,
    "related_issues": QUALITY_ISSUES_SCHEMA,
}

#: Upper bound of a valid ``minutes`` value (1 year); mirrors config ``outliers.invalid_rule``.
MINUTES_VALID_MAX = 525600

#: Recipes after InvalidValueRule + QuantileClipper (original scale, before log1p): value ranges
#: of the treated columns are checked; NaN marks values removed by the rules (imputed later).
RECIPES_AFTER_OUTLIERS_SCHEMA = LOADED_RECIPES_SCHEMA.derive(
    "recipes_after_outliers",
    add={
        "minutes": Column("float", min=0.0, max=float(MINUTES_VALID_MAX)),
        "n_steps": Column("float", min=1.0),
        "n_ingredients": Column("float", min=1.0),
        **{c: Column("float", min=0.0) for c in NUTRITION_COLUMNS},
    },
)


# --------------------------------------------------------------------------- 1.9 splits
# Tables of the `splits` stage and the checks every split must pass before it is written.

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

#: One row per interaction: which set it belongs to. ``row`` = positional row of the split input
#: (the `loaded` interactions named in the manifest).
SPLITS_SCHEMA = Schema(
    "splits",
    {
        "row": Column("int", nullable=False, unique=True, min=0),
        "user_id": Column("int", nullable=False),
        "recipe_id": Column("int", nullable=False),
        "split": Column("str", nullable=False, allowed=frozenset(SPLIT_NAMES)),
    },
    min_rows=1,
)

STRATIFICATION_LABELS_SCHEMA = Schema(
    "splits_stratification_labels",
    {
        "recipe_id": Column("int", nullable=False, unique=True),
        "label": Column("str", nullable=False),
        "split": Column("str", nullable=False, allowed=frozenset(SPLIT_NAMES)),
    },
    min_rows=1,
)

RECIPE_TARGETS_SCHEMA = Schema(
    "recipe_targets",
    {
        "recipe_id": Column("int", nullable=False, unique=True),
        "avg_rating": Column("float", nullable=False, min=1.0, max=5.0),
        "n_ratings": Column("int", nullable=False, min=1),
        "rating_class": Column("str", nullable=False),
    },
    min_rows=1,
)

SPLIT_SUMMARY_SCHEMA = Schema(
    "splits_summary",
    {
        "strategy": Column("str", nullable=False),
        "split": Column("str", nullable=False, allowed=frozenset(SPLIT_NAMES)),
        "n_interactions": Column("int", nullable=False, min=0),
        "share_interactions": Column("float", min=0.0, max=1.0),
        "n_recipes": Column("int", nullable=False, min=0),
        "n_users": Column("int", nullable=False, min=0),
        "n_rated": Column("int", nullable=False, min=0),
    },
    unique_together=(("strategy", "split"),),
    min_rows=1,
)

CLASS_BALANCE_SCHEMA = Schema(
    "splits_class_balance",
    {
        "split": Column("str", nullable=False, allowed=frozenset(SPLIT_NAMES)),
        "label": Column("str", nullable=False),
        "n_recipes": Column("int", nullable=False, min=0),
        "share": Column("float", nullable=False, min=0.0, max=1.0),
    },
    unique_together=(("split", "label"),),
    min_rows=1,
)

DEV_RECIPES_SCHEMA = Schema(
    "splits_dev_recipes",
    {
        "id": Column("int", nullable=False, unique=True),
        "label": Column("str", nullable=False),
    },
    min_rows=1,
)

RESAMPLING_SUMMARY_SCHEMA = Schema(
    "splits_resampling_summary",
    {
        "method": Column("str", nullable=False),
        "label": Column("str", nullable=False),
        "n_rows": Column("int", nullable=False, min=0),
        "class_weight": Column("float", min=0.0),
    },
    unique_together=(("method", "label"),),
    min_rows=1,
)

#: Catalogue after the fitted 1.9 pipeline: no missing numeric value is left, bins are set.
#: Value ranges are not fixed here because the scaler (1.8) decides the final scale.
RECIPE_FEATURES_SCHEMA = LOADED_RECIPES_SCHEMA.derive(
    "recipe_features",
    add={
        **{
            c: Column("float", nullable=False)
            for c in ("minutes", "n_steps", "n_ingredients", *NUTRITION_COLUMNS)
        },
        "minutes_was_missing": Column("bool", nullable=False),
        "n_steps_was_missing": Column("bool", nullable=False),
        "n_ingredients_was_missing": Column("bool", nullable=False),
        "minutes_bin": Column("category", nullable=False),
        "calories_bin": Column("category", nullable=False),
        "ingredients_clean": Column("list", nullable=False),
        "ingredients_filtered": Column("list", nullable=False),
    },
)

#: object name in the ``splits`` stage -> schema (pass as ``schemas=`` to ArtifactStore.save).
SPLITS_OUTPUT_SCHEMA: Mapping[str, Schema] = {
    "recipe_features": RECIPE_FEATURES_SCHEMA,
    "grouped_assignments": SPLITS_SCHEMA,
    "temporal_assignments": SPLITS_SCHEMA,
    "stratification_labels": STRATIFICATION_LABELS_SCHEMA,
    "targets_train": RECIPE_TARGETS_SCHEMA,
    "targets_val": RECIPE_TARGETS_SCHEMA,
    "targets_test": RECIPE_TARGETS_SCHEMA,
    "split_summary": SPLIT_SUMMARY_SCHEMA,
    "class_balance": CLASS_BALANCE_SCHEMA,
    "dev_recipes": DEV_RECIPES_SCHEMA,
    "resampling_summary": RESAMPLING_SUMMARY_SCHEMA,
}


def assert_rows_preserved(original: pd.DataFrame, frames: Mapping[str, pd.DataFrame]) -> None:
    """Raise unless ``frames`` partition ``original``: same total rows, every row exactly once.

    Rows are identified by index (splitters keep the positional row of their input).
    """
    violations: list[str] = []
    total = sum(len(f) for f in frames.values())
    if total != len(original):
        violations.append(f"{total} rows in {list(frames)} but {len(original)} in the input")
    indexes = [f.index for f in frames.values()]
    index = indexes[0].append(indexes[1:]) if indexes else pd.Index([])
    if index.has_duplicates:
        violations.append(f"{int(index.duplicated().sum())} rows appear in more than one set")
    lost = original.index.difference(index)
    if len(lost):
        violations.append(f"{len(lost)} input rows are in no set (e.g. {_examples(lost)})")
    if violations:
        raise SchemaValidationError("rows_preserved", violations)


def assert_temporal_order(
    frames: Mapping[str, pd.DataFrame], user_col: str = "user_id", date_col: str = "date"
) -> None:
    """Raise if, for any user, a later set starts before an earlier set ends.

    ``frames`` is ordered (train, val, test); equal dates are allowed at a boundary.
    """
    violations: list[str] = []
    names = list(frames)
    for earlier, later in zip(names, names[1:], strict=False):
        last = frames[earlier].groupby(user_col)[date_col].max()
        first = frames[later].groupby(user_col)[date_col].min()
        both = last.index.intersection(first.index)
        bad = both[(last.loc[both] > first.loc[both]).to_numpy()]
        if len(bad):
            violations.append(
                f"{len(bad)} users have {earlier!r} interactions after their first {later!r} "
                f"interaction (e.g. {_examples(bad)})"
            )
    if violations:
        raise SchemaValidationError("temporal_order", violations)


def assert_class_proportions(
    labels: pd.DataFrame, tolerance: float, label_col: str = "label", split_col: str = "split"
) -> None:
    """Raise if a class share differs between any two sets by more than ``tolerance``."""
    shares = (
        labels.groupby(split_col)[label_col].value_counts(normalize=True).unstack(fill_value=0.0)
    )
    spread = shares.max(axis=0) - shares.min(axis=0)
    bad = spread[spread > tolerance]
    if len(bad):
        raise SchemaValidationError(
            "class_proportions",
            [f"{c}: share differs by {v:.4f} > {tolerance} between sets" for c, v in bad.items()],
        )


# --------------------------------------------------------------------------- 1.10 kNN prototype
# Tables of the `knn_prototype` stage. Only train / val may appear: the test set is never scored
# in Phase 1 (recsys.evaluation.metrics.check_split).

_EVAL_SPLITS = frozenset({"train", "val"})

KNN_CLASSIFICATION_METRICS_SCHEMA = Schema(
    "knn_classification_metrics",
    {
        "model": Column("str", nullable=False, unique=True),
        "split": Column("str", nullable=False, allowed=_EVAL_SPLITS),
        "n_recipes": Column("int", nullable=False, min=1),
        "accuracy": Column("float", nullable=False, min=0.0, max=1.0),
        "balanced_accuracy": Column("float", nullable=False, min=0.0, max=1.0),
        "macro_f1": Column("float", nullable=False, min=0.0, max=1.0),
    },
    min_rows=1,
)

KNN_REGRESSION_METRICS_SCHEMA = Schema(
    "knn_regression_metrics",
    {
        "model": Column("str", nullable=False, unique=True),
        "split": Column("str", nullable=False, allowed=_EVAL_SPLITS),
        "n_recipes": Column("int", nullable=False, min=1),
        "mae": Column("float", nullable=False, min=0.0),
        "rmse": Column("float", nullable=False, min=0.0),
    },
    min_rows=1,
)

KNN_CONFUSION_SCHEMA = Schema(
    "knn_confusion_matrices",
    {
        "model": Column("str", nullable=False),
        "true_label": Column("str", nullable=False),
        "pred_label": Column("str", nullable=False),
        "count": Column("int", nullable=False, min=0),
    },
    unique_together=(("model", "true_label", "pred_label"),),
    min_rows=1,
)

KNN_EXAMPLE_RECOMMENDATIONS_SCHEMA = Schema(
    "knn_example_recommendations",
    {
        "query_id": Column("str", nullable=False),
        "query": Column("str", nullable=False),
        "filters": Column("str", nullable=False),
        "rank": Column("int", nullable=False, min=1),
        "recipe_id": Column("int", nullable=False),
        "name": Column("str", nullable=False),
        "similarity": Column("float", nullable=False, min=0.0, max=1.0),
        "predicted_rating": Column("float", nullable=False, min=1.0, max=5.0),
        "rating_is_estimated": Column("bool", nullable=False),
        "minutes": Column("float"),
        "minutes_bin": Column("str", nullable=False),
        "matched_ingredients": Column("str", nullable=False),
        "unknown_ingredients": Column("str", nullable=False),
    },
    unique_together=(("query_id", "rank"),),
)

#: object name in the ``knn_prototype`` stage -> schema.
KNN_PROTOTYPE_OUTPUT_SCHEMA: Mapping[str, Schema] = {
    "classification_metrics": KNN_CLASSIFICATION_METRICS_SCHEMA,
    "regression_metrics": KNN_REGRESSION_METRICS_SCHEMA,
    "confusion_matrices": KNN_CONFUSION_SCHEMA,
    "example_recommendations": KNN_EXAMPLE_RECOMMENDATIONS_SCHEMA,
}


# --------------------------------------------------------------------------- 1.12-A O3
# Output tables of stages 1.2 / 1.5 / 1.6 / 1.7 / 1.8 that were written without a schema.
# Each ``*_STAGE_SCHEMAS`` maps the object name in the stage to its schema, ready to pass as
# ``ArtifactStore.save(..., schemas=...)``.

_PCT = Column("float", nullable=False, min=0.0, max=100.0)
_COUNT = Column("int", nullable=False, min=0)

DQR_CONTINUOUS_SCHEMA = Schema(
    "dqr_continuous",
    {
        "Feature": Column("str", nullable=False, unique=True),
        "Count": _COUNT,
        "% Missing": _PCT,
        "Cardinality": _COUNT,
        **{
            k: Column("float")
            for k in ("Min", "Q1", "Mean", "Median", "Q3", "Max", "Std Dev", "Skewness", "Kurtosis")
        },
    },
    min_rows=1,
)

DQR_CATEGORICAL_SCHEMA = Schema(
    "dqr_categorical",
    {
        "Feature": Column("str", nullable=False, unique=True),
        "Count": _COUNT,
        "% Missing": _PCT,
        "Cardinality": _COUNT,
        "Mode": Column("str"),
        "Mode Freq": _COUNT,
        "Mode %": _PCT,
        "2nd Mode": Column("str"),
        "2nd Mode Freq": _COUNT,
        "2nd Mode %": _PCT,
    },
    min_rows=1,
)

DQR_LISTS_SCHEMA = Schema(
    "dqr_lists",
    {
        "Feature": Column("str", nullable=False, unique=True),
        "Count": _COUNT,
        "% Missing": _PCT,
        "Unique Items": _COUNT,
        "Min Len": _COUNT,
        "Median Len": Column("float", min=0.0),
        "Mean Len": Column("float", min=0.0),
        "Max Len": _COUNT,
    },
    min_rows=1,
)

#: object name in the ``quality`` stage -> schema.
QUALITY_STAGE_SCHEMAS: Mapping[str, Schema] = {
    "dqr_continuous": DQR_CONTINUOUS_SCHEMA,
    "dqr_categorical": DQR_CATEGORICAL_SCHEMA,
    "dqr_lists": DQR_LISTS_SCHEMA,
    "issues": QUALITY_ISSUES_SCHEMA,
}

NORMALIZATION_SUMMARY_SCHEMA = Schema(
    "normalization_summary",
    {
        "raw_unique_ingredients": _COUNT,
        "clean_unique_ingredients": _COUNT,
        "reduction_count": Column("int", nullable=False),
        "reduction_pct": Column("float", nullable=False, max=100.0),
    },
    min_rows=1,
)

MERGED_VARIANTS_SCHEMA = Schema(
    "merged_variants",
    {
        "canonical": Column("str", nullable=False, unique=True),
        "variant_count": Column("int", nullable=False, min=2),
        "examples": Column("str", nullable=False),
    },
)

RARE_THRESHOLD_COVERAGE_SCHEMA = Schema(
    "rare_threshold_coverage",
    {
        "min_count": Column("int", nullable=False, min=1, unique=True),
        "vocab_size": _COUNT,
        "vocab_retained_pct": _PCT,
        "occurrence_coverage_pct": _PCT,
    },
    min_rows=1,
)

#: object name in the ``ingredients`` stage -> schema (tables only).
INGREDIENTS_STAGE_SCHEMAS: Mapping[str, Schema] = {
    "recipes_ingredients": INGREDIENTS_OUTPUT_SCHEMA,
    "normalization_summary": NORMALIZATION_SUMMARY_SCHEMA,
    "merged_variants": MERGED_VARIANTS_SCHEMA,
    "rare_threshold_coverage": RARE_THRESHOLD_COVERAGE_SCHEMA,
}

DISTRIBUTION_SUMMARY_SCHEMA = Schema(
    "distribution_summary",
    {
        "feature": Column("str", nullable=False, unique=True),
        "distribution_type": Column(
            "str",
            nullable=False,
            allowed={
                "normal",
                "right-skewed",
                "left-skewed",
                "long-tail",
                "uniform",
                "multimodal",
                "constant",
            },
        ),
        "skewness": Column("float"),
        "kurtosis": Column("float"),
        "p99_to_median_ratio": Column("float"),
    },
    min_rows=1,
)

COLD_START_SUMMARY_SCHEMA = Schema(
    "cold_start_summary",
    {
        "entity": Column("str", nullable=False, unique=True, allowed={"user", "recipe"}),
        "total_unique": _COUNT,
        "single_review_count": _COUNT,
        "single_review_pct": _PCT,
        "median_reviews": Column("float", nullable=False, min=1.0),
        "max_reviews": Column("int", nullable=False, min=1),
    },
    min_rows=2,
)

#: object name in the ``figures`` stage -> schema (tables only).
FIGURES_STAGE_SCHEMAS: Mapping[str, Schema] = {
    "distribution_summary": DISTRIBUTION_SUMMARY_SCHEMA,
    "cold_start_summary": COLD_START_SUMMARY_SCHEMA,
}

BINNING_COMPARISON_SCHEMA = Schema(
    "binning_comparison",
    {
        "column": Column("str", nullable=False),
        "strategy": Column("str", nullable=False),
        "bin": Column("str", nullable=False),
        "count": _COUNT,
        "pct": _PCT,
    },
    unique_together=(("strategy", "bin"),),
    min_rows=1,
)

#: object name in the ``binning`` stage -> schema (tables only).
BINNING_STAGE_SCHEMAS: Mapping[str, Schema] = {
    "minutes_comparison": BINNING_COMPARISON_SCHEMA,
    "avg_rating_comparison": BINNING_COMPARISON_SCHEMA,
    "calories_comparison": BINNING_COMPARISON_SCHEMA,
}

SCALER_SUMMARY_SCHEMA = Schema(
    "scaler_summary",
    {
        "feature": Column("str", nullable=False),
        "transform": Column("str", nullable=False),
        **{k: Column("float", nullable=False) for k in ("min", "median", "mean")},
        "std": Column("float", nullable=False, min=0.0),
        "max": Column("float", nullable=False),
    },
    unique_together=(("feature", "transform"),),
    min_rows=1,
)


def combined_distance_sample_schema(n: int) -> Schema:
    """Square ``n`` x ``n`` table of combined distances (columns recipe_0 .. recipe_{n-1})."""
    return Schema(
        f"combined_distance_sample_{n}",
        {f"recipe_{i}": Column("float", nullable=False, min=0.0, max=1.0) for i in range(n)},
        min_rows=n,
    )
