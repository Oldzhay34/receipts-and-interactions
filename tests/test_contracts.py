from __future__ import annotations

import pandas as pd
import pytest

from recsys.contracts import (
    LOADED_RECIPES_SCHEMA,
    Column,
    Schema,
    SchemaValidationError,
    assert_disjoint,
    validate,
)

SCHEMA = Schema(
    name="toy",
    columns={
        "id": Column("int", nullable=False, unique=True, min=0),
        "score": Column("float", min=0.0, max=1.0),
        "label": Column("str", allowed={"a", "b"}),
        "items": Column("list"),
    },
    unique_together=(("id", "label"),),
)


def good() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "score": [0.1, None, 1.0],
            "label": ["a", "b", None],
            "items": [["x"], [], None],
        }
    )


def test_valid_frame_passes_and_is_returned() -> None:
    df = good()
    assert validate(df, SCHEMA) is df


def test_all_violations_collected_in_one_error() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 1, -5],
            "score": [0.5, 2.0, -1.0],
            "label": ["a", "a", "z"],
            "items": [["x"], "notalist", None],
            "extra": [0, 0, 0],
        }
    )
    with pytest.raises(SchemaValidationError) as exc:
        validate(df, SCHEMA)
    msgs = "\n".join(exc.value.violations)
    assert "unexpected columns: ['extra']" in msgs
    assert "id: 1 values below min" in msgs
    assert "id: 2 rows with duplicate values" in msgs
    assert "score: 1 values above max" in msgs
    assert "score: 1 values below min" in msgs
    assert "label: 1 values outside allowed set" in msgs
    assert "items: 1 non-list values" in msgs
    assert "['id', 'label']" in msgs
    assert len(exc.value.violations) >= 7


def test_missing_column_and_nullability() -> None:
    df = good().drop(columns="items")
    df.loc[0, "id"] = None  # makes id float with NaN -> dtype violation
    with pytest.raises(SchemaValidationError) as exc:
        validate(df, SCHEMA)
    msgs = "\n".join(exc.value.violations)
    assert "missing columns: ['items']" in msgs
    assert "id: expected int" in msgs


def test_non_nullable_detected() -> None:
    s = Schema("t", {"x": Column("float", nullable=False)})
    with pytest.raises(SchemaValidationError, match="1 missing values"):
        validate(pd.DataFrame({"x": [1.0, None]}), s)


def test_str_dtype_rejects_numbers_in_object_column() -> None:
    s = Schema("t", {"x": Column("str")})
    with pytest.raises(SchemaValidationError, match="expected str values"):
        validate(pd.DataFrame({"x": ["a", 3]}, dtype=object), s)


def test_derive_does_not_modify_original() -> None:
    derived = SCHEMA.derive("toy2", add={"new": Column("bool")}, drop=["items"])
    assert "items" in SCHEMA.columns and "new" not in SCHEMA.columns
    assert list(derived.columns) == ["id", "score", "label", "new"]
    with pytest.raises(KeyError):
        SCHEMA.derive("bad", drop=["nope"])


def test_schema_to_dict_is_json_friendly() -> None:
    d = LOADED_RECIPES_SCHEMA.to_dict()
    assert d["columns"]["id"] == {"dtype": "int", "nullable": False, "min": 0, "unique": True}


def test_assert_disjoint() -> None:
    a = pd.DataFrame({"recipe_id": [1, 2]})
    b = pd.DataFrame({"recipe_id": [3]})
    assert_disjoint({"train": a, "test": b}, "recipe_id")
    c = pd.DataFrame({"recipe_id": [2, 4]})
    with pytest.raises(SchemaValidationError, match="shared by 'train' and 'val'"):
        assert_disjoint({"train": a, "val": c, "test": b}, "recipe_id")


def test_validate_requires_dataframe() -> None:
    with pytest.raises(TypeError):
        validate([1, 2], SCHEMA)  # type: ignore[arg-type]
