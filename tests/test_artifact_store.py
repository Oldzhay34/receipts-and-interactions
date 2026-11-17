from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from recsys.contracts import (  # noqa: E402
    LOADED_RECIPES_SCHEMA,
    Column,
    Schema,
    SchemaValidationError,
)
from recsys.io import (  # noqa: E402
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactStore,
    describe_file,
    read_artifact_dir,
    sha256_file,
)

TOY = Schema("toy", {"a": Column("int", nullable=False)})


def df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3]})


def stage_entries(store: ArtifactStore, stage: str) -> list[str]:
    d = store.root / stage
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


# ----------------------------------------------------------------- round trip


def test_round_trip_all_kinds(store: ArtifactStore, fixture_recipes: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.bar(["a", "b"], [1, 2])
    objects = {
        "recipes": fixture_recipes,
        "matrix": sp.csr_matrix(np.eye(3)),
        "vocab": {"garlic": 0, "salt": 1},
        "notes": "hello",
        "figure": fig,
        "model": np.arange(4),  # ndarray -> joblib
    }
    saved = store.save(
        "loaded", objects, params={"k": 3}, schemas={"recipes": LOADED_RECIPES_SCHEMA}
    )
    plt.close(fig)
    assert saved.version == "v1"
    loaded = store.load("loaded")
    pd.testing.assert_frame_equal(loaded["recipes"], fixture_recipes)
    assert (loaded["matrix"] != objects["matrix"]).nnz == 0
    assert loaded["vocab"] == {"garlic": 0, "salt": 1}
    assert loaded["notes"] == "hello"
    assert isinstance(loaded["figure"], Path) and loaded["figure"].suffix == ".png"
    np.testing.assert_array_equal(loaded["model"], np.arange(4))


def test_list_columns_come_back_as_lists(store: ArtifactStore, fixture_recipes) -> None:
    store.save("loaded", {"recipes": fixture_recipes})
    back = store.load("loaded")["recipes"]
    assert back["ingredients"].map(type).eq(list).all()
    assert back.set_index("id").loc[110, "steps"] == []


def test_manifest_contents(store: ArtifactStore, data_root: Path) -> None:
    raw = describe_file(data_root / "raw" / "RAW_recipes.csv")
    first = store.save("loaded", {"t": df()}, inputs=[raw], params={"p": 1, "path": data_root})
    second = store.save("sample", {"t": df()}, inputs=[first], params={"n": 2})
    m = json.loads((second.path / "manifest.json").read_text(encoding="utf-8"))
    assert m["stage"] == "sample" and m["version"] == "v1"
    assert m["params"] == {"n": 2}
    assert m["inputs"] == [first.ref()]
    assert m["inputs"][0]["manifest_sha256"] == sha256_file(first.path / "manifest.json")
    entry = m["files"]["t"]
    assert entry["n_rows"] == 3 and entry["n_cols"] == 1
    assert entry["sha256"] == sha256_file(second.path / entry["file"])
    assert set(m["git"]) == {"commit", "dirty"}
    assert m["created_utc"].endswith("+00:00")
    m1 = first.manifest
    assert m1["inputs"][0]["kind"] == "file" and m1["inputs"][0]["sha256"] == raw["sha256"]
    assert m1["params"]["path"] == str(data_root)  # Path serialised


# ----------------------------------------------------------------- versions / no overwrite


def test_versions_increment_and_latest(store: ArtifactStore) -> None:
    a = store.save("loaded", {"t": df()})
    v1_hash = sha256_file(a.path / "t.parquet")
    b = store.save("loaded", {"t": df().assign(a=[7, 8, 9])})
    assert (a.version, b.version) == ("v1", "v2")
    assert store.versions("loaded") == ["v1", "v2"]
    assert store.latest("loaded") == "v2"
    assert (store.root / "loaded" / "LATEST").read_text(encoding="utf-8").strip() == "v2"
    assert sha256_file(a.path / "t.parquet") == v1_hash  # v1 untouched
    assert store.load("loaded", "v1")["t"]["a"].tolist() == [1, 2, 3]
    assert store.load("loaded")["t"]["a"].tolist() == [7, 8, 9]


def test_refuses_to_overwrite_existing_version(store: ArtifactStore, monkeypatch) -> None:
    a = store.save("loaded", {"t": df()})
    before = sha256_file(a.path / "t.parquet")
    monkeypatch.setattr(store, "_next_version", lambda stage: "v1")
    with pytest.raises(ArtifactError, match="Refusing to overwrite"):
        store.save("loaded", {"t": df().assign(a=[0, 0, 0])})
    assert sha256_file(a.path / "t.parquet") == before
    assert stage_entries(store, "loaded") == ["LATEST", "v1"]  # no temp folder left


# ----------------------------------------------------------------- atomicity


def test_failure_mid_write_leaves_no_version(store: ArtifactStore) -> None:
    objects = {"first": df(), "bad": lambda x: x}  # lambda cannot be pickled -> fails 2nd write
    with pytest.raises(Exception):  # noqa: B017 - pickling error type varies by Python version
        store.save("loaded", objects)
    assert stage_entries(store, "loaded") == []  # no vN, no .tmp_*, no LATEST
    assert store.latest("loaded") is None


def test_failure_keeps_previous_latest(store: ArtifactStore) -> None:
    store.save("loaded", {"t": df()})
    with pytest.raises(Exception):  # noqa: B017
        store.save("loaded", {"t": df(), "bad": lambda x: x})
    assert stage_entries(store, "loaded") == ["LATEST", "v1"]
    assert store.latest("loaded") == "v1"


def test_schema_violation_writes_nothing(store: ArtifactStore) -> None:
    with pytest.raises(SchemaValidationError):
        store.save("loaded", {"t": pd.DataFrame({"a": [1.5]})}, schemas={"t": TOY})
    assert not (store.root / "loaded").exists()


def test_failure_during_manifest_verification_cleans_up(store: ArtifactStore, monkeypatch) -> None:
    import recsys.io.artifact_store as mod

    def boom(folder, manifest):
        raise mod.ArtifactIntegrityError("simulated partial write")

    monkeypatch.setattr(mod, "_verify_dir", boom)
    with pytest.raises(ArtifactIntegrityError, match="simulated"):
        store.save("loaded", {"t": df()})
    assert stage_entries(store, "loaded") == []


# ----------------------------------------------------------------- integrity on load


def test_tampered_file_detected(store: ArtifactStore) -> None:
    a = store.save("loaded", {"t": df()})
    f = a.path / "t.parquet"
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(ArtifactIntegrityError, match="sha256 mismatch"):
        store.load("loaded")


def test_missing_file_detected(store: ArtifactStore) -> None:
    a = store.save("loaded", {"t": df(), "u": df()})
    (a.path / "u.parquet").unlink()
    with pytest.raises(ArtifactIntegrityError, match="Missing file"):
        store.load("loaded")
    assert store.load("loaded", names=["t"])["t"].shape == (3, 1)  # selective load still works


# ----------------------------------------------------------------- validation of requests


def test_unknown_stage_rejected(store: ArtifactStore) -> None:
    with pytest.raises(ArtifactError, match="Unknown stage"):
        store.save("not_a_stage", {"t": df()})


@pytest.mark.parametrize("name", ["../escape", "a/b", "", ".hidden"])
def test_bad_object_names_rejected(store: ArtifactStore, name: str) -> None:
    with pytest.raises(ArtifactError, match="Invalid object name"):
        store.save("loaded", {name: df()})


def test_load_without_versions_raises(store: ArtifactStore) -> None:
    with pytest.raises(ArtifactError, match="no saved versions"):
        store.load("loaded")


def test_bad_inputs_and_params_rejected(store: ArtifactStore) -> None:
    with pytest.raises(TypeError, match="inputs must be"):
        store.save("loaded", {"t": df()}, inputs=["loaded"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="JSON"):
        store.save("loaded", {"t": df()}, params={"f": object()})
    assert not (store.root / "loaded").exists()


def test_export_and_read_dir(store: ArtifactStore, tmp_path: Path) -> None:
    store.save("sample", {"t": df()})
    dest = tmp_path / "data_sample"
    dest.mkdir()
    (dest / ".gitkeep").touch()
    store.export("sample", dest)
    art = read_artifact_dir(dest)
    assert art.stage == "sample" and art["t"]["a"].tolist() == [1, 2, 3]
    with pytest.raises(ArtifactError, match="already contains"):
        store.export("sample", dest)
