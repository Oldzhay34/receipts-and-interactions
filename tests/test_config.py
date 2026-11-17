from __future__ import annotations

from pathlib import Path

import pytest

from recsys.config import DATA_ROOT_ENV, REPO_ROOT, ConfigError, load_config


def test_missing_data_root_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    with pytest.raises(ConfigError, match="DATA_ROOT is not set"):
        load_config()


def test_empty_data_root_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, "")
    with pytest.raises(ConfigError, match="DATA_ROOT"):
        load_config()


def test_paths_resolved_against_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    cfg = load_config()
    assert cfg.data_root == tmp_path.resolve()
    assert cfg.paths["raw_recipes"] == tmp_path.resolve() / "raw" / "RAW_recipes.csv"
    assert cfg.outputs_dir == tmp_path.resolve() / "outputs"
    # repo-relative key
    assert cfg.paths["local_sample_dir"] == REPO_ROOT / "data" / "sample"


def test_core_values(tmp_path: Path) -> None:
    cfg = load_config(data_root=tmp_path)
    assert cfg.random_state == 42
    assert cfg.split == {"train": 0.70, "val": 0.15, "test": 0.15}
    assert {"loaded", "sample"} <= set(cfg.stages)
    assert cfg.section("sample")["n_recipes"] == 5000
    with pytest.raises(ConfigError):
        cfg.section("does_not_exist")


def test_bad_split_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "paths: {outputs_dir: outputs}\nrandom_state: 1\nstages: [a]\n"
        "split: {train: 0.8, val: 0.1, test: 0.2}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(bad, data_root=tmp_path)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", data_root=tmp_path)
