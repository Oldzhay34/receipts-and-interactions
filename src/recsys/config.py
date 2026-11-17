"""Configuration loading: config/config.yaml + the DATA_ROOT environment variable.

Every path the pipeline uses comes from here. Code never hard-codes paths.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DATA_ROOT_ENV = "DATA_ROOT"
CONFIG_ENV = "RECSYS_CONFIG"

#: Repository root (src/recsys/config.py -> parents[2]); valid for editable installs and Colab.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

_REPO_RELATIVE_KEYS = frozenset({"local_sample_dir"})


class ConfigError(RuntimeError):
    """Raised when the configuration or DATA_ROOT is missing or invalid."""


@dataclass(frozen=True)
class Config:
    """Resolved project configuration.

    Attributes:
        data_root: Absolute DATA_ROOT directory (shared Google Drive folder on Colab).
        paths: Every entry of the yaml ``paths`` block resolved to an absolute Path.
        random_state: Global seed.
        split: train/val/test ratios.
        stages: Allowed stage names (= folder names under ``outputs/``).
        raw: The full parsed yaml, for blocks added by later stages (``cfg.section("outliers")``).
    """

    data_root: Path
    paths: Mapping[str, Path]
    random_state: int
    split: Mapping[str, float]
    stages: tuple[str, ...]
    raw: Mapping[str, Any] = field(repr=False)

    @property
    def outputs_dir(self) -> Path:
        return self.paths["outputs_dir"]

    def section(self, name: str) -> Any:
        """Return a top-level yaml block; raise ConfigError if it does not exist."""
        if name not in self.raw:
            raise ConfigError(f"config.yaml has no top-level block {name!r}")
        return self.raw[name]


def load_config(
    path: str | os.PathLike[str] | None = None,
    data_root: str | os.PathLike[str] | None = None,
) -> Config:
    """Load config.yaml and inject DATA_ROOT.

    Args:
        path: yaml path. Defaults to $RECSYS_CONFIG, then ``<repo>/config/config.yaml``.
        data_root: Overrides the DATA_ROOT environment variable (useful in tests).

    Raises:
        ConfigError: if DATA_ROOT is unset/empty, the yaml is missing, or required keys are absent
            or inconsistent (split ratios must sum to 1).
    """
    cfg_path = Path(path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH)
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")

    root = data_root if data_root is not None else os.environ.get(DATA_ROOT_ENV)
    if not root:
        raise ConfigError(
            f"{DATA_ROOT_ENV} is not set. On Colab: os.environ['{DATA_ROOT_ENV}'] = "
            "'/content/drive/MyDrive/CME4403/data'. Locally: point it at your synced Drive folder "
            "(or a temporary directory for tests)."
        )
    root_path = Path(root).expanduser().resolve()

    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    missing = [k for k in ("paths", "random_state", "split", "stages") if k not in raw]
    if missing:
        raise ConfigError(f"{cfg_path} is missing required keys: {missing}")

    split = {k: float(v) for k, v in raw["split"].items()}
    if set(split) != {"train", "val", "test"}:
        raise ConfigError(f"split must define exactly train/val/test, got {sorted(split)}")
    if abs(sum(split.values()) - 1.0) > 1e-9:
        raise ConfigError(f"split ratios must sum to 1.0, got {sum(split.values())}")

    paths = {
        key: (REPO_ROOT / rel if key in _REPO_RELATIVE_KEYS else root_path / rel)
        for key, rel in raw["paths"].items()
    }
    return Config(
        data_root=root_path,
        paths=paths,
        random_state=int(raw["random_state"]),
        split=split,
        stages=tuple(raw["stages"]),
        raw=raw,
    )
