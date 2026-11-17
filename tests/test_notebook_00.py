"""Runs notebooks/00_setup.ipynb locally on the fixtures (everything except the Colab cell).

Evidence for deliverable "00_setup writes loaded and sample stages". The full-data run is done
by the user on Colab.
"""

from __future__ import annotations

import json

from recsys.config import REPO_ROOT, Config
from recsys.io import ArtifactStore

NOTEBOOK = REPO_ROOT / "notebooks" / "00_setup.ipynb"


def _code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_first_cell_is_standard_colab_cell() -> None:
    first = _code_cells()[0]
    for needle in ("drive.mount", "git clone", "pip install -q -e .", "DATA_ROOT", "load_config()"):
        assert needle in first


def test_notebook_has_no_business_logic_definitions() -> None:
    for src in _code_cells()[1:]:
        assert "def " not in src and "class " not in src


def test_notebook_runs_on_fixtures(cfg: Config) -> None:
    ns: dict[str, object] = {"CFG": cfg}
    for src in _code_cells()[1:]:
        exec(compile(src, str(NOTEBOOK), "exec"), ns)  # noqa: S102 - our own notebook
    store = ArtifactStore.from_config(cfg)
    assert store.latest("loaded") == "v1" and store.latest("sample") == "v1"
    loaded = store.load("loaded")
    assert loaded["recipes"].shape[0] == 20 and loaded["interactions"].shape[0] == 62
    sample = store.load("sample")
    assert sample.manifest["inputs"][0]["stage"] == "loaded"
    assert {i["kind"] for i in loaded.manifest["inputs"]} == {"file"}
