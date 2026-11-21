"""Notebooks 05-08 run end to end on the fixtures (Colab cell skipped) and save their stage."""

from __future__ import annotations

import json
import warnings

import matplotlib
import pytest

from recsys.config import REPO_ROOT, Config
from recsys.io import ArtifactStore

matplotlib.use("Agg")

CASES = [
    ("05_ingredients.ipynb", "ingredients"),
    ("06_visualizations.ipynb", "figures"),
    ("07_binning.ipynb", "binning"),
    ("08_normalization.ipynb", "scaling"),
]


def _code_cells(name: str) -> list[str]:
    nb = json.loads((REPO_ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize(("name", "stage"), CASES, ids=[c[0] for c in CASES])
def test_notebook_runs_on_fixtures(
    name: str, stage: str, cfg: Config, fixture_recipes, fixture_interactions
) -> None:
    import matplotlib.pyplot as plt

    store = ArtifactStore.from_config(cfg)
    store.save("loaded", {"recipes": fixture_recipes, "interactions": fixture_interactions})
    ns: dict[str, object] = {"CFG": cfg, "display": lambda *a, **k: None}
    cells = _code_cells(name)
    assert 'BRANCH   = "main"' in cells[0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for src in cells[1:]:
                exec(compile(src, name, "exec"), ns)  # noqa: S102 - our own notebook
    finally:
        plt.close("all")
    art = store.load(stage)
    assert art.version == "v1"
    assert [i["stage"] for i in art.manifest["inputs"]][0] == "loaded"
    if stage == "ingredients":
        assert "tfidf_idf" in art.keys()
        assert art.manifest["files"]["recipes_ingredients"]["schema"] == "ingredients_output"
    if stage == "figures":
        assert art.manifest["params"]["input_stage_used"] == "loaded"
    if stage == "scaling":
        assert art.manifest["params"]["scaling_config"] == cfg.section("scaling")
