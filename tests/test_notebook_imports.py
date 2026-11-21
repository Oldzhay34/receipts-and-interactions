"""Every name a notebook imports from ``recsys`` must exist (catches renamed / lost symbols)."""

from __future__ import annotations

import ast
import importlib
import json

import pytest

from recsys.config import REPO_ROOT

NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))


def _recsys_imports(path) -> list[tuple[str, str]]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    found = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # drop IPython magics / shell lines, which are not Python syntax
        lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("recsys"):
                found.extend((node.module, alias.name) for alias in node.names)
    return found


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_recsys_imports_exist(notebook) -> None:
    missing = [
        f"{module}.{name}"
        for module, name in _recsys_imports(notebook)
        if not hasattr(importlib.import_module(module), name)
    ]
    assert not missing, f"{notebook.name} imports names that do not exist: {missing}"
