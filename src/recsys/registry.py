"""Name -> strategy-class registry.

Parallel-developed strategies register themselves by name; the pipeline (1.9) builds them from
config without importing concrete classes::

    from recsys.registry import register, create

    @register("imputer", "group_median")
    class GroupMedianImputer(BaseEstimator, TransformerMixin): ...

    imputer = create("imputer", "group_median", target_col="minutes")

Registration happens at import time. If a name is not found, ``create``/``get`` import every
``recsys`` submodule once (:func:`discover`) and retry, so no shared "plugin list" file has to be
edited for each new strategy.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T", bound=type)

_REGISTRY: dict[str, dict[str, type]] = {}
_discovered = False


class RegistryError(KeyError):
    """Unknown kind/name, or a name registered twice for different classes."""

    def __str__(self) -> str:  # KeyError quotes its message; keep it readable.
        return str(self.args[0]) if self.args else ""


def register(kind: str, name: str) -> Callable[[T], T]:
    """Class decorator registering ``cls`` under ``(kind, name)``.

    Re-registering the *same* class (same module + qualname, e.g. after a notebook reload) is
    allowed and replaces the entry. Registering a different class under a taken name raises.
    """

    def decorator(cls: T) -> T:
        bucket = _REGISTRY.setdefault(kind, {})
        existing = bucket.get(name)
        if existing is not None and _qualname(existing) != _qualname(cls):
            raise RegistryError(
                f"{kind}/{name} is already registered to {_qualname(existing)}; "
                f"cannot register {_qualname(cls)}"
            )
        bucket[name] = cls
        return cls

    return decorator


def get(kind: str, name: str) -> type:
    """Return the class registered under ``(kind, name)``."""
    cls = _REGISTRY.get(kind, {}).get(name)
    if cls is None:
        discover()
        cls = _REGISTRY.get(kind, {}).get(name)
    if cls is None:
        raise RegistryError(
            f"No {kind!r} named {name!r}. Available: {sorted(_REGISTRY.get(kind, {}))}"
        )
    return cls


def create(kind: str, name: str, **params: Any) -> Any:
    """Instantiate the class registered under ``(kind, name)`` with ``params``."""
    return get(kind, name)(**params)


def available(kind: str | None = None) -> dict[str, list[str]]:
    """Registered names, per kind (after discovery)."""
    discover()
    kinds = [kind] if kind is not None else sorted(_REGISTRY)
    return {k: sorted(_REGISTRY.get(k, {})) for k in kinds}


def discover(package: str = "recsys") -> None:
    """Import every ``.py`` module under ``package`` once so their ``@register`` decorators run.

    Walks the filesystem instead of using ``pkgutil``, so subfolders without ``__init__.py``
    (implicit namespace packages) are discovered too. Import errors propagate: a broken module
    must fail loudly rather than silently vanish from the registry.
    """
    global _discovered
    if _discovered and package == "recsys":
        return
    pkg = importlib.import_module(package)
    for base in map(Path, pkg.__path__):
        for py in sorted(base.rglob("*.py")):
            rel = py.relative_to(base).with_suffix("")
            parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
            if not parts or any(not p.isidentifier() for p in parts):
                continue
            importlib.import_module(".".join((package, *parts)))
    if package == "recsys":
        _discovered = True


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"
