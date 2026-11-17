from __future__ import annotations

import pytest

from recsys import registry
from recsys.registry import RegistryError, available, create, get, register


@pytest.fixture(autouse=True)
def clean_kind():
    yield
    registry._REGISTRY.pop("test_kind", None)


def test_register_and_create_with_params() -> None:
    @register("test_kind", "doubler")
    class Doubler:
        def __init__(self, factor: int = 2) -> None:
            self.factor = factor

    assert get("test_kind", "doubler") is Doubler
    obj = create("test_kind", "doubler", factor=3)
    assert isinstance(obj, Doubler) and obj.factor == 3
    assert "doubler" in available("test_kind")["test_kind"]


def test_duplicate_name_for_different_class_rejected() -> None:
    @register("test_kind", "x")
    class A:
        pass

    with pytest.raises(RegistryError, match="already registered"):

        @register("test_kind", "x")
        class B:  # noqa: F841 - different qualname
            pass


def test_reregistering_same_class_is_allowed() -> None:
    class Same:
        pass

    register("test_kind", "same")(Same)
    register("test_kind", "same")(Same)  # e.g. notebook reload
    assert get("test_kind", "same") is Same


def test_unknown_name_lists_available() -> None:
    @register("test_kind", "known")
    class K:
        pass

    with pytest.raises(RegistryError, match=r"Available: \['known'\]"):
        create("test_kind", "unknown")


def test_discover_imports_whole_package_without_error() -> None:
    registry.discover()  # must not raise on any recsys module
