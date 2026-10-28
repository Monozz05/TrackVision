import importlib

import pytest

MODULES = [
    "trackvision",
    "trackvision.config",
    "trackvision.repro",
    "trackvision.cli",
    "trackvision.io",
    "trackvision.geometry",
    "trackvision.tracking",
    "trackvision.predict",
    "trackvision.inject",
    "trackvision.metrics",
    "trackvision.stats",
    "trackvision.experiments",
    "trackvision.experiments.smoke",
    "trackvision.viz",
    "trackvision.platform",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} has no docstring stating its responsibility"


def test_package_is_imported_from_this_repository_src():
    """Guards against testing a stale installed copy instead of the editable source."""
    from pathlib import Path

    import trackvision
    from trackvision.config import REPO_ROOT

    expected = (REPO_ROOT / "src" / "trackvision" / "__init__.py").resolve()
    assert Path(trackvision.__file__).resolve() == expected
