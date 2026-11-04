"""
The package data ships every native source the package compiles at runtime (`torch.utils.cpp_extension`). A source left
out is a `FileNotFoundError` on a wheel or sdist install alone, where the kernel falls back to Triton with a warning, so
neither an editable install nor the rest of this suite can see it. MEASURED 2026-11-04: it happened twice,
`tri_math.cuh` and then 23 sources of three packages, whose 19 modules all failed to load from an installed wheel.

Reads `pyproject.toml` rather than building a wheel: the declaration is what the build follows.
"""

import fnmatch
import pathlib
import re

import pytest

try:
    import tomllib
except ModuleNotFoundError:                                            # (Python 3.10)
    tomllib = pytest.importorskip("tomli")

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
NATIVE = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)

pytestmark = pytest.mark.skipif(not ((ROOT / "pyproject.toml").is_file() and (SRC / "pyjuice").is_dir()),
                                reason = "needs the source tree")


def _sources():
    sources = sorted(p for p in (SRC / "pyjuice").rglob("*") if p.is_file() and p.suffix in NATIVE)
    assert sources, "no native sources found: the test would pass on nothing"
    return sources


def _package_data():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["setuptools"]["package-data"]


def _ships(path, package_data):
    package = ".".join(path.parent.relative_to(SRC).parts)
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in package_data.get(package, ()))


def test_every_native_source_is_package_data():
    data = _package_data()
    missing = [str(p.relative_to(ROOT)) for p in _sources() if not _ships(p, data)]
    assert not missing, f"not shipped by [tool.setuptools.package-data]: {missing}"


def test_every_local_include_is_a_shipped_file():
    """The quoted includes, from the including file's directory, as the compiler finds them from an installed tree."""
    data = _package_data()
    bad = []
    for source in _sources():
        for name in INCLUDE.findall(source.read_text(errors = "replace")):
            target = (source.parent / name).resolve()
            if not target.is_file() or not _ships(target, data):
                bad.append(f"{source.relative_to(ROOT)} -> {name}")
    assert not bad, f"includes that an installed tree would not have: {bad}"
