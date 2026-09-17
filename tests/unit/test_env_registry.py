# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit coverage for the torch_fl/_env.py registry and the reference doc.

`docs/reference/environment-variables.md` is the only description of the
environment surface a user sees, and `torch_fl/_env.py` is the only place a value
is read. Nothing used to connect the two, so the doc could name a variable no code
read, or omit one that code required, and no check noticed -- that is how it ended
up documenting `FLAGOS_USE_FLAGGEMS` and `FLAGOS_USE_FLAGGEMS_CPP` as live long
after both stopped being read. The first two tests here are that connection: the
owned section of the doc and `VARIABLES` describe the same set with the same scope
and default, and the retired table and `RETIRED` describe the same set too.

The second thing pinned here is that a retired name is really gone. "Dropped
silently" is a promise that a stale `FLAGOS_LOG_DISPATCH` export does nothing at
all, which only holds while no reader exists -- and a reader is easy to
reintroduce from a half-finished revert. The last test reads the tree and fails if
one appears, in any language, in a read position.

The module is loaded by path rather than imported. `torch_fl/_env.py` is
stdlib-only by design so that this file needs no torch install, and importing the
package would execute `torch_fl/__init__.py`.
"""

import importlib.machinery
import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "reference" / "environment-variables.md"

# A variable that defaults ON, so a test can tell "read as off" apart from
# "unset, so the default was returned".
DEFAULTS_ON = "FLAGOS_ALIAS_CUDA"


def _load_env():
    loader = importlib.machinery.SourceFileLoader(
        "torch_fl_env_under_test", str(REPO_ROOT / "torch_fl" / "_env.py")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_env = _load_env()


# ---------------------------------------------------------------------------
# The doc tables and the registry
# ---------------------------------------------------------------------------


def _section(heading: str) -> str:
    """The markdown between `heading` and the next top-level heading."""
    text = DOC.read_text(encoding="utf-8")
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end == -1 else text[start:end]


def _table_rows(markdown: str) -> list[list[str]]:
    """Cells of every table row, with the header separator dropped."""
    rows = []
    for line in markdown.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue  # |-------|---------|
        rows.append(cells)
    return rows


def _plain(cell: str) -> str:
    """A table cell with its code spans unwrapped, so it compares to the registry."""
    return cell.replace("`", "").strip()


def _documented_owned() -> dict[str, tuple[str, str]]:
    """name -> (scope, default) for every row of the owned-variables tables."""
    owned = _section("## Owned variables")
    documented = {}
    for cells in _table_rows(owned):
        name = cells[0]
        if not name.startswith("`") or len(cells) != 4:
            continue
        documented[_plain(name)] = (_plain(cells[1]), _plain(cells[2]))
    return documented


def test_owned_doc_table_matches_the_registry():
    """Every declared variable is documented, and with the registry's own wording.

    The purpose column is deliberately not compared -- it is prose, and the doc is
    allowed to say more than the registry does. Scope and default are not prose:
    they are what a reader acts on, so they come from the module.
    """
    documented = _documented_owned()
    declared = set(_env.VARIABLES)

    assert set(documented) == declared, (
        f"documented but not declared: {sorted(set(documented) - declared)}; "
        f"declared but not documented: {sorted(declared - set(documented))}"
    )

    wrong = {
        name: (documented[name], (scope, default))
        for name, (scope, default, _purpose) in _env.VARIABLES.items()
        if documented[name] != (scope, default)
    }
    assert not wrong, f"doc (scope, default) != registry for: {wrong}"


def test_retired_doc_table_matches_the_retired_set():
    """The retired table and RETIRED are the same set of names.

    Both directions matter. A name in the table but not the set is one
    check_environment() would warn about as unknown; a name in the set but not the
    table is one nothing tells the user about.
    """
    documented = set()
    for cells in _table_rows(_section("## Retired names")):
        if cells[0].startswith("`"):
            documented.update(re.findall(r"`([^`]+)`", cells[0]))

    assert documented == set(_env.RETIRED), (
        f"documented but not retired: {sorted(documented - set(_env.RETIRED))}; "
        f"retired but not documented: {sorted(set(_env.RETIRED) - documented)}"
    )


def test_every_registered_name_has_a_scope_and_a_default():
    """The registry is the doc's source of truth, so no field may be blank."""
    for name, fields in _env.VARIABLES.items():
        assert len(fields) == 3, f"{name} has {len(fields)} fields, expected 3"
        scope, default, purpose = fields
        assert scope in {
            _env.SCOPE_BUILD,
            _env.SCOPE_RUNTIME,
            _env.SCOPE_BUILD_RUNTIME,
            _env.SCOPE_CODEGEN,
            _env.SCOPE_TEST,
        }, f"{name} has an unknown scope {scope!r}"
        assert default, f"{name} has an empty default"
        assert purpose, f"{name} has an empty purpose"


# ---------------------------------------------------------------------------
# The truth table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["1", "true", "on", "yes", "TRUE", "On", "YeS"])
def test_truthy_spellings_are_on(monkeypatch, raw):
    monkeypatch.setenv(DEFAULTS_ON, raw)
    assert _env.flag(DEFAULTS_ON, default=False) is True


@pytest.mark.parametrize("raw", ["0", "false", "off", "no", "FALSE", "Off", " nO "])
def test_falsy_spellings_are_off(monkeypatch, raw):
    # default=True, so a pass cannot come from the default being used.
    monkeypatch.setenv(DEFAULTS_ON, raw)
    assert _env.flag(DEFAULTS_ON, default=True) is False


def test_an_unknown_value_warns_once_and_uses_the_default(monkeypatch, capsys):
    monkeypatch.setenv(DEFAULTS_ON, "2")
    for _ in range(3):
        assert _env.flag(DEFAULTS_ON, default=False) is False
    assert _env.flag(DEFAULTS_ON, default=True) is True

    warnings = [ln for ln in capsys.readouterr().err.splitlines() if ln]
    assert len(warnings) == 1, warnings
    assert warnings[0] == (
        f"[flagos] not a boolean; using the default ({DEFAULTS_ON}='2')"
    )


def test_empty_is_unset_rather_than_off(monkeypatch):
    """An empty value is the default, which is what a shell idiom produces."""
    monkeypatch.setenv(DEFAULTS_ON, "")
    assert _env.flag(DEFAULTS_ON, default=True) is True
    assert _env.flag(DEFAULTS_ON, default=False) is False


def test_value_treats_empty_as_unset(monkeypatch):
    monkeypatch.setenv("FLAGOS_BPU_MARCH", "")
    assert _env.value("FLAGOS_BPU_MARCH", "nash-p") == "nash-p"
    monkeypatch.setenv("FLAGOS_BPU_MARCH", " nash-e ")
    assert _env.value("FLAGOS_BPU_MARCH", "nash-p") == "nash-e"


def test_choice_names_the_alternatives(monkeypatch, capsys):
    allowed = ("flaggems", "vendor", "tileops")
    monkeypatch.setenv("FLAGOS_FORCE_BACKEND", "Vendor")
    assert _env.choice("FLAGOS_FORCE_BACKEND", allowed, "flaggems") == "vendor"

    monkeypatch.setenv("FLAGOS_FORCE_BACKEND", "flagems")
    assert _env.choice("FLAGOS_FORCE_BACKEND", allowed, "vendor") == "vendor"
    assert "expected one of flaggems, vendor, tileops" in capsys.readouterr().err


def test_listed_splits_a_comma_separated_switch(monkeypatch):
    monkeypatch.setenv("FLAGOS_LOG", " Dispatch , fallback ,,")
    assert _env.listed("FLAGOS_LOG") == frozenset({"dispatch", "fallback"})

    monkeypatch.setenv("FLAGOS_LOG", "")
    assert _env.listed("FLAGOS_LOG") == frozenset()


def test_retired_names_are_not_reported_as_unknown(monkeypatch, capsys):
    """A stale export of a retired name must stay silent, not warn.

    check_environment() is the only thing that would notice, and it is called at
    the end of _env.py, so calling it again here is the same code path.
    """
    monkeypatch.setenv("FLAGOS_LOG_DISPATCH", "1")
    _env.check_environment()
    assert capsys.readouterr().err == ""


def test_unknown_flagged_names_are_reported(monkeypatch, capsys):
    monkeypatch.setenv("FLAGOS_LOG_DISPACH", "1")
    _env.check_environment()
    err = capsys.readouterr().err
    assert "FLAGOS_LOG_DISPACH" in err and "not a torch_fl variable" in err


def test_the_dynamic_op_family_is_not_reported(monkeypatch, capsys):
    monkeypatch.setenv("FLAGOS_OP_mm__out", "flaggems_cpp")
    _env.check_environment()
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# Retired means no reader
# ---------------------------------------------------------------------------

# Scanned for readers. docs/ is excluded on purpose: the retired table, the
# integration guides and the dated measurement logs are allowed to name what was
# removed, and tests/ is scanned despite the prose because a real read would most
# plausibly be reintroduced there.
_SCANNED_ROOTS = (
    "torch_fl",
    "csrc",
    "scripts",
    "benchmarks",
    "tests",
    ".github/scripts",
    "setup.py",
    "CMakeLists.txt",
)
_SCANNED_SUFFIXES = frozenset(
    {".py", ".c", ".cc", ".cpp", ".cu", ".h", ".hh", ".hpp", ".inc", ".cmake", ".sh"}
)

# Files that name the retired set by definition rather than by reading it.
_EXEMPT = {
    "torch_fl/_env.py",
    "tests/unit/test_env_registry.py",
}

# What a read looks like. The point is that a retired name and one of these on the
# same code line is a reader, whatever the language:
#   Python      os.environ["X"] / os.environ.get("X") / _env.flag("X") / os.getenv("X")
#   C++         std::getenv("X") / EnvFlag("X", ...) / EnvValue("X") / EnvChoice("X")
#   shell       ${X} / "$X"
#   CMake       ${X}
_READ_TOKENS = (
    "environ",
    "getenv",
    "_env.",
    "EnvFlag(",
    "EnvValue(",
    "EnvChoice(",
    "EnvListed(",
    "$",
)


def _scanned_files():
    for root in _SCANNED_ROOTS:
        path = REPO_ROOT / root
        if path.is_file():
            yield path
            continue
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file():
                continue
            if "__pycache__" in candidate.parts:
                continue
            if candidate.suffix not in _SCANNED_SUFFIXES:
                continue
            yield candidate


def _code_lines(path: Path):
    """Lines that can execute, paired with their number.

    Whole-line comments are skipped. Prose is the one thing the scan cannot judge
    -- `# condition=lambda: os.environ.get("FLAGOS_USE_FLAGGEMS") == "1"` explains
    what was removed -- and a comment cannot read anything.
    """
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith(("#", "//", "*", "/*")):
            continue
        yield number, line


def test_retired_names_have_no_reader_anywhere():
    """No retired name is read from the environment in any source file.

    Dropping the old spellings without an alias or a deprecation warning is only
    honest while this holds: an exported `VENDOR_KERNEL=1` must enable nothing.
    """
    patterns = {name: re.compile(rf"\b{re.escape(name)}\b") for name in _env.RETIRED}
    offenders = []
    for path in _scanned_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in _EXEMPT:
            continue
        for number, line in _code_lines(path):
            if not any(token in line for token in _READ_TOKENS):
                continue
            for name, pattern in patterns.items():
                if pattern.search(line):
                    offenders.append(
                        f"{relative}:{number}: reads {name}: {line.strip()}"
                    )
    assert not offenders, "retired names are still read:\n" + "\n".join(offenders)
