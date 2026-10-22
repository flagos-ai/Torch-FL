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

"""The emulated-hbdk4 environment: stub discovery, library path, preload.

These are the three things that make hbdk4 importable under box64 (see
docs/vendors/bpu/integration.md). They are pure path logic, so they are tested against a fake
directory layout rather than a real x86 install -- which also means these tests
run on any platform, not only on a board that has hbdk4 set up.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from torch_fl.accelerator.bpu import compiler as C


@pytest.fixture
def fake_x86(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A directory tree shaped like scripts/vendor/setup_bpu_hbdk4.sh produces."""
    root = tmp_path / "hbdk4-x86"
    py = root / "python" / "bin" / "python3.11"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/false\n")
    libs = (
        root
        / "python"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "hbdk4"
        / "compiler"
        / "_mlir_libs"
    )
    libs.mkdir(parents=True)
    (libs / "libhbtl.so").write_bytes(b"")
    (root / "stubs" / "numba").mkdir(parents=True)

    monkeypatch.setattr(C, "X86_PYTHON", str(py))
    monkeypatch.setattr(C, "X86_STUBS", "")
    return root


def test_stub_dir_found_next_to_the_x86_python(fake_x86: Path):
    # The default location is derived from the interpreter path so that a
    # setup_bpu_hbdk4.sh install needs no extra configuration.
    assert C._stub_dir() == str(fake_x86 / "stubs")


def test_stub_dir_is_none_without_an_x86_python(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(C, "X86_PYTHON", "")
    monkeypatch.setattr(C, "X86_STUBS", "")
    assert C._stub_dir() is None


def test_explicit_stub_dir_wins(
    fake_x86: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setattr(C, "X86_STUBS", str(other))
    assert C._stub_dir() == str(other)


def test_explicit_stub_dir_that_does_not_exist_is_rejected(
    fake_x86: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(C, "X86_STUBS", str(tmp_path / "nope"))
    assert C._stub_dir() is None


def test_mlir_libs_dir_is_discovered(fake_x86: Path):
    found = C._mlir_libs_dir()
    assert found is not None
    # box64 needs this on its own search path, and libhbtl.so must be here for
    # the RTLD_GLOBAL preload to work.
    assert Path(found).name == "_mlir_libs"
    assert (Path(found) / "libhbtl.so").exists()


def test_env_carries_library_path_and_stubs(fake_x86: Path):
    env = C.x86_env()
    assert env["BOX64_LD_LIBRARY_PATH"] == C._mlir_libs_dir()
    assert env["PYTHONPATH"] == str(fake_x86 / "stubs")


def test_env_appends_stubs_rather_than_replacing_pythonpath(
    fake_x86: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PYTHONPATH", "/pre/existing")
    env = C.x86_env()
    parts = env["PYTHONPATH"].split(":")
    assert parts[0] == "/pre/existing"
    # Appended, never prepended: a real numba or torch in the guest's
    # site-packages must win over the stubs.
    assert parts[-1] == str(fake_x86 / "stubs")


def test_env_preserves_an_existing_box64_library_path(
    fake_x86: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("BOX64_LD_LIBRARY_PATH", "/opt/other")
    env = C.x86_env()
    assert env["BOX64_LD_LIBRARY_PATH"].split(":") == [
        C._mlir_libs_dir(),
        "/opt/other",
    ]


def test_driver_preloads_libhbtl():
    # Without this preload _hbdk.so fails to relocate: it needs hbtl symbols
    # that it does not list in DT_NEEDED. See _mlir_libs_dir().
    assert "libhbtl.so" in C._X86_DRIVER
    assert "RTLD_GLOBAL" in C._X86_DRIVER


def test_driver_tolerates_post_compile_validation_failure():
    # hbdk4's compile() loads the artifact back through hbrt4, which can fail
    # with AllocError under emulation after the .hbm is already written.
    assert "getsize(out_path) > 0" in C._X86_DRIVER


def test_emulator_probe_is_skipped_without_an_x86_python(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(C, "X86_PYTHON", "")
    monkeypatch.setattr(C, "_emulator", C._UNSET)
    assert C.x86_emulator(refresh=True) is None
