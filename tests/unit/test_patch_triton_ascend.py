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

"""Unit coverage for scripts/vendor/patch_triton_ascend.py flag stripping.

The rest of that script is exact-string replacement against triton-ascend's
source, which cannot be tested without the package installed. These two
functions can, and they are the ones that decide whether a kernel compiles:
USE_TORCH_NPU is what compiles the at_npu::native::OpCommand body in
npu_utils.cpp, so leaving it defined without torch_npu present fails the JIT
compile with "'at_npu' has not been declared".

That is not hypothetical. CI moved from triton-ascend 3.2.0 to 3.2.2, the
exact-string patterns stopped matching, the script still exited 0, and 11
Ascend operator tests failed 20 minutes later on precisely that error.

Run: pytest tests/unit/test_patch_triton_ascend.py -v
"""

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "vendor" / "patch_triton_ascend.py"


def _load():
    spec = importlib.util.spec_from_file_location("patch_triton_ascend", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch_triton_ascend = _load()


@pytest.fixture
def fake_triton(tmp_path):
    """A triton tree with only the three files the functions touch."""
    backend = tmp_path / "backends" / "ascend"
    backend.mkdir(parents=True)
    return tmp_path, backend


def test_strips_every_flag_spelling(fake_triton):
    """Plain, f-prefixed, single-quoted and comma-less entries all go."""
    root, backend = fake_triton
    (backend / "utils.py").write_text(
        "FLAGS = [\n"
        '    "-std=c++17",\n'
        '    "-ltorch_npu",\n'
        '    f"-DUSE_TORCH_NPU",\n'
        "    '-ltorch_npu',\n"
        '    "-DUSE_TORCH_NPU"\n'
        '    "-shared",\n'
        "]\n"
    )

    assert patch_triton_ascend.strip_torch_npu_build_flags(str(root)) is True

    result = (backend / "utils.py").read_text()
    assert "torch_npu" not in result
    # Unrelated flags must survive -- the regex matches the flag, not the list.
    assert '"-std=c++17",' in result
    assert '"-shared",' in result


def test_strip_is_idempotent(fake_triton):
    """A second run finds nothing to do.

    triton-ascend gets reinstalled (pip --force-reinstall wipes the patch), so
    this script is expected to run repeatedly against the same tree.
    """
    root, backend = fake_triton
    (backend / "driver.py").write_text('FLAGS = [\n    "-ltorch_npu",\n]\n')

    assert patch_triton_ascend.strip_torch_npu_build_flags(str(root)) is True
    assert patch_triton_ascend.strip_torch_npu_build_flags(str(root)) is False


def test_missing_files_are_tolerated(tmp_path):
    """A tree without the backend files is a no-op, not a crash."""
    (tmp_path / "backends" / "ascend").mkdir(parents=True)
    assert patch_triton_ascend.strip_torch_npu_build_flags(str(tmp_path)) is False


def test_verify_passes_when_flags_are_gone(fake_triton):
    root, backend = fake_triton
    (backend / "utils.py").write_text('FLAGS = ["-shared"]\n')
    patch_triton_ascend.verify_no_torch_npu_build_flags(str(root))


def test_verify_ignores_commented_flags(fake_triton):
    """The exact-string patches comment flags out; that is a patched state."""
    root, backend = fake_triton
    (backend / "utils.py").write_text(
        '        # "-ltorch_npu",  # removed for torch_fl\n'
    )
    patch_triton_ascend.verify_no_torch_npu_build_flags(str(root))


def test_verify_exits_nonzero_on_surviving_flag(fake_triton):
    """The postcondition CI was missing: fail loudly instead of exiting 0.

    Uses a shape the regex deliberately cannot strip (the flag is not alone on
    its line), which is exactly the case where silent success is dangerous.
    """
    root, backend = fake_triton
    (backend / "driver.py").write_text(
        'FLAGS = ["-x"] + (["-ltorch_npu"] if use_npu else [])\n'
    )

    with pytest.raises(SystemExit) as excinfo:
        patch_triton_ascend.verify_no_torch_npu_build_flags(str(root))
    assert excinfo.value.code == 1
