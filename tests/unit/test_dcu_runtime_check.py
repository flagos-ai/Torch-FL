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

"""Unit coverage for the DCU decoupled-runtime post-import checks."""

import pytest
from dcu_module_loader import load_module

check = load_module(
    "torch_fl/accelerator/dcu/_dcu_runtime_check.py", "_flagos_test_dcu_runtime_check"
)


@pytest.mark.parametrize(
    "version,expected",
    [
        ("2.10.0+cpu", "2.10.0"),
        ("2.10.0+das.opt1.dtk2604", "2.10.0"),
        ("2.10.0", "2.10.0"),
        ("2.9.1+cu128", "2.9.1"),
    ],
)
def test_base_version_strips_the_local_label(version, expected):
    assert check.base_version(version) == expected


def test_vendor_version_is_read_from_the_bundle(tmp_path):
    (tmp_path / "vendor_version.py").write_text(
        "__version__ = '2.10.0+das.opt1.dtk2604'\n"
        "debug = False\n"
        "hip: str = '6.3.26113'\n",
        encoding="utf-8",
    )
    assert check.bundled_vendor_version(str(tmp_path)) == "2.10.0+das.opt1.dtk2604"


def test_vendor_version_handles_the_annotated_form(tmp_path):
    """Newer torch writes ``__version__: str = '...'``."""
    (tmp_path / "vendor_version.py").write_text(
        "__version__: str = '2.10.0+das.dtk2604'\n", encoding="utf-8"
    )
    assert check.bundled_vendor_version(str(tmp_path)) == "2.10.0+das.dtk2604"


def test_vendor_version_is_none_without_a_bundle(tmp_path):
    assert check.bundled_vendor_version(str(tmp_path / "absent")) is None


def test_matching_versions_pass():
    check.check_version_alignment("2.10.0+cpu", "2.10.0+das.opt1.dtk2604")


def test_absent_vendor_version_is_not_an_error():
    """Source checkout with no bundle: the preload was a no-op too."""
    check.check_version_alignment("2.10.0+cpu", None)


def test_minor_mismatch_names_the_required_wheel():
    with pytest.raises(RuntimeError) as excinfo:
        check.check_version_alignment("2.9.1+cpu", "2.10.0+das.opt1.dtk2604")
    message = str(excinfo.value)
    assert "torch==2.10.0" in message
    assert "FLAGOS_DCU_SKIP_RUNTIME_CHECK=1" in message


def test_patch_mismatch_is_also_rejected():
    """dlopen accepts it -- the symbol names match -- so this must be caught here."""
    with pytest.raises(RuntimeError, match="torch==2.10.1"):
        check.check_version_alignment("2.10.0+cpu", "2.10.1+das.dtk2604")


def test_dispatch_check_passes_when_every_op_has_a_cuda_kernel():
    check.check_cuda_dispatch(lambda op: True)


def test_dispatch_check_reports_the_missing_ops():
    absent = {"aten::mm", "aten::bmm"}
    with pytest.raises(RuntimeError) as excinfo:
        check.check_cuda_dispatch(lambda op: op not in absent)
    message = str(excinfo.value)
    assert "aten::mm" in message and "aten::bmm" in message
    assert "aten::_softmax" not in message
    # The actionable half: the usual cause is import order.
    assert "before `import torch`" in message


def test_dispatch_check_covers_blas_elementwise_and_reduction():
    """A single-op probe would miss a partially registered libtorch_hip."""
    assert set(check.REQUIRED_CUDA_OPS) == {
        "aten::mm",
        "aten::add.Tensor",
        "aten::_softmax",
        "aten::bmm",
    }


def test_validate_runs_both_checks(monkeypatch, tmp_path):
    (tmp_path / "vendor_version.py").write_text(
        "__version__ = '2.10.0+das.dtk2604'\n", encoding="utf-8"
    )
    seen = []
    monkeypatch.setattr(
        check,
        "check_version_alignment",
        lambda torch_version, vendor: seen.append(("version", vendor)),
    )
    monkeypatch.setattr(
        check, "check_cuda_dispatch", lambda has_kernel: seen.append(("dispatch",))
    )
    check.validate_decoupled_runtime(str(tmp_path))
    assert seen == [("version", "2.10.0+das.dtk2604"), ("dispatch",)]
