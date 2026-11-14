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

"""Saved experimental artifacts must not bypass provider/integrity admission."""

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture
def compiler():
    path = Path(__file__).resolve().parents[2] / "torch_fl/host_compiler/__init__.py"
    spec = importlib.util.spec_from_file_location("host_admission_under_test", path)
    module = importlib.util.module_from_spec(spec)
    # Admission only needs the registration endpoint; keep these tests usable
    # in the shared CPU CI lane where PyTorch itself is not installed.
    torch = SimpleNamespace(ops=SimpleNamespace(flagos=SimpleNamespace()))
    with patch.dict(sys.modules, {"torch": torch}):
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "failure", ["source", "binary", "checksum", "no_launch", "missing"]
)
def test_reject_saved_artifact_before_native_loading(compiler, tmp_path, failure):
    source = tmp_path / "host.cpp"
    library = tmp_path / "host.so"
    source.write_text("launchKernel(original_mul, grid, stream);")
    library.write_bytes(b"original_mul")
    artifact = {
        "key": "flag_gems.mul",
        "library": str(library),
        "outputs": 1,
        "kernels": ["original_mul"],
    }
    if failure == "source":
        source.write_text("launchKernel(triton_poi_fused_clone_0, grid, stream);")
    elif failure == "binary":
        library.write_bytes(b"original_mul\0triton_poi_fused_clone_0")
    elif failure == "checksum":
        artifact["library_sha256"] = "incorrect"
    elif failure == "no_launch":
        source.write_text("// Lost the numerical launch.")
    elif failure == "missing":
        library.unlink()
    with patch.object(
        compiler.torch.ops.flagos, "host_program_register", create=True
    ) as register:
        compiler._register(artifact, tmp_path / "artifact.json")
        register.assert_not_called()
    assert (tmp_path / "admission_failure.txt").exists()


def test_accept_original_kernel_and_matching_library_hash(compiler, tmp_path):
    source = tmp_path / "host.cpp"
    library = tmp_path / "host.so"
    source.write_text("launchKernel(original_mul_0, grid, stream);")
    library.write_bytes(b"original_mul")
    artifact = {
        "key": "flag_gems.mul",
        "library": str(library),
        "outputs": 1,
        "kernels": ["original_mul"],
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    with patch.object(
        compiler.torch.ops.flagos, "host_program_register", create=True
    ) as register:
        compiler._register(artifact, tmp_path / "artifact.json")
        register.assert_called_once_with(
            "flag_gems.mul", str(library), 1, library.stat().st_size
        )


def test_resident_budget_rejection_preserves_the_original_route(compiler, tmp_path):
    (tmp_path / "host.cpp").write_text("// Metadata-only host.")
    library = tmp_path / "host.so"
    library.write_bytes(b"metadata")
    artifact = {"key": "slice", "library": str(library), "outputs": 1, "kernels": []}
    with patch.object(
        compiler.torch.ops.flagos,
        "host_program_register",
        create=True,
        side_effect=RuntimeError("resident budget exhausted"),
    ):
        compiler._register(artifact, tmp_path / "artifact.json")
    assert (
        "resident budget exhausted" in (tmp_path / "admission_failure.txt").read_text()
    )


def test_observer_failure_does_not_invalidate_a_successful_call(compiler):
    compiler._mode = "compile"
    with patch.object(
        compiler, "_encode", side_effect=TypeError("unsupported argument")
    ):
        with pytest.warns(RuntimeWarning, match="Host compilation bypassed"):
            compiler._observe(
                "key",
                test_observer_failure_does_not_invalidate_a_successful_call,
                (),
                {},
                1,
            )
    assert not compiler._local.busy


def test_attempt_budget_does_not_grow_on_new_specializations(compiler):
    compiler._mode = "compile"
    compiler._seen = set(range(512))
    with patch.object(compiler, "_encode", return_value=[]):
        compiler._observe(
            "key", test_attempt_budget_does_not_grow_on_new_specializations, (), {}, 1
        )
    assert len(compiler._seen) == 512
