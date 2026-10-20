# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""The runtime audit must detect vendor binaries behind CPU Python metadata."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "iluvatar_verify", ROOT / ".github/scripts/verify_iluvatar.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def cpu_mappings(root):
    return "\n".join(
        f"1000-2000 r-xp 00000000 08:01 123 {root}/lib/{name}"
        for name in ("libc10.so", "libtorch_cpu.so", "libtorch_python.so")
    )


def test_official_cpu_core_allows_corex_device_libraries(tmp_path):
    torch = tmp_path / "torch"
    mappings = cpu_mappings(torch)
    mappings += "\n1000-2000 r-xp 0 08:01 1 /usr/local/corex/lib64/libcudart.so"
    verifier.assert_cpu_core(torch, tmp_path / "torch_fl", mappings)


@pytest.mark.parametrize("name", ["libtorch_cuda.so", "libc10_cuda.so"])
def test_vendor_cuda_core_is_rejected(tmp_path, name):
    torch = tmp_path / "torch"
    mappings = cpu_mappings(torch)
    mappings += f"\n1000-2000 r-xp 0 08:01 1 /vendor/torch/lib/{name}"
    with pytest.raises(AssertionError, match="Vendor CUDA libtorch"):
        verifier.assert_cpu_core(torch, tmp_path / "torch_fl", mappings)


def test_vendor_cpu_core_cannot_impersonate_the_official_distribution(tmp_path):
    torch = tmp_path / "torch"
    with pytest.raises(AssertionError, match="official CPU torch"):
        verifier.assert_cpu_core(
            torch, tmp_path / "torch_fl", cpu_mappings(tmp_path / "vendor/torch")
        )


def test_a_second_loaded_cpu_core_is_rejected(tmp_path):
    torch = tmp_path / "torch"
    mappings = cpu_mappings(torch) + "\n" + cpu_mappings(tmp_path / "vendor/torch")
    with pytest.raises(AssertionError, match="second libc10.so"):
        verifier.assert_cpu_core(torch, tmp_path / "torch_fl", mappings)


def test_stale_wheel_bundle_is_rejected(tmp_path):
    package = tmp_path / "torch_fl"
    (package / "lib_iluvatar").mkdir(parents=True)
    with pytest.raises(AssertionError):
        verifier.assert_cpu_core(tmp_path / "torch", package, "")
