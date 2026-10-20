# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate the isolated CPU torch core and independent Iluvatar runtime."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def read_pins():
    path = Path(
        os.environ.get(
            "TORCH_FL_VERSION_PINS",
            Path(__file__).resolve().parents[1] / "version-pins.env",
        )
    )
    return dict(
        line.strip().split("=", 1)
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    )


def verify():
    import torch
    import triton

    pins = read_pins()
    assert pins["CPU_TORCH_VERSION_DEFAULT"] == "2.10.0", (
        "Changing CPU torch requires a rebuilt and validated CoreX CI image"
    )
    assert sys.version_info[:2] == (3, 12), sys.version
    assert torch.__version__ == "2.10.0+cpu", torch.__version__
    assert importlib.metadata.version("torch") == "2.10.0+cpu"
    assert torch.version.cuda is None, torch.version.cuda
    assert torch._C._GLIBCXX_USE_CXX11_ABI, "The untouched CPU core must be active"
    assert (
        Path(torch.__file__).resolve().is_relative_to("/opt/torch-fl-iluvatar-venv")
    ), torch.__file__
    assert not any(
        path.startswith("/usr/local/lib/python3.12/site-packages") for path in sys.path
    ), sys.path
    assert importlib.util.find_spec("vllm") is None, "Base Python packages leaked"
    config = Path(sys.prefix, "pyvenv.cfg").read_text().lower()
    assert "include-system-site-packages = false" in config, config
    assert importlib.metadata.version("flagtree") == os.environ.get(
        "FLAGTREE_VERSION", pins["FLAGTREE_VERSION_iluvatar"]
    )
    assert importlib.metadata.version("flag_gems") == os.environ.get(
        "FLAGGEMS_VERSION", pins["FLAGGEMS_VERSION_DEFAULT"]
    )
    assert (
        Path(triton.__file__).resolve().is_relative_to("/opt/torch-fl-iluvatar-venv")
    ), triton.__file__
    assert "iluvatar" in triton.backends.backends, sorted(triton.backends.backends)
    assert importlib.metadata.version("cmake") == "3.31.10"
    assert Path(shutil.which("cmake")).is_relative_to("/opt/torch-fl-iluvatar-venv")

    sdk = Path(os.environ["COREX_PATH"])
    for header in ("cuda_runtime.h", "cudnn.h", "cublas_v2.h"):
        assert (sdk / "include" / header).is_file(), header
    for library in ("libcudart.so*", "libcudnn.so*", "libcublas.so*"):
        assert any(
            path.is_file()
            for directory in ("lib", "lib64")
            for path in (sdk / directory).glob(library)
        ), library

    subprocess.run(
        ["c++", "-fsyntax-only", "-x", "c++", "-I", str(sdk / "include"), "-"],
        input=(
            "#include <cublas_v2.h>\n#include <cudnn.h>\n"
            "int main() { ixblasStatus_t s = IXBLAS_STATUS_SUCCESS; "
            "ixblasDataType_t d = IX_R_16BF; return int(s) + int(d); }\n"
        ),
        text=True,
        check=True,
    )

    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_distribution": importlib.metadata.version("torch"),
        "torch_path": torch.__file__,
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "flagtree": importlib.metadata.version("flagtree"),
        "triton": triton.__version__,
        "triton_path": triton.__file__,
        "flag_gems": importlib.metadata.version("flag_gems"),
        "cmake": shutil.which("cmake"),
        "device_checked": False,
    }
    print(json.dumps(report, indent=2))


def assert_cpu_core(torch_root, package, mappings):
    """Reject a vendor facade even if Python still reports torch+cpu."""
    assert not (package / "lib_iluvatar").exists(), package
    paths = {
        Path(fields[5]).resolve()
        for line in mappings.splitlines()
        if len(fields := line.split(None, 5)) == 6 and fields[5].startswith("/")
    }
    core = torch_root / "lib"
    for name in ("libc10.so", "libtorch_cpu.so", "libtorch_python.so"):
        expected = (core / name).resolve()
        assert expected in paths, f"{name} did not load from official CPU torch: {core}"
        assert all(path == expected for path in paths if path.name == name), (
            f"A second {name} was loaded outside official CPU torch"
        )
    assert not any(
        path.name.startswith(("libtorch_cuda", "libc10_cuda")) for path in paths
    ), "Vendor CUDA libtorch was loaded"


def verify_runtime():
    # Import CPU torch first to prove the runtime needs no core replacement.
    import torch
    import torch_fl
    import triton
    import triton.language as tl

    assert importlib.metadata.version("torch") == "2.10.0+cpu"
    assert torch.__version__ == "2.10.0+cpu", torch.__version__
    assert torch._C._GLIBCXX_USE_CXX11_ABI
    assert torch.version.cuda is None, torch.version.cuda
    assert torch_fl.flagos.is_available()
    assert torch.cuda.is_available()
    package = Path(torch_fl.__file__).resolve().parent
    torch_root = Path(torch.__file__).resolve().parent
    assert package.is_relative_to("/opt/torch-fl-iluvatar-venv"), package
    assert torch_root.is_relative_to("/opt/torch-fl-iluvatar-venv"), torch_root
    assert_cpu_core(torch_root, package, Path("/proc/self/maps").read_text())
    for library in ("libtorch_fl.so", "libtorch_bindings.so"):
        dynamic = subprocess.check_output(
            ["readelf", "-d", str(package / "lib" / library)], text=True
        )
        assert "libtorch_cuda" not in dynamic and "libc10_cuda" not in dynamic, dynamic
    capability = torch.cuda.get_device_capability()
    assert capability == (7, 1), capability
    target = triton.runtime.driver.active.get_current_target()
    assert (target.backend, target.arch, target.warp_size) == ("corex", 71, 64), target

    @triton.jit
    def add_kernel(x, y, output, size: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < size
        tl.store(
            output + offsets,
            tl.load(x + offsets, mask=mask, other=0)
            + tl.load(y + offsets, mask=mask, other=0),
            mask=mask,
        )

    x_cpu = torch.arange(1025, dtype=torch.float32)
    x = x_cpu.to("flagos")
    output = torch.empty_like(x)
    add_kernel[(triton.cdiv(x.numel(), 256),)](x, x, output, x.numel(), 256)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), x_cpu + x_cpu)
    value = torch.tensor([1.0, 2.0, 3.0], device="flagos")
    torch.testing.assert_close((value + value).cpu(), torch.tensor([2.0, 4.0, 6.0]))
    print(
        json.dumps(
            {
                "torch_fl": str(package),
                "torch": torch.__version__,
                "devices": torch_fl.flagos.device_count(),
                "triton_add": "passed",
                "flagos_add": "passed",
                "cpu_core": str(torch_root / "lib"),
                "triton_target": str(target),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", action="store_true")
    parser.add_argument("--runtime", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runtime:
        verify_runtime()
    else:
        verify()
        if args.device:
            subprocess.run([sys.executable, __file__, "--runtime"], check=True)
