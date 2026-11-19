# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Execute the real config reader and dispatcher with CPU tensors and test kernels."""

import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = r"""
#include "aten/dispatcher.h"
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>

using namespace at::native::flagos;
using Fn = at::Tensor (*)(const at::Tensor&, const at::Tensor&);
at::Tensor Gems(const at::Tensor&, const at::Tensor&) { return at::scalar_tensor(11); }
at::Tensor Vendor(const at::Tensor&, const at::Tensor&) { return at::scalar_tensor(22); }

int main(int argc, char** argv) {
  try {
    SetBackendConfigPath(argv[1]);
    if (std::string(argv[2]) == "scalar") {
      int wrong = 0;
      for (auto dtype : {at::kFloat, at::kDouble, at::kHalf, at::kBFloat16}) {
        for (double value : {-0.0, 0.1, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
          auto tensor = at::scalar_tensor(value, at::TensorOptions().dtype(dtype));
          tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
          const double a = TypedWrappedScalar(tensor).toDouble(), b = tensor.item().toDouble();
          wrong += std::memcmp(&a, &b, sizeof(a)) != 0;
        }
      }
      for (int64_t value : {INT64_MIN, int64_t(9007199254740999LL), INT64_MAX}) {
        auto tensor = at::scalar_tensor(value, at::TensorOptions().dtype(at::kLong));
        tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
        wrong += TypedWrappedScalar(tensor).toLong() != value;
      }
      auto negative = at::scalar_tensor(2.0);
      negative.unsafeGetTensorImpl()->set_wrapped_number(true);
      negative._set_neg(true);
      wrong += TypedWrappedScalar(negative).toDouble() != negative.item().toDouble();
      bool rejected = false;
      try { TypedWrappedScalar(at::ones({2})); } catch (const std::exception&) { rejected = true; }
      std::cout << wrong << ' ' << rejected << '\n';
      return 0;
    }
    if (std::string(argv[2]) == "wrapped") {
      auto scalar = at::scalar_tensor(0.5);
      auto regular = at::scalar_tensor(0.5);
      auto complex = at::scalar_tensor(c10::complex<double>(0.5, 1.0),
                                     at::TensorOptions().dtype(at::kComplexDouble));
      scalar.unsafeGetTensorImpl()->set_wrapped_number(true);
      complex.unsafeGetTensorImpl()->set_wrapped_number(true);
      const auto mask = uint64_t{1} << static_cast<unsigned>(at::kFloat);
      PublishedSourceGuard ordinary(mask), enabled(mask, false, 1, false, true);
      std::cout << ordinary.Check(scalar) << ' ' << enabled.Check(scalar) << ' '
                << enabled.Check(regular) << ' ' << enabled.HasTensor() << ' '
                << PublishedSourceAccepts<true>(mask, scalar) << ' '
                << enabled.Check(complex) << '\n';
      return 0;
    }
    Dispatcher<Fn> op("mm");
    op.RegisterKernel(Backend::kFlagGems, &Gems);
    const std::string mode(argv[2]);
    if (mode == "native" || mode == "threads") op.RegisterKernel(Backend::kNvidia, &Vendor);
    if (mode == "hygon") op.RegisterKernel(Backend::kHygon, &Vendor);
    auto bf = at::empty({1}, at::TensorOptions().dtype(at::kBFloat16));
    if (mode == "late") {
      std::cout << op(bf, bf).item<int>() << ' ';
      op.RegisterKernel(Backend::kNvidia, &Vendor);
    }
    if (mode == "threads") {
      std::atomic<int> wrong{0};
      std::vector<std::thread> threads;
      for (int i = 0; i < 4; ++i) threads.emplace_back([&] {
        for (int j = 0; j < 100; ++j) if (op(bf, bf).item<int>() != 22) ++wrong;
      });
      for (auto& thread : threads) thread.join();
      std::cout << wrong.load() << '\n';
      return 0;
    }
    for (auto dtype : {at::kFloat, at::kBFloat16, at::kDouble}) {
      auto input = at::empty({1}, at::TensorOptions().dtype(dtype));
      std::cout << op(input, input).item<int>() << ' ';
    }
    std::cout << op.DispatchConfigured(bf, bf).item<int>() << ' '
              << op.DispatchAs("other", bf, bf).item<int>() << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
"""


@pytest.fixture(scope="module")
def executable(tmp_path_factory):
    torch = pytest.importorskip("torch")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    root = tmp_path_factory.mktemp("configured-routes")
    source, binary = root / "main.cc", root / "routes"
    entry_source = ROOT / "torch_fl/native_host/compiler/entries.py"
    header = next(
        ast.literal_eval(node.value)
        for node in ast.parse(entry_source.read_text()).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "HEADER"
            for target in node.targets
        )
    )
    source.write_text("#include <ATen/ATen.h>\n" + header + SOURCE)
    torch_root = Path(torch.__file__).parent
    command = [
        compiler,
        "-std=c++17",
        "-O0",
        "-pthread",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        f'-DFLAGOS_SOURCE_ROOT="{ROOT}"',
        "-I" + str(ROOT / "csrc"),
        "-I" + str(ROOT / "csrc/include"),
        "-I" + str(torch_root / "include"),
        "-I" + str(torch_root / "include/torch/csrc/api/include"),
        str(source),
        str(ROOT / "csrc/aten/common.cc"),
        "-L" + str(torch_root / "lib"),
        "-Wl,-rpath," + str(torch_root / "lib"),
        "-ltorch_cpu",
        "-lc10",
        "-o",
        str(binary),
    ]
    if sys.platform.startswith("linux"):
        command.append("-ldl")
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return binary


def run(executable, tmp_path, config, mode="native", **overrides):
    path = tmp_path / "backends.conf"
    path.write_text(config + "\nother = flaggems\n")
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("FLAGOS_")
    }
    env.update(overrides)
    return subprocess.run(
        [str(executable), str(path), mode], env=env, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "config,mode,expected",
    [
        ("mm = flaggems", "native", "11 11 11 11 11"),
        ("mm = flaggems; BFloat16:hygon", "hygon", "11 22 11 22 11"),
        ("mm = flaggems; BFloat16:hygon", "native", "11 11 11 11 11"),
        ("mm = flaggems; BFloat16:nvidia", "hygon", "11 11 11 11 11"),
        ("mm = flaggems; BFloat16:nvidia", "native", "11 22 11 22 11"),
        ("mm = flaggems; Float:nvidia", "native", "22 11 11 11 11"),
        ("mm = flaggems; Float,BFloat16:nvidia", "native", "22 22 11 22 11"),
        ("mm = flaggems; BFloat16:nvidia", "none", "11 11 11 11 11"),
        ("mm = flaggems; BFloat16:nvidia", "late", "11 11 22 11 22 11"),
        ("mm = flaggems; BFloat16:nvidia\nmm = flaggems", "native", "11 11 11 11 11"),
        ("mm = flaggems; BFloat16:nvidia", "threads", "0"),
    ],
)
def test_config_is_the_only_dtype_route_source(
    executable, tmp_path, config, mode, expected
):
    result = run(executable, tmp_path, config, mode)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    "override", [{"FLAGOS_OP_mm": "flaggems"}, {"FLAGOS_FORCE_BACKEND": "flaggems"}]
)
def test_explicit_flaggems_override_removes_native_conditions(
    executable, tmp_path, override
):
    result = run(executable, tmp_path, "mm = flaggems; BFloat16:nvidia", **override)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "11 11 11 11 11"


def test_vendor_force_retains_the_measured_dtype_boundary(executable, tmp_path):
    result = run(
        executable,
        tmp_path,
        "mm = flaggems; BFloat16:nvidia",
        FLAGOS_FORCE_BACKEND="vendor",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "11 22 11 22 11"


def test_shipped_conf_selects_the_measured_native_dtype(executable, tmp_path):
    config = (ROOT / "torch_fl/configs/backends_nvidia.conf").read_text()
    result = run(executable, tmp_path, config)
    assert result.returncode == 0, result.stderr
    assert "unknown backend" not in result.stderr
    assert result.stdout.strip() == "11 22 11 22 11"


def test_plain_override_replaces_an_included_dtype_rule(executable, tmp_path):
    (tmp_path / "included.conf").write_text("mm = flaggems; BFloat16:nvidia\n")
    result = run(executable, tmp_path, "include included.conf\nmm = flaggems")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "11 11 11 11 11"


@pytest.mark.parametrize(
    "clause",
    [
        "",
        ":nvidia",
        "Float,:nvidia",
        "Unknown:nvidia",
        "Float:typo",
        "Float:none",
        "Float:nvidia; Float:flaggems",
    ],
)
def test_invalid_dtype_rules_fail_explicitly(executable, tmp_path, clause):
    result = run(executable, tmp_path, "mm = flaggems; " + clause)
    assert result.returncode != 0
    assert "flagos:" in result.stderr


def test_published_source_scalar_admission_requires_a_real_device_tensor(
    executable, tmp_path
):
    result = run(executable, tmp_path, "mm = flaggems", mode="wrapped")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0 1 0 0 0 0"


def test_wrapped_scalar_conversion_preserves_bits_and_item_errors(executable, tmp_path):
    result = run(executable, tmp_path, "mm = flaggems", mode="scalar")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0 1"
