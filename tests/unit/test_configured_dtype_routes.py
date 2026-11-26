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
    const std::string mode(argv[2]);
    if (mode == "name") {
      std::cout << BackendLogName(GetBackendForOp("mm")) << '\n';
      return 0;
    }
    if (mode == "scalar") {
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
    if (mode == "wrapped") {
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
    if (mode == "host-disabled") {
      SetStrictHostRoutes(false);
      TORCH_CHECK(!CpuFallbackAllowed());
      return 0;
    }
    if (mode == "host") {
      auto value = at::ones({2, 2}, at::TensorOptions().dtype(at::kDouble));
      TORCH_CHECK(CpuFallbackAllowed());
      auto expected = MatmulOnHost(value, value);
      SetStrictHostRoutes(true);
      TORCH_CHECK(!CpuFallbackAllowed());
      bool rejected = false;
      try {
        MatmulOnHost(value, value);
      } catch (const c10::Error& error) {
        rejected = std::string(error.what()).find("host matmul escape") != std::string::npos;
      }
      TORCH_CHECK(rejected, "strict host matmul was not rejected");
      SetStrictHostRoutes(false);
      TORCH_CHECK(CpuFallbackAllowed());
      TORCH_CHECK(at::equal(MatmulOnHost(value, value), expected));
      std::cout << "allowed rejected restored\n";
      return 0;
    }
    if (mode == "native" || mode == "threads") op.RegisterKernel(Backend::kNvidia, &Vendor);
    if (mode == "hygon") op.RegisterKernel(Backend::kHygon, &Vendor);
    auto bf = at::empty({1}, at::TensorOptions().dtype(at::kBFloat16));
    if (mode == "decoded") {
      op.RegisterKernel(Backend::kFlagGemsCpp, &Vendor);
      op.RegisterKernel(Backend::kHygon, &Vendor);
      op.RegisterKernel(Backend::kNvidia, &Vendor);
      std::cout << op.DecodedKernel("mm")(bf, bf).item<int>() << '\n';
      return 0;
    }
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
        '-DFLAGOS_BUILTIN_KERNELS="flaggems,vendor"',
        "-I" + str(ROOT / "csrc"),
        "-I" + str(ROOT / "csrc/include"),
        "-I" + str(torch_root / "include"),
        "-I" + str(torch_root / "include/torch/csrc/api/include"),
        str(source),
        str(ROOT / "csrc/aten/common.cc"),
        str(ROOT / "csrc/aten/route.cc"),
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
    "key,log_name",
    [
        ("FLAGOS", "flagos"),
        ("flaggems_python", "flagos_python"),
        ("FLAGOS_PYTHON", "flagos_python"),
        ("musa", "musa"),
        ("gcu", "gcu"),
        ("mps", "mps"),
        ("bpu", "drobotics"),
        ("thead", "none"),
        ("nvidia", "nvidia"),
        ("hygon", "hygon"),
    ],
)
def test_legacy_conf_names_resolve_to_the_expected_backend(
    executable, tmp_path, key, log_name
):
    result = run(executable, tmp_path, f"mm = {key}", mode="name")
    assert result.returncode == 0, result.stderr
    assert "unknown backend" not in result.stderr
    assert result.stdout.strip() == log_name


@pytest.mark.parametrize("key", ["boxing", "CuDa"])
@pytest.mark.parametrize("source", ["base", "dtype", "override", "decoded"])
def test_retired_boxing_route_requires_migration(executable, tmp_path, key, source):
    config = "mm = flaggems"
    mode = "name"
    environment = {}
    if source == "base":
        config = f"mm = {key}"
    elif source == "dtype":
        config += f"; Float:{key}"
    elif source == "override":
        environment["FLAGOS_OP_mm"] = key
    else:
        config = "mm = metax"
        mode = "decoded"
        environment["FLAGOS_SOFT_LOWP_STAY_ON_ROUTE"] = key
    result = run(executable, tmp_path, config, mode=mode, **environment)
    assert result.returncode != 0
    assert "boxing route has been removed" in result.stderr


def test_path_from_binding_takes_priority_over_the_env(executable, tmp_path):
    env_conf = tmp_path / "env.conf"
    env_conf.write_text("mm = none\n")
    result = run(
        executable,
        tmp_path,
        "mm = flaggems_cpp",
        mode="name",
        FLAGOS_BACKEND_CONFIG=str(env_conf),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "flagos"


@pytest.mark.parametrize(
    "key,log_name",
    [
        ("flagos", "flagos"),
        ("hygon", "hygon"),
        ("nvidia", "nvidia"),
    ],
)
def test_decoded_route_pins_only_supported_kernel_slots(
    executable, tmp_path, key, log_name
):
    result = run(
        executable,
        tmp_path,
        "mm = metax",
        mode="decoded",
        FLAGOS_SOFT_LOWP_STAY_ON_ROUTE=key,
        FLAGOS_LOG="dispatch",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "22"
    assert f"[flagos dispatch] mm -> {log_name}" in result.stderr


def test_decoded_route_rejects_a_valid_conf_key_for_an_ineligible_slot(
    executable, tmp_path
):
    result = run(
        executable,
        tmp_path,
        "mm = metax",
        mode="decoded",
        FLAGOS_SOFT_LOWP_STAY_ON_ROUTE="mps",
    )
    assert result.returncode != 0
    assert 'FLAGOS_SOFT_LOWP_STAY_ON_ROUTE="mps" names no live slot' in result.stderr


def test_uncompiled_kernel_set_is_named_in_dispatch_error(executable, tmp_path):
    result = run(executable, tmp_path, "mm = tileops")
    assert result.returncode != 0
    assert "the 'tileops' kernel set was not compiled into this wheel" in result.stderr


def test_forced_vendor_route_reports_a_missing_kernel(executable, tmp_path):
    result = run(
        executable,
        tmp_path,
        "mm = flaggems # nvidia",
        mode="none",
        FLAGOS_FORCE_BACKEND="vendor",
    )
    assert result.returncode != 0
    assert "FLAGOS_FORCE_BACKEND=vendor but no vendor kernel" in result.stderr


def test_host_matmul_uses_the_shared_strict_fallback_policy(executable, tmp_path):
    result = run(executable, tmp_path, "mm = flaggems", mode="host")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "allowed rejected restored"


def test_fallback_disable_cannot_be_overridden_by_host_routes(executable, tmp_path):
    result = run(
        executable,
        tmp_path,
        "mm = flaggems",
        mode="host-disabled",
        FLAGOS_DISABLE_CPU_FALLBACK="1",
    )
    assert result.returncode == 0, result.stderr


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
