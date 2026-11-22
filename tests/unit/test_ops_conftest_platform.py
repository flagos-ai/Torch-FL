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

"""Unit coverage for tests/integration/ops/conftest.py platform detection.

The ops gate skips backend-specific tests per detected platform
(_PLATFORM_SKIP_MARKERS). These tests cover platform detection from the build
record, older PPU wheels and backend-specific test selection.

Run: pytest tests/unit/test_ops_conftest_platform.py -v
"""

import importlib.util
from importlib import import_module
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPS_CONFTEST_PATH = _REPO_ROOT / "tests" / "integration" / "ops" / "conftest.py"


def _load_ops_conftest():
    """Import ops/conftest.py without going through the tests/integration
    pytest-plugin machinery (which requires the full integration conftest)."""
    spec = importlib.util.spec_from_file_location(
        "ops_conftest_under_test", _OPS_CONFTEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ops_conftest = _load_ops_conftest()


@pytest.mark.parametrize(
    "platform",
    ["ascend", "dcu", "gcu", "iluvatar", "kunlunxin", "metax", "mps", "musa", "ppu"],
)
@pytest.mark.parametrize(
    "filename,reason",
    [
        ("test_flaggems_mul_route.py", "CUDA FlagGems route regression"),
        ("test_dtype_cast_performance.py", "CUDA dtype-cast performance regression"),
    ],
)
def test_cuda_profiling_skips_other_platforms_before_runtime_access(
    monkeypatch, platform, filename, reason
):
    package = ModuleType("torch_fl")
    package.__path__ = []
    detector = ModuleType("torch_fl._platform")
    detector.detect_platform = lambda: platform
    monkeypatch.setitem(sys.modules, "torch_fl", package)
    monkeypatch.setitem(sys.modules, "torch_fl._platform", detector)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    # Importing this optional vendor package would produce a different skip.
    monkeypatch.setitem(sys.modules, "flag_gems", None)
    path = _OPS_CONFTEST_PATH.parent / filename
    spec = importlib.util.spec_from_file_location("cuda_mul_route_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(pytest.skip.Exception, match=reason):
        spec.loader.exec_module(module)


def test_enflame_hosts_collect_without_optional_backend_packages(monkeypatch):
    torch = ModuleType("torch")
    for name in (
        "float32",
        "float16",
        "bfloat16",
        "int64",
        "add",
        "sub",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
    ):
        setattr(torch, name, object())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_fl", ModuleType("torch_fl"))
    for name in ("flag_gems", "triton"):
        monkeypatch.setitem(sys.modules, name, None)
    path = _OPS_CONFTEST_PATH.parent / "test_enflame_native_hosts.py"
    spec = importlib.util.spec_from_file_location("enflame_hosts_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Collection needs no backend package, but executing an admitted GCU test
    # must still report a missing FlagGems installation as an error.
    with pytest.raises(ModuleNotFoundError, match="flag_gems"):
        module.test_warm_silu_dispatch_uses_the_compiled_vendor_host()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FLAGOS_BACKEND_CONFIG", raising=False)
    monkeypatch.delenv("PPU_SDK", raising=False)


@pytest.fixture
def isolated_torch_fl_import():
    """Temporarily hide the real torch_fl module so _platform.detect_platform()'s
    ``import torch_fl`` resolves a throwaway stub instead.

    Same pattern as test_platform_support_detect.py: the real torch_fl.__init__
    has import-time side effects, and swapping sys.modules out and back in
    avoids re-triggering that import entirely.
    """
    real_torch_fl = sys.modules.pop("torch_fl", None)
    try:
        yield
    finally:
        sys.modules.pop("torch_fl", None)
        if real_torch_fl is not None:
            sys.modules["torch_fl"] = real_torch_fl


def _fake_torch_fl(
    tmp_path, *, record=None, marker=None, bundle_lib_thead=False, resolved_conf=None
):
    """Point ``torch_fl`` at a stub package with the given layout.

    ``record`` writes the _build_config.py setup.py would have, which is where
    detect_platform() reads the accelerator from. ``resolved_conf`` writes the
    backend_config_path() accessor -- the conf torch_fl would have selected; a
    stub without it stands for a package that predates the accessor.
    """
    package = tmp_path / "torch_fl"
    package.mkdir()
    init = ""
    if resolved_conf is not None:
        init = f"def backend_config_path():\n    return {str(resolved_conf)!r}\n"
    (package / "__init__.py").write_text(init)
    if record is not None:
        (package / "_build_config.py").write_text(
            "# AUTO-GENERATED by setup.py at build time. Do not edit.\n"
            f'ACCELERATOR = "{record}"\n'
            "KERNELS = ()\n"
        )
    if marker is not None:
        lib_dir = package / "lib"
        lib_dir.mkdir()
        (lib_dir / "flagos_platform").write_text(marker + "\n")
    if bundle_lib_thead:
        (package / "lib_thead").mkdir()
    sys.path.insert(0, str(tmp_path))
    try:
        if resolved_conf is not None:
            import_module("torch_fl")
        yield package
    finally:
        sys.path.remove(str(tmp_path))


@pytest.mark.parametrize(
    "accelerator,expected",
    [
        ("ascend", "ascend"),
        ("metax", "metax"),
        ("maca", "metax"),
        ("musa", "musa"),
        ("dcu", "dcu"),
        ("ppu", "ppu"),
        ("iluvatar", "iluvatar"),
    ],
)
def test_record_values_still_map(
    isolated_torch_fl_import, tmp_path, accelerator, expected
):
    for _ in _fake_torch_fl(tmp_path, record=accelerator):
        assert ops_conftest._platform.detect_platform() == expected


def test_ppu_record_identifies_ppu(isolated_torch_fl_import, tmp_path):
    """PPU has its own record value; no SDK probe needed."""
    for _ in _fake_torch_fl(tmp_path, record="ppu"):
        assert ops_conftest._platform.detect_platform() == "ppu"


def test_legacy_ppu_sdk_env_identifies_ppu_under_cuda_record(
    monkeypatch, isolated_torch_fl_import, tmp_path
):
    """A wheel built before PPU had a record value of its own says cuda and
    carries PPU_SDK; the env signal still tells the gate apart."""
    monkeypatch.setenv("PPU_SDK", "/usr/local/PPU_SDK")
    for _ in _fake_torch_fl(tmp_path, record="cuda"):
        assert ops_conftest._platform.detect_platform() == "ppu"


def test_ppu_home_env_does_not_identify_ppu(
    monkeypatch, isolated_torch_fl_import, tmp_path
):
    """PPU_HOME was an invented alias and is no longer read; only PPU_SDK is."""
    monkeypatch.setenv("PPU_HOME", "/opt/ppu")
    for _ in _fake_torch_fl(tmp_path):
        assert ops_conftest._platform.detect_platform() == "cuda"


def test_cuda_without_ppu_signals_stays_cuda(
    monkeypatch, isolated_torch_fl_import, tmp_path
):
    """No PPU signal anywhere: same bucket as before this change."""
    for _ in _fake_torch_fl(tmp_path, record="cuda"):
        assert ops_conftest._platform.detect_platform() == "cuda"


def test_lib_thead_bundle_dir_identifies_ppu_without_env(
    monkeypatch, isolated_torch_fl_import, tmp_path
):
    """An installed PPU wheel with no PPU_SDK in the environment
    (dev pod with the baked SDK) is still recognized by its bundle dir."""
    for _ in _fake_torch_fl(tmp_path, record="cuda", bundle_lib_thead=True):
        assert ops_conftest._platform.detect_platform() == "ppu"


def test_marker_stays_authoritative_over_bundle_dir(
    monkeypatch, isolated_torch_fl_import, tmp_path
):
    """A flagos_platform marker wins even if lib_thead/ is also present, so the
    native-kernel platforms keep their existing precedence."""
    for _ in _fake_torch_fl(
        tmp_path, record="cuda", marker="gcu", bundle_lib_thead=True
    ):
        assert ops_conftest._platform.detect_platform() == "gcu"


def test_resolved_ppu_backend_config_identifies_ppu(
    monkeypatch, isolated_torch_fl_import, tmp_path
):
    """A wheel whose conf resolves to backends_thead.conf is identified as PPU even
    when neither PPU_SDK nor lib_thead/ says so. This is the name-based last
    resort, read from torch_fl.backend_config_path() -- the import no longer
    writes FLAGOS_BACKEND_CONFIG, so there is nothing to read from the
    environment."""
    conf = "/opt/venv/lib/torch_fl/configs/backends_thead.conf"
    for _ in _fake_torch_fl(tmp_path, record="cuda", resolved_conf=conf):
        assert ops_conftest._platform.detect_platform() == "ppu"


def test_explicit_backend_config_identifies_platform_without_the_accessor(
    monkeypatch, isolated_torch_fl_import, tmp_path
):
    """A stub package with no backend_config_path() falls back to the variable,
    which is what the accessor itself falls back to: FLAGOS_BACKEND_CONFIG is
    still honored as the user's explicit override."""
    monkeypatch.setenv(
        "FLAGOS_BACKEND_CONFIG", "/opt/venv/lib/torch_fl/configs/backends_thead.conf"
    )
    for _ in _fake_torch_fl(tmp_path, record="cuda"):
        assert ops_conftest._platform.detect_platform() == "ppu"


def test_ppu_skip_set_matches_cuda():
    """Behavior-neutrality pin: PPU skips exactly what the old implicit
    cuda bucket skipped. Any future divergence must be a deliberate,
    separately reviewed edit to this tuple."""
    assert (
        ops_conftest._PLATFORM_SKIP_MARKERS["ppu"]
        == ops_conftest._PLATFORM_SKIP_MARKERS["cuda"]
    )


def test_iluvatar_keeps_flaggems():
    markers = ops_conftest._PLATFORM_SKIP_MARKERS["iluvatar"]
    assert "flaggems" not in markers
    assert "flaggems_python" not in markers


def test_no_platform_skips_the_plain_flaggems_mark():
    """The retired FLAGOS_USE_FLAGGEMS switch used to gate `flaggems`-marked
    tests; with routing stated per op in backends_<platform>.conf there is no
    switch left, so no platform may skip the plain mark. (The suffixed
    flaggems_python / flaggems_cpp marks are separate build-gated paths and
    are unaffected by this invariant.)"""
    for platform, markers in ops_conftest._PLATFORM_SKIP_MARKERS.items():
        assert "flaggems" not in markers, platform


def test_every_non_mps_platform_skips_the_mps_mark():
    """`@pytest.mark.mps` names Apple's Metal runtime; nothing else serves it.

    The migrated Apple files (tests/integration/ops/test_mps_*.py) keep a
    module-level `mps` mark and mark their portable contract cases
    `anyplatform`, so every other lane selects them into its shared cohort and
    depends on this gate to skip them. The MUSA row was missing the entry, and
    13 argmin/argmax NaN cases from test_mps_argmin_arpextreme.py ran on Moore
    Threads hardware instead of being skipped.
    """
    for platform, markers in ops_conftest._PLATFORM_SKIP_MARKERS.items():
        if platform == "mps":
            continue
        assert "mps" in markers, platform


@pytest.mark.parametrize(
    "version,cuda,kernels,expected",
    [
        ("2.10.0+cpu", None, ("flaggems", "vendor"), True),
        ("2.10.0+cpu", None, ("vendor",), False),
        ("2.10.0+cu130", "13.0", ("flaggems", "vendor"), False),
        ("2.9.0+cpu", None, ("flaggems", "vendor"), False),
    ],
)
@pytest.mark.parametrize("native_hosts", [False, True])
def test_host_compiler_gate_reads_the_installed_build(
    monkeypatch, version, cuda, kernels, expected, native_hosts
):
    torch = ModuleType("torch")
    torch.__version__ = version
    torch.version = SimpleNamespace(cuda=cuda)
    package = ModuleType("torch_fl")
    package._env = SimpleNamespace(build_kernels=lambda: kernels)
    package._build_config = SimpleNamespace(NATIVE_HOSTS=native_hosts)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_fl", package)
    assert ops_conftest._host_compiler_enabled() is (expected and not native_hosts)
    assert ops_conftest._native_source_host_enabled() is native_hosts


@pytest.mark.parametrize("enabled", [False, True])
def test_native_vendor_precision_gate_reads_the_installed_build(monkeypatch, enabled):
    package = ModuleType("torch_fl")
    package._build_config = SimpleNamespace(NATIVE_VENDOR=enabled)
    monkeypatch.setitem(sys.modules, "torch_fl", package)
    assert ops_conftest._native_vendor_precision_enabled() is enabled


@pytest.mark.parametrize(
    "platform,compiler,marks,gate",
    [
        ("cuda", True, ("nvidia", "host_compiler"), None),
        ("cuda", False, ("nvidia", "host_compiler"), "build_record"),
        ("cuda", False, ("nvidia",), None),
        ("mps", False, ("nvidia",), "platform_support"),
        ("ppu", True, ("host_compiler",), "platform_support"),
        ("cuda", True, ("nvidia", "native_source_host"), None),
        ("cuda", False, ("nvidia", "native_source_host"), "build_record"),
        ("metax", True, ("native_source_host",), None),
        ("metax", False, ("native_source_host",), "build_record"),
        ("metax", True, ("metax", "native_source_host"), None),
        ("metax", False, ("metax", "native_source_host"), "build_record"),
        ("metax", True, ("metax",), "backend_marker"),
        ("metax", True, ("nvidia", "native_source_host"), "platform_support"),
        ("ascend", True, ("ascend", "native_source_host"), None),
        ("ascend", False, ("ascend", "native_source_host"), "build_record"),
        ("ascend", True, ("nvidia", "native_source_host"), "platform_support"),
        ("mps", True, ("native_source_host",), "platform_support"),
        ("ppu", True, ("native_source_host",), "platform_support"),
        ("dcu", True, ("dcu", "native_source_host"), None),
        ("dcu", False, ("dcu", "native_source_host"), "build_record"),
        ("dcu", True, ("nvidia", "native_source_host"), "platform_support"),
        ("gcu", True, ("native_source_host",), "platform_support"),
        ("gcu", False, ("native_source_host",), "platform_support"),
        ("gcu", True, ("enflame_native_host",), None),
        ("gcu", False, ("enflame_native_host",), "build_record"),
        ("cuda", True, ("enflame_native_host",), "platform_support"),
        ("mps", False, ("enflame_native_host",), "platform_support"),
        ("cuda", True, ("native_vendor_precision",), None),
        ("cuda", False, ("native_vendor_precision",), "build_record"),
        ("mps", True, ("native_vendor_precision",), "platform_support"),
        ("ppu", True, ("native_vendor_precision",), "platform_support"),
    ],
)
def test_native_host_cases_keep_one_collection_shape(
    monkeypatch, platform, compiler, marks, gate
):
    class Item:
        def __init__(self):
            self.own_markers = [getattr(pytest.mark, name).mark for name in marks]

        def get_closest_marker(self, name):
            return next((m for m in self.own_markers if m.name == name), None)

        def add_marker(self, marker):
            self.own_markers.append(marker.mark)

    monkeypatch.setattr(ops_conftest._platform, "detect_platform", lambda: platform)
    monkeypatch.setattr(ops_conftest, "_host_compiler_enabled", lambda: compiler)
    monkeypatch.setattr(ops_conftest, "_native_source_host_enabled", lambda: compiler)
    monkeypatch.setattr(
        ops_conftest, "_native_vendor_precision_enabled", lambda: compiler
    )
    monkeypatch.setattr(ops_conftest, "_flaggems_cpp_enabled", lambda: False)
    item = Item()
    ops_conftest.pytest_collection_modifyitems(None, [item])
    marker = item.get_closest_marker("flagos_platform_gate")
    assert (marker.kwargs["gate"] if marker else None) == gate
    assert bool(item.get_closest_marker("skip")) == (gate is not None)


@pytest.mark.parametrize(
    "enabled,devices,implementation,version,gate",
    [
        (True, 2, "cpython", (3, 12), None),
        (True, 1, "cpython", (3, 12), "device_count"),
        (True, 2, "cpython", (3, 11), "platform_support"),
        (True, 2, "pypy", (3, 12), "platform_support"),
        (False, 0, "cpython", (3, 11), "build_record"),
    ],
)
def test_source_host_runtime_requirements_are_collection_gates(
    monkeypatch, enabled, devices, implementation, version, gate
):
    class Item:
        own_markers = [
            pytest.mark.native_source_host.mark,
            pytest.mark.min_devices(2).mark,
            pytest.mark.cpython_dict_watchers.mark,
        ]

        def get_closest_marker(self, name):
            return next((m for m in self.own_markers if m.name == name), None)

        def add_marker(self, marker):
            self.own_markers.append(marker.mark)

    def device_count():
        assert enabled, "ordinary builds must skip before device discovery"
        return devices

    package = ModuleType("torch_fl")
    package.flagos = SimpleNamespace(device_count=device_count)
    monkeypatch.setitem(sys.modules, "torch_fl", package)
    monkeypatch.setattr(ops_conftest._platform, "detect_platform", lambda: "cuda")
    monkeypatch.setattr(ops_conftest, "_native_source_host_enabled", lambda: enabled)
    monkeypatch.setattr(ops_conftest, "_flaggems_cpp_enabled", lambda: False)
    monkeypatch.setattr(
        ops_conftest,
        "sys",
        SimpleNamespace(
            implementation=SimpleNamespace(name=implementation), version_info=version
        ),
    )
    item = Item()
    ops_conftest.pytest_collection_modifyitems(None, [item])
    marker = item.get_closest_marker("flagos_platform_gate")
    assert (marker.kwargs["gate"] if marker else None) == gate
    assert bool(item.get_closest_marker("skip")) == (gate is not None)
