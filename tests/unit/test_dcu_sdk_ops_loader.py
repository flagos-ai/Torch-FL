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

"""Unit coverage for the DCU SDK-native plugin loader.

No DTK, no GPU and no plugin binary are needed: everything the loader decides
before it touches the hardware -- which switches enable it, what a manifest must
contain, and whether a vendor libtorch slipped into the process -- is exercised
against fakes here. The on-device behaviour of the kernels themselves is covered
by the integration suite.
"""

import json
from pathlib import Path

import pytest
from dcu_module_loader import load_module

sdk = load_module(
    "torch_fl/accelerator/dcu/_dcu_sdk_ops.py", "_flagos_test_dcu_sdk_ops"
)


def _manifest(**overrides):
    manifest = {
        "schema_version": 2,
        "torch_base": "2.10.0",
        "torch_version": {"major": 2, "minor": 10, "patch": 0},
        "torch_abi": "cxx11",
        "sdk": "dtk",
        "sdk_version": "25.10.0",
        "registration_abi": 2,
        "dispatch_key": "PrivateUse1",
        "library": "libdcu_aten_ops.so",
        "route_count": 2034,
        "operators": ["aten::mm", "aten::mm.out"],
        "dtypes": ["float32"],
        "layouts": ["contiguous", "transposed"],
        "stream_behavior": "asynchronous",
        "fallback": "hybrid-only",
        "sdk_only": True,
    }
    manifest.update(overrides)
    return manifest


def _write_manifest(tmp_path, monkeypatch, manifest):
    path = tmp_path / "dcu_sdk_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("FLAGOS_DCU_SDK_MANIFEST", str(path))
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "FLAGOS_DCU_SDK_OPS",
        "FLAGOS_DCU_SDK_ONLY",
        "FLAGOS_DCU_SDK_OPS_LIB",
        "FLAGOS_DCU_SDK_MANIFEST",
        "FLAGOS_DCU_SDK_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)


# --- enablement ------------------------------------------------------------


def test_disabled_by_default():
    assert sdk.enabled() is False
    assert sdk.sdk_only() is False


@pytest.mark.parametrize("value", ["1", "on", "true", "yes", "YES", " 1 "])
def test_sdk_ops_switch_accepts_truthy_spellings(monkeypatch, value):
    monkeypatch.setenv("FLAGOS_DCU_SDK_OPS", value)
    assert sdk.enabled() is True


@pytest.mark.parametrize("value", ["0", "", "off", "false", "no"])
def test_sdk_ops_switch_rejects_falsy_spellings(monkeypatch, value):
    monkeypatch.setenv("FLAGOS_DCU_SDK_OPS", value)
    assert sdk.enabled() is False


def test_sdk_only_implies_enabled(monkeypatch):
    """The load-bearing case: SDK-only alone must still load the plugin.

    Otherwise the process picks backends_dcu_sdk.conf and skips the vendor
    preload, leaving the six SDK routes with no kernel behind them at all.
    """
    monkeypatch.setenv("FLAGOS_DCU_SDK_ONLY", "1")
    assert sdk.sdk_only() is True
    assert sdk.enabled() is True


# --- manifest validation ---------------------------------------------------


def test_manifest_round_trips(tmp_path, monkeypatch):
    _write_manifest(tmp_path, monkeypatch, _manifest())
    assert sdk.load_manifest()["operators"] == ["aten::mm", "aten::mm.out"]


def test_absent_manifest_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("FLAGOS_DCU_SDK_MANIFEST", str(tmp_path / "nope.json"))
    monkeypatch.setattr(sdk, "_package_dir", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="dcu_sdk_manifest.json"):
        sdk.load_manifest()


@pytest.mark.parametrize(
    "field", ["torch_base", "operators", "dtypes", "sdk_only", "dispatch_key"]
)
def test_missing_required_field_is_named(tmp_path, monkeypatch, field):
    manifest = _manifest()
    del manifest[field]
    _write_manifest(tmp_path, monkeypatch, manifest)
    with pytest.raises(RuntimeError, match=field):
        sdk.load_manifest()


def test_unexpected_dispatch_key_is_rejected(tmp_path, monkeypatch):
    _write_manifest(tmp_path, monkeypatch, _manifest(dispatch_key="CUDA"))
    with pytest.raises(RuntimeError, match="dispatch key"):
        sdk.load_manifest()


def test_future_schema_is_rejected(tmp_path, monkeypatch):
    _write_manifest(tmp_path, monkeypatch, _manifest(schema_version=3))
    with pytest.raises(RuntimeError, match="schema"):
        sdk.load_manifest()


def test_schema_two_requires_registration_metadata(tmp_path, monkeypatch):
    manifest = _manifest()
    del manifest["registration_abi"]
    _write_manifest(tmp_path, monkeypatch, manifest)
    with pytest.raises(RuntimeError, match="registration_abi"):
        sdk.load_manifest()


def test_schema_two_rejects_invalid_torch_version(tmp_path, monkeypatch):
    _write_manifest(tmp_path, monkeypatch, _manifest(torch_version={"major": 2}))
    with pytest.raises(RuntimeError, match="torch_version"):
        sdk.load_manifest()


def test_schema_two_rejects_empty_route_registry(tmp_path, monkeypatch):
    _write_manifest(tmp_path, monkeypatch, _manifest(route_count=0))
    with pytest.raises(RuntimeError, match="route_count"):
        sdk.load_manifest()


def test_schema_two_rejects_unknown_registration_abi(tmp_path, monkeypatch):
    _write_manifest(tmp_path, monkeypatch, _manifest(registration_abi=1))
    with pytest.raises(RuntimeError, match="registration ABI"):
        sdk.load_manifest()


def test_schema_two_accepts_generated_manifest(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    generated = json.loads(
        (repo_root / "torch_fl/configs/dcu_sdk_manifest.json").read_text()
    )
    _write_manifest(tmp_path, monkeypatch, generated)
    assert sdk.load_manifest()["route_count"] == 2034


def test_wrong_library_name_is_rejected(tmp_path, monkeypatch):
    _write_manifest(tmp_path, monkeypatch, _manifest(library="libsomething.so"))
    with pytest.raises(RuntimeError, match="libdcu_aten_ops.so"):
        sdk.load_manifest()


def test_empty_operator_list_is_rejected(tmp_path, monkeypatch):
    """An empty cohort would claim SDK-native coverage while providing none."""
    _write_manifest(tmp_path, monkeypatch, _manifest(operators=[]))
    with pytest.raises(RuntimeError, match="operators"):
        sdk.load_manifest()


def test_corrupt_manifest_names_the_file(tmp_path, monkeypatch):
    path = tmp_path / "dcu_sdk_manifest.json"
    path.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("FLAGOS_DCU_SDK_MANIFEST", str(path))
    with pytest.raises(RuntimeError, match="dcu_sdk_manifest.json"):
        sdk.load_manifest()


# --- SDK-only process assertions ------------------------------------------


_OFFICIAL_MAPS = (
    "7f0000000000-7f0000001000 r-xp 00000000 00:1b 101   "
    "/usr/lib/python3/site-packages/torch/lib/libtorch_cpu.so\n"
    "7f0000002000-7f0000003000 r-xp 00000000 00:1b 102   "
    "/usr/lib/python3/site-packages/torch/lib/libc10.so\n"
    "7f0000004000-7f0000005000 r-xp 00000000 00:1b 103   /opt/dtk/lib/librocblas.so.4\n"
    "7f0000006000-7f0000007000 rw-p 00000000 00:00 0     [heap]\n"
)


def test_official_core_plus_sdk_is_accepted(monkeypatch):
    """rocBLAS under /opt/dtk is the whole point; only DTK's *torch* is forbidden."""
    assert sdk._vendor_mappings(_OFFICIAL_MAPS) == []


def test_vendor_device_library_is_reported():
    maps = _OFFICIAL_MAPS + (
        "7f0000008000-7f0000009000 r-xp 00000000 00:1b 104   "
        "/some/env/torch/lib/libtorch_hip.so\n"
    )
    found = sdk._vendor_mappings(maps)
    assert len(found) == 1
    assert "libtorch_hip.so" in found[0]


def test_vendor_core_from_the_bundle_dir_is_reported():
    """libtorch_cpu.so exists in both wheels, so the loading path decides."""
    maps = _OFFICIAL_MAPS + (
        "7f000000a000-7f000000b000 r-xp 00000000 00:1b 105   "
        "/x/torch_fl/lib_dcu/libtorch_cpu.so\n"
    )
    found = sdk._vendor_mappings(maps)
    assert len(found) == 1
    assert "lib_dcu/libtorch_cpu.so" in found[0]


def test_assert_is_a_noop_outside_sdk_only(monkeypatch):
    monkeypatch.setattr(
        sdk,
        "_vendor_mappings",
        lambda maps: pytest.fail("must not inspect maps in hybrid mode"),
    )
    sdk.assert_sdk_only_process()


def test_assert_raises_in_sdk_only_when_vendor_torch_is_mapped(monkeypatch):
    monkeypatch.setenv("FLAGOS_DCU_SDK_ONLY", "1")
    monkeypatch.setattr(sdk, "_vendor_mappings", lambda maps: ["libtorch_hip.so (/x)"])
    with pytest.raises(RuntimeError) as excinfo:
        sdk.assert_sdk_only_process()
    message = str(excinfo.value)
    assert "libtorch_hip.so" in message
    assert "FLAGOS_DCU_SDK_ONLY" in message


# --- discovery -------------------------------------------------------------


def test_explicit_plugin_override_wins(tmp_path, monkeypatch):
    plugin = tmp_path / "libdcu_aten_ops.so"
    plugin.write_bytes(b"\x7fELF fake")
    monkeypatch.setenv("FLAGOS_DCU_SDK_OPS_LIB", str(plugin))
    assert sdk.discover_plugin() == str(plugin)


def test_manifest_is_found_next_to_the_plugin(tmp_path, monkeypatch):
    plugin = tmp_path / "libdcu_aten_ops.so"
    plugin.write_bytes(b"\x7fELF fake")
    (tmp_path / "dcu_sdk_manifest.json").write_text(
        json.dumps(_manifest()), encoding="utf-8"
    )
    assert sdk.discover_manifest(str(plugin)) == str(tmp_path / "dcu_sdk_manifest.json")


def test_missing_plugin_is_reported_as_none(tmp_path, monkeypatch):
    monkeypatch.setattr(sdk, "_package_dir", lambda: tmp_path)
    monkeypatch.setattr(sdk, "_sdk_root", lambda: str(tmp_path))
    assert sdk.discover_plugin() is None
