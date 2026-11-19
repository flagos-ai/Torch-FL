# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise native-only config selection and generated registration without Torch."""

import ast
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def functions(relative, names, namespace):
    source = ROOT / relative
    tree = ast.parse(source.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace
    )
    return namespace


def routes(path):
    return dict(re.findall(r"^([^#=\n]+?)\s*=\s*(\w+)", path.read_text(), re.M))


def test_native_routes_and_claims_are_generated_from_one_policy():
    generator = load("scripts/codegen/gen_vendor_confs.py", "metax_native_confs")
    registration = load(
        "scripts/codegen/codegen_boxing_flaggems_register.py", "metax_native_claims"
    )
    conf_dir = ROOT / "torch_fl/configs"
    ordinary = (conf_dir / "backends_metax.conf").read_bytes()
    built = generator.build_all(conf_dir)
    native_text, native, _ = built["metax_native_host"]
    assert native_text == (conf_dir / "backends_metax_native_host.conf").read_text()
    base = built["metax"][1]
    policy = generator.NATIVE_HOST_FLAGGEMS_OPS["metax"]
    assert native.keys() == base.keys()
    changed = {op for op in native if native[op] != base[op]}
    assert (
        changed - policy
        == {op for op in base if base[op] == "flaggems_cpp" and op not in policy}
        | generator.METAX_COMPOSITE_FLAGGEMS
    )
    assert "flaggems_cpp" not in native.values()
    assert {native[op] for op in policy} == {"flaggems"}
    assert (conf_dir / "backends_metax.conf").read_bytes() == ordinary
    text = registration.render_native_hosts(
        "metax", registration.wrapper_map(), registration.flaggems_kernel_ops()
    )
    path = ROOT / "csrc/aten/generated/register_flaggems_metax_native_host.inc"
    assert text == path.read_text()
    claimed = set(re.findall(r'm\.impl\("([^"]+)"', text))
    assert claimed == {op for op in policy if base[op] == "none"}
    assert not (
        claimed
        & set(
            re.findall(
                r'm\.impl\("([^"]+)"',
                (path.parent / "register_flaggems_metax.inc").read_text(),
            )
        )
    )
    assert not ({"boxing", "cuda", "nvidia", "metax"} & set(native.values()))


def test_native_claim_generation_rejects_an_unavailable_kernel():
    registration = load(
        "scripts/codegen/codegen_boxing_flaggems_register.py", "metax_native_missing"
    )
    with pytest.raises(SystemExit, match="no FlagGems kernel"):
        registration.render_native_hosts("metax", registration.wrapper_map(), set())


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("runtime_value", ["0", "1"])
def test_build_record_selects_native_config_before_torch_import(
    tmp_path, monkeypatch, native, runtime_value
):
    package = ModuleType("torch_fl")
    package._build_config = SimpleNamespace(NATIVE_HOSTS=native)
    monkeypatch.setitem(sys.modules, "torch_fl", package)
    monkeypatch.setenv("FLAGOS_BUILD_NATIVE_HOST", runtime_value)
    monkeypatch.delenv("FLAGOS_BACKEND_CONFIG", raising=False)
    directory = tmp_path / "configs"
    directory.mkdir()
    for name in ("backends_metax.conf", "backends_metax_native_host.conf"):
        (directory / name).write_text("")
    namespace = functions(
        "torch_fl/__init__.py",
        ["_select_backend_config"],
        {
            "os": os,
            "__file__": str(tmp_path / "__init__.py"),
            "_env": SimpleNamespace(value=os.getenv),
            "_platform": SimpleNamespace(
                platform_for=lambda value: value, vendor_for=lambda value: value
            ),
            "_build_accelerator": lambda: "metax",
            "_is_ppu_build": lambda: False,
        },
    )
    namespace["_select_backend_config"]()
    name = "backends_metax_native_host.conf" if native else "backends_metax.conf"
    assert namespace["_BACKEND_CONFIG_PATH"] == str(directory / name)
    assert "FLAGOS_BACKEND_CONFIG" not in os.environ
    if native:
        (directory / name).unlink()
        with pytest.raises(RuntimeError, match="missing its generated config"):
            namespace["_select_backend_config"]()
        override = tmp_path / "explicit.conf"
        override.write_text("")
        monkeypatch.setenv("FLAGOS_BACKEND_CONFIG", str(override))
        namespace["_select_backend_config"]()
        assert namespace["_BACKEND_CONFIG_PATH"] == str(override)


@pytest.mark.parametrize(
    "vendor,native,expected",
    [
        ("metax", False, False),
        ("metax", True, True),
        ("nvidia", False, True),
        ("hygon", False, True),
        ("iluvatar", True, False),
    ],
)
def test_device_identity_alignment_is_scoped_to_the_measured_build(
    monkeypatch, vendor, native, expected
):
    package = ModuleType("flag_gems")
    package.__spec__ = importlib.machinery.ModuleSpec(
        "flag_gems", loader=None, is_package=True
    )
    package.__path__ = ["/synthetic/flag_gems"]
    package.device = "cuda"
    operation = ModuleType("flag_gems.ops.mul")
    operation._DEVICE_NAME = "cuda"
    unrelated = ModuleType("unrelated_metax_test_module")
    unrelated.device = "cuda"
    detector = SimpleNamespace(vendor_name=vendor, name="cuda")
    finder = ModuleType("flag_gems.runtime.backend.device_finder")
    finder.DeviceDetector = lambda: detector
    for module in (package, operation, unrelated, finder):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = functions(
        "torch_fl/accelerator/nvidia/_cuda_compat.py",
        [
            "patch_flaggems_device_name",
            "_flag_gems_package_root",
            "_is_flag_gems_module",
        ],
        {
            "os": os,
            "_ALIGNED_VENDORS": ("nvidia", "hygon"),
            "_ALIGNED_NATIVE_HOST_VENDORS": ("metax",),
            "_FLAGGEMS_DEVICE_NAME": "flagos",
            "_VENDOR_DEVICE_NAME": "cuda",
            "torch": SimpleNamespace(cuda=SimpleNamespace(get_device_properties=None)),
        },
    )
    assert namespace["patch_flaggems_device_name"](native_hosts=native) is expected
    assert detector.name == ("flagos" if expected else "cuda")
    assert package.device == operation._DEVICE_NAME == detector.name
    assert unrelated.device == "cuda"
