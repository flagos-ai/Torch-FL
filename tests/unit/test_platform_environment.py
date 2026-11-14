# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime policies retain native checks and failure evidence."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from manifest_composer import environment_check as CHECKS, platform_config


PLATFORMS = [
    "ascend",
    "iluvatar",
    "kunlunxin",
    "mthreads",
    "thead",
    "enflame",
    "metax",
    "hygon",
]


@pytest.mark.parametrize("platform", PLATFORMS)
def test_migrated_lanes_share_isolation_without_borrowing_another_core_policy(platform):
    config = platform_config(platform)
    for key in ("environment_check", "boxing_environment_check"):
        if key not in config:
            continue
        spec = config[key]
        CHECKS.validate_spec(spec)
        assert spec["venv_isolation"] is True
        if platform != "iluvatar":
            assert "loaded_torch_libraries" not in spec
            assert "triton_target" not in spec


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("devices", [1, 2, 4, 8, 16])
def test_migrated_lanes_require_two_devices_without_a_count_ceiling(
    platform, devices, monkeypatch
):
    dependencies = SimpleNamespace(
        torch=SimpleNamespace(
            __file__="/venv/torch/__init__.py",
            __version__="2.10.0+cpu",
            version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(
                is_available=lambda: True, device_count=lambda: devices
            ),
        ),
        torch_fl=SimpleNamespace(
            __file__="/venv/torch_fl/__init__.py",
            flagos=SimpleNamespace(
                is_available=lambda: True, device_count=lambda: devices
            ),
        ),
    )
    monkeypatch.setattr(CHECKS, "import_dependencies", lambda _: dependencies)
    monkeypatch.delenv("FLAGCICD_REPORT_DIR", raising=False)
    config = platform_config(platform)
    for key in ("environment_check", "boxing_environment_check"):
        if key not in config:
            continue
        policy = config[key]
        assert policy["flagos_devices"] == {"at_least": 2}
        if "cuda_devices" in policy:
            assert policy["cuda_devices"] == {"at_least": 2}
        spec = {
            key: value
            for key, value in policy.items()
            if key in ("flagos_devices", "cuda_devices", "cuda_available")
        }
        if devices < 2:
            with pytest.raises(AssertionError, match="device count at_least 2"):
                CHECKS.validate_environment(spec)
        else:
            assert CHECKS.validate_environment(spec) == 0


def test_migrated_lanes_expose_all_runner_devices_with_the_ppu_access_exception():
    import shlex

    for platform in PLATFORMS:
        config = platform_config(platform)
        options = shlex.split(config["container_options"])
        if platform == "thead":
            assert "--privileged" not in options
            assert "--device=/dev/alixpu" in options
            assert "--device=/dev/alixpu_ctl" in options
            assert all(f"--device=/dev/alixpu_ppu{i}" in options for i in range(16))
        else:
            assert "--privileged" in options
            assert not any(option.startswith("--device=") for option in options)
        for name, value in config.get("integration_environment", {}).items():
            if name.endswith("_VISIBLE_DEVICES"):
                assert value == "all", (platform, name, value)
        for option in options:
            if "_VISIBLE_DEVICES=" in option:
                assert option.endswith("=all"), (platform, option)


@pytest.mark.parametrize(
    "invalid",
    [
        {"venv_isolation": "true"},
        {"torch_first": 1},
        {"loaded_torch_libraries": ["../libc10.so"]},
        {"forbidden_loaded_library_prefixes": []},
        {
            "elf_dependencies": {
                "files": ["../escape.so"],
                "forbidden": ["libtorch_cuda"],
            }
        },
        {"elf_dependencies": {"files": ["lib/test.so"], "forbiden": ["libtorch_cuda"]}},
        {"triton_target": {"backend": "corex", "arch": 71, "warp_size": 64}},
    ],
)
def test_invalid_native_policies_fail_before_import(invalid, monkeypatch, tmp_path):
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(tmp_path))
    monkeypatch.setenv("REPORT_PLATFORM", "iluvatar")
    monkeypatch.setattr(
        CHECKS,
        "import_dependencies",
        lambda _: pytest.fail("invalid policy imported the device stack"),
    )
    with pytest.raises(ValueError):
        CHECKS.validate_environment(invalid)
    report = json.loads((tmp_path / "hardware-environment-iluvatar.json").read_text())
    assert report["valid"] is False
    assert report["expected"] == invalid
    assert report["checks"][0]["name"] == "policy"
    assert report["checks"][0]["status"] == "failed"
    assert report["checks"][0]["error"]


@pytest.mark.parametrize("raw", [None, "broken JSON", "[]", '{"unknown": true}'])
def test_unreadable_environment_policy_retains_failure_evidence(
    raw, tmp_path, monkeypatch
):
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(tmp_path))
    monkeypatch.setenv("REPORT_PLATFORM", "iluvatar")
    if raw is None:
        monkeypatch.delenv(CHECKS.ENVIRONMENT_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(CHECKS.ENVIRONMENT_VARIABLE, raw)
    monkeypatch.setattr(
        CHECKS, "import_dependencies", lambda _: pytest.fail("imported invalid policy")
    )
    with pytest.raises(SystemExit):
        CHECKS.validate_environment()
    report = json.loads((tmp_path / "hardware-environment-iluvatar.json").read_text())
    assert report["valid"] is False
    assert report["checks"][0]["expected"] == raw
    assert report["checks"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "problem", [None, "backend", "arch", "warp_size", "capability"]
)
def test_corex_target_is_checked_against_platform_data(problem):
    expected = platform_config("iluvatar")["environment_check"]["triton_target"]
    target = SimpleNamespace(backend="corex", arch=71, warp_size=64)
    if problem in ("backend", "arch", "warp_size"):
        setattr(target, problem, "wrong")
    dependencies = SimpleNamespace(
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_capability=lambda: (
                    (0, 0) if problem == "capability" else (7, 1)
                )
            )
        ),
        triton=SimpleNamespace(
            runtime=SimpleNamespace(
                driver=SimpleNamespace(
                    active=SimpleNamespace(get_current_target=lambda: target)
                )
            )
        ),
    )
    if problem:
        with pytest.raises(AssertionError):
            CHECKS.assert_triton_target(dependencies, expected)
    else:
        assert CHECKS.assert_triton_target(dependencies, expected) == expected


@pytest.mark.parametrize("forbidden", [False, True])
def test_extension_dependency_policy_keeps_the_original_vendor_core_rejection(
    tmp_path, monkeypatch, forbidden
):
    policy = platform_config("iluvatar")["environment_check"]["elf_dependencies"]
    calls = []

    def read(command, **kwargs):
        calls.append(command)
        return "NEEDED libtorch_cuda.so" if forbidden else "NEEDED libc10.so"

    monkeypatch.setattr(CHECKS.subprocess, "check_output", read)
    if forbidden:
        with pytest.raises(AssertionError, match="libtorch_cuda"):
            CHECKS.assert_elf_dependencies(tmp_path, policy)
    else:
        assert len(CHECKS.assert_elf_dependencies(tmp_path, policy)) == 2
        assert all(Path(command[-1]).is_relative_to(tmp_path) for command in calls)


@pytest.mark.parametrize(
    "failure", [None, "foreign-package", "wrong-target", "device", "import"]
)
def test_hardware_validation_retains_success_and_failure_without_changing_dependency_evidence(
    tmp_path, monkeypatch, failure
):
    root = tmp_path / "venv"
    root.mkdir()
    (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    monkeypatch.setenv("VIRTUAL_ENV", str(root))
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(tmp_path))
    monkeypatch.setenv("REPORT_PLATFORM", "iluvatar")
    snapshot = tmp_path / "environment-iluvatar.json"
    snapshot.write_text('{"valid": true, "wheel": {"sha256": "artifact"}}')
    before = snapshot.read_bytes()
    monkeypatch.setattr(
        CHECKS, "sys", SimpleNamespace(prefix=str(root), base_prefix="/usr", path=[])
    )
    target = SimpleNamespace(
        backend="wrong" if failure == "wrong-target" else "corex", arch=71, warp_size=64
    )
    dependencies = SimpleNamespace(
        torch=SimpleNamespace(
            __file__=str(root / "site-packages/torch/__init__.py"),
            __version__="2.10.0+cpu",
            version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(get_device_capability=lambda: (7, 1)),
        ),
        torch_fl=SimpleNamespace(
            __file__=str(
                (tmp_path if failure == "foreign-package" else root)
                / "site-packages/torch_fl/__init__.py"
            ),
            flagos=SimpleNamespace(
                is_available=lambda: failure != "device", device_count=lambda: 2
            ),
        ),
        triton=SimpleNamespace(
            __file__=str(root / "site-packages/triton/__init__.py"),
            runtime=SimpleNamespace(
                driver=SimpleNamespace(
                    active=SimpleNamespace(get_current_target=lambda: target)
                )
            ),
        ),
        flag_gems=None,
    )

    def import_stack(_):
        if failure == "import":
            raise ImportError("device SDK library cannot be loaded")
        return dependencies

    monkeypatch.setattr(CHECKS, "import_dependencies", import_stack)
    spec = {
        "venv_isolation": True,
        "torch_version_prefix": "2.10.0+cpu",
        "cuda_version": None,
        "flagos_devices": {"at_least": 1},
        "triton_target": platform_config("iluvatar")["environment_check"][
            "triton_target"
        ],
    }
    if failure:
        with pytest.raises(ImportError if failure == "import" else AssertionError):
            CHECKS.validate_environment(spec)
    else:
        assert CHECKS.validate_environment(spec) == 0
    report = json.loads((tmp_path / "hardware-environment-iluvatar.json").read_text())
    assert report["valid"] is (failure is None)
    assert report["phase"] == "hardware-environment"
    assert report["expected"] == spec
    assert report["checks"]
    if failure:
        assert report["checks"][-1]["status"] == "failed"
        assert report["checks"][-1]["error"]
    else:
        assert report["checks"][-1]["actual"] == spec["triton_target"]
    assert snapshot.read_bytes() == before
    assert list(tmp_path.glob("tmp*")) == []


def test_iluvatar_computations_follow_the_common_environment_gate():
    from manifest_composer import composed_steps

    steps = composed_steps("iluvatar")
    environment = next(
        step for step in steps if "check_platform_environment.py" in step["command"]
    )
    smoke = next(
        step for step in steps if "test_iluvatar_runtime.py" in step["command"]
    )
    assert steps.index(environment) < steps.index(smoke)
    assert "python -m pytest" in smoke["command"]
    expected = json.loads(environment["environment"][CHECKS.ENVIRONMENT_VARIABLE])
    assert expected["torch_first"] is True
    assert expected["torch_version_exact"] == "2.10.0+cpu"
    assert expected["torch_distribution_version"] == "2.10.0+cpu"
    assert expected["loaded_torch_libraries"] == [
        "libc10.so",
        "libtorch_cpu.so",
        "libtorch_python.so",
    ]
    assert expected["triton_target"] == {
        "backend": "corex",
        "arch": 71,
        "warp_size": 64,
        "device_capability": [7, 1],
    }


def test_private_vendor_facade_preserves_the_venvs_python_package_ownership(
    tmp_path, monkeypatch
):
    """PPU/MetaX may select bundled cores through a facade outside the venv."""
    root = tmp_path / "venv"
    root.mkdir()
    (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    torch = root / "site-packages/torch/__init__.py"
    torch.parent.mkdir(parents=True)
    torch.touch()
    overlay = tmp_path / "private-facade"
    (overlay / "torch").mkdir(parents=True)
    (overlay / "torch/__init__.py").symlink_to(torch)
    monkeypatch.setenv("VIRTUAL_ENV", str(root))
    monkeypatch.setattr(
        CHECKS,
        "sys",
        SimpleNamespace(
            prefix=str(root),
            base_prefix="/usr",
            path=[str(overlay), str(root / "site-packages")],
        ),
    )
    dependencies = SimpleNamespace(
        torch=SimpleNamespace(__file__=str(overlay / "torch/__init__.py")),
        torch_fl=SimpleNamespace(
            __file__=str(root / "site-packages/torch_fl/__init__.py")
        ),
        triton=None,
        flag_gems=None,
    )
    actual = CHECKS.assert_venv_isolation(dependencies)
    assert actual["packages"]["torch"] == str(torch)


def assert_cpu_core(torch_root, package, mappings):
    spec = platform_config("iluvatar")["environment_check"]
    CHECKS.assert_package_files(
        {"forbid_package_files": spec["forbid_package_files"]}, package
    )
    CHECKS.assert_loaded_torch_libraries(
        torch_root,
        spec["loaded_torch_libraries"],
        spec["forbidden_loaded_library_prefixes"],
        mappings,
    )


def cpu_mappings(root):
    return "\n".join(
        f"1000-2000 r-xp 00000000 08:01 123 {root}/lib/{name}"
        for name in ("libc10.so", "libtorch_cpu.so", "libtorch_python.so")
    )


def test_official_cpu_core_allows_corex_device_libraries(tmp_path):
    torch = tmp_path / "torch"
    mappings = cpu_mappings(torch)
    mappings += "\n1000-2000 r-xp 0 08:01 1 /usr/local/corex/lib64/libcudart.so"
    assert_cpu_core(torch, tmp_path / "torch_fl", mappings)


@pytest.mark.parametrize("name", ["libtorch_cuda.so", "libc10_cuda.so"])
def test_vendor_cuda_core_is_rejected(tmp_path, name):
    torch = tmp_path / "torch"
    mappings = cpu_mappings(torch)
    mappings += f"\n1000-2000 r-xp 0 08:01 1 /vendor/torch/lib/{name}"
    with pytest.raises(AssertionError, match="Forbidden vendor Torch libraries"):
        assert_cpu_core(torch, tmp_path / "torch_fl", mappings)


def test_vendor_cpu_core_cannot_impersonate_the_official_distribution(tmp_path):
    torch = tmp_path / "torch"
    with pytest.raises(AssertionError, match="expected torch/lib"):
        assert_cpu_core(
            torch, tmp_path / "torch_fl", cpu_mappings(tmp_path / "vendor/torch")
        )


def test_a_second_loaded_cpu_core_is_rejected(tmp_path):
    torch = tmp_path / "torch"
    mappings = cpu_mappings(torch) + "\n" + cpu_mappings(tmp_path / "vendor/torch")
    with pytest.raises(AssertionError, match="second libc10.so"):
        assert_cpu_core(torch, tmp_path / "torch_fl", mappings)


def test_stale_wheel_bundle_is_rejected(tmp_path):
    package = tmp_path / "torch_fl"
    (package / "lib_iluvatar").mkdir(parents=True)
    with pytest.raises(AssertionError):
        assert_cpu_core(tmp_path / "torch", package, "")


def test_runtime_venv_root_uses_the_job_environment(tmp_path, monkeypatch):
    root = tmp_path / "job-venv"
    root.mkdir()
    (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    monkeypatch.setenv("VIRTUAL_ENV", str(root))
    monkeypatch.setattr(
        CHECKS, "sys", SimpleNamespace(prefix=str(root), base_prefix="/usr")
    )
    assert CHECKS.isolated_venv_root() == root
    CHECKS.assert_venv_package(root, root / "lib/python3.12/site-packages/torch")
    with pytest.raises(AssertionError, match="outside CI venv"):
        CHECKS.assert_venv_package(root, tmp_path / "image/torch")


@pytest.mark.parametrize("problem", ["system-site", "wrong-prefix", "base-python"])
def test_runtime_rejects_shared_or_misidentified_python(tmp_path, monkeypatch, problem):
    root = tmp_path / "job-venv"
    root.mkdir()
    (root / "pyvenv.cfg").write_text(
        f"include-system-site-packages = {'true' if problem == 'system-site' else 'false'}\n"
    )
    monkeypatch.setenv(
        "VIRTUAL_ENV", str(root / "other" if problem == "wrong-prefix" else root)
    )
    monkeypatch.setattr(
        CHECKS,
        "sys",
        SimpleNamespace(
            prefix=str(root),
            base_prefix=str(root) if problem == "base-python" else "/usr",
        ),
    )
    with pytest.raises(AssertionError):
        CHECKS.isolated_venv_root()
