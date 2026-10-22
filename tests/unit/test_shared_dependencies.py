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

"""The shared dependency install must reject incomplete metadata cohorts."""

import importlib.metadata
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/verify_shared_dependencies.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_shared_dependencies", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_follows_installed_transitive_dependencies(monkeypatch):
    module = _load()

    def distribution(name, requires=()):
        return SimpleNamespace(
            metadata={"Name": name}, version="1", requires=list(requires)
        )

    installed = {
        "root": distribution("root", ["parent>=1"]),
        "parent": distribution("parent", ["leaf"]),
        "numpy": distribution("numpy"),
    }
    numpy = installed["numpy"]
    numpy.version = "1.26.4"
    monkeypatch.setattr(module, "PACKAGES", {"root": "1", "numpy": "1.26.4"})
    monkeypatch.setattr(
        module.importlib.metadata, "distributions", lambda: installed.values()
    )
    with pytest.raises(SystemExit, match="leaf"):
        module.validate_cohort()

    installed["leaf"] = distribution("leaf")
    module.validate_cohort()


def test_typer_lists_annotated_doc_as_an_explicit_pin():
    module = _load()
    assert module.PACKAGES["annotated-doc"] == "0.0.5"
    spec = importlib.util.spec_from_file_location(
        "imports", SCRIPT.with_name("verify_shared_dependency_imports.py")
    )
    imports = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imports)
    assert imports.MODULE_IMPORT_NAMES["annotated-doc"] == "annotated_doc"
    assert imports.MODULE_IMPORT_NAMES["markdown-it-py"] == "markdown_it"
    probes = imports.module_imports(module.PACKAGES)
    assert set(probes) == set(module.PACKAGES)
    assert probes["pyyaml"] == "yaml"
    assert probes["numpy"] == "numpy"
    with pytest.raises(SystemExit, match="new-package"):
        imports.module_imports({**module.PACKAGES, "new-package": "1"})


def test_onnxruntime_metadata_dependencies_are_explicit_pins():
    module = _load()
    assert module.PACKAGES["sympy"] == "1.14.0"
    assert module.PACKAGES["typing-extensions"] == "4.16.0"
    assert module.PACKAGES["mpmath"] == "1.3.0"


def test_common_pins_satisfy_installed_package_metadata(monkeypatch):
    module = _load()
    installed = [
        SimpleNamespace(metadata={"Name": name}, version=version, requires=[])
        for name, version in module.PACKAGES.items()
    ]
    requirements = {
        "anyio": ["typing_extensions>=4.16.0; python_version < '3.15'"],
        "torch": ["typing-extensions>=4.10.0"],
        "sympy": ["mpmath<1.4,>=1.1.0"],
    }
    installed.append(
        SimpleNamespace(
            metadata={"Name": "torch"},
            version="2.10.0",
            requires=requirements["torch"],
        )
    )
    anyio = next(
        dist
        for dist in installed
        if module._canonical_name(dist.metadata["Name"]) == "anyio"
    )
    anyio.requires = requirements["anyio"]
    sympy = next(
        dist
        for dist in installed
        if module._canonical_name(dist.metadata["Name"]) == "sympy"
    )
    sympy.requires = requirements["sympy"]
    monkeypatch.setattr(module.importlib.metadata, "distributions", lambda: installed)

    module.validate_cohort()


def test_active_site_roots_reads_the_running_interpreters_site_directories(
    monkeypatch, tmp_path
):
    module = _load()
    purelib = tmp_path / "site-packages"
    execute = module.subprocess.run

    def run(command, **kwargs):
        assert command[0] == module.sys.executable
        if "sysconfig" in command[-1]:
            return module.subprocess.CompletedProcess(command, 0, str(purelib), "")
        # Run this fragment as written to catch a missing import in CI.
        return execute(
            [module.sys.executable, "-c", command[-1]],
            text=True,
            capture_output=True,
            check=True,
        )

    monkeypatch.setattr(module, "_active_purelib", lambda: purelib)
    monkeypatch.setattr(module.subprocess, "run", run)

    roots = module._active_site_roots()
    assert roots[0] == purelib.resolve()
    assert tmp_path.resolve() not in roots


def test_unloadable_flag_gems_fails_after_torch_startup(monkeypatch):
    module = _load()
    imports = []
    execute = module.__builtins__["__import__"]

    def import_module(name, *args, **kwargs):
        if name == "flag_gems":
            imports.append(name)
            raise ModuleNotFoundError("No module named 'flag_gems'", name=name)
        return execute(name, *args, **kwargs)

    monkeypatch.setitem(module.__builtins__, "__import__", import_module)
    monkeypatch.setattr(
        module, "_import_torch_fl", lambda: imports.append("torch_fl") or True
    )
    monkeypatch.setattr(
        module,
        "probe_environment",
        lambda: imports.append("environment") or {},
    )
    with pytest.raises(SystemExit, match="FlagGems is required"):
        module.check({}, platform="cuda", flag_gems_required=True)
    assert imports == ["torch_fl", "environment", "flag_gems"]


def test_qwen_rotation_none_is_not_treated_as_an_import_error(monkeypatch):
    module = _load()
    monkeypatch.setattr(
        module,
        "_import_torch_fl",
        lambda: _raise(AssertionError("startup not reached")),
    )
    monkeypatch.setattr(
        module,
        "flag_gems_vendor",
        lambda: _raise(
            ModuleNotFoundError("No module named 'flag_gems'", name="flag_gems")
        ),
    )
    with pytest.raises(AssertionError, match="startup not reached"):
        module.check({}, platform="cuda", flag_gems_required=True)


def test_flag_gems_vendor_comes_from_the_startup_contract(monkeypatch):
    module = _load()
    flag_gems = ModuleType("flag_gems")
    flag_gems.vendor_name = "Iluvatar"
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    assert module.flag_gems_vendor() == "iluvatar"


def test_ascend_registers_flag_tree_tle_before_loading_flag_gems(monkeypatch):
    module = _load()

    monkeypatch.delitem(sys.modules, "shmem", raising=False)
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "flag_gems",
        SimpleNamespace(
            vendor_name="ascend",
            __getattr__=lambda name: calls.append(name),
        ),
    )
    helper = ModuleType("torch_fl.compile.flagtree_ascend_tle")
    helper.ensure_tle_importable = lambda: sys.modules.__setitem__("shmem", object())
    monkeypatch.setitem(sys.modules, "torch_fl.compile.flagtree_ascend_tle", helper)

    module.prepare_flag_gems_import("ascend")
    assert sys.modules["shmem"]
    flag_gems = module.import_flag_gems()
    assert flag_gems.vendor_name == "ascend"
    assert calls == []


def test_qwen_rotation_is_checked_for_its_startup_vendors(monkeypatch, tmp_path):
    module = _load()
    wheel = tmp_path / "site-packages"
    torch_package = wheel / "torch"
    torch_package.mkdir(parents=True)
    source = '"""torch"""\n__version__ = "2.10.0"\n__file__ = __file__\n'
    (torch_package / "__init__.py").write_text(source, encoding="utf-8")
    diffusers = wheel / "diffusers" / "models" / "transformers"
    (wheel / "diffusers").mkdir()
    (wheel / "diffusers" / "models").mkdir()
    diffusers.mkdir(parents=True)
    (wheel / "diffusers" / "__init__.py").write_text("", encoding="utf-8")
    (wheel / "diffusers" / "models" / "__init__.py").write_text("", encoding="utf-8")
    (diffusers / "__init__.py").write_text("", encoding="utf-8")
    (diffusers / "transformer_qwenimage.py").write_text(
        'ROPE_PER_DEVICE = {"cuda": object()}\n'
        "QwenEmbedLayer3DRope = object()\n"
        "def apply_rotary_emb_qwen_neuron(x, freqs):\n"
        "    return x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(locate_file=lambda path: torch_package),
    )
    monkeypatch.setattr(
        module.importlib.metadata,
        "version",
        lambda name: "2.10.0" if name == "torch" else "",
    )
    monkeypatch.setattr(module, "PACKAGES", {})
    monkeypatch.setattr(module, "MODULE_IMPORTS", {})
    monkeypatch.setattr(module, "TORCH_STARTUP_IMPORTS", set())
    monkeypatch.setattr(module, "_import_torch_fl", lambda: True)
    monkeypatch.setattr(module, "_installed_package_root", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(__file__=str(torch_package / "__init__.py")),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.syspath_prepend(wheel)
    monkeypatch.setattr(module, "flag_gems_vendor", lambda **kwargs: "iluvatar")
    transformer = SimpleNamespace(
        ROPE_PER_DEVICE={"cuda": object()},
        QwenEmbedLayer3DRope=object(),
        apply_rotary_emb_qwen_neuron=lambda x, freqs: x,
    )
    monkeypatch.setattr(module.importlib, "import_module", lambda name: transformer)
    rotator = ModuleType("verify_shared_dependency_rotator")
    rotator.rotator = lambda: lambda x, freqs: x
    monkeypatch.setitem(sys.modules, "verify_shared_dependency_rotator", rotator)
    with pytest.raises(SystemExit, match="iluvatar: torch_fl did not install"):
        module.check({}, platform="cuda", flag_gems_required=True)


def _raise(error):
    raise error


@pytest.mark.parametrize("platform", ["cuda", "metax", "ppu"])
def test_other_platforms_do_not_load_the_ascend_tle_helper(monkeypatch, platform):
    module = _load()
    monkeypatch.setitem(sys.modules, "torch_fl.compile.flagtree_ascend_tle", None)

    module.prepare_flag_gems_import(platform)


def test_import_probe_failure_reports_the_child_error(monkeypatch):
    module = _load()
    result = SimpleNamespace(
        returncode=1,
        args=["python", "-c", "import torch_fl; import onnx"],
        stderr="ImportError: shared probe failed\n",
    )
    monkeypatch.setattr(module, "_import_torch_fl", lambda: True)
    monkeypatch.setattr(module, "_installed_package_root", lambda: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(SystemExit) as error:
        module.check({"onnx": "onnx"}, platform="cuda", flag_gems_required=False)
    assert "onnx: shared import probe failed" in str(error.value)
    assert "ImportError: shared probe failed" in str(error.value)


def test_numpy_two_does_not_match_the_common_pin(monkeypatch):
    module = _load()
    numpy = SimpleNamespace(metadata={"Name": "numpy"}, version="2.4.0", requires=[])
    monkeypatch.setattr(
        module.importlib.metadata,
        "distributions",
        lambda: [numpy],
    )
    with pytest.raises(SystemExit, match="numpy is 2.4.0, expected 1.26.4"):
        module.validate_cohort()


def test_shared_transformers_pin_matches_its_installed_requirements(monkeypatch):
    module = _load()
    installed = {
        "numpy": SimpleNamespace(
            metadata={"Name": "numpy"}, version="1.26.4", requires=[]
        ),
        "transformers": SimpleNamespace(
            metadata={"Name": "transformers"},
            version=module.PACKAGES["transformers"],
            requires=["tokenizers<0.24.0,>=0.23.1", "huggingface-hub<3.0,>=1.31.0"],
        ),
        "tokenizers": SimpleNamespace(
            metadata={"Name": "tokenizers"},
            version=module.PACKAGES["tokenizers"],
            requires=[],
        ),
        "huggingface-hub": SimpleNamespace(
            metadata={"Name": "huggingface-hub"},
            version=module.PACKAGES["huggingface-hub"],
            requires=[],
        ),
    }
    monkeypatch.setattr(
        module,
        "PACKAGES",
        {
            "transformers": module.PACKAGES["transformers"],
            "tokenizers": module.PACKAGES["tokenizers"],
            "huggingface-hub": module.PACKAGES["huggingface-hub"],
            "numpy": module.PACKAGES["numpy"],
        },
    )
    monkeypatch.setattr(
        module.importlib.metadata, "distributions", lambda: installed.values()
    )
    module.validate_cohort()


def test_every_common_pin_has_an_import_probe():
    module = _load()
    spec = importlib.util.spec_from_file_location(
        "verify_shared_dependency_imports",
        SCRIPT.with_name("verify_shared_dependency_imports.py"),
    )
    imports = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imports)
    with pytest.raises(SystemExit, match="missing.*new-package"):
        imports.module_imports(
            {**module.PACKAGES, "new-package": ""},
            pinned={*module.PACKAGES, "new-package"},
        )


def test_an_exact_dependency_is_checked_against_the_shared_pin(monkeypatch):
    module = _load()
    installed = {
        "root": SimpleNamespace(
            metadata={"Name": "root"}, version="1", requires=["leaf==1.0"]
        ),
        "numpy": SimpleNamespace(
            metadata={"Name": "numpy"}, version="1.26.4", requires=[]
        ),
    }
    monkeypatch.setattr(module, "PACKAGES", {"root": "1", "numpy": "1.26.4"})
    monkeypatch.setattr(
        module.importlib.metadata, "distributions", lambda: installed.values()
    )
    with pytest.raises(
        SystemExit, match="shared dependency cohort is incomplete: leaf"
    ):
        module.validate_cohort()

    installed["leaf"] = SimpleNamespace(
        metadata={"Name": "leaf"}, version="2.0", requires=[]
    )
    with pytest.raises(SystemExit, match=r"root 1 requires leaf==1\.0, found 2\.0"):
        module.validate_cohort()


def test_flag_gems_absence_fails_explicitly(monkeypatch):
    module = _load()
    original_import = __import__

    def missing(name, *args, **kwargs):
        if name == "flag_gems":
            raise ModuleNotFoundError(name="flag_gems")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", missing)
    with pytest.raises(SystemExit, match="FlagGems is required"):
        module.import_flag_gems()
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "flag_gems", ModuleType("flag_gems"))
    assert module.import_flag_gems() is sys.modules["flag_gems"]


@pytest.mark.parametrize(
    ("contract", "expected"),
    [
        ({"require_build_kernels": ["flaggems"]}, "required"),
        ({"forbid_build_kernels": ["flaggems"]}, "forbidden"),
        ({}, "optional"),
    ],
)
def test_flag_gems_requirement_comes_from_build_contract(contract, expected):
    module = _load()
    assert module.flag_gems_mode("platform", contract) == expected


def test_flag_gems_contract_rejects_conflicting_kernel_sets():
    module = _load()
    with pytest.raises(SystemExit, match="both requires and forbids"):
        module.requires_flag_gems(
            "platform",
            {
                "require_build_kernels": ["flaggems"],
                "forbid_build_kernels": ["flaggems"],
            },
        )


def test_dependency_script_allows_only_explicit_flag_gems_cohorts(monkeypatch):
    source = SCRIPT.read_text()
    assert 'choices=("required", "forbidden")' in source
    module = _load()
    monkeypatch.setattr(sys, "argv", ["script", "--platform", "mps"])
    with pytest.raises(SystemExit):
        module.arguments()


def test_only_non_flaggems_platforms_may_omit_flag_gems():
    module = _load()
    yaml = pytest.importorskip("yaml")
    configs = {
        path.stem: yaml.safe_load(path.read_text())["environment_check"]
        for path in (ROOT / ".github/configs").glob("*.yml")
    }
    cohort = {
        platform
        for platform, spec in configs.items()
        if module.requires_flag_gems(platform, spec)
    }
    assert set(configs) - cohort == {"mps"}
    assert "flaggems" in configs["mps"]["forbid_build_kernels"]


def test_shared_dependency_checks_use_the_loaded_platform_contract():
    yaml = pytest.importorskip("yaml")
    loader = yaml.safe_load(
        (ROOT / ".github/workflows/load-platform-config.yml").read_text()
    )
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/integration-tests.yml").read_text()
    )
    step = next(
        step
        for step in workflow["jobs"]["integration-tests"]["steps"]
        if step.get("name") == "Install shared test dependencies"
    )
    assert "--flag-gems" in step["run"]
    assert step["env"]["FLAG_GEMS_REQUIREMENT"] == (
        "${{ needs.configuration.outputs.flag_gems_requirement }}"
    )
    # PyYAML 1.1 reads the unquoted `on` key as a boolean.
    workflow_call = loader.get("on") or loader[True]
    assert "flag_gems_requirement" in workflow_call["workflow_call"]["outputs"]
    outputs = loader["jobs"]["configuration"]["outputs"]
    assert (
        "${{ steps.config.outputs.flag_gems_requirement }}"
        in outputs["flag_gems_requirement"]
    )
    assert "torch_fl_package_root" in workflow_call["workflow_call"]["outputs"]
    check = next(
        step
        for step in workflow["jobs"]["integration-tests"]["steps"]
        if step.get("name") == "Verify shared test dependencies"
    )
    assert check["env"]["TORCH_FL_INSTALLED_PACKAGE_ROOT"] == (
        "${{ needs.configuration.outputs.torch_fl_package_root }}"
    )
    config_step = next(
        step
        for step in loader["jobs"]["configuration"]["steps"]
        if step.get("id") == "config"
    )
    assert "FLAG_GEMS_REQUIRED=$(yq -r" in config_step["run"]
    assert "FLAG_GEMS_FORBIDDEN=$(yq -r" in config_step["run"]


def test_shared_dependency_workflow_ignores_setup_constraints():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/integration-tests.yml").read_text()
    )
    steps = workflow["jobs"]["integration-tests"]["steps"]
    install = next(
        step for step in steps if step.get("name") == "Install shared test dependencies"
    )
    check = next(
        step for step in steps if step.get("name") == "Verify shared test dependencies"
    )

    for step, verify in ((install, False), (check, True)):
        assert "unset PIP_CONSTRAINT" in step["run"].splitlines()
        command = next(
            line
            for line in step["run"].splitlines()
            if "verify_shared_dependencies.py" in line
        )
        assert "force-reinstall" not in command
        assert ("--check" in step["run"]) is verify


def test_shared_dependency_workflow_invokes_the_current_repository_script(
    monkeypatch, tmp_path
):
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/integration-tests.yml").read_text()
    )
    install = next(
        step
        for step in workflow["jobs"]["integration-tests"]["steps"]
        if step.get("name") == "Install shared test dependencies"
    )
    checkout = tmp_path / "checkout"
    (checkout / ".github" / "scripts").mkdir(parents=True)
    (checkout / ".github" / "scripts" / "verify_shared_dependencies.py").write_text(
        "print('current repository script')\n"
    )
    run = (
        f'cd "{checkout}"\n'
        "unset PIP_CONSTRAINT\n"
        "python .github/scripts/verify_shared_dependencies.py \\\n"
        '  --platform "cuda" \\\n'
        '  --flag-gems "required"\n'
    )
    assert "verify_shared_dependencies.py" in install["run"]
    completed = subprocess.run(
        ["bash", "-c", run],
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
        },
    )
    assert completed.stdout == "current repository script\n"

    module = _load()
    commands = []
    assert hasattr(module, "install")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )
    monkeypatch.setattr(module, "validate_cohort", lambda: None)
    monkeypatch.setattr(
        sys, "argv", ["script", "--platform", "cuda", "--flag-gems", "required"]
    )
    monkeypatch.setattr(module, "PACKAGES", {"numpy": "1.26.4"})
    monkeypatch.setattr(
        module.importlib.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": "numpy"}, version="1.26.4", requires=[])
        ],
    )
    monkeypatch.setattr(module.os, "environ", {})
    module.main()
    assert "--force-reinstall" not in commands[0]
    assert commands[0][-1] == "numpy==1.26.4"


def test_common_install_replaces_existing_versions_without_touching_torch(
    monkeypatch, tmp_path
):
    module = _load()
    assert module.PACKAGES["numpy"] == "1.26.4"
    assert module.PACKAGES["pyyaml"] == "6.0.1"
    commands = []

    def record(command, *, check, env=None, **_):
        assert check
        assert "PIP_CONSTRAINT" not in env
        commands.append(command)

    monkeypatch.setattr(module.subprocess, "run", record)
    monkeypatch.setattr(module, "pins", lambda: ["numpy==1.26.4"])
    monkeypatch.setattr(module, "PACKAGES", {"numpy": "1.26.4"})
    monkeypatch.setattr(
        module.importlib.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": "numpy"}, version="1.26.4", requires=[])
        ],
    )
    monkeypatch.setattr(
        module.os,
        "environ",
        {
            "PIP_INDEX_URL": "https://index.test",
            "PIP_CONSTRAINT": "/checkout/constraints.txt",
        },
    )
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))

    module.install()

    command = commands[0]
    assert command[:6] == [
        str(tmp_path / "python"),
        "-m",
        "pip",
        "install",
        "--index-url",
        "https://index.test",
    ]
    assert "--no-deps" not in command
    assert "--force-reinstall" not in command
    assert command[-1] == "numpy==1.26.4"
    assert "torch==" not in module.pins()


def test_common_install_runs_the_reconciliation(monkeypatch):
    module = _load()
    calls = []
    monkeypatch.setattr(module, "reconcile", lambda: calls.append("reconcile"))

    module.install()

    assert calls == ["reconcile"]


def test_common_install_resolves_pins_before_replacing_mismatches(
    monkeypatch, tmp_path
):
    module = _load()
    calls = []
    installed = {
        "numpy": SimpleNamespace(
            metadata={"Name": "numpy"}, version="2.5.4", requires=[]
        )
    }

    def record(command, *, check, env=None, **_):
        assert check
        calls.append(command)

    monkeypatch.setattr(module.subprocess, "run", record)
    monkeypatch.setattr(module, "validate_cohort", lambda: None)
    monkeypatch.setattr(
        module,
        "PACKAGES",
        {"numpy": "1.26.4", "pyyaml": "6.0.1"},
    )
    monkeypatch.setattr(
        module.importlib.metadata,
        "distributions",
        lambda: installed.values(),
    )
    monkeypatch.setattr(module.os, "environ", {})
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))

    module.install()

    assert len(calls) == 2
    assert "--force-reinstall" not in calls[0]
    assert "--constraint" not in calls[0]
    assert calls[0][-2:] == ["numpy==1.26.4", "pyyaml==6.0.1"]
    assert calls[1][-2:] == ["--force-reinstall", "numpy==1.26.4"]


def test_install_replaces_image_versions_before_validating_the_cohort(
    monkeypatch,
):
    module = _load()
    calls = []

    def image_version():
        calls.append("image")
        return {}

    def check_imports():
        calls.append("validate")

    monkeypatch.setattr(importlib.metadata, "distributions", image_version)
    monkeypatch.setattr(
        module, "_run_pip", lambda *_args, **_kwargs: calls.append("pip")
    )
    monkeypatch.setattr(module, "pins", lambda: ["numpy==1.26.4"])
    monkeypatch.setattr(module, "PACKAGES", {"numpy": "1.26.4"})
    monkeypatch.setattr(module, "validate_cohort", check_imports)

    module.reconcile()

    # validate_cohort reads installed metadata independently of the repair.
    # Its failure remains the final, populated dependency report.
    assert calls == ["pip", "image", "validate"]


def test_install_resolves_before_reading_image_constraints(monkeypatch):
    module = _load()
    calls = []

    def image_version():
        calls.append("image")
        raise SystemExit("filelock is 4.1.0, expected 3.32.3")

    def check_imports():
        calls.append("validate")

    monkeypatch.setattr(importlib.metadata, "distributions", image_version)
    monkeypatch.setattr(
        module, "_run_pip", lambda *_args, **_kwargs: calls.append("pip")
    )
    monkeypatch.setattr(module, "pins", lambda: ["numpy==1.26.4"])
    monkeypatch.setattr(module, "PACKAGES", {"numpy": "1.26.4"})
    monkeypatch.setattr(
        module,
        "_installed_distributions",
        lambda: {
            "numpy": SimpleNamespace(
                metadata={"Name": "numpy"}, version="1.26.4", requires=[]
            )
        },
    )
    monkeypatch.setattr(module, "validate_cohort", check_imports)
    monkeypatch.setattr(
        sys, "argv", ["script", "--platform", "cuda", "--flag-gems", "required"]
    )

    module.main()

    assert calls == ["pip", "validate"]


def test_common_install_pins_versions_newer_than_the_installed_packages(
    monkeypatch, tmp_path
):
    module = _load()
    commands = []
    installed = {
        "transformers": SimpleNamespace(
            metadata={"Name": "transformers"}, version="4.57.6", requires=[]
        ),
        "typing-extensions": SimpleNamespace(
            metadata={"Name": "typing-extensions"}, version="4.15.0", requires=[]
        ),
    }

    def record(command, *, check, env=None, **_):
        assert check
        assert "PIP_CONSTRAINT" not in env
        commands.append(command)

    monkeypatch.setattr(module.subprocess, "run", record)
    monkeypatch.setattr(module, "validate_cohort", lambda: None)
    monkeypatch.setattr(
        module,
        "PACKAGES",
        {"transformers": "5.20.2", "typing-extensions": "4.16.0"},
    )
    monkeypatch.setattr(module, "pins", lambda: ["transformers==5.20.2"])
    monkeypatch.setattr(
        module.importlib.metadata,
        "distributions",
        lambda: installed.values(),
    )
    monkeypatch.setattr(module.os, "environ", {})
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))

    module.install()

    assert len(commands) == 2
    assert "--force-reinstall" not in commands[0]
    assert "transformers==5.20.2" in commands[0]
    assert "--force-reinstall" in commands[1]
    assert "transformers==5.20.2" in commands[1]
    assert "typing-extensions==4.16.0" in commands[1]


def test_image_version_newer_than_shared_pin_is_replaced(monkeypatch, tmp_path):
    module = _load()
    commands = []

    def record(command, *, check, env=None, **_):
        assert check
        assert env["PIP_CONFIG_FILE"] == module.os.devnull
        commands.append(command)

    monkeypatch.setattr(module.subprocess, "run", record)
    monkeypatch.setattr(module, "validate_cohort", lambda: None)
    monkeypatch.setattr(module, "PACKAGES", {"typing-extensions": "4.16.0"})
    monkeypatch.setattr(
        module.importlib.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(
                metadata={"Name": "typing-extensions"},
                version="4.17.0",
                requires=[],
            )
        ],
    )
    monkeypatch.setattr(module.os, "environ", {})
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))

    module.install()

    assert len(commands) == 2
    assert "--force-reinstall" not in commands[0]
    assert commands[0][-1] == "typing-extensions==4.16.0"
    assert "--force-reinstall" in commands[1]
    assert commands[1][-1] == "typing-extensions==4.16.0"


def test_installed_flag_gems_pins_must_match_the_shared_cohort(monkeypatch):
    module = _load()
    installed = {
        "root": SimpleNamespace(metadata={"Name": "root"}, version="1", requires=[]),
        "numpy": SimpleNamespace(
            metadata={"Name": "numpy"}, version="1.26.4", requires=[]
        ),
        "flag-gems": SimpleNamespace(
            metadata={"Name": "flag-gems"},
            version="5.4.0",
            requires=["PyYAML==6.0.3"],
        ),
        "pyyaml": SimpleNamespace(
            metadata={"Name": "PyYAML"}, version="6.0.1", requires=[]
        ),
    }
    monkeypatch.setattr(
        module,
        "PACKAGES",
        {"root": "1", "numpy": "1.26.4", "pyyaml": "6.0.1"},
    )
    monkeypatch.setattr(
        module.importlib.metadata, "distributions", lambda: installed.values()
    )
    with pytest.raises(SystemExit, match="flag-gems 5.4.0 requires PyYAML==6.0.3"):
        module.validate_cohort()


def test_pin_reconciliation_uses_distribution_name_aliases(monkeypatch, tmp_path):
    module = _load()
    commands = []
    installed = [
        SimpleNamespace(
            metadata={"Name": "ml-dtypes"},
            version="0.5.5",
            requires=[],
        )
    ]

    def record(command, *, check, env=None, **_):
        assert check
        assert "PIP_CONSTRAINT" not in env
        commands.append(command)

    monkeypatch.setattr(module.subprocess, "run", record)
    monkeypatch.setattr(module, "validate_cohort", lambda: None)
    monkeypatch.setattr(module, "PACKAGES", {"ml_dtypes": "0.5.4"})
    monkeypatch.setattr(
        module.importlib.metadata,
        "distributions",
        lambda: installed,
    )
    monkeypatch.setattr(module.os, "environ", {})
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))

    module.install()

    assert len(commands) == 2
    assert commands[1][-1] == "ml-dtypes==0.5.4"


def test_pin_reconciliation_ignores_installed_but_unpinned_distributions(
    monkeypatch, tmp_path
):
    module = _load()
    commands = []
    installed = [
        SimpleNamespace(metadata={"Name": "ml-dtypes"}, version="0.5.5", requires=[]),
        SimpleNamespace(metadata={"Name": "setuptools"}, version="80.0.0", requires=[]),
        SimpleNamespace(metadata={"Name": "pip"}, version="25.0", requires=[]),
        SimpleNamespace(metadata={"Name": "build"}, version="1.2.0", requires=[]),
    ]

    def record(command, *, check, env=None, **_):
        assert check
        assert "PIP_CONSTRAINT" not in env
        commands.append(command)

    monkeypatch.setattr(module.subprocess, "run", record)
    monkeypatch.setattr(module, "validate_cohort", lambda: None)
    monkeypatch.setattr(module, "PACKAGES", {"ml_dtypes": "0.5.4"})
    monkeypatch.setattr(module.importlib.metadata, "distributions", lambda: installed)
    monkeypatch.setattr(module.os, "environ", {})
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))

    module.install()

    assert len(commands) == 2
    assert commands[1][-1] == "ml-dtypes==0.5.4"


def test_pin_reconciliation_does_not_replace_image_only_distributions(
    monkeypatch, tmp_path
):
    """A missing shared pin is installed; an unpinned image package is left."""
    module = _load()
    commands = []
    installed = [
        SimpleNamespace(metadata={"Name": "numpy"}, version="2.0.0", requires=[]),
        SimpleNamespace(metadata={"Name": "pip"}, version="25.0", requires=[]),
    ]

    def record(command, *, check, env=None, **_):
        assert check
        assert "PIP_CONSTRAINT" not in env
        commands.append(command)

    monkeypatch.setattr(module.subprocess, "run", record)
    monkeypatch.setattr(module, "validate_cohort", lambda: None)
    monkeypatch.setattr(module, "PACKAGES", {"numpy": "1.26.4"})
    monkeypatch.setattr(
        module.importlib.metadata,
        "distributions",
        lambda: installed,
    )
    monkeypatch.setattr(module.os, "environ", {})
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))

    module.install()

    assert len(commands) == 2
    assert commands[1][-1] == "numpy==1.26.4"
    assert not any("pip" in command[-1] for command in commands)


def test_pin_reconciliation_uses_record_name_when_metadata_has_no_name(
    monkeypatch, tmp_path
):
    module = _load()
    commands = []
    metadata = importlib.metadata.PathDistribution(
        tmp_path / "filelock-4.2.4.dist-info"
    )
    installed = [
        SimpleNamespace(
            metadata={"Name": None},
            version="4.2.4",
            requires=[],
            _path=metadata._path,
        )
    ]

    def record(command, *, check, env=None, **_):
        assert check
        commands.append(command)

    monkeypatch.setattr(module.subprocess, "run", record)
    monkeypatch.setattr(module, "validate_cohort", lambda: None)
    monkeypatch.setattr(module, "pins", lambda: ["filelock==3.32.3"])
    monkeypatch.setattr(module, "PACKAGES", {"filelock": "3.32.3"})
    monkeypatch.setattr(
        module.importlib.metadata,
        "distributions",
        lambda: installed,
    )
    monkeypatch.setattr(module.os, "environ", {})
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))

    module.install()

    assert len(commands) == 2
    assert commands[1][-1] == "filelock==3.32.3"
    assert not any("None==" in argument for argument in commands[1])


def test_metadata_without_a_name_or_distribution_record_is_ignored(monkeypatch):
    module = _load()
    anonymous = SimpleNamespace(
        metadata={"Name": None},
        version="4.2.4",
        requires=[],
        _path=Path("/tmp/unknown.egg-link"),
    )

    monkeypatch.setattr(
        module.importlib.metadata,
        "distributions",
        lambda: [anonymous],
    )

    assert module._installed_distributions() == {}


def test_probe_locates_torch_fl_in_the_active_interpreter(monkeypatch, tmp_path):
    module = _load()
    image = tmp_path / "image" / "dist-packages"
    image_package = image / "torch_fl"
    image_package.mkdir(parents=True)
    (image_package / "__init__.py").write_text("", encoding="utf-8")
    active = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    package = active / "torch_fl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / f"_C{module.EXTENSION_SUFFIXES[0]}").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        module.os, "environ", {"TORCH_FL_INSTALLED_PACKAGE_ROOT": str(image)}
    )
    monkeypatch.setattr(module, "_active_purelib", lambda: active)

    def run(command, **kwargs):
        if "sysconfig" in command[-1]:
            return module.subprocess.CompletedProcess(command, 0, str(active), "")
        if "join(sys.path)" in command[-1]:
            return module.subprocess.CompletedProcess(command, 0, "/absent\n", "")
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", run)

    assert module._installed_package_root() == active


def test_shared_import_probe_uses_the_installed_wheel(tmp_path):
    root = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    package = root / "torch_fl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "_C.test.so").write_text("")
    checkout = tmp_path / "checkout"
    source_package = checkout / "torch_fl"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_text("")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join([str(checkout), str(root)]),
            "TORCH_FL_INSTALLED_PACKAGE_ROOT": str(root),
            "TORCH_FL_SOURCE_ROOT": str(checkout),
            "TORCH_FL_TEST_SITE_PACKAGE": "1",
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util\n"
                f"path = {str(SCRIPT.with_name('_torch_fl_test_site.py'))!r}\n"
                "spec = importlib.util.spec_from_file_location("
                "'_torch_fl_test_site', path)\n"
                "site = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(site)\n"
                "site.install()\n"
                "import torch_fl\n"
                "print(torch_fl.__file__)"
            ),
        ],
        cwd=checkout,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(package / "__init__.py")


@pytest.mark.parametrize(
    ("package", "installed", "expected"),
    [
        ("filelock", "4.1.0", ["filelock==3.32.3"]),
        ("typing-extensions", "4.15.0", ["typing-extensions==4.16.0"]),
        ("transformers", None, ["transformers==5.20.2"]),
    ],
)
def test_common_install_forces_every_exact_pin_mismatch(
    monkeypatch, tmp_path, package, installed, expected
):
    module = _load()
    commands = []

    def record(command, *, check, env=None, **_):
        assert check
        assert "PIP_CONSTRAINT" not in env
        assert env["PIP_CONFIG_FILE"] == module.os.devnull
        commands.append(command)

    distributions = SimpleNamespace(
        metadata={"Name": package},
        version=installed or "0",
        requires=[],
    )
    monkeypatch.setattr(module.subprocess, "run", record)
    monkeypatch.setattr(module, "validate_cohort", lambda: None)
    monkeypatch.setattr(module, "pins", lambda: expected)
    monkeypatch.setattr(
        module,
        "PACKAGES",
        {package: expected[0].split("==")[1]},
    )
    monkeypatch.setattr(
        module.importlib.metadata,
        "distributions",
        lambda: [] if installed is None else [distributions],
    )
    monkeypatch.setattr(
        module.os,
        "environ",
        {
            "PIP_CONFIG_FILE": "/image/pip.conf",
        },
    )
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))

    module.install()

    assert len(commands) == (2 if installed else 1)
    assert "--force-reinstall" not in commands[0]
    assert expected[0] in commands[0]
    if installed:
        assert "--force-reinstall" in commands[1]
        assert expected[0] in commands[1]
    assert "torch==" not in module.pins()


def test_flagtree_triton_uses_the_imported_version_without_dist_info(monkeypatch):
    module = _load()
    installed = {"flagtree": SimpleNamespace(version="0.7.0")}
    triton = ModuleType("triton")
    triton.__version__ = "3.5.1"
    triton.__file__ = "/vendor/flagtree/triton/__init__.py"
    monkeypatch.setitem(sys.modules, "triton", triton)

    active = module._flagtree_triton(installed)

    assert active["module"] is triton
    assert str(active["version"]) == "3.5.1"


def test_flagtree_triggers_a_named_constraint_and_checks_its_install(
    monkeypatch, tmp_path
):
    module = _load()
    triton = ModuleType("triton")
    triton.__version__ = "3.5.1"
    triton.__file__ = "/vendor/flagtree/triton/__init__.py"
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setattr(
        module,
        "_flagtree_triton",
        lambda: {"module": triton, "version": module.Version("3.5.1")},
    )
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(
        module.os, "environ", {"PIP_CONSTRAINT": "/image/constraints.txt"}
    )
    commands, constraints, constraint_paths = [], [], []
    calls = []

    def check_installed():
        calls.append("checked")

    def record(command, *, check, env):
        assert check
        assert env["PIP_CONSTRAINT"] != "/image/constraints.txt"
        commands.append(command)
        constraint_paths.append(Path(env["PIP_CONSTRAINT"]))
        constraints.append(Path(env["PIP_CONSTRAINT"]).read_text(encoding="utf-8"))
        Path(env["PIP_CONSTRAINT"]).write_text("deleted", encoding="utf-8")

    monkeypatch.setattr(module.subprocess, "run", record)
    monkeypatch.setattr(module, "_assert_flagtree_triton", lambda _: check_installed())

    module._run_pip(["preserved-dependency==1"], no_deps=False)
    module._run_pip(["pinned-dependency==2"])

    assert calls == ["checked", "checked"]
    assert constraints == ["triton==3.5.1\n"] * 2
    assert "--no-deps" not in commands[0]
    assert "--constraint" in commands[0]
    assert Path(commands[0][commands[0].index("--constraint") + 1]).name == (
        "constraints.txt"
    )
    assert "--no-deps" in commands[1]
    assert constraint_paths[0] != constraint_paths[1]
    assert not constraint_paths[0].parent.exists()


def test_replaced_flagtree_triton_fails_with_both_locations(monkeypatch):
    module = _load()
    flagtree = ModuleType("triton")
    flagtree.__file__ = "/vendor/flagtree/triton/__init__.py"
    flagtree.__version__ = "3.5.1"
    stock = ModuleType("triton")
    stock.__file__ = "/venv/site-packages/triton/__init__.py"
    stock.__version__ = "3.6.0"
    monkeypatch.setitem(sys.modules, "triton", stock)

    with pytest.raises(SystemExit) as error:
        module._assert_flagtree_triton(
            {"module": flagtree, "version": module.Version("3.5.1")}
        )
    assert "3.6.0 at /venv/site-packages/triton/__init__.py" in str(error.value)
    assert "expected 3.5.1 at /vendor/flagtree/triton/__init__.py" in str(error.value)


def _build_wheel(source, directory):
    (source / "setup.py").write_text(
        "from setuptools import setup\n\nsetup()\n", encoding="utf-8"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(directory),
            str(source),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )


def _build_triton_wheel(name, version, directory, root):
    source = root / name
    package = source / "triton"
    package.mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""\
            [build-system]
            requires = ["setuptools"]
            build-backend = "setuptools.build_meta"

            [project]
            name = "triton"
            version = "{version}"

            [tool.setuptools.package-data]
            triton = ["experimental/{name}.py"]
            """
        ),
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )
    (package / "experimental").mkdir()
    (package / "experimental" / f"{name}.py").write_text(
        "__all__ = []\n", encoding="utf-8"
    )
    _build_wheel(source, directory)


def _build_dependency_wheel(directory, root):
    source = root / "dependency"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [build-system]
            requires = ["setuptools"]
            build-backend = "setuptools.build_meta"

            [project]
            name = "shared-dependency"
            version = "1"
            dependencies = ["triton>=3.5.1"]
            """
        ),
        encoding="utf-8",
    )
    _build_wheel(source, directory)


def test_real_pip_keeps_flagtree_triton_when_installing_a_dependent(
    monkeypatch, tmp_path
):
    """A real dependency install retains FlagTree's active Triton package."""

    module = _load()
    wheels = tmp_path / "wheels"
    _build_triton_wheel("flagtree", "3.5.1", wheels, tmp_path)
    _build_triton_wheel("stock", "3.6.0", wheels, tmp_path)
    _build_dependency_wheel(wheels, tmp_path)
    site = tmp_path / "site-packages"
    with zipfile.ZipFile(next(wheels.glob("triton-3.5.1-*"))) as wheel:
        wheel.extractall(site)
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    monkeypatch.syspath_prepend(str(site))
    triton = importlib.import_module("triton")
    original_file = triton.__file__
    assert triton.__version__ == "3.5.1"

    def isolated_pip_command():
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheels),
            "--target",
            str(site),
            "--no-deps",
        ]

    real_run = subprocess.run

    def isolated_run(command, **kwargs):
        assert "--target" in command, command
        assert command[command.index("--target") + 1] == str(site), command
        return real_run(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", isolated_run)
    monkeypatch.setattr(
        module,
        "_pip_install_command",
        isolated_pip_command,
    )
    monkeypatch.setattr(
        module,
        "_flagtree_triton",
        lambda: {"module": triton, "version": module.Version("3.5.1")},
    )

    module._run_pip([str(next(wheels.glob("shared_dependency-1-*")))], no_deps=False)

    active = importlib.import_module("triton")
    assert active.__version__ == "3.5.1"
    assert active.__file__ == original_file
    assert (
        Path(active.__file__).parent.joinpath("experimental", "flagtree.py").is_file()
    )


def test_flagtree_does_not_hide_other_flag_gems_metadata_conflicts(monkeypatch):
    module = _load()
    triton = ModuleType("triton")
    triton.__version__ = "3.5.1"
    triton.__file__ = "/vendor/flagtree/triton/__init__.py"
    monkeypatch.setitem(sys.modules, "triton", triton)
    installed = {
        "flagtree": SimpleNamespace(
            metadata={"Name": "flagtree"}, version="0.7.0", requires=[]
        ),
        "triton": SimpleNamespace(
            metadata={"Name": "triton"}, version="3.6.0", requires=[]
        ),
        "flag_gems": SimpleNamespace(
            metadata={"Name": "flag-gems"},
            version="5.4.0",
            requires=["triton>=3.5.1", "PyYAML==6.0.3"],
        ),
        "pyyaml": SimpleNamespace(
            metadata={"Name": "PyYAML"}, version="6.0.1", requires=[]
        ),
    }
    monkeypatch.setattr(module, "PACKAGES", {"pyyaml": "6.0.1"})
    monkeypatch.setattr(
        module.importlib.metadata, "distributions", lambda: installed.values()
    )

    with pytest.raises(SystemExit, match="flag-gems 5.4.0 requires PyYAML==6.0.3"):
        module.validate_cohort()
    installed["flag_gems"].requires = ["triton>=3.5.1", "PyYAML==6.0.1"]
    module.validate_cohort()
