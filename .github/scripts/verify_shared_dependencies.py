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

"""Install and import the dependency cohort shared by hardware test jobs."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


# Every import is explicit because the CI installs this cohort into the vendor's
# torch environment. Image constraints are removed for this install so they
# cannot select a mismatching dependency or another torch build. Every pin has
# a CPython 3.10 Linux wheel because the hardware images use both 3.10 and 3.12.
PACKAGES = {
    "onnx": "1.23.2",
    "onnxruntime": "1.23.2",
    "jq": "1.13.0",
    "transformers": "5.20.2",
    "huggingface-hub": "1.33.0",
    # FlagGems 5.4.0 also needs SQLAlchemy at module import. Keep these explicit
    # so a change to pip's resolver does not leave the setup probe without a
    # dependency that pip metadata reports.
    "sqlalchemy": "2.0.48",
    "greenlet": "3.5.6",
    "hf-xet": "1.7.0",
    "tokenizers": "0.23.3",
    "safetensors": "0.8.0",
    "protobuf": "6.33.4",
    "ml_dtypes": "0.5.4",
    "coloredlogs": "15.0.1",
    "humanfriendly": "10.0",
    "flatbuffers": "25.12.19",
    "regex": "2026.9.29",
    "numpy": "1.26.4",
    "requests": "2.34.2",
    "urllib3": "2.8.0",
    "charset-normalizer": "3.5.2",
    "certifi": "2026.10.19",
    "idna": "3.20",
    "httpx": "0.28.1",
    "httpcore": "1.0.9",
    "h11": "0.16.0",
    "anyio": "4.15.1",
    "diffusers": "0.40.0",
    "pillow": "12.2.0",
    "pyyaml": "6.0.1",
    "typer": "0.27.3",
    "shellingham": "1.5.4",
    "annotated-doc": "0.0.5",
    "rich": "15.0.0",
    "markdown-it-py": "4.2.0",
    "pygments": "2.21.0",
    "mdurl": "0.1.2",
    "tqdm": "4.70.1",
    # Images can install a different version before this cohort is applied.
    # install() first resolves the exact pins, then replaces every remaining
    # mismatch with the pin below.
    # Kunlun's configured index does not publish 4.2.4. 3.32.3 has no
    # dependencies and satisfies the other packages in this cohort.
    "filelock": "3.32.3",
    "packaging": "26.3",
    "importlib-metadata": "4.13.0",
    "zipp": "4.1.1",
    "click": "8.5.0",
    "fsspec": "2026.9.0",
    # anyio 4.15.x requires typing_extensions>=4.16.0 on every supported
    # Python. torch 2.10 only requires >=4.10.0, so the newer runtime satisfies
    # both packages in the shared cohort.
    "typing-extensions": "4.16.0",
    "sympy": "1.14.0",
    "mpmath": "1.3.0",
}

# numpy stays below 2 because several vendor torch extensions and the stock CPU
# torch build require that range. MPS and PPU import torch before the cohort is
# installed, so their hooks install this same pin sooner; the validation below
# still requires it to survive into the tested environment.
# PyYAML 6.0.1 is FlagGems 5.4.0's exact runtime requirement. A different
# FlagGems release whose exact pin changes fails validation below until the
# shared cohort is updated alongside it.
# The modules include imports made inside functions by upstream code. Checking
# them here catches an incomplete cohort before any hardware test starts.
MODULE_IMPORTS = {
    "huggingface_hub.utils._http": "huggingface-hub",
}

# These packages import torch during their own initialization. Check them in
# this process so they use the facade selected by torch_fl above. Hardware runs
# additionally execute a child probe to prove that the facade is inherited.
TORCH_STARTUP_IMPORTS = {"diffusers", "transformers"}
FLAGGEMS_KERNEL_SET = "flaggems"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-packages", action="store_true")
    parser.add_argument(
        "--platform",
        required=True,
        help="Platform whose environment contract this cohort must satisfy",
    )
    parser.add_argument(
        "--flag-gems",
        choices=("required", "forbidden"),
        required=True,
        help="FlagGems requirement supplied by the loaded platform contract",
    )
    return parser.parse_args()


def pins():
    return [f"{name}=={version}" for name, version in PACKAGES.items() if version]


def pip_environment(environment=None):
    """Ignore image constraints while leaving shared image indexes available."""

    environment = (os.environ if environment is None else environment).copy()
    environment.pop("PIP_CONSTRAINT", None)
    environment.pop("PIP_CONSTRAINTS", None)
    environment["PIP_CONFIG_FILE"] = os.devnull
    return environment


def _matches_pin(actual, expected):
    """Compare installed versions with pip's normalized exact-pin semantics."""

    try:
        return Version(actual) == Version(expected)
    except InvalidVersion:
        return False


def _installed_distributions():
    installed = {}
    for distribution in importlib.metadata.distributions():
        name = _distribution_name(distribution)
        if name is not None:
            installed[canonicalize_name(name)] = distribution
    return installed


def _distribution_name(distribution):
    """Use the filename when importlib.metadata cannot parse a package name."""

    name = distribution.metadata.get("Name")
    if name:
        return name
    # Some distributions omit Name even though their record still identifies
    # the package. Several records can share the normalized package name;
    # retain the first and let the exact-pin reconciliation replace it.
    try:
        path = Path(str(distribution._path))
    except AttributeError:
        return None
    if path.suffix == ".dist-info":
        return path.name[: -len(".dist-info")].split("-", 1)[0]
    if path.suffix == ".egg-info":
        return path.name[: -len(".egg-info")].split("-", 1)[0]
    return None


def _flagtree_triton(installed=None):
    """Return FlagTree's active Triton, or None when FlagTree is absent."""

    installed = _installed_distributions() if installed is None else installed
    if "flagtree" not in installed:
        return None

    # FlagTree ships Triton as part of its wheel. Some releases also leave a
    # triton distribution record; others expose only the imported package.
    # Keep the module that pip would install over unchanged.
    import triton as triton_module

    version = getattr(triton_module, "__version__", None)
    if not isinstance(version, str):
        raise SystemExit("FlagTree's Triton has no version")
    try:
        return {"module": triton_module, "version": Version(version)}
    except InvalidVersion:
        raise SystemExit(f"invalid FlagTree Triton version: {version}") from None


def _mismatched_distributions(installed):
    """Return mismatching pins except packages supplied by FlagTree itself."""

    # The protection covers the active Python module because a stock Triton can
    # replace it without leaving a triton distribution record. Its misleading
    # metadata is then the only remaining dependency conflict.
    pins = snapshot_pins()
    exclude = {"triton"} if _flagtree_triton(installed) is not None else set()
    return {
        name: distribution
        for name, distribution in installed.items()
        if name in pins
        and name not in exclude
        and not _matches_pin(distribution.version, pins[name])
    }


def _pip_install_command():
    """Install only pinned wheels into the vendor's active interpreter."""

    index = os.environ.get("PIP_INDEX_URL", "")
    command = [sys.executable, "-m", "pip", "install", "--no-deps"]
    if index:
        command.extend(["--index-url", index])
    return command


def _active_purelib():
    """Return the site directory of the interpreter that installed the wheel."""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(completed.stdout.strip()).resolve()


def _package_root(package):
    return any((package / f"_C{suffix}").is_file() for suffix in EXTENSION_SUFFIXES)


def _active_site_roots():
    """Return the sequence in which the active interpreter imports packages."""

    completed = subprocess.run(
        [sys.executable, "-c", "import sys; print('\\n'.join(sys.path))"],
        check=True,
        capture_output=True,
        text=True,
    )
    roots = [_active_purelib()]
    roots.extend(
        Path(entry).resolve() for entry in completed.stdout.splitlines() if entry
    )
    return list(dict.fromkeys(roots))


def _installed_package_root():
    """Select the built package even when the current checkout is importable."""

    # The setup hook can change interpreters after recording the image's root.
    # Treat the recording as a request to select a wheel, then find the wheel
    # in the current interpreter. An image may expose the venv through the
    # purelib path, its sys.path, or both.
    selected = os.environ.get("TORCH_FL_INSTALLED_PACKAGE_ROOT")
    roots = _active_site_roots()
    for root in roots:
        package = root / "torch_fl"
        if _package_root(package):
            return root if selected else None
    if selected or any((root / "torch_fl" / "__init__.py").is_file() for root in roots):
        raise SystemExit(
            "installed torch_fl package has no native _C in active interpreter: "
            + ", ".join(str(root / "torch_fl") for root in roots)
        )
    return None


def probe_environment(environment=None):
    """Run dependency probes from the installed package, not its source tree."""

    environment = (os.environ if environment is None else environment).copy()
    root = _installed_package_root()
    if root is not None:
        entries = [
            entry
            for entry in environment.get("PYTHONPATH", "").split(os.pathsep)
            if entry and Path(entry).resolve() != root
        ]
        environment["PYTHONPATH"] = os.pathsep.join([str(root), *entries])
        environment["TORCH_FL_INSTALLED_PACKAGE_ROOT"] = str(root)
        environment["TORCH_FL_SOURCE_ROOT"] = str(Path.cwd())
        environment["TORCH_FL_TEST_SITE_PACKAGE"] = "1"
    return environment


def _run_pip(requirements, *, no_deps=True):
    """Install pins without letting pip replace FlagTree's Triton package."""

    flagtree_triton = _flagtree_triton()
    environment = pip_environment()
    with tempfile.TemporaryDirectory(prefix="torch-fl-pip-") as temporary:
        constraint = Path(temporary, "constraints.txt")
        if flagtree_triton is not None:
            constraint.write_text(
                f"triton=={flagtree_triton['version']}\n", encoding="utf-8"
            )
            environment["PIP_CONSTRAINT"] = str(constraint)
        command = _pip_install_command()
        if not no_deps:
            command.remove("--no-deps")
        if flagtree_triton is not None:
            command.extend(["--constraint", str(constraint)])
        command.extend(requirements)
        print(" ".join(command))
        subprocess.run(command, check=True, env=environment)
        if flagtree_triton is not None:
            _assert_flagtree_triton(flagtree_triton)


def _assert_flagtree_triton(flagtree_triton):
    """Fail if a dependency install replaced FlagTree's Triton package."""

    active = importlib.import_module("triton")
    expected_file = flagtree_triton["module"].__file__
    expected_version = str(flagtree_triton["version"])
    if active.__file__ != expected_file or active.__version__ != expected_version:
        raise SystemExit(
            "shared dependency install replaced FlagTree's Triton with "
            f"{active.__version__} at {active.__file__}; expected "
            f"{expected_version} at {expected_file}"
        )


def reconcile():
    """Make every pinned package present at exactly the shared version."""

    # Resolve the pins together before replacing the image versions. Force-
    # reinstalling every package lets a newer image version satisfy another
    # package's exact requirement, so the resolver rejects the cohort.
    print("installing shared dependency cohort with exact pins")
    _run_pip(pins(), no_deps=False)
    # The first install can add or change dependencies, so compare every pin
    # again before forcing image versions over the resolved cohort.
    mismatched = _mismatched_distributions(_installed_distributions())
    if mismatched:
        print("replacing mismatching shared dependency versions")
        _run_pip(
            [
                "--force-reinstall",
                *(
                    f"{_distribution_name(distribution)}=={snapshot_pins()[name]}"
                    for name, distribution in mismatched.items()
                ),
            ],
            no_deps=False,
        )
    validate_cohort()


def _canonical_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def snapshot_pins():
    """Return the pins keyed by the normalized metadata distribution name."""

    return {canonicalize_name(name): version for name, version in PACKAGES.items()}


def requires_flag_gems(platform, spec):
    """Read the FlagGems requirement from the build the lane checks."""

    required = spec.get("require_build_kernels") or []
    forbidden = spec.get("forbid_build_kernels") or []
    for key, value in (
        ("require_build_kernels", required),
        ("forbid_build_kernels", forbidden),
    ):
        if not isinstance(value, list) or not all(
            isinstance(name, str) for name in value
        ):
            raise SystemExit(f"{platform}: environment_check.{key} must be a list")
    if FLAGGEMS_KERNEL_SET in required and FLAGGEMS_KERNEL_SET in forbidden:
        raise SystemExit(
            f"{platform}: environment_check both requires and forbids "
            f"{FLAGGEMS_KERNEL_SET}"
        )
    return FLAGGEMS_KERNEL_SET in required


def flag_gems_mode(platform, spec):
    """Turn a platform contract into an explicit dependency mode."""

    required = spec.get("require_build_kernels") or []
    forbidden = spec.get("forbid_build_kernels") or []
    if FLAGGEMS_KERNEL_SET in required:
        return "required"
    if FLAGGEMS_KERNEL_SET in forbidden:
        return "forbidden"
    return "optional"


def _current_platform_requirements(distribution):
    """Return only requirements pip would install for this interpreter."""

    requires = distribution.requires
    if not requires:
        return []
    requirements = [Requirement(item) for item in requires]
    return [
        requirement
        for requirement in requirements
        if not requirement.marker or requirement.marker.evaluate({"extra": ""})
    ]


def validate_cohort():
    """Verify the common pins satisfy every installed package's metadata."""

    installed = _installed_distributions()
    flagtree_triton = _flagtree_triton(installed)
    if flagtree_triton is not None:
        # FlagTree owns the Triton runtime and publishes the version its vendor
        # backend needs. Its metadata is not comparable with the stock Triton
        # that pip would install, so the Triton requirement is not checked
        # here. The install step proves the active module is still FlagTree's.
        installed.pop("triton", None)
    names = {_canonical_name(name): name for name in PACKAGES}
    pins = {
        _canonical_name(name): Version(version) for name, version in PACKAGES.items()
    }
    missing = {name for name in names if name not in installed}
    for name, pin in pins.items():
        distribution = installed.get(name)
        if distribution is None:
            continue
        try:
            actual = Version(distribution.version)
        except InvalidVersion:
            raise SystemExit(
                f"invalid version for {name}: {distribution.version}"
            ) from None
        if actual != pin:
            raise SystemExit(f"{name} is {distribution.version}, expected {pin}")

    # Follow metadata instead of scanning upstream source. FlagGems is the one
    # package absent from PACKAGES because its build varies by platform. Use
    # its metadata so a new runtime pin is visible before pip changes the
    # versions supplied by an image.
    roots = dict(names)
    for name in ("flag_gems",):
        key = _canonical_name(name)
        if key in installed:
            roots[key] = _distribution_name(installed[key])
    pending = list(roots.values())
    checked = set()
    conflicts = []
    platform_packages = {"torch"}
    if flagtree_triton is not None:
        platform_packages.add("triton")
    while pending:
        name = pending.pop()
        if name in checked:
            continue
        checked.add(name)
        distribution = installed.get(name)
        if distribution is None:
            continue
        for requirement in _current_platform_requirements(distribution):
            required = _canonical_name(requirement.name)
            if required in platform_packages:
                continue
            target = installed.get(required)
            if target is None:
                missing.add(required)
                continue
            if target.version not in requirement.specifier:
                conflicts.append(
                    f"{name} {distribution.version} requires "
                    f"{requirement.name}{requirement.specifier}, found "
                    f"{target.version}"
                )
            pending.append(required)

    if missing:
        raise SystemExit(
            "shared dependency cohort is incomplete: " + ", ".join(sorted(missing))
        )
    if conflicts:
        raise SystemExit("shared dependency conflicts: " + "; ".join(conflicts))


def _import_torch_fl():
    """Load the vendor torch through torch_fl in an installed CI environment."""

    try:
        import torch_fl  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "torch_fl._C":
            raise
        # A development checkout has the Python package but no compiled
        # extension. CI reaches the import checks after installing the wheel.
        print("torch_fl: source checkout has no compiled extension")
        return False
    return True


def import_flag_gems():
    try:
        import flag_gems
    except ModuleNotFoundError as error:
        if error.name != "flag_gems":
            raise
        raise SystemExit("FlagGems is required by this platform's build")
    return flag_gems


def prepare_flag_gems_import(platform):
    """Apply the same Triton compatibility setup used by torch_fl startup."""

    if platform != "ascend":
        return
    # FlagGems imports its Ascend vendor backend during package initialization.
    # That backend imports FlagTree's TLE package, which may need the same inert
    # shmem placeholder that torch_fl registers before its own FlagGems import.
    from torch_fl.compile.flagtree_ascend_tle import ensure_tle_importable

    ensure_tle_importable()


def flag_gems_vendor(*, flag_gems_required=False):
    """Return the accelerator selected by FlagGems, or None when optional."""

    try:
        import flag_gems
    except ModuleNotFoundError as error:
        if error.name != "flag_gems":
            raise
        if flag_gems_required:
            raise SystemExit("FlagGems is required by this platform's build") from error
        return None
    value = getattr(flag_gems, "vendor_name", None)
    if not isinstance(value, str):
        raise SystemExit("FlagGems does not publish its selected vendor")
    return value.lower()


def check(module_imports, *, platform, flag_gems_required):
    # Vendors such as MetaX and PPU replace torch during torch_fl startup. Load
    # torch_fl before any dependency that can trigger torch initialization:
    # FlagGems imports torch as part of its own initialization.
    imported_torch_fl = _import_torch_fl()
    probe = probe_environment()

    prepare_flag_gems_import(platform)
    vendor = flag_gems_vendor(flag_gems_required=flag_gems_required)
    if flag_gems_required and vendor is None:
        raise SystemExit("FlagGems is required by this platform's build")

    modules = {name: package for package, name in module_imports.items()}
    modules.update(MODULE_IMPORTS)
    for module, distribution in modules.items():
        if module.split(".", 1)[0] in TORCH_STARTUP_IMPORTS:
            # A fresh CPU interpreter cannot provide the vendor torch selected
            # above. Check these after the independent subprocess probes below.
            continue
        # The setup facade is published by torch_fl. A child that selects it from
        # PYTHONPATH but loads torch without torch_fl keeps a stock CPU runtime.
        source = (
            "import importlib.util; "
            "spec = importlib.util.spec_from_file_location("
            "'_torch_fl_test_site', "
            f"{str(Path(__file__).with_name('_torch_fl_test_site.py'))!r}); "
            "site = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(site); "
            "site.install(); "
            f"import torch_fl; import {module}"
        )
        result = subprocess.run(
            [sys.executable, "-c", source],
            check=False,
            env=probe,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode:
            if not imported_torch_fl and "torch_fl._C" in result.stderr:
                # A development checkout has the package source but no
                # compiled extension. CI reaches this step after installing it.
                print(f"{module}: source checkout has no torch_fl extension")
                continue
            raise SystemExit(
                f"{module}: shared import probe failed with exit code "
                f"{result.returncode}\n{result.stderr.rstrip()}"
            )
        if not PACKAGES[distribution]:
            print(f"{distribution}: present")
            continue
        actual = importlib.metadata.version(distribution)
        expected = PACKAGES[distribution]
        if actual != expected:
            raise SystemExit(f"{distribution} is {actual}, expected {expected}")
    for module, distribution in modules.items():
        if module.split(".", 1)[0] not in TORCH_STARTUP_IMPORTS:
            continue
        __import__(module)
        if not PACKAGES[distribution]:
            print(f"{distribution}: present")
            continue
        actual = importlib.metadata.version(distribution)
        expected = PACKAGES[distribution]
        if actual != expected:
            raise SystemExit(f"{distribution} is {actual}, expected {expected}")
    import torch
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenEmbedLayer3DRope,
        ROPE_PER_DEVICE,
        apply_rotary_emb_qwen_neuron,
    )

    expected_torch = Path(
        importlib.metadata.distribution("torch").locate_file("torch")
    ).resolve()
    if Path(torch.__file__).resolve() != expected_torch / "__init__.py":
        raise SystemExit(
            f"shared dependencies changed torch's location: "
            f"{torch.__file__}, expected {expected_torch / '__init__.py'}"
        )

    if not callable(apply_rotary_emb_qwen_neuron):
        raise SystemExit("diffusers does not export a Qwen-Image rotary consumer")
    if "cuda" not in ROPE_PER_DEVICE or QwenEmbedLayer3DRope is None:
        raise SystemExit("diffusers does not expose the Qwen-Image rope contract")
    # These builds register the real-valued rotation after _C replaces the
    # table's flagos entry. The other vendors retain their own startup behavior,
    # which the platform tests cover.
    if vendor in {"ascend", "enflame", "iluvatar"}:
        # This helper defers its import until after torch_fl has initialized.
        from verify_shared_dependency_rotator import rotator

        flagos_qwenimage_rotary_emb = rotator()
        if ROPE_PER_DEVICE.get("flagos") is not flagos_qwenimage_rotary_emb:
            raise SystemExit(
                f"{vendor}: torch_fl did not install the flagos Qwen rotation"
            )
    else:
        print(f"{vendor or 'torch_fl'}: Qwen rotation is not part of its startup")
    print(
        f"torch {torch.__version__} at {torch.__file__}; "
        f"python {sys.version.split()[0]}; "
        f"shared cohort imports {len(modules)} modules; "
        f"{vendor or 'torch_fl'} startup imports passed"
    )


def install():
    """Install the cohort before validating versions supplied by an image."""

    # Images can already fail validation with their uncohort versions. The
    # install step must remain able to replace those packages so a runner that
    # reused an older script is visible from the operation log instead of
    # stopping with only one package's version mismatch.
    reconcile()


def main():
    options = arguments()
    if options.print_packages:
        print(" ".join(pins()))
        return
    if options.check:
        validate_cohort()
        module_imports = importlib.util.spec_from_file_location(
            "verify_shared_dependency_imports",
            Path(__file__).with_name("verify_shared_dependency_imports.py"),
        )
        imports = importlib.util.module_from_spec(module_imports)
        module_imports.loader.exec_module(imports)
        check(
            imports.module_imports(PACKAGES),
            platform=options.platform,
            flag_gems_required=options.flag_gems == "required",
        )
        return
    install()


if __name__ == "__main__":
    main()
