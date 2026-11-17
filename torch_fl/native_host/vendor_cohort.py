# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Read published vendor-wheel provenance without importing either Torch build."""

import ast
from email.parser import Parser
import hashlib
import importlib.util
from pathlib import Path
import shutil

from packaging.requirements import Requirement

LIBRARIES = ("libtorch_nvshmem.so", "libc10_cuda.so", "libtorch_cuda.so")


def load_build_policy(source_root):
    """Read the codegen policy during a build, without importing torch_fl."""
    path = Path(source_root) / "scripts/codegen/native_host_policy.py"
    spec = importlib.util.spec_from_file_location("native_host_build_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_policy_for_build(build_config, source_root):
    if not getattr(build_config, "NATIVE_VENDOR", False):
        return None
    return load_build_policy(source_root).source_policy(build_config.ACCELERATOR)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def version_fields(path):
    values = {}
    for statement in ast.parse(Path(path).read_text()).body:
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
            if isinstance(statement, ast.AnnAssign)
            else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {
                "__version__",
                "git_version",
                "cuda",
            }:
                values[target.id] = ast.literal_eval(statement.value)
    return values


def describe(library_dir, policy):
    library_dir = Path(library_dir).resolve()
    fields = version_fields(library_dir.parent / "version.py")
    if (
        fields.get("__version__") != policy["vendor_torch"]
        or fields.get("git_version") != policy["torch_git_version"]
        or fields.get("cuda") is None
    ):
        raise ValueError(
            "native vendor precision requires its validated published Torch cohort"
        )
    for name in LIBRARIES:
        if not (library_dir / name).is_file():
            raise ValueError(f"missing published vendor library: {name}")
    matches = []
    for candidate in library_dir.parent.parent.glob("torch-*.dist-info"):
        metadata = Parser().parsestr((candidate / "METADATA").read_text())
        if metadata["Name"] == "torch" and metadata["Version"] == fields["__version__"]:
            matches.append((candidate, metadata))
    if len(matches) != 1:
        raise ValueError("published vendor Torch metadata is missing or ambiguous")
    distribution, metadata = matches[0]
    requirements = []
    for text in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(text)
        if not requirement.name.lower().replace("_", "-").startswith("nvidia-"):
            continue
        pins = list(requirement.specifier)
        if len(pins) != 1 or pins[0].operator != "==" or "*" in pins[0].version:
            raise ValueError(
                "vendor runtime dependencies must have verified exact pins"
            )
        requirements.append(str(requirement))
    if not requirements:
        raise ValueError("vendor Torch declares no NVIDIA runtime dependencies")
    licenses = sorted((distribution / "licenses").glob("*"))
    if not any(path.is_file() for path in licenses):
        raise ValueError("published vendor Torch license files are missing")
    return {
        "version": fields["__version__"],
        "git_version": fields["git_version"],
        "cuda": fields["cuda"],
        "flag_gems": policy["flag_gems"],
        "library_dir": str(library_dir),
        "metadata_sha256": digest(distribution / "METADATA"),
        "requirements": sorted(requirements),
        "licenses": [str(path) for path in licenses if path.is_file()],
    }


def bundle(cohort, package):
    package = Path(package)
    source = Path(cohort["library_dir"])
    destination = package / "lib"
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in LIBRARIES:
        expected = digest(source / name)
        target = destination / name
        if not target.exists() or digest(target) != expected:
            shutil.copy2(source / name, target)
        hashes[name] = expected
    licenses = package / "native_host/_generated/licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    for path in cohort["licenses"]:
        source_license = Path(path)
        shutil.copy2(source_license, licenses / ("vendor_torch_" + source_license.name))
    return hashes
