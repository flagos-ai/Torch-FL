# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Select and verify published host sources without importing a vendor backend."""

import _imp
import ast
import hashlib
import importlib.metadata
import importlib.util
from importlib._bootstrap_external import _code_to_hash_pyc
from importlib.machinery import SourcelessFileLoader
from pathlib import Path
import re
import sys


_SHARED_LERP_MODULE = "flag_gems.runtime.backend._kunlunxin.ops.lerp"
BYTECODE_DIRECTORY = "published_sources"


def validate(policy):
    vendor_requirements = {
        "nvidia": True,
        "hygon": True,
        "metax": False,
        "ascend": False,
        "mthreads": False,
        "iluvatar": False,
        "thead": False,
        "tsingmicro": False,
    }
    if (
        policy.get("schema_version") != 1
        or policy.get("vendor") not in vendor_requirements
        or policy.get("requires_native_vendor")
        is not vendor_requirements.get(policy.get("vendor"))
    ):
        raise ValueError("unsupported published source policy")
    operators, identifiers = set(), set()
    for row in policy["entries"]:
        if row["op"] in operators or row["id"] in identifiers:
            raise ValueError("duplicate published source selection")
        operators.add(row["op"])
        identifiers.add(row["id"])
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]*", row["id"])
            or not re.fullmatch(r"aten::[A-Za-z_][\w.]*", row["op"])
            or not re.fullmatch(r"flag_gems(?:\.[A-Za-z_]\w*)+", row["module"])
            or not re.fullmatch(r"[A-Za-z_]\w*", row["function"])
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))
            or not row.get("dtypes")
            or row.get("adaptation") is not None
            or set(row["dtypes"]) - {"Bool", "Float", "BFloat16", "Long"}
            or type(row.get("allow_wrapped_numbers", False)) is not bool
            or type(row.get("shared_pointwise", False)) is not bool
            or (
                row.get("shared_pointwise", False)
                and (
                    row["module"] != _SHARED_LERP_MODULE
                    or row["function"] != "lerp_scalar_"
                )
            )
            or not row.get("reason")
            or type(row.get("same_tensor_metadata", False)) is not bool
            or type(row.get("unit_stride_axis", False)) is not bool
            or type(row.get("min_tensors", 1)) is not int
            or row.get("min_tensors", 1) < 1
        ):
            raise ValueError("invalid published source selection")
    return policy


def shared_lerp_source(path, *, filename=None):
    """Load the published scalar JIT definitions with the common host generator."""
    source = ast.parse(importlib.util.decode_source(path.read_bytes()))
    host = ast.parse(
        "import logging\n"
        "import triton\n"
        "import triton.language as tl\n"
        "from flag_gems.utils.pointwise_dynamic import pointwise_dynamic\n"
        "logger = logging.getLogger(__name__)\n"
    )
    names = {"lerp_scalar_kernel_head", "lerp_scalar_kernel_tail", "lerp_scalar_"}
    host.body.extend(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    )
    return compile(host, str(path) if filename is None else filename, "exec")


def _published_root(policy):
    validate(policy)
    if importlib.metadata.version("flag_gems") != policy["flag_gems"]:
        raise RuntimeError("published source policy FlagGems cohort changed")
    import flag_gems

    return Path(flag_gems.__file__).resolve().parent


def _published_source(row, root, expected_sources=None):
    name = row["module"]
    path = root.joinpath(*name.split(".")[1:]).with_suffix(".py")
    if not path.resolve().is_relative_to(root):
        raise RuntimeError("published source escaped its installed package")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != row["sha256"]:
        raise RuntimeError(f"published source changed: {name}")
    if expected_sources is not None:
        expected = expected_sources.get(name)
        prefix = "flag_gems.runtime.backend."
        if expected is None and name.startswith(prefix):
            # Published vendor registrars expose the same source under a short name.
            expected = expected_sources.get(name.removeprefix(prefix))
        if expected is None or digest != expected["sha256"]:
            raise RuntimeError(f"published source changed: {name}")
    return path, payload


def build_bytecode(policy, package):
    """Stage the shared pointwise hosts during the wheel build."""
    if policy is None:
        return {}
    root = _published_root(policy)
    bytecode = {}
    for row in policy["entries"]:
        if not row.get("shared_pointwise", False):
            continue
        source, payload = _published_source(row, root)
        path = package / BYTECODE_DIRECTORY / f"{row['id']}.pyc"
        path.parent.mkdir(exist_ok=True)
        code = shared_lerp_source(source, filename=source.name)
        pyc = _code_to_hash_pyc(code, importlib.util.source_hash(payload))
        path.write_bytes(pyc)
        bytecode[row["id"]] = hashlib.sha256(pyc).hexdigest()
    return bytecode


def _load_bytecode(row, source, package, bytecode):
    entry = row["id"]
    if entry not in bytecode:
        raise RuntimeError(f"Missing prebuilt published Native Host source: {entry}")
    path = package / BYTECODE_DIRECTORY / f"{entry}.pyc"
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != bytecode[entry]
    ):
        raise RuntimeError(f"Published Native Host bytecode changed: {entry}")
    name = row["module"]
    code = SourcelessFileLoader(name, str(path)).get_code(name)
    # Triton inspects the original published JIT source and line numbers.
    _imp._fix_co_filename(code, str(source))
    return code


def load(policy, expected_sources=None, *, package=None, bytecode=None):
    """Load exact published modules without executing unrelated backend imports."""
    if policy is None:
        return {}
    root = _published_root(policy)
    functions = {}
    for row in policy["entries"]:
        name = row["module"]
        path, _ = _published_source(row, root, expected_sources)
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                if row.get("shared_pointwise", False):
                    code = (
                        shared_lerp_source(path)
                        if bytecode is None
                        else _load_bytecode(row, path, package, bytecode)
                    )
                    exec(code, vars(module))
                else:
                    spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[name]
                raise
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError(f"published source loaded from another path: {name}")
        functions[row["id"]] = getattr(module, row["function"])
    return functions
