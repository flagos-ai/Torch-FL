# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Select and verify published host sources without importing a vendor backend."""

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import sys


def validate(policy):
    if (
        policy.get("schema_version") != 1
        or policy.get("vendor") != "nvidia"
        or policy.get("requires_native_vendor") is not True
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
            or set(row["dtypes"]) - {"Float", "BFloat16"}
            or not row.get("reason")
        ):
            raise ValueError("invalid published source selection")
    return policy


def selected(build_config):
    """Read the opt-in cohort's policy during wheel construction."""
    if not getattr(build_config, "NATIVE_VENDOR", False):
        return None
    policy = validate(json.loads(Path(__file__).with_suffix(".json").read_text()))
    if build_config.ACCELERATOR != policy["vendor"]:
        raise RuntimeError("published source policy vendor mismatch")
    return policy


def load(policy, expected_sources=None):
    """Load exact published modules without executing unrelated backend imports."""
    if policy is None:
        return {}
    validate(policy)
    if importlib.metadata.version("flag_gems") != policy["flag_gems"]:
        raise RuntimeError("published source policy FlagGems cohort changed")
    import flag_gems

    root = Path(flag_gems.__file__).resolve().parent
    functions = {}
    for row in policy["entries"]:
        name = row["module"]
        path = root.joinpath(*name.split(".")[1:]).with_suffix(".py")
        if not path.resolve().is_relative_to(root):
            raise RuntimeError("published source escaped its installed package")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"] or (
            expected_sources is not None and digest != expected_sources[name]["sha256"]
        ):
            raise RuntimeError(f"published source changed: {name}")
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[name]
                raise
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError(f"published source loaded from another path: {name}")
        functions[row["id"]] = getattr(module, row["function"])
    return functions
