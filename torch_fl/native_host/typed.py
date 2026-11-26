# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Bind prebuilt typed hosts to the original FlagGems kernel definitions."""

import hashlib
import importlib
import inspect
import json
from pathlib import Path


def activate(hosts):
    """Install the wheel's source-generated callbacks without compiling code."""
    import torch

    directory = hosts.path / "typed"
    manifest = json.loads((directory / "manifest.json").read_text())
    versions = {
        "abi": 2,
        "source_abi": 2,
        "global_abi": 1,
        "foreign_abi": 1,
        "hook_abi": 2,
        "pointwise_at_abi": 1,
        "pointwise_launch_abi": 1,
    }
    if any(manifest.get(name) != version for name, version in versions.items()):
        raise RuntimeError("typed host ABI does not match the installed runtime")
    for name, digest in manifest.get("metadata_sources", {}).items():
        module = importlib.import_module(name)
        source = Path(inspect.getsourcefile(module))
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"build-time host metadata source changed: {name}")
    for name, digest in manifest["artifacts"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"typed host artifact changed: {name}")

    launcher = importlib.import_module(hosts.package + "._launcher")

    def resolve(symbol):
        if "builtin_module" in symbol:
            value = importlib.import_module(symbol["builtin_module"])
        elif "operator" in symbol:
            namespace, name = symbol["operator"].split("::")
            operator = getattr(getattr(torch.ops, namespace), name)
            return getattr(getattr(operator, symbol["overload"]), symbol["method"])
        elif "generated" in symbol:
            value = hosts.modules[symbol["generated"]]
        else:
            value = hosts.originals[symbol["module"]]
        for name in symbol["path"]:
            value = getattr(value, name)
        return value

    launcher.configure_descriptor(
        hosts.modules["triton.tools.tensor_descriptor"].TensorDescriptor
    )
    slots = {}
    for kind, records in manifest["records"].items():
        values = []
        for record in records:
            if kind == "global":
                value = resolve(record["symbol"])
                if type(value).__name__ != record["type"]:
                    raise RuntimeError(
                        "initialized host global changed type: " + str(record["symbol"])
                    )
                values.append(value)
                continue
            value = hosts.resolve(resolve(record["symbol"]))
            if kind == "foreign":
                if "attribute" in record:
                    function, attribute = value, record["attribute"]

                    def value(*args, _fn=function, _attr=attribute, **kwargs):
                        return getattr(_fn(*args, **kwargs), _attr)

                slot = launcher.register_foreign(
                    value, tuple(resolve(symbol) for symbol in record["prefix"])
                )
            elif kind == "kernel":
                slot = launcher.register_script_kernel(value)
            elif kind == "pointwise":
                slot = launcher.register_pointwise(value)
            else:
                raise RuntimeError(f"unknown typed host binding: {kind}")
            values.append(slot)
        slots[kind] = values

    module = importlib.import_module(hosts.package + ".typed._source_hosts")
    module.bind_globals(slots["global"])
    module.bind_api(launcher.launch_packed_api, launcher.pointwise_packed_api)
    module.bind_foreign(launcher.foreign_packed_api)
    module.bind_pointwise_launch(launcher.pointwise_launch_api)
    module.bind_pointwise_at(launcher.pointwise_at_api)
    module.bind_slots(slots["kernel"], slots["pointwise"], slots["foreign"])
    for hook in manifest["native_hooks"]:
        launcher.register_native_hook_v2(
            hosts.resolve(resolve(hook["symbol"])), module.metadata_hooks[hook["name"]]
        )
    for name, address in module.callbacks.items():
        torch.ops.flagos.native_source_register(name, address, 1, 2)
    if manifest.get("typed_entries"):
        if manifest.get("typed_entry_abi") != 1:
            raise RuntimeError("typed entry ABI does not match the installed runtime")
        for entry in manifest["typed_entries"]:
            torch.ops.flagos.native_source_register_typed(
                entry["op"],
                entry["host"],
                entry["abi"],
                module.typed_callbacks[entry["op"]],
            )
    return manifest
