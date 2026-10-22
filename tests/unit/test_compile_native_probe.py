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

"""CPU-only coverage for Inductor's CUDA-shaped device capability probe."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


_SOURCE = Path(__file__).resolve().parents[2] / "torch_fl/compile/inductor_backend.py"
_MODULE_NAMES = (
    "torch_fl",
    "torch_fl._build_config",
    "torch_fl.compile",
    "torch_fl.compile.device_interface",
    "torch_fl.compile.flagtree_shim",
    "torch_fl._env",
)
_CUDA_PROBE_NAMES = (
    "_flagos_native_patched",
    "_flagos_native_probe_extended",
)


@contextmanager
def _backend_modules(flagos):
    """Keep imports made by the probe inside the stubbed package."""
    environment = SimpleNamespace(set_foreign=lambda _name, _value: False)
    torch_fl = SimpleNamespace(_env=environment)
    compile_package = SimpleNamespace(
        device_interface=SimpleNamespace(
            is_native_accelerator=lambda: True,
            _torch_has_cuda_kernels=lambda: True,
        )
    )
    torch_fl.compile = compile_package
    modules = {
        "torch_fl": torch_fl,
        "torch_fl._build_config": SimpleNamespace(ACCELERATOR="cuda"),
        "torch_fl.compile": compile_package,
        "torch_fl.compile.device_interface": compile_package.device_interface,
        "torch_fl.compile.flagtree_shim": SimpleNamespace(flagtree_backend=lambda: ""),
        "torch_fl._env": environment,
    }
    import torch

    previous = {name: sys.modules.get(name) for name in _MODULE_NAMES}
    previous_torch_flagos = getattr(torch, "flagos", None)
    torch.flagos = flagos
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location("native_probe_backend", _SOURCE)
        assert spec is not None and spec.loader is not None
        backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backend)
        yield backend
    finally:
        if previous_torch_flagos is None:
            delattr(torch, "flagos")
        else:
            torch.flagos = previous_torch_flagos
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@contextmanager
def _cuda_probe_state():
    """Restore the process's CUDA probe, including attributes the patch adds."""
    import torch

    previous = {
        name: (hasattr(torch.cuda, name), getattr(torch.cuda, name, None))
        for name in _CUDA_PROBE_NAMES
    }
    try:
        yield torch.cuda
    finally:
        for name, (existed, value) in previous.items():
            if existed:
                setattr(torch.cuda, name, value)
            elif hasattr(torch.cuda, name):
                delattr(torch.cuda, name)


def test_native_probe_redirects_inductor_cuda_queries(monkeypatch):
    def current_device():
        return 7

    def properties(device):
        return SimpleNamespace(major=9, minor=4, index=device)

    flagos = SimpleNamespace(
        is_available=lambda: True,
        current_device=current_device,
        get_device_properties=properties,
    )
    import torch

    with _backend_modules(flagos) as backend, _cuda_probe_state():
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        for attribute in (
            *_CUDA_PROBE_NAMES,
            "current_device",
            "get_device_properties",
            "get_device_capability",
        ):
            monkeypatch.delattr(torch.cuda, attribute, raising=False)

        backend._patch_native_cuda_probe()

        assert torch.cuda.is_available()
        assert torch.cuda.current_device() == 7
        assert torch.cuda.get_device_properties(7).index == 7
        assert torch.cuda.get_device_capability(7) == (9, 4)
        assert torch.cuda._flagos_native_probe_extended


def test_an_older_native_redirect_also_gets_the_extended_probe(monkeypatch):
    def properties(device):
        return SimpleNamespace(major=8, minor=0, index=device)

    flagos = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 2,
        get_device_properties=properties,
    )
    import torch

    with _backend_modules(flagos) as backend, _cuda_probe_state():
        monkeypatch.setattr(torch.cuda, "_flagos_native_patched", True)
        monkeypatch.setattr(torch.cuda, "_flagos_native_probe_extended", False)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: None)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (0, 0))

        backend._patch_native_cuda_probe()

        assert torch.cuda.current_device() == 2
        assert torch.cuda.get_device_capability(2) == (8, 0)
