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

"""Run the original Cosmos attention kernels on CUDA views of FlagOS tensors."""

import functools
import importlib.machinery
import sys

import torch

from torch_fl import _C, _platform
from torch_fl.compat import cuda


_PACKAGE = "cosmos_framework.model.attention"
_FRONTEND = f"{_PACKAGE}.frontend"


def _convert(value, device_type, view):
    if isinstance(value, torch.Tensor) and value.device.type == device_type:
        return view(value)
    if isinstance(value, tuple):
        return tuple(_convert(item, device_type, view) for item in value)
    if isinstance(value, list):
        return [_convert(item, device_type, view) for item in value]
    if isinstance(value, dict):
        return {key: _convert(item, device_type, view) for key, item in value.items()}
    return value


def _patch_frontend(module):
    original = module.attention
    if getattr(original, "_flagos_cuda_bridge", False):
        return

    @functools.wraps(original)
    def attention(query, *args, **kwargs):
        if query.device.type != "flagos":
            return original(query, *args, **kwargs)
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Cosmos CUDA attention compatibility requires eager inference"
            )

        index = query.device.index
        torch.flagos.synchronize(index)
        views = {}

        def to_cuda(tensor):
            key = id(tensor)
            if key not in views:
                views[key] = _C._flagos_to_cuda_view(tensor)
            return views[key]

        with torch.cuda.device(index):
            result = original(
                to_cuda(query),
                *_convert(args, "flagos", to_cuda),
                **_convert(kwargs, "flagos", to_cuda),
            )
            torch.cuda.synchronize(index)
        return _convert(result, "cuda", _C._cuda_to_flagos_view)

    attention._flagos_cuda_bridge = True
    module.attention = attention
    package = sys.modules.get(_PACKAGE)
    if package is not None and getattr(package, "attention", None) is original:
        package.attention = attention


class _FrontendLoader:
    def __init__(self, loader):
        self._loader = loader

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _patch_frontend(module)


class _FrontendFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _FRONTEND:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _FrontendLoader(spec.loader)
        return spec


def install():
    """Enable the native CUDA attention bridge in a dedicated NVIDIA process."""
    if (
        _platform.build_info().vendor != "nvidia"
        or torch.version.cuda is None
        or not cuda.has_real_cuda()
        or not torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::empty.memory_format", "CUDA"
        )
    ):
        raise RuntimeError(
            "Cosmos CUDA attention compatibility requires NVIDIA CUDA PyTorch"
        )

    # Cosmos still uses Tensor.cuda() for packed metadata. Its model tensors
    # stay on FlagOS and only the attention inputs and outputs use CUDA views.
    cuda._install_tensor_cuda_alias("flagos")
    module = sys.modules.get(_FRONTEND)
    if module is not None:
        _patch_frontend(module)
    elif not any(isinstance(finder, _FrontendFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _FrontendFinder())
