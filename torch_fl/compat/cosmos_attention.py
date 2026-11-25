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

"""Run the original packed, SDPA, and FlexAttention kernels on NVIDIA."""

import functools
import importlib.machinery
import sys

import torch

from torch_fl import _C, _platform
from torch_fl.compat import cuda
from torch_fl.compat.cosmos_numerics import suspend_cuda_numerics


_PACKAGE = "cosmos_framework.model.attention"
_FRONTEND = f"{_PACKAGE}.frontend"
_SMALL_ACTION = "cosmos_framework.model.generator.mot.small_action_attention"


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


def _native_attention(original, first_argument):
    if getattr(original, "_flagos_cuda_bridge", False):
        return original

    @functools.wraps(original)
    def attention(*args, **kwargs):
        tensor = args[0] if args else kwargs.get(first_argument)
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "flagos":
            return original(*args, **kwargs)
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Cosmos CUDA attention compatibility requires eager inference"
            )

        index = tensor.device.index
        torch.flagos.synchronize(index)
        views = {}

        def to_cuda(tensor):
            key = id(tensor)
            if key not in views:
                views[key] = _C._flagos_to_cuda_view(tensor)
            return views[key]

        with suspend_cuda_numerics(), torch.cuda.device(index):
            result = original(
                *_convert(args, "flagos", to_cuda),
                **_convert(kwargs, "flagos", to_cuda),
            )
            torch.cuda.synchronize(index)
        return _convert(result, "cuda", _C._cuda_to_flagos_view)

    attention._flagos_cuda_bridge = True
    return attention


def _patch_frontend(module):
    original = module.attention
    module.attention = _native_attention(original, "query")
    package = sys.modules.get(_PACKAGE)
    if package is not None and getattr(package, "attention", None) is original:
        package.attention = module.attention


def _patch_small_action(module):
    # Build both the BlockMask and its captured CUDA indices on the device where
    # the original sparse attention compiles and consumes them.
    module.sample_block_mask = _native_attention(module.sample_block_mask, "query_ids")
    module.sparse_attend = _native_attention(module.sparse_attend, "q")


_PATCHES = {_FRONTEND: _patch_frontend, _SMALL_ACTION: _patch_small_action}


class _AttentionLoader:
    def __init__(self, loader, patch):
        self._loader = loader
        self._patch = patch

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def exec_module(self, module):
        self._loader.exec_module(module)
        self._patch(module)


class _AttentionFinder:
    def find_spec(self, fullname, path=None, target=None):
        patch = _PATCHES.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _AttentionLoader(spec.loader, patch)
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
    functional = torch.nn.functional
    functional.scaled_dot_product_attention = _native_attention(
        functional.scaled_dot_product_attention, "query"
    )
    for name, patch in _PATCHES.items():
        module = sys.modules.get(name)
        if module is not None:
            patch(module)
    if not any(isinstance(finder, _AttentionFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AttentionFinder())
