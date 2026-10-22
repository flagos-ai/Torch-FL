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

"""Optional PyTorch flex-attention compatibility for the flagos device.

``torch.nn.attention.flex_attention`` checks its inputs against a hard-coded
device set -- ``{"cuda", "cpu", "xpu", "hpu"}`` -- before anything else runs, so
a ``flagos`` tensor raises ``ValueError`` before a kernel is chosen even though
the device serves every operator the two implementations downstream of the check
call. Upstream's own docstring marks the set as a placeholder ("TODO: Remove
once non cuda/cpu devices support is added"), and what the check performs is a
device *name* membership test, not a capability test.

The patch relaxes that one test for the flagos device and leaves every other
check alone. It is a capability gate, not a route: which backend serves the
attention is still decided by the op-routing conf, and a platform that cannot
serve the operators still fails -- at the kernel it cannot serve, naming it,
instead of at a device-name comparison.

Measured 2026-09-26 on an H100 with the CUDA-boxing build (torch 2.10.0+cpu plus
the bundled ``libtorch_cuda.so``, FlagTree Triton 3.6.0), patched, against the
same call on CPU at B=2, H=4, S=256, D=64, fp32::

    eager (CompositeExplicitAutograd -> math_attention)
        forward  max|diff| = 9.5e-07     backward  out 7.2e-07
        grads    max|diff| = 1.78e-06 / 1.79e-06 / 1.91e-06
    fused (torch.compile, the Triton template)
        forward  max|diff| = 4.6e-03 (mask_mod only) / 5.2e-03 (score_mod)
        grads    max|diff| = 2.30e-03 / 2.52e-03 / 1.96e-03

The fused rows are ``torch.compile(flex_attention, backend="flagos")``; the
default ``torch.compile`` backend reaches the same kernel and the same error once
``torch_fl.compile.inductor_backend.pin_static_cuda_launcher()`` has run. The
fused rows' ``1e-3`` is not a defect: the generated kernel runs at
``FLOAT32_PRECISION='tf32'``, which is what ``torch.backends.cuda.matmul``
reports on both devices for this build. At bfloat16 the eager path differs on
one element out of 131072 in each case -- one bf16 ulp at magnitude 2.4
(``1.6e-02``) for the mask-only case, ``1.2e-04`` with a score_mod -- and the
fused path sits at bf16/tf32 resolution (``3.1e-02``).

Two things this deliberately does not touch, both still open and both upstream of
the fused kernel:

* ``create_block_mask`` carries no device gate of its own, so it needs no patch.
  ``transformers`` does call it with ``_compile=True``, which compiles the mask
  construction; on GCU that pass takes the process down with SIGSEGV rather than
  raising, which is why ``tests/manual/hf_flagos_shims.py`` builds the mask
  eagerly for the HF suite. That is a vendor-compiler result, not a device gate,
  and stays out of this module.
* ``_validate_device`` is the first of upstream's device assumptions, not the
  only one. A shape or dtype the fused template cannot express fails inside the
  template, with the template's own message -- which is the behaviour this patch
  restores.

Disable with ``FLAGOS_DISABLE_FLEX_ATTENTION_COMPAT=1`` to measure a platform
against upstream's gate unchanged.
"""

from __future__ import annotations

import functools
import importlib
import warnings
from typing import Any

from torch_fl import _env


_DISABLE_ENV = "FLAGOS_DISABLE_FLEX_ATTENTION_COMPAT"

_MODULE = "torch.nn.attention.flex_attention"

#: The two spellings a PrivateUse1 tensor's ``device.type`` can carry. The device
#: is registered under the backend name ``flagos``, which is what ``.type``
#: reports; ``privateuseone`` is the generic name it can still be written as.
_DEVICE_TYPES = frozenset({"flagos", "privateuseone"})

_PATCHED_ATTR = "_flagos_flex_attention_compat_patched"
_ORIGINAL_ATTR = "_flagos_flex_attention_compat_original"


def _is_disabled() -> bool:
    return _env.flag(_DISABLE_ENV)


def is_flex_attention_compat_available() -> bool:
    """Return whether the flex-attention device gate may be relaxed."""
    return not _is_disabled()


def _patch_validate_device(module: Any) -> bool:
    """Admit the flagos device to ``module._validate_device``. Idempotent.

    The wrapper delegates every other device to the function it replaced, so a
    device upstream already rejects is still rejected with upstream's own
    message, and the whitelist keeps growing with upstream rather than being
    frozen here.
    """
    original = getattr(module, "_validate_device", None)
    if original is None:
        return False
    if getattr(original, _PATCHED_ATTR, False):
        return True

    @functools.wraps(original)
    def patched(query, key, value):
        # Testing `query` alone is upstream's own contract, not a shortcut it
        # does not take: `flex_attention` calls `_validate_sdpa_input` first,
        # which raises unless query, key and value are on the identical device.
        if query.device.type in _DEVICE_TYPES:
            return
        return original(query, key, value)

    setattr(patched, _PATCHED_ATTR, True)
    setattr(patched, _ORIGINAL_ATTR, original)
    module._validate_device = patched
    return True


def patch_flex_attention() -> bool:
    """Patch the flex-attention module if it is importable.

    A missing module (an older torch, or a build without the attention package)
    leaves the normal Torch-FL path unchanged.
    """
    try:
        module = importlib.import_module(_MODULE)
    except (ImportError, ModuleNotFoundError):
        return False

    try:
        return _patch_validate_device(module)
    except (AttributeError, TypeError) as exc:
        warnings.warn(
            f"[torch_fl] flex-attention compatibility patch was skipped: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False


def install_flex_attention_compat() -> bool:
    """Install the flex-attention device-gate patch, and report whether it holds.

    Safe to call more than once. Unlike the Apex shim this installs no import
    hook and needs none: the patch target is a module-level function that
    ``flex_attention`` looks up on every call, so replacing it on the module
    object takes effect however late the module was imported -- including when
    ``transformers.masking_utils`` imports it inside the model's first forward,
    which is well after ``import torch_fl``. Importing it here is what makes the
    patch present *before* the first call rather than depending on that.
    """
    if not is_flex_attention_compat_available():
        return False
    return patch_flex_attention()


__all__ = [
    "install_flex_attention_compat",
    "is_flex_attention_compat_available",
    "patch_flex_attention",
]
