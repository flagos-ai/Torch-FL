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

"""
Which compiler a flagos build actually talks to.

flagos has no Triton backend of its own, so the compile path has to name the
*hardware's* backend at two boundaries:

* ``DeviceProperties.type`` -- inductor forwards this to Triton as
  ``GPUTarget.backend`` (hints.py -> triton_heuristics.py -> triton's
  make_backend), so it must be a name the installed Triton claims to support.
* the Triton package's backend key -- ``triton.backends.backends``, used to
  check the toolchain is present before compiling.

The two are not the same string on every platform, which is why this is a table
rather than a single name. Ascend is the case that forced the split: its driver
reports ``GPUTarget(backend="npu", ...)`` and its compiler accepts only
``target.backend == "npu"``, but the Triton package registers that backend under
the key ``ascend``. Measured on a real 910:

    >>> from triton.runtime.driver import driver
    >>> driver.active.get_current_target()
    GPUTarget(backend='npu', arch='Ascend910_9382', warp_size=0)
    >>> list(triton.backends.backends)
    ['ascend']

Ascend also differs in *kind*, not just in naming: it is not a CUDA-shaped GPU
behind a shim, so its device properties, raw streams and generated device
snippets come from the ACL runtime rather than from torch.cuda. That is what
``is_cuda_like`` selects between.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformProfile:
    """How one accelerator family is presented to inductor and Triton.

    Attributes:
        triton_device_type: the name reported as ``DeviceProperties.type``, and
            therefore as ``GPUTarget.backend``. Must satisfy the vendor
            compiler's ``supports_target``.
        triton_backend_key: the key the vendor backend is registered under in
            ``triton.backends.backends``.
        is_cuda_like: whether hardware queries, raw streams and generated device
            code can go through torch.cuda / the CUDA codegen snippets. False on
            runtimes with no CUDA at all (Ascend, MUSA).
        vendor: the ``ACCELERATOR`` this profile describes, used to pick the
            vendor runtime when ``is_cuda_like`` is False and the two non-CUDA
            paths diverge (ACL streams on Ascend, musa streams on MThreads).
    """

    triton_device_type: str
    triton_backend_key: str
    is_cuda_like: bool
    vendor: str = ""


# NVIDIA and every CUDA-compatible vendor build (MetaX excepted, below): flagos
# storage *is* CUDA memory and torch_fl ships a torch.cuda shim over the same
# physical GPU, so both names are the CUDA ones.
_CUDA_PROFILE = PlatformProfile(
    triton_device_type="cuda",
    triton_backend_key="nvidia",
    is_cuda_like=True,
)

# MetaX: triton-metax's MACABackend.supports_target checks for "maca", and it is
# registered under the key "metax". Still CUDA-like underneath (boxing mode).
_METAX_PROFILE = PlatformProfile(
    triton_device_type="maca",
    triton_backend_key="metax",
    is_cuda_like=True,
)

# Ascend: AscendBackend.supports_target checks for "npu"; the package key is
# "ascend". No CUDA runtime, so nothing may route through torch.cuda.
_ASCEND_PROFILE = PlatformProfile(
    triton_device_type="npu",
    triton_backend_key="ascend",
    is_cuda_like=False,
    vendor="ascend",
)

# MThreads: FlagTree calls the target "musa" but registers the backend package
# under "mthreads". torch.cuda is deliberately unavailable on this build, so
# hardware queries go to the vendor runtime like they do on Ascend.
_MUSA_PROFILE = PlatformProfile(
    triton_device_type="musa",
    triton_backend_key="mthreads",
    is_cuda_like=False,
    vendor="musa",
)

# Enflame: triton_gcu names both the target and backend package "gcu". Like
# Ascend and MUSA it has no CUDA runtime, so device and stream queries must go
# through torch_fl's vendor runtime.
_GCU_PROFILE = PlatformProfile(
    triton_device_type="gcu",
    triton_backend_key="gcu",
    is_cuda_like=False,
    vendor="gcu",
)

_PROFILES = {
    "metax": _METAX_PROFILE,
    "ascend": _ASCEND_PROFILE,
    "musa": _MUSA_PROFILE,
    "gcu": _GCU_PROFILE,
}


def platform_profile() -> PlatformProfile:
    """Return the compile profile for the accelerator this build targets."""
    from torch_fl._build_config import ACCELERATOR

    return _PROFILES.get(ACCELERATOR, _CUDA_PROFILE)
