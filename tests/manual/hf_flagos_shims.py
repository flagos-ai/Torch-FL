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
"""Corrections for upstream assumptions that a ``flagos`` device breaks.

The runner copies this file into the child's working directory and loads it with
``-p``, so it applies to every test in the process. It sits beside
``hf_device_spec.py`` without being part of that contract: the spec is what
transformers merges into its ``BACKEND_*`` tables, while everything here
corrects behaviour that assumes an NVIDIA or Intel accelerator.

Two assumptions need correcting, both because the device is neither ``cpu`` nor
``cuda``, which are the only two names transformers branches on:

1. The eager-vs-SDPA tolerance tables are keyed on the device name, so a device
   outside the table is compared with fp32's tolerances whatever its dtype.
   Measured on GCU: fp16 is the difference between 8 failing parametrizations
   and none (flagos-ai/Torch-FL#248).
2. ``create_block_mask`` is called with ``_compile=True``, whose generated
   kernel the GCU300 target cannot legalize; the failed pass then takes the
   interpreter down with SIGSEGV instead of raising, so a single test ends the
   pytest process and every test still queued behind it loses its result
   (flagos-ai/Torch-FL#249, root cause flagos-ai/Torch-FL#400).

Each shim stops applying on its own once the assumption it corrects is no
longer true -- see the condition in each -- so none of them can outlive the
problem it works around. ``HF_TEST_NO_DEVICE_SHIMS=1`` disables all of them, for
measuring what the platform does uncorrected.
"""

import os
import sys

import torch

# transformers' fallback for a device it does not recognise: fp32 tolerances for
# every dtype (tests/test_modeling_common.py:486).
FALLBACK_TOLERANCES = (1e-7, 1e-4)

# The tolerance transformers grants a CUDA accelerator for the same dtype and
# SDPA kernel setting. This is the table it applies to hpu and npu as well,
# which is the precedent for applying it to a third non-CUDA device: a device
# family's arithmetic, not the individual operator's, is what these limits
# excuse. Keyed by (sdpa kernels enabled, dtype).
DEVICE_TOLERANCES = {
    (False, "fp32"): (1e-6, 1e-4),
    (True, "fp32"): (1e-6, 1e-4),
    (False, "fp16"): (5e-3, 5e-3),
    (True, "fp16"): (5e-3, 5e-3),
    (False, "bf16"): (1e-2, 1e-2),
    (True, "bf16"): (1e-2, 3e-2),
}


def _active() -> bool:
    """Whether these corrections apply to the device this process runs on."""
    if os.environ.get("HF_TEST_NO_DEVICE_SHIMS") == "1":
        return False
    try:
        from transformers.testing_utils import torch_device
    except Exception:
        return False
    # The name is resolved by the device spec, so this is the authoritative
    # answer to "is this a flagos run", not an environment variable the runner
    # could set for another platform by accident.
    return torch_device == "flagos"


def _patch_block_mask() -> None:
    """Build the block mask eagerly instead of compiling it on GCU300."""
    import torch.nn.attention.flex_attention as flex

    if getattr(flex.create_block_mask, "_flagos_shim", False):
        shim = flex.create_block_mask
    else:
        original = flex.create_block_mask

        def shim(*args, **kwargs):
            kwargs["_compile"] = False
            return original(*args, **kwargs)

        shim._flagos_shim = True
        flex.create_block_mask = shim

    # transformers binds the name when masking_utils is first imported
    # (masking_utils.py:31-33), which is inside the model's forward, so a module
    # imported earlier in the session still holds the unpatched function. That
    # import may not have happened yet when this first runs, which is why the
    # callers re-invoke this after collection.
    module = sys.modules.get("transformers.masking_utils")
    if module is not None and not getattr(
        module.create_block_mask, "_flagos_shim", False
    ):
        module.create_block_mask = shim


def _patch_sdpa_tolerances() -> None:
    """Hold this device to its own dtype's tolerance, not to fp32's."""
    try:
        from tests import test_modeling_common as common
    except ImportError:
        return

    if getattr(common._test_eager_matches_sdpa_inference, "_flagos_shim", False):
        return
    original = common._test_eager_matches_sdpa_inference

    def wrapper(
        self,
        name,
        dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
        atols=None,
        rtols=None,
    ):
        # Only the call that brought no tolerances of its own is the one that
        # fell through to the unknown-device branch. A test that passes its own
        # limits keeps them, and if transformers ever lists this device the way
        # it lists hpu and npu the fall-through stops happening and this becomes
        # a pass-through with nothing to remove.
        row = None
        if atols is None and rtols is None:
            row = DEVICE_TOLERANCES.get((enable_kernels, dtype))
        if row is None:
            return original(
                self,
                name,
                dtype,
                padding_side,
                use_attention_mask,
                output_attentions,
                enable_kernels,
                atols,
                rtols,
            )

        atol, rtol = row
        real_allclose = torch.allclose

        def allclose(a, b, *args, **kwargs):
            # Widen only. The helper is the only caller on this path, and it
            # performs a single comparison; `max` preserves the magnitude-based
            # bf16 guard it raises atol with for large outputs.
            kwargs["atol"] = max(atol, kwargs.get("atol", FALLBACK_TOLERANCES[0]))
            kwargs["rtol"] = max(rtol, kwargs.get("rtol", FALLBACK_TOLERANCES[1]))
            return real_allclose(a, b, *args, **kwargs)

        torch.allclose = allclose
        try:
            return original(
                self,
                name,
                dtype,
                padding_side,
                use_attention_mask,
                output_attentions,
                enable_kernels,
                atols,
                rtols,
            )
        finally:
            torch.allclose = real_allclose

    wrapper._flagos_shim = True
    common._test_eager_matches_sdpa_inference = wrapper


def _install() -> None:
    if not _active():
        return
    _patch_block_mask()
    _patch_sdpa_tolerances()


def pytest_configure(config) -> None:
    _install()


def pytest_runtest_setup(item) -> None:
    # transformers.masking_utils is imported by the model under test, so the
    # block-mask patch can only reach its namespace once collection is done.
    _install()
