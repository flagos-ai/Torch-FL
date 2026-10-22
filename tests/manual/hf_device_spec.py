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
"""HuggingFace test device specification for the ``flagos`` accelerator.

``transformers.testing_utils`` imports this module when
``TRANSFORMERS_TEST_DEVICE_SPEC`` points at it, and merges the hooks below into
its ``BACKEND_*`` dispatch tables. That is the whole custom-device contract: HF
validates only that ``DEVICE_NAME`` is constructible by ``torch.device``, so no
patch of the official tests is needed.

Importing ``torch_fl`` here is what registers the device; without it
``torch.flagos`` does not exist and HF raises while loading the spec.

The gating split this produces is deliberate. ``require_torch_accelerator``
passes because the device is neither CPU nor None, so device-agnostic tests run
on the accelerator. ``require_torch_gpu`` compares against ``cuda`` literally
and therefore skips, which is correct: those cases are CUDA-specific and are not
coverage gaps for this platform.
"""

import torch

import torch_fl  # noqa: F401 - registers the flagos device on torch

DEVICE_NAME = "flagos"

MANUAL_SEED_FN = torch.flagos.manual_seed
EMPTY_CACHE_FN = torch.flagos.empty_cache
DEVICE_COUNT_FN = torch.flagos.device_count
