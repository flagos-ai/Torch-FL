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

"""scaled_dot_product_attention must stay traceable under FakeTensorMode.

The flagos override is registered for PrivateUse1 and AutogradPrivateUse1,
which outrank Meta. Without a guard at its entry, Dynamo's fake pass reaches
the override with meta-backed FakeTensors, boxes them into real device tensors,
and the next op fails with "tensor's device must be `meta`, got flagos
instead". That is what blocked torch.compile on Qwen-Image-2.1.
"""

import pytest
import torch_fl  # noqa: F401  -- must precede torch: relinks the vendor libtorch
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.mark.skipif(
    torch_fl.flagos.device_count() == 0, reason="flagos device not available"
)
def test_sdpa_shape_inference_runs_on_fake_tensors():
    with FakeTensorMode():
        query = torch.randn(1, 8, 16, 64, device="flagos:0", dtype=torch.bfloat16)

        out = torch.nn.functional.scaled_dot_product_attention(query, query, query)

        assert out.shape == query.shape
        assert out.dtype == query.dtype
