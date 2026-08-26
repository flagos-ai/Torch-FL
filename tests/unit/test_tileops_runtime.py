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

"""Unit coverage for TileOPs runtime fallback argument reconstruction."""

import torch

from torch_fl.tileops import runtime


def test_keyword_only_reduction_fallback_preserves_schema():
    """Declined var routes must call ATen with keyword-only keepdim."""
    impl = runtime._reduce_impl(
        object,
        (torch.float16,),
        {"keepdim": True, "correction": True},
        "var.correction",
    )
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    got = impl(x, [-1], True, correction=1)
    want = torch.ops.aten.var.correction(x, [-1], correction=1, keepdim=True)

    torch.testing.assert_close(got, want)


def test_keyword_only_reduction_fallback_handles_full_reduction():
    """Unsupported full reductions must not pass keepdim positionally."""
    impl = runtime._reduce_impl(
        object,
        (torch.float16,),
        {"keepdim": True, "correction": True},
        "std.correction",
    )
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    got = impl(x, None, True, correction=1)
    want = torch.ops.aten.std.correction(x, correction=1, keepdim=True)

    torch.testing.assert_close(got, want)
