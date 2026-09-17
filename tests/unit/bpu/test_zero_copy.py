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

"""The zero-copy numpy view over flagos device storage.

UCP device memory is mapped into this process, so a tensor on the flagos device
can be handed to hbm_runtime as a numpy array that *is* its storage -- no
device-to-host copy. These tests pin that property down, since it is easy to
regress into a silent `.cpu()` that still passes every correctness test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

torch_fl = pytest.importorskip("torch_fl")

from torch_fl.accelerator.bpu.runtime import _as_numpy, _device_view  # noqa: E402

on_bpu = pytest.mark.skipif(
    torch_fl._build_accelerator() != "bpu",
    reason="requires a build with FLAGOS_ACCELERATOR=bpu",
)


@on_bpu
def test_view_shares_storage_with_the_tensor():
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4).to("flagos")
    arr = _device_view(t)
    assert arr is not None
    assert arr.__array_interface__["data"][0] == t.data_ptr()
    assert arr.shape == (3, 4)


@on_bpu
def test_view_reads_the_tensors_values():
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    arr = _device_view(t.to("flagos"))
    np.testing.assert_array_equal(arr, t.numpy())


@on_bpu
def test_writes_through_the_view_are_visible_to_torch():
    """The mapping is writable, which is what would let an output land straight
    in a torch tensor's storage."""
    t = torch.zeros(4, dtype=torch.float32).to("flagos")
    arr = _device_view(t)
    arr[:] = [1.0, 2.0, 3.0, 4.0]
    torch.testing.assert_close(t.cpu(), torch.tensor([1.0, 2.0, 3.0, 4.0]))


@on_bpu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.int8, torch.uint8, torch.int32, torch.int64]
)
def test_supported_dtypes_get_a_view(dtype):
    t = torch.ones(8, dtype=dtype).to("flagos")
    arr = _device_view(t)
    assert arr is not None
    assert arr.__array_interface__["data"][0] == t.data_ptr()


@on_bpu
def test_non_contiguous_falls_back():
    """A view would misread a strided tensor, so it must be declined."""
    t = torch.randn(4, 6).to("flagos").t()
    assert not t.is_contiguous()
    assert _device_view(t) is None
    # ...but _as_numpy still produces correct data via the copy path.
    np.testing.assert_allclose(_as_numpy(t), t.cpu().numpy())


def test_cpu_tensor_falls_back_to_copy():
    t = torch.randn(3, 3)
    assert _device_view(t) is None
    arr = _as_numpy(t)
    np.testing.assert_allclose(arr, t.numpy())


def test_unsupported_dtype_falls_back():
    t = torch.randn(4, dtype=torch.complex64)
    assert _device_view(t) is None


@on_bpu
def test_as_numpy_avoids_the_copy_on_device_tensors():
    t = torch.randn(16).to("flagos")
    assert _as_numpy(t).__array_interface__["data"][0] == t.data_ptr()


@on_bpu
def test_empty_tensor_falls_back():
    """A 0-element buffer has no address worth wrapping."""
    t = torch.empty(0, dtype=torch.float32).to("flagos")
    assert _device_view(t) is None
