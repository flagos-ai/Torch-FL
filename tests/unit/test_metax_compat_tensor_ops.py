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

"""Unit coverage for the MetaX `zero_`/`fill_` guard.

`_patch_cuda_tensor_ops` replaces `Tensor.zero_` and `Tensor.fill_` with
stand-ins that hand a byte range to the MetaX runtime's memset, and both of them
ask `is_contiguous()` on the way. A sparse tensor raises out of that question,
so a sparse tensor has to be sent to stock torch instead of into the memset
path. The guard that decides it landed as a device check only in #539, which is
why the two `TestBufferMovement` cases in
`tests/integration/ops/test_sparse_csr_dispatch.py` passed before it on a flagos
device and raised after it:

    RuntimeError: Sparse CSR tensors do not have is_contiguous

No MetaX hardware is needed to pin the decision: the guard reads three
attributes (`layout`, `is_cuda`, `device.type`), so a stub carries the whole
truth table, and a real sparse tensor confirms that the `layout` half is the one
that keeps a sparse operand out. What no test here can check is the memset
itself -- that needs the device.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "torch_fl" / "accelerator" / "metax" / "_metax_compat.py"

_spec = importlib.util.spec_from_file_location("metax_compat_tensor_ops", MODULE)
metax = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metax)


class _StubTensor:
    """The three attributes `_is_metax_device_tensor` reads.

    The device is a stub rather than a real `torch.device` because the guard
    compares `.type` against "flagos", which is a torch_fl alias for
    PrivateUse1 and not a string a CPU-only torch build will parse.
    """

    class _Device:
        def __init__(self, type):
            self.type = type

    def __init__(self, layout=torch.strided, is_cuda=False, device_type="flagos"):
        self.layout = layout
        self.is_cuda = is_cuda
        self.device = self._Device(device_type)


@pytest.mark.parametrize(
    ("layout", "is_cuda", "device_type", "expected"),
    [
        # The device half: a cuda tensor and a flagos/privateuseone tensor are
        # both served, whichever way `is_cuda` answers.
        (torch.strided, True, "cuda", True),
        (torch.strided, False, "flagos", True),
        (torch.strided, False, "privateuseone", True),
        # Anything else is stock torch's: `is_cuda` decides a dense tensor, so a
        # tensor that answers neither way stays with stock torch's `zero_`.
        (torch.strided, False, "cpu", False),
        # The layout half, which is what #539 dropped: a sparse operand on a
        # flagos device used to fall through on `is_cuda` alone, and the memset
        # path is the one place it must not land.
        (torch.sparse_csr, False, "flagos", False),
        (torch.sparse_csr, True, "cuda", False),
        (torch.sparse_coo, False, "privateuseone", False),
    ],
)
def test_guard_truth_table(layout, is_cuda, device_type, expected):
    tensor = _StubTensor(layout=layout, is_cuda=is_cuda, device_type=device_type)
    assert metax._is_metax_device_tensor(tensor) is expected


def test_a_real_sparse_tensor_is_kept_out_of_the_memset_path():
    # The stub above asserts the rule; this asserts that a sparse tensor really
    # reports the layout the rule keys off, so the two cannot drift apart.
    csr = torch.sparse_csr_tensor(
        torch.tensor([0, 1, 2]), torch.tensor([0, 1]), torch.tensor([1.0, 2.0])
    )
    assert csr.layout == torch.sparse_csr
    assert not metax._is_metax_device_tensor(csr)
    # And that stock torch zeroes it rather than raises. Zeroing a sparse CSR
    # empties the structure -- nnz drops to zero, which is what
    # `TestBufferMovement::test_zero_empties_the_structure` asserts.
    assert csr.zero_() is not None
    assert csr._nnz() == 0


def test_a_dense_strided_tensor_that_is_not_ours_is_kept_out_too():
    # A CPU tensor is not MetaX's, so `zero_` on it stays byte-for-byte the
    # stock implementation; this is the path every non-MetaX host takes.
    tensor = torch.ones(4, 4)
    assert not metax._is_metax_device_tensor(tensor)
    tensor.zero_()
    assert torch.equal(tensor, torch.zeros(4, 4))
