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

from torch._subclasses.fake_tensor import FakeTensor
from flag_gems.utils.tensor_wrapper import StridedBuffer


class AlignedPointer:
    """Only alignment queries are legal; the native guard enforces 16 bytes."""

    def __mod__(self, divisor):
        if isinstance(divisor, int) and divisor > 0 and 16 % divisor == 0:
            return 0
        raise RuntimeError("pointer query requires a stronger alignment guard")

    def __bool__(self):
        raise RuntimeError("raw pointer truth tests require an explicit capture rule")

    def __eq__(self, other):
        raise RuntimeError("raw pointer comparisons require an explicit capture rule")


def fake_data_ptr(self):
    return AlignedPointer()


FakeTensor.data_ptr = fake_data_ptr

_original_init = StridedBuffer.__init__
_original_ptr = StridedBuffer.data_ptr


def init_metadata(self, base, shape=None, strides=None, dtype=None, offset=0):
    if not isinstance(base, FakeTensor):
        return _original_init(self, base, shape, strides, dtype, offset)
    if offset != 0 or (dtype is not None and dtype != base.dtype):
        raise RuntimeError(
            "capture: pointer offset/reinterpret needs an explicit IR rule"
        )
    self._base = base
    self.dtype = dtype or base.dtype
    self.offset = offset
    self.shape = tuple(shape if shape is not None else base.shape)
    self._strides = tuple(strides if strides is not None else base.stride())
    self.device = base.device
    self.ndim = len(self.shape)


def data_ptr(self):
    if isinstance(self._base, FakeTensor):
        raise RuntimeError("capture cannot materialize a raw device pointer")
    return _original_ptr(self)


StridedBuffer.__init__ = init_metadata
StridedBuffer.data_ptr = data_ptr


def tensor_argument(arg):
    if isinstance(arg, StridedBuffer):
        if arg.offset != 0 or arg.dtype != arg._base.dtype:
            raise RuntimeError("capture: nontrivial pointer reinterpretation")
        return arg._base
    return arg
